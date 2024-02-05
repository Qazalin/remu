use crate::alu_modifiers::VOPModifier;
use crate::dtype::IEEEClass;
use crate::memory::VecDataStore;
use crate::state::{Register, Value, WaveValue, VGPR};
use crate::todo_instr;
use crate::utils::{as_signed, f16_hi, f16_lo, nth, Colorize, DebugLevel, DEBUG, END_PRG};
use half::f16;
use num_traits::Float;

pub const SGPR_COUNT: usize = 105;
pub const VGPR_COUNT: usize = 256;
const NULL_SRC: u32 = 124;

pub struct CPU<'a> {
    pub scalar_reg: &'a mut Vec<u32>,
    pub scc: &'a mut u32,

    pub vec_reg: &'a mut VGPR,
    pub vcc: &'a mut WaveValue,
    pub exec: &'a mut WaveValue,

    pub lds: &'a mut VecDataStore,
    pub sds: &'a mut VecDataStore,

    pub pc: u64,
    pub prg: Vec<u32>,
    pub simm: Option<u32>,
}

impl<'a> CPU<'a> {
    pub fn interpret(&mut self, prg: &Vec<u32>) {
        self.pc = 0;
        self.prg = prg.to_vec();

        loop {
            let instruction = prg[self.pc as usize];
            self.pc += 1;

            if instruction == END_PRG {
                break;
            }
            if instruction == 0xbfb60003 || instruction >> 20 == 0xbf8 {
                continue;
            }

            self.exec(instruction);
            self.simm = None
        }
    }

    fn u64_instr(&mut self) -> u64 {
        let msb = self.prg[self.pc as usize] as u64;
        let instr = msb << 32 | self.prg[self.pc as usize - 1] as u64;
        self.pc += 1;
        return instr;
    }

    fn exec(&mut self, instruction: u32) {
        if (instruction >> 25 == 0b0111111
            || instruction >> 26 == 0b110010
            || instruction >> 25 == 0b0111110
            || instruction >> 31 == 0b0
            || instruction >> 26 == 0b110101)
            && !self.exec.read()
        {
            return;
        }
        // smem
        if instruction >> 26 == 0b111101 {
            let instr = self.u64_instr();
            /* addr: s[sbase:sbase+1] */
            let sbase = (instr & 0x3f) * 2;
            let sdata = ((instr >> 6) & 0x7f) as usize;
            let op = (instr >> 18) & 0xff;
            let offset = as_signed((instr >> 32) & 0x1fffff, 21);
            let soffset = match self.val(((instr >> 57) & 0x7f) as usize) {
                NULL_SRC => 0,
                val => val,
            };

            if *DEBUG >= DebugLevel::INSTRUCTION {
                println!(
                    "{} sbase={sbase} sdata={sdata} op={op} offset={offset} soffset={soffset}",
                    "SMEM".color("blue"),
                );
            }
            let base_addr = self.scalar_reg.read64(sbase as usize);
            let addr = (base_addr as i64 + offset + soffset as i64) as u64;

            match op {
                0..=4 => (0..2_usize.pow(op as u32)).for_each(|i| unsafe {
                    self.scalar_reg[sdata + i] = *((addr + (4 * i as u64)) as *const u32);
                }),
                _ => todo_instr!(instruction),
            }
        }
        // sop1
        else if instruction >> 23 == 0b10_1111101 {
            let src = (instruction & 0xFF) as usize;
            let op = (instruction >> 8) & 0xFF;
            let sdst = (instruction >> 16) & 0x7F;

            if *DEBUG >= DebugLevel::INSTRUCTION {
                println!("{} src={src} sdst={sdst} op={op}", "SOP1".color("blue"));
            }

            match op {
                1 => {
                    let s0 = self.val(src);
                    let ret = match op {
                        1 => s0,
                        _ => panic!(),
                    };
                    self.scalar_reg.write64(sdst as usize, ret);
                }
                _ => {
                    let s0 = self.val(src);
                    let ret = match op {
                        0 => s0,
                        10 => self.clz_i32_u32(s0),
                        12 => self.cls_i32(s0),
                        14 => s0 as i8 as i32 as u32,
                        16 | 18 => {
                            let sdst: u32 = self.val(sdst as usize);
                            if op == 16 {
                                sdst & !(1 << (s0 & 0x1f))
                            } else {
                                sdst | (1 << (s0 & 0x1f))
                            }
                        }
                        30 => {
                            let ret = !s0;
                            *self.scc = (ret != 0) as u32;
                            ret
                        }
                        32 | 34 | 48 => {
                            let saveexec = self.exec.read() as u32;
                            self.exec.value = match op {
                                32 => s0 & saveexec,
                                34 => s0 | saveexec,
                                48 => s0 & !saveexec,
                                _ => panic!(),
                            };
                            *self.scc = self.exec.read() as u32;
                            saveexec
                        }
                        _ => todo_instr!(instruction),
                    };

                    self.write_to_sdst(sdst, ret);
                }
            };
        }
        // sopc
        else if (instruction >> 23) & 0x3ff == 0b101111110 {
            let s0 = (instruction & 0xff) as usize;
            let s1 = ((instruction >> 8) & 0xff) as usize;
            let op = (instruction >> 16) & 0x7f;

            if *DEBUG >= DebugLevel::INSTRUCTION {
                println!("{} s0={s0} ssrc1={s1} op={op}", "SOPC".color("blue"));
            }

            fn scmp<T>(s0: T, s1: T, offset: u32, op: u32) -> bool
            where
                T: PartialOrd + PartialEq,
            {
                match op - offset {
                    0 => s0 == s1,
                    1 => s0 != s1,
                    2 => s0 > s1,
                    3 => s0 >= s1,
                    4 => s0 < s1,
                    _ => s0 <= s1,
                }
            }
            *self.scc = match op {
                0..=5 => {
                    let (s0, s1): (u32, u32) = (self.val(s0), self.val(s1));
                    scmp(s0 as i32, s1 as i32, 0, op)
                }
                6..=11 => {
                    let (s0, s1): (u32, u32) = (self.val(s0), self.val(s1));
                    scmp(s0, s1, 6, op)
                }
                16 | 17 => {
                    let (s0, s1): (u64, u64) = (self.val(s0), self.val(s1));
                    scmp(s0, s1, 16, op)
                }
                _ => todo_instr!(instruction),
            } as u32;
        }
        // sopp
        else if instruction >> 23 == 0b10_1111111 {
            let simm16 = (instruction & 0xffff) as i16;
            let op = (instruction >> 16) & 0x7f;
            if *DEBUG >= DebugLevel::INSTRUCTION {
                println!(
                    "{} simm16={simm16} op={op} pc={}",
                    "SOPP".color("blue"),
                    self.pc,
                );
            }

            match op {
                32..=42 => {
                    let should_jump = match op {
                        32 => true,
                        33 => *self.scc == 0,
                        34 => *self.scc == 1,
                        35 => self.vcc.value == 0,
                        36 => self.vcc.value != 0,
                        37 => self.exec.value == 0,
                        38 => self.exec.value != 0,
                        _ => todo_instr!(instruction),
                    };
                    if should_jump {
                        self.pc = (self.pc as i64 + simm16 as i64) as u64;
                    }
                }
                _ => todo_instr!(instruction),
            }
        }
        // sopk
        else if instruction >> 28 == 0b1011 {
            let simm = instruction & 0xffff;
            let sdst = ((instruction >> 16) & 0x7f) as usize;
            let op = (instruction >> 23) & 0x1f;
            let s0: u32 = self.val(sdst);

            if *DEBUG >= DebugLevel::INSTRUCTION {
                println!(
                    "{} simm={simm} sdst={sdst} s0={s0} op={op}",
                    "SOPK".color("blue"),
                );
            }

            match op {
                0 => self.write_to_sdst(sdst as u32, simm as i16 as i32 as u32),
                3..=8 => {
                    let s1 = simm as i16 as i64;
                    let s0 = s0 as i32 as i64;
                    *self.scc = match op {
                        3 => s0 == s1,
                        4 => s0 != s1,
                        5 => s0 > s1,
                        7 => s0 < s1,
                        _ => todo_instr!(instruction),
                    } as u32
                }
                9..=14 => {
                    let s1 = simm as u16 as u32;
                    *self.scc = match op {
                        9 => s0 == s1,
                        13 => s0 < s1,
                        _ => todo_instr!(instruction),
                    } as u32
                }
                15 => {
                    let temp = s0 as i32;
                    let simm16 = simm as i16;
                    let dest = (temp as i64 + simm16 as i64) as i32;
                    self.write_to_sdst(sdst as u32, dest as u32);
                    let temp_sign = ((temp >> 31) & 1) as u32;
                    let simm_sign = ((simm16 >> 15) & 1) as u32;
                    let dest_sign = ((dest >> 31) & 1) as u32;
                    *self.scc = ((temp_sign == simm_sign) && (temp_sign != dest_sign)) as u32;
                }
                16 => {
                    let simm16 = simm as i16;
                    let ret = (s0 as i32 * simm16 as i32) as u32;
                    self.write_to_sdst(sdst as u32, ret);
                }
                _ => todo_instr!(instruction),
            }
        }
        // sop2
        else if instruction >> 30 == 0b10 {
            let s0 = (instruction & 0xFF) as usize;
            let s1 = ((instruction >> 8) & 0xFF) as usize;
            let sdst = (instruction >> 16) & 0x7F;
            let op = (instruction >> 23) & 0xFF;

            if *DEBUG >= DebugLevel::INSTRUCTION {
                println!(
                    "{} s0={s0} s1={s1} sdst={sdst} op={op}",
                    "SOP2".color("blue"),
                );
            }

            match op {
                27 => {
                    let (s0, s1): (u64, u64) = (self.val(s0), self.val(s1));
                    let ret = s0 ^ s1;
                    self.scalar_reg.write64(sdst as usize, ret);
                    *self.scc = (ret != 0) as u32;
                }
                9 | 13 | 40 | 41 => {
                    let (s0, s1): (u64, u32) = (self.val(s0), self.val(s1));
                    let ret = match op {
                        9 => {
                            let ret = s0 << (s1 & 0x3f);
                            (ret, Some(ret != 0))
                        }
                        13 => {
                            let ret = (s0 as i64) >> (s1 & 0x3f);
                            (ret as u64, Some(ret != 0))
                        }
                        40 => {
                            let ret = (s0 >> (s1 & 0x3f)) & ((1 << ((s1 >> 16) & 0x7f)) - 1);
                            (ret as u64, Some(ret != 0))
                        }
                        41 => {
                            let s0 = s0 as i64;
                            let mut ret = (s0 >> (s1 & 0x3f)) & ((1 << ((s1 >> 16) & 0x7f)) - 1);
                            let shift = 64 - ((s1 >> 16) & 0x7f);
                            ret = (ret << shift) >> shift;
                            (ret as u64, Some(ret != 0))
                        }
                        _ => panic!(),
                    };
                    self.scalar_reg.write64(sdst as usize, ret.0);
                    if let Some(val) = ret.1 {
                        *self.scc = val as u32
                    }
                }
                _ => {
                    let (s0, s1): (u32, u32) = (self.val(s0), self.val(s1));
                    let ret = match op {
                        0 | 4 => {
                            let (s0, s1) = (s0 as u64, s1 as u64);
                            let ret = match op {
                                0 => s0 + s1,
                                4 => s0 + s1 + *self.scc as u64,
                                _ => panic!(),
                            };
                            (ret as u32, Some(ret >= 0x100000000))
                        }
                        1 => (s0 - s1, Some(s1 > s0)),
                        5 => (
                            s0 - s1 - *self.scc,
                            Some((s1 as u64 + *self.scc as u64) > s0 as u64),
                        ),
                        2 | 3 => {
                            let s0 = s0 as i32 as i64;
                            let s1 = s1 as i32 as i64;
                            let ret = match op {
                                2 => s0 + s1,
                                3 => s0 - s1,
                                _ => panic!(),
                            };
                            let overflow = (nth(s0 as u32, 31) == nth(s1 as u32, 31))
                                && (nth(s0 as u32, 31) != nth(ret as u32, 31));

                            (ret as i32 as u32, Some(overflow))
                        }
                        (8..=17) => {
                            let s1 = s1 & 0x1f;
                            let ret = match op {
                                8 => s0 << s1,
                                10 => s0 >> s1,
                                12 => ((s0 as i32) >> (s1 as i32)) as u32,
                                _ => todo_instr!(instruction),
                            };
                            (ret, Some(ret != 0))
                        }
                        (18..=21) => {
                            let scc = match op {
                                18 => (s0 as i32) < (s1 as i32),
                                19 => s0 < s1,
                                20 => (s0 as i32) > (s1 as i32),
                                _ => panic!(),
                            };
                            let ret = match scc {
                                true => s0,
                                false => s1,
                            };
                            (ret, Some(scc))
                        }
                        (22..=26) | 34 | 36 => {
                            let ret = match op {
                                22 => s0 & s1,
                                24 => s0 | s1,
                                26 => s0 ^ s1,
                                34 => s0 & !s1,
                                36 => s0 | !s1,
                                _ => panic!(),
                            };
                            (ret, Some(ret != 0))
                        }
                        38 => {
                            let ret = (s0 >> (s1 & 0x1f)) & ((1 << ((s1 >> 16) & 0x7f)) - 1);
                            (ret, Some(ret != 0))
                        }
                        39 => {
                            let s0 = s0 as i32;
                            let mut ret = (s0 >> (s1 & 0x1f)) & ((1 << ((s1 >> 16) & 0x1f)) - 1);
                            let shift = 32 - ((s1 >> 16) & 0x7f);
                            ret = (ret << shift) >> shift;
                            (ret as u32, Some(ret != 0))
                        }
                        44 => (((s0 as i32) * (s1 as i32)) as u32, None),
                        45 => (((s0 as u64) * (s1 as u64) >> 32) as u32, None),
                        46 => (
                            (((s0 as i32 as i64 * s1 as i32 as i64) as u64) >> 32u64) as i32 as u32,
                            None,
                        ),
                        48 => match *self.scc != 0 {
                            true => (s0, None),
                            false => (s1, None),
                        },
                        _ => todo_instr!(instruction),
                    };

                    self.write_to_sdst(sdst, ret.0);
                    if let Some(val) = ret.1 {
                        *self.scc = val as u32
                    }
                }
            }
        }
        // vopp
        else if instruction >> 24 == 0b11001100 {
            let instr = self.u64_instr();
            let vdst = (instr & 0xff) as usize;
            let clmp = (instr >> 15) & 0x1;
            assert_eq!([clmp], [0]);
            let op = (instr >> 16) & 0x7f;

            let mut src = |x: usize| -> (u16, u16, u32) {
                let val: u32 = self.val(x);
                match x {
                    255 => {
                        let val_lo: u16 = self.val(x);
                        (val_lo, val_lo, val)
                    }
                    (240..=247) => {
                        let val_lo: u16 = self.val(x);
                        (val_lo, f16::from_bits(0).to_bits(), val)
                    }
                    _ => ((val & 0xffff) as u16, ((val >> 16) & 0xffff) as u16, val),
                }
            };

            let src_fields = [32, 41, 50]
                .iter()
                .map(|x| ((instr >> x) & 0x1ff) as usize)
                .collect::<Vec<_>>();
            let src_parts = src_fields.iter().map(|x| src(*x)).collect::<Vec<_>>();

            let b = |i: usize| (instr >> i) & 0x1 != 0;
            let neg_hi = ((instr >> 8) & 0x7) as usize;
            let neg = ((instr >> 61) & 0x7) as usize;
            let opsel = [b(11), b(12), b(13)];
            let opsel_hi = [b(59), b(60), b(14)];
            if *DEBUG >= DebugLevel::INSTRUCTION {
                println!("{} op={op} vdst={vdst} src2={:?} opsel={:?} opsel_hi={:?} neg={:03b} neg_hi={:03b}", "VOPP".color("blue"), src_parts, opsel, opsel_hi, neg, neg_hi);
            }

            match op {
                0..=18 => {
                    let fxn = |x, y, z| match op {
                        1 => x * y,
                        4 => y << (x & 0xf),
                        10 => x + y,
                        9 => x * y + z,
                        11 => x - y,
                        _ => {
                            let (x, y, z) =
                                (f16::from_bits(x), f16::from_bits(y), f16::from_bits(z));
                            match op {
                                14 => f16::mul_add(x, y, z),
                                15 => x + y,
                                16 => x * y,
                                _ => todo_instr!(instruction),
                            }
                            .to_bits()
                        }
                    };
                    let src = |opsel: [bool; 3]| {
                        opsel
                            .iter()
                            .enumerate()
                            .map(|(i, sel)| {
                                if (14..=19).contains(&op) {
                                    let half = |x, n| f16::from_bits(x).negate(i, n).to_bits();
                                    match sel {
                                        true => half(src_parts[i].1, neg),
                                        false => half(src_parts[i].0, neg_hi),
                                    }
                                } else {
                                    match sel {
                                        true => src_parts[i].1,
                                        false => src_parts[i].0,
                                    }
                                }
                            })
                            .collect::<Vec<u16>>()
                    };
                    let (src_hi, src_lo) = (src(opsel_hi), src(opsel));
                    let ret = ((fxn(src_hi[0], src_hi[1], src_hi[2]) as u32) << 16)
                        | (fxn(src_lo[0], src_lo[1], src_lo[2]) as u32);
                    self.vec_reg[vdst] = ret;
                }
                32..=34 => {
                    let src: Vec<f32> = src_parts
                        .iter()
                        .enumerate()
                        .map(|(i, (lo, hi, full))| {
                            if !opsel_hi[i] {
                                f32::from_bits(*full).absolute(i, neg_hi)
                            } else if opsel[i] {
                                f32::from(f16::from_bits(*hi)).absolute(i, neg_hi)
                            } else {
                                f32::from(f16::from_bits(*lo)).absolute(i, neg_hi)
                            }
                        })
                        .collect();
                    let ret = match op {
                        32 => f32::mul_add(src[0], src[1], src[2]).to_bits(),
                        33 | 34 => {
                            let ret = f16::from_f32(f32::mul_add(src[0], src[1], src[2])).to_bits();
                            match op {
                                33 => (self.vec_reg[vdst] & 0xffff0000) | (ret as u32),
                                34 => (self.vec_reg[vdst] & 0x0000ffff) | ((ret as u32) << 16),
                                _ => panic!(),
                            }
                        }
                        _ => todo_instr!(instruction),
                    };
                    self.vec_reg[vdst] = ret;
                }
                _ => todo_instr!(instruction),
            }
        }
        // vop1
        else if instruction >> 25 == 0b0111111 {
            let s0 = (instruction & 0x1ff) as usize;
            let op = (instruction >> 9) & 0xff;
            let vdst = ((instruction >> 17) & 0xff) as usize;

            if *DEBUG >= DebugLevel::INSTRUCTION {
                println!("{} src={s0} op={op} vdst={vdst}", "VOP1".color("blue"),);
            }

            match op {
                3 | 15 | 21 | 23 | 26 => {
                    let s0: u64 = self.val(s0);
                    match op {
                        3 | 15 | 21 | 23 | 26 => {
                            let s0 = f64::from_bits(s0);
                            match op {
                                23 | 26 => {
                                    let ret = match op {
                                        23 => f64::trunc(s0),
                                        26 => f64::floor(s0),
                                        _ => panic!(),
                                    };
                                    self.vec_reg.write64(vdst, ret.to_bits())
                                }
                                _ => {
                                    self.vec_reg[vdst] = match op {
                                        3 => s0 as i32 as u32,
                                        15 => (s0 as f32).to_bits(),
                                        21 => s0 as u32,
                                        _ => panic!(),
                                    };
                                }
                            }
                        }
                        _ => panic!(),
                    }
                }
                84..=97 => {
                    let s0 = f16::from_bits(self.val(s0));
                    let ret = match op {
                        85 => f16::sqrt(s0),
                        87 => f16::log2(s0),
                        88 => f16::exp2(s0),
                        _ => todo_instr!(instruction),
                    };
                    self.vec_reg[vdst] = ret.to_bits() as u32;
                }
                _ => {
                    let s0: u32 = self.val(s0);
                    match op {
                        4 | 16 | 22 => {
                            let ret = match op {
                                4 => (s0 as i32 as f64).to_bits(),
                                22 => (s0 as f64).to_bits(),
                                16 => (f32::from_bits(s0) as f64).to_bits(),
                                _ => panic!(),
                            };
                            self.vec_reg.write64(vdst, ret)
                        }
                        2 => {
                            assert!(self.exec.read());
                            self.scalar_reg[vdst] = s0;
                        }
                        _ => {
                            self.vec_reg[vdst] = match op {
                                1 => s0,
                                5 => (s0 as i32 as f32).to_bits(),
                                6 => (s0 as f32).to_bits(),
                                7 => f32::from_bits(s0) as u32,
                                8 => f32::from_bits(s0) as i32 as u32,
                                10 => f16::from_f32(f32::from_bits(s0)).to_bits() as u32,
                                11 => f32::from(f16::from_bits(s0 as u16)).to_bits(),
                                17 => ((s0 & 0xff) as f32).to_bits(),
                                18 => (((s0 >> 8) & 0xff) as f32).to_bits(),
                                19 => (((s0 >> 16) & 0xff) as f32).to_bits(),
                                20 => (((s0 >> 24) & 0xff) as f32).to_bits(),
                                56 => s0.reverse_bits(),
                                57 => self.clz_i32_u32(s0),
                                35..=51 => {
                                    let s0 = f32::from_bits(s0);
                                    match op {
                                        35 => {
                                            let mut temp = f32::floor(s0 + 0.5);
                                            if f32::floor(s0) % 2.0 != 0.0 && f32::fract(s0) == 0.5
                                            {
                                                temp -= 1.0;
                                            }
                                            temp
                                        }
                                        37 => f32::exp2(s0),
                                        39 => f32::log2(s0),
                                        42 => 1.0 / s0,
                                        43 => 1.0 / s0,
                                        51 => f32::sqrt(s0),
                                        _ => panic!(),
                                    }
                                    .to_bits()
                                }
                                55 => !s0,
                                59 => self.cls_i32(s0),
                                80 => f16::from_f32(s0 as u16 as f32).to_bits() as u32,
                                81 => f16::from_f32(s0 as i16 as u16 as f32).to_bits() as u32,
                                82 => f32::from(f16::from_bits(s0 as u16)) as i16 as u32,
                                83 => f32::from(f16::from_bits(s0 as u16)) as u32,
                                _ => todo_instr!(instruction),
                            }
                        }
                    }
                }
            }
        }
        // vopd
        else if instruction >> 26 == 0b110010 {
            let instr = self.u64_instr();
            let sx = instr & 0x1ff;
            let vx = (instr >> 9) & 0xff;
            let srcx0 = self.val(sx as usize);
            let vsrcx1 = self.vec_reg[(vx) as usize] as u32;
            let opy = (instr >> 17) & 0x1f;
            let sy = (instr >> 32) & 0x1ff;
            let vy = (instr >> 41) & 0xff;
            let opx = (instr >> 22) & 0xf;
            let srcy0 = match sy {
                255 => match sx {
                    255 => srcx0,
                    _ => self.val(sy as usize),
                },
                _ => self.val(sy as usize),
            };
            let vsrcy1 = self.vec_reg[(vy) as usize];

            let vdstx = ((instr >> 56) & 0xff) as usize;
            // LSB is the opposite of VDSTX[0]
            let vdsty = (((instr >> 49) & 0x7f) << 1 | ((vdstx as u64 & 1) ^ 1)) as usize;

            if *DEBUG >= DebugLevel::INSTRUCTION {
                println!(
                    "{} X=[op={opx}, dest={vdstx} src({sx})={srcx0}, vsrc({vx})={vsrcx1}] Y=[op={opy}, dest={vdsty}, src({sy})={srcy0}, vsrc({vy})={vsrcy1}]",
                    "VOPD".color("blue"),
                );
            }

            ([(opx, srcx0, vsrcx1, vdstx), (opy, srcy0, vsrcy1, vdsty)])
                .iter()
                .for_each(|(op, s0, s1, dst)| {
                    self.vec_reg[*dst] = match *op {
                        0 | 1 | 2 | 3 | 4 | 5 | 6 | 10 => {
                            let s0 = f32::from_bits(*s0 as u32);
                            let s1 = f32::from_bits(*s1 as u32);
                            match *op {
                                0 => f32::mul_add(s0, s1, f32::from_bits(self.vec_reg[*dst])),
                                1 => f32::mul_add(s0, s1, f32::from_bits(self.simm())),
                                2 => f32::mul_add(s0, f32::from_bits(self.simm()), s1),
                                3 => s0 * s1,
                                4 => s0 + s1,
                                5 => s0 - s1,
                                6 => s1 - s0,
                                10 => f32::max(s0, s1),
                                _ => panic!(),
                            }
                            .to_bits()
                        }
                        _ => match op {
                            8 => *s0,
                            9 => match self.vcc.read() {
                                true => *s1,
                                false => *s0,
                            },
                            16 => s0 + s1,
                            17 => s1 << s0,
                            18 => s0 & s1,
                            _ => todo_instr!(instruction),
                        },
                    }
                });
        }
        // vopc
        else if instruction >> 25 == 0b0111110 {
            let s0 = (instruction & 0x1ff) as usize;
            let s1 = ((instruction >> 9) & 0xff) as usize;
            let op = (instruction >> 17) & 0xff;

            if *DEBUG >= DebugLevel::INSTRUCTION {
                println!("{} src={:?} op={}", "VOPC".color("blue"), (s0, s1), op);
            }

            let dest_offset = if op >= 128 { 128 } else { 0 };
            let ret = match op {
                (0..=15) | 125 | (128..=143) => {
                    let s0 = f16::from_bits(self.val(s0));
                    let s1 = f16::from_bits(self.vec_reg[s1] as u16);
                    match op {
                        125 => self.cmp_class_f16(s0, s1.to_bits()),
                        _ => self.cmpf(s0, s1, op - dest_offset),
                    }
                }
                (16..=31) | 126 | (144..=159) => {
                    let s0 = f32::from_bits(self.val(s0));
                    let s1 = f32::from_bits(self.vec_reg[s1]);
                    match op {
                        126 => self.cmp_class_f32(s0, s1.to_bits()),
                        _ => self.cmpf(s0, s1, op - 16 - dest_offset),
                    }
                }
                (32..=47) | (160..=174) => {
                    let s0 = f64::from_bits(self.val(s0));
                    let s1 = f64::from_bits(self.vec_reg.read64(s1));
                    self.cmpf(s0, s1, op - 32 - dest_offset)
                }
                (49..=54) | (177..=182) => {
                    let (s0, s1): (u16, u16) = (self.val(s0), self.vec_reg[s1] as u16);
                    self.cmpi(s0 as i16, s1 as i16, op - 48 - dest_offset)
                }
                (57..=62) | (185..=190) => {
                    let (s0, s1): (u16, u16) = (self.val(s0), self.vec_reg[s1] as u16);
                    self.cmpi(s0, s1, op - 56 - dest_offset)
                }
                (64..=71) | (192..=199) => {
                    let (s0, s1): (u32, u32) = (self.val(s0), self.vec_reg[s1]);
                    self.cmpi(s0 as i32, s1 as i32, op - 64 - dest_offset)
                }
                (72..=79) | (200..=207) => {
                    let (s0, s1): (u32, u32) = (self.val(s0), self.vec_reg[s1]);
                    self.cmpi(s0, s1, op - 72 - dest_offset)
                }
                (80..=87) | (208..=215) => {
                    let (s0, s1): (u64, u64) = (self.val(s0), self.vec_reg.read64(s1));
                    self.cmpi(s0 as i64, s1 as i64, op - 80 - dest_offset)
                }
                (88..=95) | (216..=223) => {
                    let (s0, s1): (u64, u64) = (self.val(s0), self.vec_reg.read64(s1));
                    self.cmpi(s0, s1, op - 88 - dest_offset)
                }
                _ => todo_instr!(instruction),
            };

            match op >= 128 {
                true => self.exec.mut_lane(ret),
                false => self.vcc.mut_lane(ret),
            };
        }
        // vop2
        else if instruction >> 31 == 0b0 {
            let s0 = (instruction & 0x1FF) as usize;
            let s1 = self.vec_reg[((instruction >> 9) & 0xFF) as usize];
            let vdst = ((instruction >> 17) & 0xFF) as usize;
            let op = (instruction >> 25) & 0x3F;

            if *DEBUG >= DebugLevel::INSTRUCTION {
                println!(
                    "{} s0={s0} s1={s1} vdst={vdst} op={op}",
                    "VOP2".color("blue"),
                );
            }

            match op {
                54 | 56 => {
                    let (s0, s1) = (f16::from_bits(self.val(s0)), f16::from_bits(s1 as u16));
                    let ret = match op {
                        54 => f16::mul_add(s0, s1, f16::from_bits(self.vec_reg[vdst] as u16)),
                        56 => f16::mul_add(s0, s1, f16::from_bits(self.simm() as u16)),
                        _ => todo_instr!(instruction),
                    };
                    self.vec_reg[vdst] = ret.to_bits() as u32;
                }
                _ => {
                    let s0 = self.val(s0);
                    self.vec_reg[vdst] = match op {
                        1 => match self.vcc.read() {
                            true => s1,
                            false => s0,
                        },
                        2 => {
                            let mut acc = f32::from_bits(self.vec_reg[vdst]);
                            acc += f32::from(f16_lo(s0)) * f32::from(f16_lo(s1));
                            acc += f32::from(f16_hi(s0)) * f32::from(f16_hi(s1));
                            acc.to_bits()
                        }
                        50..=60 => {
                            let (s0, s1) = (f16::from_bits(s0 as u16), f16::from_bits(s1 as u16));
                            match op {
                                _ => match op {
                                    50 => s0 + s1,
                                    51 => s0 - s1,
                                    53 => s0 * s1,
                                    _ => todo_instr!(instruction),
                                }
                                .to_bits() as u32,
                            }
                        }

                        3 | 4 | 5 | 8 | 16 | 43 | 44 | 45 => {
                            let (s0, s1) = (f32::from_bits(s0), f32::from_bits(s1));
                            match op {
                                3 => s0 + s1,
                                4 => s0 - s1,
                                5 => s1 - s0,
                                8 => s0 * s1,
                                16 => f32::max(s0, s1),
                                43 => f32::mul_add(s0, s1, f32::from_bits(self.vec_reg[vdst])),
                                44 => f32::mul_add(s0, f32::from_bits(self.simm()), s1),
                                45 => f32::mul_add(s0, s1, f32::from_bits(self.simm())),
                                _ => panic!(),
                            }
                            .to_bits()
                        }
                        9 => {
                            let s0 = as_signed((s0 & 0xffffff) as u64, 24) as i32;
                            let s1 = as_signed((s1 & 0xffffff) as u64, 24) as i32;
                            (s0 * s1) as u32
                        }
                        18 | 26 => {
                            let (s0, s1) = (s0 as i32, s1 as i32);
                            (match op {
                                18 => i32::max(s0, s1),
                                26 => s1 >> s0,
                                _ => panic!(),
                            }) as u32
                        }
                        32 => {
                            let temp = s0 as u64 + s1 as u64 + self.vcc.read() as u64;
                            self.vcc.mut_lane(temp >= 0x100000000);
                            temp as u32
                        }
                        33 | 34 => {
                            let temp = match op {
                                33 => s0 - s1 - self.vcc.read() as u32,
                                34 => s1 - s0 - self.vcc.read() as u32,
                                _ => panic!(),
                            };
                            self.vcc
                                .mut_lane((s1 as u64 + self.vcc.read() as u64) > s0 as u64);
                            temp
                        }
                        11 => s0 * s1,
                        19 => u32::min(s0, s1),
                        24 => s1 << s0,
                        29 => s0 ^ s1,
                        25 => s1 >> s0,
                        27 => s0 & s1,
                        28 => s0 | s1,
                        37 => s0 + s1,
                        38 => s0 - s1,
                        39 => s1 - s0,
                        _ => todo_instr!(instruction),
                    }
                }
            };
        }
        // vop3
        else if instruction >> 26 == 0b110101 {
            let instr = self.u64_instr();

            let op = ((instr >> 16) & 0x3ff) as u32;
            match op {
                764 | 288 | 289 | 766 | 768 | 769 => {
                    let vdst = (instr & 0xff) as usize;
                    let sdst = ((instr >> 8) & 0x7f) as usize;
                    let mut s = |i: u32| -> u32 { self.val(((instr >> i) & 0x1ff) as usize) };
                    let (s0, s1, s2) = (s(32), s(41), s(50));
                    let omod = (instr >> 59) & 0x3;
                    let _neg = (instr >> 61) & 0x7;
                    let clmp = (instr >> 15) & 0x1;
                    assert_eq!(omod, 0);
                    assert_eq!(clmp, 0);

                    if *DEBUG >= DebugLevel::INSTRUCTION {
                        println!(
                            "{} vdst={vdst} sdst={sdst} op={op} src={:?}",
                            "VOPSD".color("blue"),
                            (s0, s1, s2)
                        );
                    }

                    let (ret, vcc) = match op {
                        288 => {
                            let ret = s0 as u64 + s1 as u64 + (s2 != 0) as u64;
                            (ret as u32, ret >= 0x100000000)
                        }
                        289 => {
                            let vcc = (s2 != 0) as u64;
                            let ret = s0 as u64 - s1 as u64 - vcc;
                            (ret as u32, s1 as u64 + vcc > s0 as u64)
                        }
                        764 => (0, false), // NOTE: div scaling isn't required
                        766 => {
                            let ret = s0 as u64 * s1 as u64 + s2 as u64;
                            assert!(sdst as u32 == NULL_SRC, "not yet implemented");
                            (ret as u32, false)
                        }
                        768 => {
                            let ret = s0 as u64 + s1 as u64;
                            (ret as u32, ret >= 0x100000000)
                        }
                        769 => {
                            let ret = s0.wrapping_sub(s1);
                            (ret as u32, s1 > s0)
                        }
                        _ => todo_instr!(instruction),
                    };
                    match sdst {
                        106 => self.vcc.mut_lane(vcc),
                        124 => {}
                        _ => self.scalar_reg[sdst] = vcc as u32,
                    }
                    self.vec_reg[vdst] = ret;
                }
                _ => {
                    let vdst = (instr & 0xff) as usize;
                    let abs = ((instr >> 8) & 0x7) as usize;
                    let opsel = ((instr >> 11) & 0xf) as usize;
                    let cm = (instr >> 15) & 0x1;

                    let s = |n: usize| ((instr >> n) & 0x1ff) as usize;
                    let src = (s(32), s(41), s(50));

                    let omod = (instr >> 59) & 0x3;
                    let neg = ((instr >> 61) & 0x7) as usize;
                    assert_eq!(omod, 0);
                    assert_eq!(cm, 0);
                    assert_eq!(opsel, 0);

                    if *DEBUG >= DebugLevel::INSTRUCTION {
                        println!(
                            "{} vdst={vdst} abs={abs} opsel={opsel} op={op} src={:?} neg=0b{:03b}",
                            "VOP3".color("blue"),
                            src,
                            neg
                        );
                    }

                    match op {
                        // VOPC using VOP3 encoding
                        0..=255 => {
                            let dest_offset = if op >= 128 { 128 } else { 0 };
                            let ret = match op {
                                (0..=15) | 125 | (128..=143) => {
                                    let (s0, s1) = (self.val(src.0), self.val(src.1));
                                    let s0 = f16::from_bits(s0).negate(0, neg).absolute(0, abs);
                                    let s1 = f16::from_bits(s1).negate(1, neg).absolute(1, abs);
                                    match op {
                                        125 => self.cmp_class_f16(s0, s1.to_bits()),
                                        _ => self.cmpf(s0, s1, op - dest_offset),
                                    }
                                }
                                (16..=31) | 126 | (144..=159) => {
                                    let (s0, s1) = (self.val(src.0), self.val(src.1));
                                    let s0 = f32::from_bits(s0).negate(0, neg).absolute(0, abs);
                                    let s1 = f32::from_bits(s1).negate(1, neg).absolute(1, abs);
                                    match op {
                                        126 => self.cmp_class_f32(s0, s1.to_bits()),
                                        _ => self.cmpf(s0, s1, op - 16 - dest_offset),
                                    }
                                }
                                (32..=47) | (160..=174) => {
                                    let (s0, s1) = (self.val(src.0), self.val(src.1));
                                    let s0 = f64::from_bits(s0).negate(0, neg).absolute(0, abs);
                                    let s1 = f64::from_bits(s1).negate(1, neg).absolute(1, abs);
                                    self.cmpf(s0, s1, op - 32 - dest_offset)
                                }
                                (49..=54) | (177..=182) => {
                                    let (s0, s1): (u16, u16) = (self.val(src.0), self.val(src.1));
                                    self.cmpi(s0 as i16, s1 as i16, op - 48 - dest_offset)
                                }
                                (57..=62) | (185..=190) => {
                                    let (s0, s1): (u16, u16) = (self.val(src.0), self.val(src.1));
                                    self.cmpi(s0, s1, op - 56 - dest_offset)
                                }
                                (64..=71) | (192..=199) => {
                                    let (s0, s1): (u32, u32) = (self.val(src.0), self.val(src.1));
                                    self.cmpi(s0 as i32, s1 as i32, op - 64 - dest_offset)
                                }
                                (72..=79) | (200..=207) => {
                                    let (s0, s1): (u32, u32) = (self.val(src.0), self.val(src.1));
                                    self.cmpi(s0, s1, op - 72 - dest_offset)
                                }
                                (80..=87) | (208..=215) => {
                                    let (s0, s1): (u64, u64) = (self.val(src.0), self.val(src.1));
                                    self.cmpi(s0 as i64, s1 as i64, op - 80 - dest_offset)
                                }
                                (88..=95) | (216..=223) => {
                                    let (s0, s1): (u64, u64) = (self.val(src.0), self.val(src.1));
                                    self.cmpi(s0, s1, op - 88 - dest_offset)
                                }
                                _ => todo_instr!(instruction),
                            } as u32;

                            match vdst {
                                0..=SGPR_COUNT => self.scalar_reg[vdst] = ret,
                                106 => self.vcc.mut_lane(ret != 0),
                                126 => self.exec.mut_lane(ret != 0),
                                _ => todo_instr!(instruction),
                            }
                        }
                        828..=830 => {
                            let (s0, s1, _s2): (u32, u64, u64) =
                                (self.val(src.0), self.val(src.1), self.val(src.2));
                            let shift = s0 & 0x3f;
                            let ret = match op {
                                828 => s1 << shift,
                                829 => s1 >> shift,
                                830 => ((s1 as i64) >> shift) as u64,
                                _ => todo_instr!(instruction),
                            };
                            self.vec_reg.write64(vdst, ret)
                        }
                        808 | 807 | 811 | 532 => {
                            let (s0, s1, s2): (u64, u64, u64) =
                                (self.val(src.0), self.val(src.1), self.val(src.2));
                            let ret = match op {
                                532 | 808 | 807 | 811 => {
                                    let (s0, s1, s2) = (
                                        f64::from_bits(s0).negate(0, neg).absolute(0, abs),
                                        f64::from_bits(s1).negate(1, neg).absolute(1, abs),
                                        f64::from_bits(s2).negate(2, neg).absolute(2, abs),
                                    );

                                    match op {
                                        532 => f64::mul_add(s0, s1, s2),
                                        808 => s0 * s1,
                                        811 => (s0 * 2.0).powi(s1.to_bits() as i32),
                                        807 => s0 + s1,
                                        _ => panic!(),
                                    }
                                    .to_bits()
                                }
                                _ => panic!(),
                            };
                            self.vec_reg.write64(vdst, ret)
                        }
                        306 | 596 | 584 => {
                            let (s0, s1, s2) = (self.val(src.0), self.val(src.1), self.val(src.2));
                            let s0 = f16::from_bits(s0).negate(0, neg).absolute(0, abs);
                            let s1 = f16::from_bits(s1).negate(1, neg).absolute(1, abs);
                            let s2 = f16::from_bits(s2).negate(1, neg).absolute(1, abs);
                            self.vec_reg[vdst] = match op {
                                306 => s0 + s1,
                                584 => f16::mul_add(s0, s1, s2),
                                596 => s2 / s1,
                                _ => panic!(),
                            }
                            .to_bits() as u32
                        }
                        395 => {
                            let s0 = f16::from_bits(self.val(src.0))
                                .negate(0, neg)
                                .absolute(0, abs);
                            self.vec_reg[vdst] = f32::from(s0).to_bits();
                        }
                        785 => {
                            let (s0, s1) = (self.val(src.0), self.val(src.1));
                            self.vec_reg[vdst] = (f16::from_bits(s1).to_bits() as u32) << 16
                                | f16::from_bits(s0).to_bits() as u32
                        }
                        _ => {
                            let (s0, s1, s2) = (self.val(src.0), self.val(src.1), self.val(src.2));
                            match op {
                                865 => {
                                    self.vec_reg.get_lane_mut(s1 as usize)[vdst] = s0;
                                    return;
                                }
                                864 => {
                                    let val =
                                        self.vec_reg.get_lane(s1 as usize)[src.0 - VGPR_COUNT];
                                    self.write_to_sdst(vdst as u32, val);
                                    return;
                                }
                                826 => {
                                    self.vec_reg[vdst].mut_lo16(((s1 as i16) >> (s0 & 0xf)) as u16);
                                    return;
                                }
                                577 | 771 | 772 | 773 | 824 | 825 => {
                                    let (s0, s1, s2) = (s0 as u16, s1 as u16, s2 as u16);
                                    let ret = match op {
                                        577 => s0 * s1 + s2,
                                        771 => s0 + s1,
                                        772 => s0 - s1,
                                        773 => s0 * s1,
                                        824 => s1 << s0,
                                        825 => s1 >> s0,
                                        _ => panic!(),
                                    };
                                    self.vec_reg[vdst].mut_lo16(ret);
                                    return;
                                }
                                _ => {}
                            }

                            self.vec_reg[vdst] = match op {
                                257 | 259 | 299 | 260 | 264 | 272 | 531 | 537 | 540 | 551 | 567
                                | 796 => {
                                    let s0 = f32::from_bits(s0).negate(0, neg).absolute(0, abs);
                                    let s1 = f32::from_bits(s1).negate(1, neg).absolute(1, abs);
                                    let s2 = f32::from_bits(s2).negate(2, neg).absolute(2, abs);
                                    match op {
                                        259 => s0 + s1,
                                        260 => s0 - s1,
                                        264 => s0 * s1,
                                        272 => f32::max(s0, s1),
                                        299 => {
                                            f32::mul_add(s0, s1, f32::from_bits(self.vec_reg[vdst]))
                                        }
                                        531 => f32::mul_add(s0, s1, s2),
                                        537 => f32::min(f32::min(s0, s1), s2),
                                        540 => f32::max(f32::max(s0, s1), s2),
                                        551 => s2 / s1,
                                        567 => {
                                            let ret = f32::mul_add(s0, s1, s2);
                                            match self.vcc.read() {
                                                true => 2.0_f32.powi(32) * ret,
                                                false => ret,
                                            }
                                        }
                                        796 => s0 * 2f32.powi(s1.to_bits() as i32),
                                        // cnd_mask isn't a float only ALU but supports neg
                                        257 => match s2.to_bits() != 0 {
                                            true => s1,
                                            false => s0,
                                        },
                                        _ => panic!(),
                                    }
                                    .to_bits()
                                }
                                _ => {
                                    assert!(neg == 0);
                                    match op {
                                        529 => {
                                            let s0 = s0 as i32;
                                            let shift = 32 - (s2 & 0x1f);
                                            let mask: i32 = 1 << (s2 & 0x1f);
                                            let ret = (s0 >> (s1 & 0x1f)) & (mask.wrapping_sub(1));
                                            ((ret << shift) >> shift) as u32
                                        }
                                        522 | 541 | 544 | 814 => {
                                            let (s0, s1, s2) = (s0 as i32, s1 as i32, s2 as i32);
                                            (match op {
                                                522 => {
                                                    let s0 = as_signed((s0 & 0xffffff) as u64, 24)
                                                        as i32;
                                                    let s1 = as_signed((s1 & 0xffffff) as u64, 24)
                                                        as i32;
                                                    s0 * s1 + s2
                                                }
                                                541 => i32::max(i32::max(s0, s1), s2),
                                                544 => {
                                                    if (i32::max(i32::max(s0, s1), s2)) == s0 {
                                                        i32::max(s1, s2)
                                                    } else if (i32::max(i32::max(s0, s1), s2)) == s1
                                                    {
                                                        i32::max(s0, s2)
                                                    } else {
                                                        i32::max(s0, s1)
                                                    }
                                                }
                                                814 => ((s0 as i64) * (s1 as i64) >> 32) as i32,
                                                _ => panic!(),
                                            }) as u32
                                        }
                                        283 => s0 & s1,
                                        523 => s0 * s1 + s2, // TODO 24 bit trunc
                                        528 => (s0 >> s1) & ((1 << s2) - 1),
                                        534 => {
                                            let val = ((s0 as u64) << 32) | (s1 as u64);
                                            let shift = (s2 & 0x1F) as u64;
                                            ((val >> shift) & 0xffffffff) as u32
                                        }
                                        576 => s0 ^ s1 ^ s2,
                                        582 => (s0 << s1) + s2,
                                        597 => s0 + s1 + s2,
                                        580 => {
                                            fn byte_permute(data: u64, sel: u32) -> u8 {
                                                let bytes = data.to_ne_bytes();
                                                match sel {
                                                    13..=u32::MAX => 0xff,
                                                    12 => 0x00,
                                                    11 => ((bytes[7] & 0x80) != 0) as u8 * 0xff,
                                                    10 => ((bytes[5] & 0x80) != 0) as u8 * 0xff,
                                                    9 => ((bytes[3] & 0x80) != 0) as u8 * 0xff,
                                                    8 => ((bytes[1] & 0x80) != 0) as u8 * 0xff,
                                                    _ => bytes[sel as usize],
                                                }
                                            }
                                            let combined = ((s0 as u64) << 32) | s1 as u64;
                                            let d0 = ((byte_permute(combined, s2 >> 24) as u32)
                                                << 24)
                                                | ((byte_permute(combined, (s2 >> 16) & 0xFF)
                                                    as u32)
                                                    << 16)
                                                | ((byte_permute(combined, (s2 >> 8) & 0xFF)
                                                    as u32)
                                                    << 8)
                                                | (byte_permute(combined, s2 & 0xFF) as u32);
                                            d0
                                        }
                                        581 => (s0 ^ s1) + s2,
                                        583 => (s0 + s1) << s2,
                                        598 => (s0 << s1) | s2,
                                        599 => (s0 & s1) | s2,
                                        812 => s0 * s1,
                                        813 => ((s0 as u64) * (s1 as u64) >> 32) as u32,
                                        _ => todo_instr!(instruction),
                                    }
                                }
                            }
                        }
                    };
                }
            }
        } else if instruction >> 26 == 0b110110 {
            let instr = self.u64_instr();
            let offset0 = instr & 0xff;
            let offset1 = (instr >> 8) & 0xff;
            let op = (instr >> 18) & 0xff;
            let addr = (instr >> 32) & 0xff;
            let data0 = ((instr >> 40) & 0xff) as usize;
            let data1 = (instr >> 48) & 0xff;
            let vdst = ((instr >> 56) & 0xff) as usize;

            if *DEBUG >= DebugLevel::INSTRUCTION {
                println!(
                    "{} offset0={offset0} offset1={offset1} op={op} addr={addr} data0={data0} data1={data1} vdst={vdst}",
                    "LDS".color("blue"),
                );
            }
            let addr = (self.vec_reg[addr as usize] as u64 + offset0) as usize;

            match op {
                // load
                255 => (0..4).for_each(|i| self.vec_reg[vdst + i] = self.lds.read(addr + 4 * i)),
                // store
                13 | 223 => {
                    let iters = if op == 223 { 4 } else { 1 };
                    (0..iters).for_each(|i| self.lds.write(addr + 4 * i, self.vec_reg[data0 + i]))
                }
                _ => todo_instr!(instruction),
            }
        }
        // global
        // flat
        else if instruction >> 26 == 0b110111 {
            let instr = self.u64_instr();
            let offset = as_signed(instr & 0x1fff, 13);
            let seg = (instr >> 16) & 0x3;
            let op = ((instr >> 18) & 0x7f) as usize;
            let addr = ((instr >> 32) & 0xff) as usize;
            let data = ((instr >> 40) & 0xff) as usize;
            let saddr = ((instr >> 48) & 0x7f) as usize;
            let vdst = ((instr >> 56) & 0xff) as usize;

            let saddr_val: u32 = self.val(saddr);
            let saddr_off = saddr_val == 0x7F || saddr as u32 == NULL_SRC;

            match seg {
                1 => {
                    let sve = ((instr >> 50) & 0x1) != 0;
                    if *DEBUG >= DebugLevel::INSTRUCTION {
                        println!("{} offset={offset} op={op} addr={addr} data={data} saddr={saddr} vdst={vdst} sve={sve}", "SCRATCH".color("blue"));
                    }
                    let addr = match (sve, saddr_off) {
                        (true, true) => offset as u64 as usize,
                        _ => todo_instr!(instruction),
                    };
                    match op {
                        // load
                        20..=23 => (0..op - 19).for_each(|i| {
                            self.vec_reg[vdst + i] = self.sds.read(addr + 4 * i);
                        }),
                        // store
                        26..=29 => (0..op - 25).for_each(|i| {
                            self.sds.write(addr + 4 * i, self.vec_reg[data + i]);
                        }),
                        _ => todo_instr!(instruction),
                    }
                }
                2 => {
                    if *DEBUG >= DebugLevel::INSTRUCTION {
                        println!("{} offset={offset} op={op} addr={addr} data={data} saddr={saddr} vdst={vdst}", "GLOBAL".color("blue"));
                    }

                    let addr = match saddr_off {
                        true => self.vec_reg.read64(addr) as i64 + (offset as i64),
                        false => {
                            let scalar_addr = self.scalar_reg.read64(saddr);
                            let vgpr_offset = self.vec_reg[addr];
                            scalar_addr as i64 + vgpr_offset as i64 + offset
                        }
                    } as u64;
                    unsafe {
                        match op {
                            // load
                            16 => self.vec_reg[vdst] = *(addr as *const u8) as u32,
                            17 => self.vec_reg[vdst] = *(addr as *const i8) as u32,
                            18 => self.vec_reg[vdst] = *(addr as *const u16) as u32,
                            19 => self.vec_reg[vdst] = *(addr as *const i16) as u32,

                            20..=23 => (0..op - 19).for_each(|i| {
                                self.vec_reg[vdst + i] = *((addr + 4 * i as u64) as *const u32);
                            }),
                            35 => self.vec_reg[vdst].mut_hi16(*(addr as *const u16)),
                            // store
                            24 => *(addr as *mut u8) = self.vec_reg[data] as u8,
                            25 => *(addr as *mut u16) = self.vec_reg[data] as u16,
                            26..=29 => (0..op - 25).for_each(|i| {
                                *((addr + 4 * i as u64) as u64 as *mut u32) =
                                    self.vec_reg[data + i];
                            }),
                            _ => todo_instr!(instruction),
                        };
                    }
                }
                _ => todo_instr!(instruction),
            };
        } else {
            todo_instr!(instruction);
        }
    }

    fn cmpf<T>(&self, s0: T, s1: T, offset: u32) -> bool
    where
        T: PartialOrd + PartialEq,
    {
        return match offset {
            0 => true,
            1 => s0 < s1,
            2 => s0 == s1,
            3 => s0 <= s1,
            4 => s0 > s1,
            5 => s0 != s1,
            6 => s0 >= s1,
            9 => !(s0 >= s1),
            10 => !(s0 != s1),
            11 => !(s0 > s1),
            12 => !(s0 <= s1),
            13 => !(s0 == s1),
            14 => !(s0 < s1),
            15 => true,
            _ => panic!("{offset}"),
        };
    }
    fn cmp_class_f32(&self, s0: f32, s1: u32) -> bool {
        let offset = match s0 {
            _ if (s0 as f64).is_nan() => 1,
            _ if s0.exponent() == 255 => match s0.signum() == -1.0 {
                true => 2,
                false => 9,
            },
            _ if s0.exponent() > 0 => match s0.signum() == -1.0 {
                true => 3,
                false => 8,
            },
            _ if s0.abs() as f64 > 0.0 => match s0.signum() == -1.0 {
                true => 4,
                false => 7,
            },
            _ => match s0.signum() == -1.0 {
                true => 5,
                false => 6,
            },
        };
        ((s1 >> offset) & 1) != 0
    }
    fn cmp_class_f16(&self, s0: f16, s1: u16) -> bool {
        let offset = match s0 {
            _ if (f64::from(s0)).is_nan() => 1,
            _ if s0.exponent() == 31 => match s0.signum() == f16::NEG_ONE {
                true => 2,
                false => 9,
            },
            _ if s0.exponent() > 0 => match s0.signum() == f16::NEG_ONE {
                true => 3,
                false => 8,
            },
            _ if f64::from(s0.abs()) > 0.0 => match s0.signum() == f16::NEG_ONE {
                true => 4,
                false => 7,
            },
            _ => match s0.signum() == f16::NEG_ONE {
                true => 5,
                false => 6,
            },
        };
        ((s1 >> offset) & 1) != 0
    }
    fn cmpi<T>(&self, s0: T, s1: T, offset: u32) -> bool
    where
        T: PartialOrd + PartialEq,
    {
        return match offset {
            0 => false,
            1 => s0 < s1,
            2 => s0 == s1,
            3 => s0 <= s1,
            4 => s0 > s1,
            5 => s0 != s1,
            6 => s0 >= s1,
            7 => true,
            _ => panic!("{offset}"),
        };
    }
    fn cls_i32(&self, s0: u32) -> u32 {
        let mut ret: i32 = -1;
        let s0 = s0 as i32;
        for i in (1..=31).into_iter() {
            if s0 >> (31 - i as u32) != s0 >> 31 {
                ret = i;
                break;
            }
        }
        ret as u32
    }
    fn clz_i32_u32(&self, s0: u32) -> u32 {
        let mut ret: i32 = -1;
        for i in (0..=31).into_iter() {
            if s0 >> (31 - i as u32) == 1 {
                ret = i;
                break;
            }
        }
        ret as u32
    }

    /* ALU utils */
    fn _common_srcs(&mut self, code: u32) -> u32 {
        match code {
            106 => self.vcc.read() as u32,
            126 => self.exec.read() as u32,
            128 => 0,
            124 => NULL_SRC,
            255 => self.simm(),
            _ => todo!("resolve_src={code}"),
        }
    }
    fn simm(&mut self) -> u32 {
        if let Some(val) = self.simm {
            val
        } else {
            self.pc += 1;
            let val = self.prg[self.pc as usize - 1];
            self.simm = Some(val);
            val
        }
    }
    fn write_to_sdst(&mut self, sdst_bf: u32, val: u32) {
        match sdst_bf as usize {
            0..=SGPR_COUNT => self.scalar_reg[sdst_bf as usize] = val,
            106 => self.vcc.value = val,
            126 => self.exec.value = val,
            _ => todo!("write to sdst {}", sdst_bf),
        }
    }
}

pub trait ALUSrc<T> {
    fn val(&mut self, code: usize) -> T;
}
impl ALUSrc<u16> for CPU<'_> {
    fn val(&mut self, code: usize) -> u16 {
        match code {
            0..=SGPR_COUNT => self.scalar_reg[code] as u16,
            VGPR_COUNT..=511 => self.vec_reg[code - VGPR_COUNT] as u16,
            129..=192 => (code - 128) as u16,
            193..=208 => ((code - 192) as i16 * -1) as u16,
            240..=247 => f16::from_f32(
                [
                    (240, 0.5_f32),
                    (241, -0.5_f32),
                    (242, 1_f32),
                    (243, -1.0_f32),
                    (244, 2.0_f32),
                    (245, -2.0_f32),
                    (246, 4.0_f32),
                    (247, -4.0_f32),
                ]
                .iter()
                .find(|x| x.0 == code)
                .unwrap()
                .1,
            )
            .to_bits(),
            _ => self._common_srcs(code as u32) as u16,
        }
    }
}
impl ALUSrc<u32> for CPU<'_> {
    fn val(&mut self, code: usize) -> u32 {
        match code {
            0..=SGPR_COUNT => self.scalar_reg[code],
            VGPR_COUNT..=511 => self.vec_reg[code - VGPR_COUNT],
            129..=192 => (code - 128) as u32,
            193..=208 => ((code - 192) as i32 * -1) as u32,
            240..=247 => [
                (240, 0.5_f32),
                (241, -0.5_f32),
                (242, 1_f32),
                (243, -1.0_f32),
                (244, 2.0_f32),
                (245, -2.0_f32),
                (246, 4.0_f32),
                (247, -4.0_f32),
            ]
            .iter()
            .find(|x| x.0 == code)
            .unwrap()
            .1
            .to_bits(),
            _ => self._common_srcs(code as u32),
        }
    }
}
impl ALUSrc<u64> for CPU<'_> {
    fn val(&mut self, code: usize) -> u64 {
        match code {
            0..=SGPR_COUNT => self.scalar_reg.read64(code),
            VGPR_COUNT..=511 => self.vec_reg.read64(code - VGPR_COUNT),
            129..=192 => (code - 128) as u64,
            193..=208 => ((code - 192) as i64 * -1) as u64,
            240..=247 => [
                (240, 0.5_f64),
                (241, -0.5_f64),
                (242, 1_f64),
                (243, -1.0_f64),
                (244, 2.0_f64),
                (245, -2.0_f64),
                (246, 4.0_f64),
                (247, -4.0_f64),
            ]
            .iter()
            .find(|x| x.0 == code)
            .unwrap()
            .1
            .to_bits(),
            _ => self._common_srcs(code as u32) as u64,
        }
    }
}

fn _helper_test_cpu() -> CPU<'static> {
    let static_lds: &'static mut VecDataStore = Box::leak(Box::new(VecDataStore::new()));
    let static_sgpr: &'static mut Vec<u32> = Box::leak(Box::new(vec![0; 256]));
    let static_vgpr: &'static mut VGPR = Box::leak(Box::new(VGPR::new()));
    let static_scc: &'static mut u32 = Box::leak(Box::new(0));
    let static_exec: &'static mut WaveValue = Box::leak(Box::new(WaveValue::new(u32::MAX)));
    let static_vcc: &'static mut WaveValue = Box::leak(Box::new(WaveValue::new(0)));
    let static_sds: &'static mut VecDataStore = Box::leak(Box::new(VecDataStore::new()));

    let cpu = CPU {
        scalar_reg: static_sgpr,
        vec_reg: static_vgpr,
        scc: static_scc,
        vcc: static_vcc,
        exec: static_exec,
        lds: static_lds,
        sds: static_sds,
        simm: None,
        pc: 0,
        prg: vec![],
    };
    cpu.vec_reg.default_lane = Some(0);
    cpu.vcc.default_lane = Some(0);
    cpu.exec.default_lane = Some(0);
    return cpu;
}
#[cfg(test)]
mod test_alu_utils {
    use super::*;

    #[test]
    fn test_write_to_sdst_sgpr() {
        let mut cpu = _helper_test_cpu();
        cpu.write_to_sdst(10, 200);
        assert_eq!(cpu.scalar_reg[10], 200);
    }

    #[test]
    fn test_write_to_sdst_vcc_val() {
        let mut cpu = _helper_test_cpu();
        let val = 0b1011101011011011111011101111;
        cpu.write_to_sdst(106, val);
        assert_eq!(cpu.vcc.value, 195935983);
    }

    #[test]
    fn test_clz_i32_u32() {
        let cpu = _helper_test_cpu();
        assert_eq!(cpu.clz_i32_u32(0x00000000), 0xffffffff);
        assert_eq!(cpu.clz_i32_u32(0x0000cccc), 16);
        assert_eq!(cpu.clz_i32_u32(0xffff3333), 0);
        assert_eq!(cpu.clz_i32_u32(0x7fffffff), 1);
        assert_eq!(cpu.clz_i32_u32(0x80000000), 0);
        assert_eq!(cpu.clz_i32_u32(0xffffffff), 0);
    }

    #[test]
    fn test_cls_i32() {
        let cpu = _helper_test_cpu();
        assert_eq!(cpu.cls_i32(0x00000000), 0xffffffff);
        assert_eq!(cpu.cls_i32(0x0000cccc), 16);
        assert_eq!(cpu.cls_i32(0xffff3333), 16);
        assert_eq!(cpu.cls_i32(0x7fffffff), 1);
        assert_eq!(cpu.cls_i32(0x80000000), 1);
    }
}

#[cfg(test)]
mod test_sop1 {
    use super::*;

    #[test]
    fn test_s_mov_b64() {
        let mut cpu = _helper_test_cpu();
        cpu.scalar_reg.write64(16, 5236523008);
        cpu.interpret(&vec![0xBE880110, END_PRG]);
        assert_eq!(cpu.scalar_reg.read64(8), 5236523008)
    }

    #[test]
    fn test_s_mov_b32() {
        let mut cpu = _helper_test_cpu();
        cpu.scalar_reg[15] = 42;
        cpu.interpret(&vec![0xbe82000f, END_PRG]);
        assert_eq!(cpu.scalar_reg[2], 42);
    }

    #[test]
    fn test_s_bitset0_b32() {
        [
            [
                0b11111111111111111111111111111111,
                0b00000000000000000000000000000001,
                0b11111111111111111111111111111101,
            ],
            [
                0b11111111111111111111111111111111,
                0b00000000000000000000000000000010,
                0b11111111111111111111111111111011,
            ],
        ]
        .iter()
        .for_each(|[a, b, ret]| {
            let mut cpu = _helper_test_cpu();
            cpu.scalar_reg[20] = *a;
            cpu.scalar_reg[10] = *b;
            cpu.interpret(&vec![0xBE94100A, END_PRG]);
            assert_eq!(cpu.scalar_reg[20], *ret);
        });
    }

    #[test]
    fn test_s_bitset1_b32() {
        [
            [
                0b00000000000000000000000000000000,
                0b00000000000000000000000000000001,
                0b00000000000000000000000000000010,
            ],
            [
                0b00000000000000000000000000000000,
                0b00000000000000000000000000000010,
                0b00000000000000000000000000000100,
            ],
        ]
        .iter()
        .for_each(|[a, b, ret]| {
            let mut cpu = _helper_test_cpu();
            cpu.scalar_reg[20] = *a;
            cpu.scalar_reg[10] = *b;
            cpu.interpret(&vec![0xbe94120a, END_PRG]);
            assert_eq!(cpu.scalar_reg[20], *ret);
        });
    }

    #[test]
    fn test_s_not_b32() {
        [[0, 4294967295, 1], [1, 4294967294, 1], [u32::MAX, 0, 0]]
            .iter()
            .for_each(|[a, ret, scc]| {
                let mut cpu = _helper_test_cpu();
                cpu.scalar_reg[10] = *a;
                cpu.interpret(&vec![0xBE8A1E0A, END_PRG]);
                assert_eq!(cpu.scalar_reg[10], *ret);
                assert_eq!(*cpu.scc, *scc);
            });
    }
}

#[cfg(test)]
mod test_sopk {
    use super::*;

    #[test]
    fn test_cmp_zero_extend() {
        let mut cpu = _helper_test_cpu();
        cpu.scalar_reg[20] = 0xcd14;
        cpu.interpret(&vec![0xB494CD14, END_PRG]);
        assert_eq!(*cpu.scc, 1);

        cpu.interpret(&vec![0xB194CD14, END_PRG]);
        assert_eq!(*cpu.scc, 0);
    }

    #[test]
    fn test_cmp_sign_extend() {
        let mut cpu = _helper_test_cpu();
        cpu.scalar_reg[6] = 0x2db4;
        cpu.interpret(&vec![0xB1862DB4, END_PRG]);
        assert_eq!(*cpu.scc, 1);

        cpu.interpret(&vec![0xB1862DB4, END_PRG]);
        assert_eq!(*cpu.scc, 1);
    }
}

#[cfg(test)]
mod test_sop2 {
    use super::*;

    #[test]
    fn test_s_add_u32() {
        [
            [10, 20, 30, 0],
            [u32::MAX, 10, 9, 1],
            [u32::MAX, 0, u32::MAX, 0],
        ]
        .iter()
        .for_each(|[a, b, expected, scc]| {
            let mut cpu = _helper_test_cpu();
            cpu.scalar_reg[2] = *a;
            cpu.scalar_reg[6] = *b;
            cpu.interpret(&vec![0x80060206, END_PRG]);
            assert_eq!(cpu.scalar_reg[6], *expected);
            assert_eq!(*cpu.scc, *scc);
        });
    }

    #[test]
    fn test_s_addc_u32() {
        [
            [10, 20, 31, 1, 0],
            [10, 20, 30, 0, 0],
            [u32::MAX, 10, 10, 1, 1],
        ]
        .iter()
        .for_each(|[a, b, expected, scc_before, scc_after]| {
            let mut cpu = _helper_test_cpu();
            *cpu.scc = *scc_before;
            cpu.scalar_reg[7] = *a;
            cpu.scalar_reg[3] = *b;
            cpu.interpret(&vec![0x82070307, END_PRG]);
            assert_eq!(cpu.scalar_reg[7], *expected);
            assert_eq!(*cpu.scc, *scc_after);
        });
    }

    #[test]
    fn test_s_add_i32() {
        [[-10, 20, 10, 0], [i32::MAX, 1, -2147483648, 1]]
            .iter()
            .for_each(|[a, b, expected, scc]| {
                let mut cpu = _helper_test_cpu();
                cpu.scalar_reg[14] = *a as u32;
                cpu.scalar_reg[10] = *b as u32;
                cpu.interpret(&vec![0x81060E0A, END_PRG]);
                assert_eq!(cpu.scalar_reg[6], *expected as u32);
                assert_eq!(*cpu.scc, *scc as u32);
            });
    }

    #[test]
    fn test_s_sub_i32() {
        [[-10, 20, -30, 0], [i32::MAX, -1, -2147483648, 1]]
            .iter()
            .for_each(|[a, b, expected, scc]| {
                let mut cpu = _helper_test_cpu();
                cpu.scalar_reg[13] = *a as u32;
                cpu.scalar_reg[8] = *b as u32;
                cpu.interpret(&vec![0x818C080D, END_PRG]);
                assert_eq!(cpu.scalar_reg[12], *expected as u32);
                assert_eq!(*cpu.scc, *scc as u32);
            });
    }

    #[test]
    fn test_s_lshl_b32() {
        [[20, 40, 1], [0, 0, 0]]
            .iter()
            .for_each(|[a, expected, scc]| {
                let mut cpu = _helper_test_cpu();
                cpu.scalar_reg[15] = *a as u32;
                cpu.interpret(&vec![0x8408810F, END_PRG]);
                assert_eq!(cpu.scalar_reg[8], *expected as u32);
                assert_eq!(*cpu.scc, *scc as u32);
            });
    }

    #[test]
    fn test_s_lshl_b64() {
        let mut cpu = _helper_test_cpu();
        cpu.scalar_reg.write64(2, u64::MAX - 30);
        cpu.interpret(&vec![0x84828202, END_PRG]);
        assert_eq!(cpu.scalar_reg[2], 4294967172);
        assert_eq!(cpu.scalar_reg[3], 4294967295);
        assert_eq!(*cpu.scc, 1);
    }

    #[test]
    fn test_s_ashr_i32() {
        let mut cpu = _helper_test_cpu();
        cpu.scalar_reg[2] = 36855;
        cpu.interpret(&vec![0x86039F02, END_PRG]);
        assert_eq!(cpu.scalar_reg[3], 0);
        assert_eq!(*cpu.scc, 0);
    }

    #[test]
    fn test_s_min_i32() {
        let mut cpu = _helper_test_cpu();
        cpu.scalar_reg[2] = -42i32 as u32;
        cpu.scalar_reg[3] = -92i32 as u32;
        cpu.interpret(&vec![0x89020203, END_PRG]);
        assert_eq!(cpu.scalar_reg[2], -92i32 as u32);
        assert_eq!(*cpu.scc, 1);
    }

    #[test]
    fn test_s_mul_hi_u32() {
        [[u32::MAX, 10, 9], [u32::MAX / 2, 4, 1]]
            .iter()
            .for_each(|[a, b, expected]| {
                let mut cpu = _helper_test_cpu();
                cpu.scalar_reg[0] = *a;
                cpu.scalar_reg[8] = *b;
                cpu.interpret(&vec![0x96810800, END_PRG]);
                assert_eq!(cpu.scalar_reg[1], *expected);
            });
    }

    #[test]
    fn test_s_mul_hi_i32() {
        [[(u64::MAX) as i32, (u64::MAX / 2) as i32, 0], [2, -2, -1]]
            .iter()
            .for_each(|[a, b, expected]| {
                let mut cpu = _helper_test_cpu();
                cpu.scalar_reg[0] = *a as u32;
                cpu.scalar_reg[8] = *b as u32;
                cpu.interpret(&vec![0x97010800, END_PRG]);
                assert_eq!(cpu.scalar_reg[1], *expected as u32);
            });
    }

    #[test]
    fn test_s_mul_i32() {
        [[40, 2, 80], [-10, -10, 100]]
            .iter()
            .for_each(|[a, b, expected]| {
                let mut cpu = _helper_test_cpu();
                cpu.scalar_reg[0] = *a as u32;
                cpu.scalar_reg[6] = *b as u32;
                cpu.interpret(&vec![0x96000600, END_PRG]);
                assert_eq!(cpu.scalar_reg[0], *expected as u32);
            });
    }

    #[test]
    fn test_s_bfe_u64() {
        [
            [2, 4, 2, 0],
            [800, 400, 32, 0],
            [-10i32 as u32, 3, 246, 0],
            [u32::MAX, u32::MAX, 255, 0],
        ]
        .iter()
        .for_each(|[a_lo, a_hi, ret_lo, ret_hi]| {
            let mut cpu = _helper_test_cpu();
            cpu.scalar_reg[6] = *a_lo;
            cpu.scalar_reg[7] = *a_hi;
            cpu.interpret(&vec![0x940cff06, 524288, END_PRG]);
            assert_eq!(cpu.scalar_reg[12], *ret_lo);
            assert_eq!(cpu.scalar_reg[13], *ret_hi);
        });
    }

    #[test]
    fn test_s_bfe_i64() {
        [
            [131073, 0, 1, 0, 0x100000],
            [-2, 0, -2, -1, 524288],
            [2, 0, 2, 0, 524288],
        ]
        .iter()
        .for_each(|[a_lo, a_hi, ret_lo, ret_hi, shift]| {
            let mut cpu = _helper_test_cpu();
            cpu.scalar_reg[6] = *a_lo as u32;
            cpu.scalar_reg[7] = *a_hi as u32;
            cpu.interpret(&vec![0x948cff06, *shift as u32, END_PRG]);
            assert_eq!(cpu.scalar_reg[12], *ret_lo as u32);
            assert_eq!(cpu.scalar_reg[13], *ret_hi as u32);
        });
    }

    #[test]
    fn test_s_bfe_u32() {
        [
            [67305985, 2],
            [0b100000000110111111100000001, 0b1111111],
            [0b100000000100000000000000001, 0b0],
            [0b100000000111000000000000001, 0b10000000],
            [0b100000000111111111100000001, 0b11111111],
        ]
        .iter()
        .for_each(|[a, ret]| {
            let mut cpu = _helper_test_cpu();
            cpu.scalar_reg[0] = *a;
            cpu.interpret(&vec![0x9303FF00, 0x00080008, END_PRG]);
            assert_eq!(cpu.scalar_reg[3], *ret);
        });
    }
}

#[cfg(test)]
mod test_vopd {
    use super::*;

    #[test]
    fn test_inline_const_vopx_only() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[0] = f32::to_bits(0.5);
        let constant = f32::from_bits(0x39a8b099);
        cpu.vec_reg[1] = 10;
        cpu.interpret(&vec![0xC8D000FF, 0x00000080, 0x39A8B099, END_PRG]);
        assert_eq!(f32::from_bits(cpu.vec_reg[0]), 0.5 * constant);
        assert_eq!(cpu.vec_reg[1], 0);
    }

    #[test]
    fn test_inline_const_vopy_only() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[0] = 10;
        cpu.vec_reg[1] = 10;
        cpu.interpret(&vec![0xCA100080, 0x000000FF, 0x3E15F480, END_PRG]);
        assert_eq!(cpu.vec_reg[0], 0);
        assert_eq!(cpu.vec_reg[1], 0x3e15f480);

        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[18] = f32::to_bits(2.0);
        cpu.vec_reg[32] = f32::to_bits(4.0);
        cpu.vec_reg[7] = 10;
        cpu.interpret(&vec![0xC9204112, 0x00060EFF, 0x0000006E, END_PRG]);
        assert_eq!(f32::from_bits(cpu.vec_reg[0]), 2.0f32 + 4.0f32);
        assert_eq!(cpu.vec_reg[7], 120);
    }

    #[test]
    fn test_inline_const_shared() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[2] = f32::to_bits(2.0);
        cpu.vec_reg[3] = f32::to_bits(4.0);
        let constant = f32::from_bits(0x3e800000);
        cpu.interpret(&vec![0xC8C604FF, 0x020206FF, 0x3E800000, END_PRG]);
        assert_eq!(f32::from_bits(cpu.vec_reg[2]), 2.0 * constant);
        assert_eq!(f32::from_bits(cpu.vec_reg[3]), 4.0 * constant);
    }

    #[test]
    fn test_simm_op_shared_1() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[23] = f32::to_bits(4.0);
        cpu.vec_reg[12] = f32::to_bits(2.0);

        cpu.vec_reg[13] = f32::to_bits(10.0);
        cpu.vec_reg[24] = f32::to_bits(3.0);

        let simm = f32::from_bits(0x3e000000);
        cpu.interpret(&vec![0xC8841917, 0x0C0C1B18, 0x3E000000, END_PRG]);
        assert_eq!(f32::from_bits(cpu.vec_reg[12]), 4.0 * simm + 2.0);
        assert_eq!(f32::from_bits(cpu.vec_reg[13]), 3.0 * simm + 10.0);
    }

    #[test]
    fn test_simm_op_shared_2() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[29] = f32::to_bits(4.0);
        cpu.vec_reg[10] = f32::to_bits(2.0);

        cpu.vec_reg[11] = f32::to_bits(10.0);
        cpu.vec_reg[26] = f32::to_bits(6.5);

        let simm = 0.125;
        cpu.interpret(&vec![0xC880151D, 0x0A0A34FF, 0x3E000000, END_PRG]);
        assert_eq!(f32::from_bits(cpu.vec_reg[10]), 4.0 * simm + 2.0);
        assert_eq!(f32::from_bits(cpu.vec_reg[11]), simm * 6.5 + 10.0);
    }

    #[test]
    fn test_add_mov() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[0] = f32::to_bits(10.5);
        cpu.interpret(&vec![0xC9100300, 0x00000080, END_PRG]);
        assert_eq!(f32::from_bits(cpu.vec_reg[0]), 10.5);
        assert_eq!(cpu.vec_reg[1], 0);
    }

    #[test]
    fn test_max_add() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[0] = f32::to_bits(5.0);
        cpu.vec_reg[3] = f32::to_bits(2.0);
        cpu.vec_reg[1] = f32::to_bits(2.0);
        cpu.interpret(&vec![0xCA880280, 0x01000700, END_PRG]);
        assert_eq!(f32::from_bits(cpu.vec_reg[0]), 7.0);
        assert_eq!(f32::from_bits(cpu.vec_reg[1]), 2.0);
    }
}
#[cfg(test)]
mod test_vop1 {
    use super::*;

    #[test]
    fn test_v_mov_b32_srrc_const0() {
        let mut cpu = _helper_test_cpu();
        cpu.interpret(&vec![0x7e000280, END_PRG]);
        assert_eq!(cpu.vec_reg[0], 0);
        cpu.interpret(&vec![0x7e020280, END_PRG]);
        assert_eq!(cpu.vec_reg[1], 0);
        cpu.interpret(&vec![0x7e040280, END_PRG]);
        assert_eq!(cpu.vec_reg[2], 0);
    }

    #[test]
    fn test_v_mov_b32_srrc_register() {
        let mut cpu = _helper_test_cpu();
        cpu.scalar_reg[6] = 31;
        cpu.interpret(&vec![0x7e020206, END_PRG]);
        assert_eq!(cpu.vec_reg[1], 31);
    }

    fn helper_test_fexp(val: f32) -> f32 {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[6] = val.to_bits();
        cpu.interpret(&vec![0x7E0C4B06, END_PRG]);
        f32::from_bits(cpu.vec_reg[6])
    }

    #[test]
    fn test_fexp_1ulp() {
        let test_values = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        for &val in test_values.iter() {
            let expected = (2.0_f32).powf(val);
            assert!((helper_test_fexp(val) - expected).abs() <= f32::EPSILON);
        }
    }

    #[test]
    fn test_fexp_flush_denormals() {
        assert_eq!(helper_test_fexp(f32::from_bits(0xff800000)), 0.0);
        assert_eq!(helper_test_fexp(f32::from_bits(0x80000000)), 1.0);
        assert_eq!(
            helper_test_fexp(f32::from_bits(0x7f800000)),
            f32::from_bits(0x7f800000)
        );
    }

    #[test]
    fn test_cast_f32_i32() {
        let mut cpu = _helper_test_cpu();
        [(10.42, 10i32), (-20.08, -20i32)]
            .iter()
            .for_each(|(src, expected)| {
                cpu.scalar_reg[2] = f32::to_bits(*src);
                cpu.interpret(&vec![0x7E001002, END_PRG]);
                assert_eq!(cpu.vec_reg[0] as i32, *expected);
            })
    }

    #[test]
    fn test_cast_f32_u32() {
        let mut cpu = _helper_test_cpu();
        cpu.scalar_reg[4] = 2;
        cpu.interpret(&vec![0x7E000C04, END_PRG]);
        assert_eq!(cpu.vec_reg[0], 1073741824);
    }

    #[test]
    fn test_cast_u32_f32() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[0] = 1325400062;
        cpu.interpret(&vec![0x7E000F00, END_PRG]);
        assert_eq!(cpu.vec_reg[0], 2147483392);
    }

    #[test]
    fn test_cast_i32_f32() {
        let mut cpu = _helper_test_cpu();
        [(10.0, 10i32), (-20.0, -20i32)]
            .iter()
            .for_each(|(expected, src)| {
                cpu.vec_reg[0] = *src as u32;
                cpu.interpret(&vec![0x7E000B00, END_PRG]);
                assert_eq!(f32::from_bits(cpu.vec_reg[0]), *expected);
            })
    }

    #[test]
    fn test_v_readfirstlane_b32_basic() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[0] = 2147483392;
        cpu.interpret(&vec![0x7E060500, END_PRG]);
        assert_eq!(cpu.scalar_reg[3], 2147483392);
    }

    #[test]
    fn test_v_cls_i32() {
        fn t(val: u32) -> u32 {
            let mut cpu = _helper_test_cpu();
            cpu.vec_reg[2] = val;
            cpu.interpret(&vec![0x7E087702, END_PRG]);
            return cpu.vec_reg[4];
        }

        assert_eq!(t(0x00000000), 0xffffffff);
        assert_eq!(t(0x40000000), 1);
        assert_eq!(t(0x80000000), 1);
        assert_eq!(t(0x0fffffff), 4);
        assert_eq!(t(0xffff0000), 16);
        assert_eq!(t(0xfffffffe), 31);
    }

    #[test]
    fn test_v_rndne_f32() {
        [
            [1.2344, 1.0],
            [2.3, 2.0], // [0.5f32, 0.0f32],
            [0.51, 1.0],
            [f32::from_bits(1186963295), f32::from_bits(1186963456)],
        ]
        .iter()
        .for_each(|[a, ret]| {
            let mut cpu = _helper_test_cpu();
            cpu.vec_reg[0] = f32::to_bits(*a);
            println!("a={} ret={}", a.to_bits(), ret.to_bits());
            cpu.interpret(&vec![0x7E024700, END_PRG]);
            assert_eq!(f32::from_bits(cpu.vec_reg[1]), *ret);
        })
    }
}

#[cfg(test)]
mod test_vopc {
    use super::*;

    #[test]
    fn test_v_cmp_gt_i32() {
        let mut cpu = _helper_test_cpu();

        cpu.vec_reg[1] = (4_i32 * -1) as u32;
        cpu.interpret(&vec![0x7c8802c1, END_PRG]);
        assert_eq!(cpu.vcc.read(), true);

        cpu.vec_reg[1] = 4;
        cpu.interpret(&vec![0x7c8802c1, END_PRG]);
        assert_eq!(cpu.vcc.read(), false);
    }

    #[test]
    fn test_v_cmpx_nlt_f32() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[0] = f32::to_bits(0.9);
        cpu.vec_reg[3] = f32::to_bits(0.4);
        cpu.interpret(&vec![0x7D3C0700, END_PRG]);
        assert_eq!(cpu.exec.read(), true);
    }

    #[test]
    fn test_cmp_class_f32() {
        let cpu = _helper_test_cpu();
        assert!(!cpu.cmp_class_f32(f32::NAN, 0b00001));
        assert!(cpu.cmp_class_f32(f32::NAN, 0b00010));

        assert!(cpu.cmp_class_f32(f32::INFINITY, 0b00000000000000000000001000000000));
        assert!(!cpu.cmp_class_f32(f32::INFINITY, 0b00000000000000000000000000000010));

        assert!(cpu.cmp_class_f32(f32::NEG_INFINITY, 0b00000000000000000000000000000100));
        assert!(!cpu.cmp_class_f32(f32::NEG_INFINITY, 0b00000000000000000000010000000000));

        assert!(!cpu.cmp_class_f32(0.752, 0b00000000000000000000000000000000));
        assert!(cpu.cmp_class_f32(0.752, 0b00000000000000000000000100000000));

        assert!(!cpu.cmp_class_f32(-0.752, 0b00000000000000000000010000000000));
        assert!(cpu.cmp_class_f32(-0.752, 0b00000000000000000000010000001000));

        assert!(!cpu.cmp_class_f32(1.0e-42, 0b11111111111111111111111101111111));
        assert!(cpu.cmp_class_f32(1.0e-42, 0b00000000000000000000000010000000));

        assert!(cpu.cmp_class_f32(-1.0e-42, 0b00000000000000000000000000010000));
        assert!(!cpu.cmp_class_f32(-1.0e-42, 0b11111111111111111111111111101111));

        assert!(cpu.cmp_class_f32(-0.0, 0b00000000000000000000000000100000));
        assert!(cpu.cmp_class_f32(0.0, 0b00000000000000000000000001000000));
    }
}
#[cfg(test)]
mod test_vop2 {
    use super::*;

    #[test]
    fn test_v_add_f32_e32() {
        let mut cpu = _helper_test_cpu();
        cpu.scalar_reg[2] = f32::to_bits(42.0);
        cpu.vec_reg[0] = f32::to_bits(1.0);
        cpu.interpret(&vec![0x06000002, END_PRG]);
        assert_eq!(f32::from_bits(cpu.vec_reg[0]), 43.0);
    }

    #[test]
    fn test_v_mul_f32_e32() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[2] = f32::to_bits(21.0);
        cpu.vec_reg[4] = f32::to_bits(2.0);
        cpu.interpret(&vec![0x10060504, END_PRG]);
        assert_eq!(f32::from_bits(cpu.vec_reg[3]), 42.0);
    }

    #[test]
    fn test_v_ashrrev_i32() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[0] = 4294967295;
        cpu.interpret(&vec![0x3402009F, END_PRG]);
        assert_eq!(cpu.vec_reg[1] as i32, -1);
    }

    #[test]
    fn test_v_mul_i32_i24() {
        [
            [18, 0x64, 1800],
            [0b10000000000000000000000000, 0b1, 0],
            [
                0b100000000000000000000000,
                0b1,
                0b11111111100000000000000000000000,
            ],
        ]
        .iter()
        .for_each(|[a, b, ret]| {
            let mut cpu = _helper_test_cpu();
            cpu.vec_reg[1] = *a;
            cpu.interpret(&vec![0x124E02FF, *b, END_PRG]);
            assert_eq!(cpu.vec_reg[39], *ret);
        });
    }

    #[test]
    fn test_v_add_nc_u32_const() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[18] = 7;
        cpu.interpret(&vec![0x4A3024B8, END_PRG]);
        assert_eq!(cpu.vec_reg[24], 63);
    }

    #[test]
    fn test_v_add_nc_u32_sint() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[14] = 7;
        cpu.vec_reg[6] = 4294967279;
        cpu.interpret(&vec![0x4A0C1D06, END_PRG]);
        assert_eq!(cpu.vec_reg[6], 4294967286);
    }
}

#[cfg(test)]
mod test_vopsd {
    use super::*;

    #[test]
    fn test_v_sub_co_u32() {
        [[69, 0, 69, 0], [100, 200, 4294967196, 1]]
            .iter()
            .for_each(|[a, b, ret, scc]| {
                let mut cpu = _helper_test_cpu();
                cpu.vec_reg[4] = *a;
                cpu.vec_reg[15] = *b;
                cpu.interpret(&vec![0xD7016A04, 0x00021F04, END_PRG]);
                assert_eq!(cpu.vec_reg[4], *ret);
                assert_eq!(cpu.vcc.read(), *scc != 0);
            })
    }
}

#[cfg(test)]
mod test_vop3 {
    use super::*;
    use float_cmp::approx_eq;

    fn helper_test_vop3(op: u32, a: f32, b: f32) -> f32 {
        let mut cpu = _helper_test_cpu();
        cpu.scalar_reg[0] = f32::to_bits(a);
        cpu.scalar_reg[6] = f32::to_bits(b);
        cpu.interpret(&vec![op, 0x00000006, END_PRG]);
        return f32::from_bits(cpu.vec_reg[0]);
    }

    #[test]
    fn test_v_add_f32() {
        assert_eq!(helper_test_vop3(0xd5030000, 0.4, 0.2), 0.6);
    }

    #[test]
    fn test_v_max_f32() {
        assert_eq!(helper_test_vop3(0xd5100000, 0.4, 0.2), 0.4);
        assert_eq!(helper_test_vop3(0xd5100000, 0.2, 0.8), 0.8);
    }

    #[test]
    fn test_v_mul_f32() {
        assert_eq!(helper_test_vop3(0xd5080000, 0.4, 0.2), 0.4 * 0.2);
    }

    #[test]
    fn test_signed_src() {
        // v0, max(s2, s2)
        let mut cpu = _helper_test_cpu();
        cpu.scalar_reg[2] = f32::to_bits(0.5);
        cpu.interpret(&vec![0xd5100000, 0x00000402, END_PRG]);
        assert_eq!(f32::from_bits(cpu.vec_reg[0]), 0.5);

        // v1, max(-s2, -s2)
        let mut cpu = _helper_test_cpu();
        cpu.scalar_reg[2] = f32::to_bits(0.5);
        cpu.interpret(&vec![0xd5100001, 0x60000402, END_PRG]);
        assert_eq!(f32::from_bits(cpu.vec_reg[1]), -0.5);
    }

    #[test]
    fn test_cnd_mask_cond_src_sgpr() {
        let mut cpu = _helper_test_cpu();
        cpu.scalar_reg[3] = 30;
        cpu.interpret(&vec![0xD5010000, 0x000D0280, END_PRG]);
        assert_eq!(cpu.vec_reg[0], 1);

        cpu.scalar_reg[3] = 0;
        cpu.interpret(&vec![0xD5010000, 0x000D0280, END_PRG]);
        assert_eq!(cpu.vec_reg[0], 0);
    }

    #[test]
    fn test_cnd_mask_cond_src_vcclo() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[2] = 20;
        cpu.vec_reg[0] = 100;
        cpu.interpret(&vec![0xD5010002, 0x41AA0102, END_PRG]);
        assert_eq!(cpu.vec_reg[2], 20);
    }

    #[test]
    fn test_v_cndmask_b32_e64_neg() {
        [[0.0f32, 0.0], [1.0f32, -1.0], [-1.0f32, 1.0]]
            .iter()
            .for_each(|[input, ret]| {
                let mut cpu = _helper_test_cpu();
                cpu.scalar_reg[0] = false as u32;
                cpu.vec_reg[3] = input.to_bits();
                cpu.interpret(&vec![0xD5010003, 0x2001FF03, 0x80000000, END_PRG]);
                assert_eq!(cpu.vec_reg[3], ret.to_bits());
            });
    }

    #[test]
    fn test_v_mul_hi_i32() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[2] = -2i32 as u32;
        cpu.interpret(&vec![0xD72E0003, 0x000204FF, 0x2E8BA2E9, END_PRG]);
        assert_eq!(cpu.vec_reg[3] as i32, -1);

        cpu.vec_reg[2] = 2;
        cpu.interpret(&vec![0xD72E0003, 0x000204FF, 0x2E8BA2E9, END_PRG]);
        assert_eq!(cpu.vec_reg[3], 0);
    }

    #[test]
    fn test_v_writelane_b32() {
        let mut cpu = _helper_test_cpu();
        cpu.scalar_reg[8] = 25056;
        cpu.interpret(&vec![0xD7610004, 0x00010008, END_PRG]);
        assert_eq!(cpu.vec_reg.get_lane(0)[4], 25056);

        cpu.scalar_reg[9] = 25056;
        cpu.interpret(&vec![0xD7610004, 0x00010209, END_PRG]);
        assert_eq!(cpu.vec_reg.get_lane(1)[4], 25056);
    }

    #[test]
    fn test_v_readlane_b32() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg.get_lane_mut(15)[4] = 0b1111;
        cpu.interpret(&vec![0xD760006A, 0x00011F04, END_PRG]);
        assert_eq!(cpu.vcc.read(), true);
    }

    #[test]
    fn test_v_lshlrev_b64() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg.write64(2, 100);
        cpu.vec_reg[4] = 2;
        cpu.interpret(&vec![0xD73C0002, 0x00020504, END_PRG]);
        assert_eq!(cpu.vec_reg.read64(2), 400);
    }

    #[test]
    fn test_v_lshrrev_b64() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg.write64(2, 100);
        cpu.vec_reg[4] = 2;
        cpu.interpret(&vec![0xd73d0002, 0x00020504, END_PRG]);
        assert_eq!(cpu.vec_reg.read64(2), 25);
    }

    #[test]
    fn test_v_cvt_f32_f16_abs_modifier() {
        [[0.4, 0.4], [-0.4, 0.4]].iter().for_each(|[a, ret]| {
            let mut cpu = _helper_test_cpu();
            cpu.vec_reg[1] = f16::from_f32_const(*a).to_bits() as u32;
            cpu.interpret(&vec![0xD58B0102, 0x00000101, END_PRG]);
            assert!(approx_eq!(
                f32,
                f32::from_bits(cpu.vec_reg[2]),
                *ret,
                (0.01, 2)
            ));
        });
    }

    #[test]
    fn test_v_alignbit_b32() {
        let mut cpu = _helper_test_cpu();
        cpu.scalar_reg[4] = 5340353;
        cpu.scalar_reg[10] = 3072795146;
        cpu.vec_reg[0] = 8;
        cpu.interpret(&vec![0xD6160001, 0x04001404, END_PRG]);
        assert_eq!(cpu.vec_reg[1], 3250005794);
    }

    #[test]
    fn test_v_bfe_i32() {
        [
            [0b00000000000000000000000000000001, -1],
            [0b00000000000000000000000000000000, 0],
            [0b00000000000000000000000000000010, 0],
        ]
        .iter()
        .for_each(|[a, ret]| {
            let mut cpu = _helper_test_cpu();
            cpu.vec_reg[2] = *a as u32;
            cpu.interpret(&vec![0xD6110005, 0x02050102, END_PRG]);
            assert_eq!(cpu.vec_reg[5] as i32, *ret);
        });

        [
            [0b00000000000000000000000000000010, -2],
            [0b00000000000000000000000000000001, 1],
            [0b00000000000000000000000000000100, 0],
        ]
        .iter()
        .for_each(|[a, ret]| {
            let mut cpu = _helper_test_cpu();
            cpu.vec_reg[2] = *a as u32;
            cpu.interpret(&vec![0xD6110005, 0x02090102, END_PRG]);
            assert_eq!(cpu.vec_reg[5] as i32, *ret);
        });

        [
            [
                0b00100000000000000000000000000000,
                0b100000000000000000000000000000,
            ],
            [0b00000000000000001000000000000000, 0b1000000000000000],
            [-1, -1],
        ]
        .iter()
        .for_each(|[a, ret]| {
            let mut cpu = _helper_test_cpu();
            cpu.vec_reg[2] = *a as u32;
            cpu.interpret(&vec![0xD6110005, 0x03050102, END_PRG]);
            assert_eq!(cpu.vec_reg[5] as i32, *ret);
        });
    }

    #[test]
    fn test_v_ashrrev_i16() {
        let mut cpu = _helper_test_cpu();
        [
            [0b10000000000000000000000000000000, 0],
            [0b10000000000000000000000000000111, 3],
            [0b0000000000000000, 0],
            [0b1000000000000000, 0b1100000000000000],
            [0b0100000000000000, 0b0010000000000000],
            [0b0010000000000000, 0b0001000000000000],
            [0b1010000000000000, 0b1101000000000000],
            [0b1110000000000000, 0b1111000000000000],
            [0b0110000000000000, 0b0011000000000000],
        ]
        .iter()
        .for_each(|[a, ret]| {
            cpu.vec_reg[2] = *a;
            cpu.scalar_reg[1] = 1;
            cpu.interpret(&vec![0xd73a0005, 0b11000001100000010000000001, END_PRG]);
            assert_eq!(cpu.vec_reg[5], *ret);
        });

        [
            [0b1000000000000000, 0b1111, 0b1111111111111111],
            [0b1000000000000000, 0b11111, 0b1111111111111111],
            [0b1000000000000000, 0b0111, 0b1111111100000000],
        ]
        .iter()
        .for_each(|[a, shift, ret]| {
            cpu.vec_reg[2] = *a;
            cpu.scalar_reg[1] = *shift;
            cpu.interpret(&vec![0xd73a0005, 0b11000001100000010000000001, END_PRG]);
            assert_eq!(cpu.vec_reg[5], *ret);
        });

        cpu.vec_reg[5] = 0b11100000000000001111111111111111;
        cpu.vec_reg[2] = 0b0100000000000000;
        cpu.scalar_reg[1] = 1;
        cpu.interpret(&vec![0xd73a0005, 0b11000001100000010000000001, END_PRG]);
        assert_eq!(cpu.vec_reg[5], 0b11100000000000000010000000000000);
    }

    #[test]
    fn test_v_add_nc_u16() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[5] = 10;
        cpu.vec_reg[8] = 20;
        cpu.interpret(&vec![0xD7030005, 0x00021105, END_PRG]);
        assert_eq!(cpu.vec_reg[5], 30);
    }

    #[test]
    fn test_v_mul_lo_u16() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[5] = 2;
        cpu.vec_reg[15] = 0;
        cpu.interpret(&vec![0xD705000F, 0x00010B05, END_PRG]);
        assert_eq!(cpu.vec_reg[15], 10);

        cpu.vec_reg[5] = 2;
        cpu.vec_reg[15] = 0b10000000000000000000000000000000;
        cpu.interpret(&vec![0xD705000F, 0x00010B05, END_PRG]);
        assert_eq!(cpu.vec_reg[15], 0b10000000000000000000000000000000 + 10);
    }

    #[test]
    fn test_v_cmp_gt_u16() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[1] = 52431;
        cpu.scalar_reg[5] = 0;
        cpu.interpret(&vec![0xD43C0005, 0x000202FF, 0x00003334, END_PRG]);
        assert_eq!(cpu.scalar_reg[5], 0);
    }

    #[test]
    fn test_v_cmp_ngt_f32_abs() {
        [
            (0.5f32, 0.5f32, 1),
            (-0.5, 0.5, 1),
            (0.1, 0.2, 0),
            (-0.1, 0.2, 0),
        ]
        .iter()
        .for_each(|(x, y, ret)| {
            let mut cpu = _helper_test_cpu();
            cpu.scalar_reg[2] = x.to_bits();
            cpu.interpret(&vec![0xD41B0203, 0x000004FF, y.to_bits(), END_PRG]);
            assert_eq!(cpu.scalar_reg[3], *ret);
        })
    }
    #[test]
    fn test_fma() {
        fn v_fma_f32(a: u32, b: u32, c: u32, ret: u32) {
            let mut cpu = _helper_test_cpu();
            cpu.vec_reg[1] = b;
            cpu.scalar_reg[3] = c;
            cpu.interpret(&vec![0xD6130000, 0x000E02FF, a, END_PRG]);
            assert_eq!(cpu.vec_reg[0], ret);
        }
        fn v_fmac_f32(a: u32, b: u32, c: u32, ret: u32) {
            let mut cpu = _helper_test_cpu();
            cpu.scalar_reg[1] = a;
            cpu.scalar_reg[2] = b;
            cpu.vec_reg[0] = c;
            cpu.interpret(&vec![0xd52b0000, 0x401, END_PRG]);
            assert_eq!(cpu.vec_reg[0], ret);
        }
        [[0xbfc90fda, 1186963456, 1192656896, 3204127872]]
            .iter()
            .for_each(|[a, b, c, ret]| {
                v_fma_f32(*a, *b, *c, *ret);
                v_fmac_f32(*a, *b, *c, *ret);
            })
    }

    #[test]
    fn test_v_perm_b32() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[1] = 15944;
        cpu.vec_reg[0] = 84148480;
        cpu.interpret(&vec![0xD644000F, 0x03FE0101, 0x05040100, END_PRG]);
        assert_eq!(cpu.vec_reg[15], 1044906240);
    }
}

#[cfg(test)]
mod test_vopp {
    use super::*;

    #[test]
    fn test_v_fma_mix_f32() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[2] = 1065353216;
        cpu.scalar_reg[2] = 3217620992;
        cpu.vec_reg[1] = 15360;
        cpu.interpret(&vec![0xCC204403, 0x04040502, END_PRG]);
        assert_eq!(cpu.vec_reg[3], 3205627904);

        cpu.vec_reg[2] = 1065353216;
        cpu.scalar_reg[2] = 3217620992;
        cpu.vec_reg[1] = 48128;
        cpu.interpret(&vec![0xCC204403, 0x04040502, END_PRG]);
        assert_eq!(cpu.vec_reg[3], 3205627904);
    }

    #[test]
    fn test_packed_opsel_000_op_000() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[1] = 1;
        cpu.vec_reg[2] = 2;
        cpu.vec_reg[3] = 3;
        cpu.interpret(&vec![0xCC090004, 0x040E0501, 0xBFB00000, END_PRG]);
        assert_eq!(cpu.vec_reg[4], 0b1010000000000000101);
    }

    #[test]
    fn test_packed_opsel_001_op_100() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[1] = 1;
        cpu.vec_reg[2] = 2;
        cpu.vec_reg[3] = 3;
        cpu.interpret(&vec![0xCC092004, 0x0C0E0501, 0xBFB00000, END_PRG]);
        assert_eq!(cpu.vec_reg[4], 0b110000000000000010);
    }

    #[test]
    fn test_packed_inline_const_int() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[1] = 1;
        cpu.vec_reg[2] = 2;
        cpu.vec_reg[3] = 3;

        cpu.interpret(&vec![0xCC090004, 0x020E0501, 0xBFB00000, END_PRG]);
        assert_eq!(cpu.vec_reg[4], 0b1010000000000000101);

        cpu.interpret(&vec![0xCC090804, 0x0A0E0501, 0xBFB00000, END_PRG]);
        assert_eq!(cpu.vec_reg[4], 0b110000000000000011);

        cpu.interpret(&vec![0xCC096004, 0x020E0501, 0xBFB00000, END_PRG]);
        assert_eq!(cpu.vec_reg[4], 0b100000000000000010);

        cpu.interpret(&vec![0xCC090004, 0x03FE0501, 0x00000080, END_PRG]);
        assert_eq!(cpu.vec_reg[4], 8519810);
    }

    #[test]
    fn test_pk_fma_f16_inline_const() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[2] = 0x393a35f6;
        cpu.vec_reg[3] = 0x2800;

        cpu.interpret(&vec![0xCC0E0004, 0x03FE0702, 0x0000A400, END_PRG]);
        assert_eq!(cpu.vec_reg[4], 2618596372);

        cpu.interpret(&vec![0xCC0E0004, 0x0BFE0702, 0x0000A400, END_PRG]);
        assert_eq!(cpu.vec_reg[4], 485006356);

        cpu.interpret(&vec![0xCC0E0004, 0x1BFE0702, 0x0000A400, END_PRG]);
        assert_eq!(cpu.vec_reg[4], 2751503380);

        cpu.interpret(&vec![0xCC0E0804, 0x03FE0702, 0x0000A400, END_PRG]);
        assert_eq!(cpu.vec_reg[4], 2618563816);

        cpu.interpret(&vec![0xCC0E1804, 0x03FE0702, 0x0000A400, END_PRG]);
        assert_eq!(cpu.vec_reg[4], 2618598400);
    }

    #[test]
    fn test_v_fma_mixhilo_f16() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[11] = 1065353216;
        cpu.vec_reg[7] = 3047825943;
        cpu.vec_reg[16] = 3047825943;

        cpu.vec_reg[14] = 0b10101010101010101111111111111111;
        cpu.interpret(&vec![0xCC21000E, 0x04420F0B, END_PRG]);
        assert_eq!(cpu.vec_reg[14], 0b10101010101010101000000000101011);

        cpu.vec_reg[14] = 0b10101010101010101111111111111111;
        cpu.interpret(&vec![0xCC22000E, 0x04420F0B, END_PRG]);
        assert_eq!(cpu.vec_reg[14], 0b10000000001010111111111111111111);
    }

    #[test]
    fn test_v_pk_lshlrev_b16() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[3] = 0b1010101011101101;

        cpu.interpret(&vec![0xCC044004, 0x0002068E, END_PRG]);
        assert_eq!(cpu.vec_reg[4], 0b1000000000000000100000000000000);

        cpu.interpret(&vec![0xCC044004, 0x1002068E, END_PRG]);
        assert_eq!(cpu.vec_reg[4], 0b100000000000000);

        cpu.interpret(&vec![0xCC044004, 0x100206FF, 0x00010002, END_PRG]);
        assert_eq!(cpu.vec_reg[4], 0b1010101110110100);
        cpu.interpret(&vec![0xCC044004, 0x100206FF, 0x05012002, END_PRG]);
        assert_eq!(cpu.vec_reg[4], 0b1010101110110100);

        cpu.interpret(&vec![0xCC044004, 0x100206FF, 0x0503E00F, END_PRG]);
        assert_eq!(cpu.vec_reg[4], 0b1000000000000000);
        cpu.interpret(&vec![0xCC044004, 0x100206FF, 0x0503E007, END_PRG]);
        assert_eq!(cpu.vec_reg[4], 0b111011010000000);
        cpu.interpret(&vec![0xCC044004, 0x100206FF, 0x0503E01F, END_PRG]);
        assert_eq!(cpu.vec_reg[4], 0b1000000000000000);
    }

    #[test]
    fn test_pk_fma_with_neg() {
        let mut cpu = _helper_test_cpu();
        let a1 = f16::from_f32(1.0);
        let b1 = f16::from_f32(2.0);
        let c1 = f16::from_f32(3.0);

        let a2 = f16::from_f32(4.0);
        let b2 = f16::from_f32(5.0);
        let c2 = f16::from_f32(6.0);

        cpu.vec_reg[0] = (a1.to_bits() as u32) << 16 | (a2.to_bits() as u32);
        cpu.vec_reg[9] = (b1.to_bits() as u32) << 16 | (b2.to_bits() as u32);
        cpu.vec_reg[10] = (c1.to_bits() as u32) << 16 | (c2.to_bits() as u32);

        cpu.interpret(&vec![0xCC0E3805, 0x042A1300, END_PRG]);
        assert_eq!(cpu.vec_reg[5], 1317029120);

        cpu.interpret(&vec![0xCC0E3805, 0x242A1300, END_PRG]);
        assert_eq!(cpu.vec_reg[5], 1317026816);

        cpu.interpret(&vec![0xCC0E3B05, 0x042A1300, END_PRG]);
        assert_eq!(cpu.vec_reg[5], 1317029120);

        cpu.interpret(&vec![0xCC0E3905, 0x042A1300, END_PRG]);
        assert_eq!(cpu.vec_reg[5], 3405792512);
    }

    #[test]
    fn test_pk_add_f16_with_float_const() {
        let mut cpu = _helper_test_cpu();
        let a1 = f16::from_f32(5.0);
        let a2 = f16::from_f32(10.0);

        cpu.vec_reg[1] = (a1.to_bits() as u32) << 16 | (a2.to_bits() as u32);
        cpu.interpret(&vec![0xCC0F4002, 0x0001E501, END_PRG]);
        assert_eq!(cpu.vec_reg[2], 1233144192);

        cpu.interpret(&vec![0xCC0F5002, 0x0001E501, END_PRG]);
        assert_eq!(cpu.vec_reg[2], 1233144064);

        cpu.interpret(&vec![0xCC0F5002, 0x1001E501, END_PRG]);
        assert_eq!(cpu.vec_reg[2], 1224755456);

        cpu.interpret(&vec![0xCC0F5802, 0x1801E501, END_PRG]);
        assert_eq!(cpu.vec_reg[2], 1157645568);
    }
}

#[cfg(test)]
mod test_flat {
    use super::*;
    use std::alloc::{alloc, handle_alloc_error, Layout};

    #[test]
    fn test_scratch_swap_values() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[13] = 42;
        cpu.vec_reg[14] = 10;
        cpu.interpret(&vec![
            0xDC690096, 0x007C0D00, 0xDC69001E, 0x007C0E00, 0xDC51001E, 0x0D7C0000, 0xDC510096,
            0x0E7C0000, END_PRG,
        ]);
        assert_eq!(cpu.vec_reg[13], 10);
        assert_eq!(cpu.vec_reg[14], 42);
    }

    #[test]
    fn test_scratch_load_dword_offset() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[14] = 14;
        cpu.vec_reg[15] = 23;
        cpu.interpret(&vec![
            0xDC6D000A, 0x007C0E00, 0xDC51000A, 0x0E7C0000, END_PRG,
        ]);
        assert_eq!(cpu.vec_reg[14], 14);

        cpu.interpret(&vec![
            0xDC6D000A, 0x007C0E00, 0xDC51000E, 0x0E7C0000, END_PRG,
        ]);
        assert_eq!(cpu.vec_reg[14], 23);
    }

    #[test]
    fn test_global_load_d16_hi_b16() {
        let mut cpu = _helper_test_cpu();
        cpu.vec_reg[13] = 0b10101011101101001111111111111111;
        unsafe {
            let layout = Layout::new::<u16>();
            let ptr = alloc(layout);
            if ptr.is_null() {
                handle_alloc_error(layout)
            }
            *(ptr as *mut u16) = 42;
            cpu.vec_reg.write64(10, ptr as u64);
        }
        cpu.interpret(&vec![0xDC8E0000, 0x0D7C000A, END_PRG]);
        assert_eq!(cpu.vec_reg[13], 0b00000000001010101111111111111111);
    }
}
