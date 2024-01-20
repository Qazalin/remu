use crate::allocator::BumpAllocator;
use crate::alu_modifiers::VOPModifier;
use crate::state::{Assign, RegisterGroup, VCC};
use crate::todo_instr;
use crate::utils::{as_signed, Colorize, DebugLevel, DEBUG};

const SGPR_COUNT: u32 = 105;
const VGPR_COUNT: u32 = 256;
const NULL_SRC: u32 = 124;
pub const END_PRG: u32 = 0xbfb00000;
const NOOPS: [u32; 1] = [0xbfb60003];

pub struct CPU {
    pub pc: u64,
    pub gds: BumpAllocator,
    pub lds: BumpAllocator,
    pub scalar_reg: RegisterGroup,
    pub vec_reg: RegisterGroup,
    pub scc: u32,
    pub vcc: VCC,
    pub exec: u32,
    prg: Vec<u32>,
}

impl CPU {
    pub fn new(gds: BumpAllocator, lds: BumpAllocator) -> Self {
        return CPU {
            pc: 0,
            scc: 0,
            vcc: VCC::from(0),
            exec: 0,
            gds,
            lds,
            scalar_reg: RegisterGroup::new(105, "SGPR"),
            vec_reg: RegisterGroup::new(256, "VGPR"),
            prg: vec![],
        };
    }

    pub fn interpret(&mut self, prg: &Vec<u32>) {
        self.pc = 0;
        self.vcc.assign(0);
        self.exec = 1;
        self.prg = prg.to_vec();

        loop {
            let instruction = prg[self.pc as usize];
            self.pc += 1;

            if instruction == END_PRG {
                break;
            }
            if NOOPS.contains(&instruction) || instruction >> 20 == 0xbf8 {
                continue;
            }

            self.exec(instruction);
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
            && self.exec == 0
        {
            return;
        }
        // smem
        if instruction >> 26 == 0b111101 {
            let instr = self.u64_instr();
            // NOTE: sbase has an implied LSB of zero
            /*
             * In reads, the address-base comes from an SGPR-pair, it's always
             * even-aligned. s[sbase:sbase+1]
             */
            let sbase = (instr & 0x3f) * 2;
            let sdata = ((instr >> 6) & 0x7f) as usize;
            let dlc = (instr >> 13) & 0x1;
            let glc = (instr >> 14) & 0x1;
            let op = (instr >> 18) & 0xff;
            let offset = as_signed((instr >> 32) & 0x1fffff, 21);
            let soffset = match instr & 0x7F {
                _ => 0, // TODO soffset is not implemented
            };

            if *DEBUG >= DebugLevel::INSTRUCTION {
                println!(
                    "{} sbase={} sdata={} dlc={} glc={} op={} offset={} soffset={}",
                    "SMEM".color("blue"),
                    sbase,
                    sdata,
                    dlc,
                    glc,
                    op,
                    offset,
                    soffset
                );
            }
            let base_addr = self.scalar_reg.read64(sbase as usize);
            let effective_addr = (base_addr as i64 + offset + soffset as i64) as u64;

            match op {
                0..=4 => (0..2_usize.pow(op as u32)).for_each(|i| {
                    self.scalar_reg[sdata + i] = self.gds.read(effective_addr + (4 * i as u64));
                }),
                _ => todo_instr!(instruction),
            }
        }
        // sop1
        else if instruction >> 23 == 0b10_1111101 {
            let s0 = self.resolve_src(instruction & 0xFF) as u32;
            let op = (instruction >> 8) & 0xFF;
            let sdst = (instruction >> 16) & 0x7F;

            if *DEBUG >= DebugLevel::INSTRUCTION {
                println!("{} s0={} sdst={} op={}", "SOP1".color("blue"), s0, sdst, op,);
            }

            match op {
                0 => self.write_to_sdst(sdst, s0),
                1 => self.scalar_reg.write64(sdst as usize, s0 as u64),
                16 => {
                    let sdst_val = self.resolve_src(sdst) as u32;
                    self.write_to_sdst(sdst, sdst_val >> s0);
                }
                30 => self.write_to_sdst(sdst, !s0),
                32 | 34 | 48 => {
                    let saveexec = self.exec;
                    self.exec = match op {
                        32 => s0 & self.exec,
                        34 => s0 | self.exec,
                        48 => s0 & !self.exec,
                        _ => panic!(),
                    };
                    self.scc = (self.exec != 0) as u32;
                    self.write_to_sdst(sdst, saveexec)
                }
                _ => todo_instr!(instruction),
            };
        }
        // sopc
        else if (instruction >> 23) & 0x3ff == 0b101111110 {
            let s0 = self.resolve_src(instruction & 0xff) as u32;
            let s1 = self.resolve_src((instruction >> 8) & 0xff) as u32;
            let op = (instruction >> 16) & 0x7f;

            if *DEBUG >= DebugLevel::INSTRUCTION {
                println!(
                    "{} ssrc0={} ssrc1={} op={}",
                    "SOPC".color("blue"),
                    s0,
                    s1,
                    op
                );
            }

            self.scc = match op {
                0..=4 => {
                    let s0 = s0 as i32;
                    let s1 = s1 as i32;
                    match op {
                        2 => s0 > s1,
                        4 => s0 < s1,
                        _ => todo_instr!(instruction),
                    }
                }
                5..=11 => match op {
                    6 => s0 == s1,
                    8 => s0 > s1,
                    9 => s0 >= s1,
                    10 => s0 < s1,
                    _ => todo_instr!(instruction),
                },
                _ => todo_instr!(instruction),
            } as u32;

            if *DEBUG >= DebugLevel::STATE {
                println!("{} {}", "SCC".color("pink"), self.scc);
            }
        }
        // sopp
        else if instruction >> 23 == 0b10_1111111 {
            let simm16 = (instruction & 0xffff) as i16;
            let op = (instruction >> 16) & 0x7f;

            if *DEBUG >= DebugLevel::INSTRUCTION {
                println!(
                    "{} simm16={} op={} pc={}",
                    "SOPP".color("blue"),
                    simm16,
                    op,
                    self.pc
                );
            }

            match op {
                32..=42 => {
                    let should_jump = match op {
                        32 => true,
                        33 => self.scc == 0,
                        34 => self.scc == 1,
                        35 => *self.vcc == 0,
                        36 => *self.vcc != 0,
                        37 => self.exec == 0,
                        38 => self.exec != 0,
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
            let simm16 = (instruction & 0xffff) as i16;
            let sdst = (instruction >> 16) & 0x7f;
            let op = (instruction >> 23) & 0x1f;
            let s0 = self.resolve_src(sdst) as u32;

            if *DEBUG >= DebugLevel::INSTRUCTION {
                println!(
                    "{} simm16={} sdst={} op={}",
                    "SOPK".color("blue"),
                    simm16,
                    s0,
                    op
                );
            }

            match op {
                0 => self.scalar_reg[sdst as usize] = simm16 as i32 as u32,
                3 => self.scc = (s0 as i32 as i64 == simm16 as i64) as u32,
                4 => self.scc = (s0 as i32 as i64 != simm16 as i64) as u32,
                15 => {
                    let temp = s0 as i32;
                    let dest = (temp as i64 + simm16 as i64) as i32;
                    self.write_to_sdst(sdst, dest as u32);
                    let temp_sign = ((temp >> 31) & 1) as u32;
                    let simm_sign = ((simm16 >> 15) & 1) as u32;
                    let dest_sign = ((dest >> 31) & 1) as u32;
                    self.scc = ((temp_sign == simm_sign) && (temp_sign != dest_sign)) as u32;
                }
                16 => {
                    let ret = (s0 as i32 * simm16 as i32) as u32;
                    self.write_to_sdst(sdst, ret);
                }
                _ => todo_instr!(instruction),
            }
        }
        // sop2
        else if instruction >> 30 == 0b10 {
            let s0 = self.resolve_src(instruction & 0xFF) as u32;
            let s1 = self.resolve_src((instruction >> 8) & 0xFF) as u32;
            let sdst = (instruction >> 16) & 0x7F;
            let op = (instruction >> 23) & 0xFF;

            if *DEBUG >= DebugLevel::INSTRUCTION {
                println!(
                    "{} ssrc0={} ssrc1={} sdst={} op={}",
                    "SOP2".color("blue"),
                    s0,
                    s1,
                    sdst,
                    op
                );
            }

            let ret64 = match op {
                9 => {
                    let s0 = self.scalar_reg.read64(instruction as usize & 0xFF);
                    Some(s0 << (s1 as u64))
                }
                _ => None,
            };
            if let Some(ret64) = ret64 {
                self.scalar_reg.write64(sdst as usize, ret64);
                return;
            }

            let ret = match op {
                0 | 4 => {
                    let s0 = s0 as u64;
                    let s1 = s1 as u64;
                    let temp = match op {
                        0 => s0 + s1,
                        4 => s0 + s1 + self.scc as u64,
                        _ => panic!(),
                    };
                    self.scc = (temp >= 0x100000000) as u32;
                    temp as u32
                }
                2 | 3 | 44 | 46 => {
                    let s0 = s0 as i32;
                    let s1 = s1 as i32;
                    let temp = match op {
                        2 => s0 + s1,
                        3 => s0 - s1,
                        44 => s0 * s1,
                        46 => ((s0 as i64 * s1 as i64) >> 32) as i32,
                        _ => panic!(),
                    };
                    temp as u32
                }
                8 | 10 | 9 | 12 => {
                    let ret = match op {
                        8 => s0 << s1,
                        9 => ((s0 as u64) << (s1 as u64 & 0x1F)) as u32,
                        10 => s0 >> s1,
                        12 => ((s0 as i32) >> (s1 as i32)) as u32,
                        _ => panic!(),
                    };
                    self.scc = (ret != 0) as u32;
                    ret
                }
                18 | 20 => {
                    self.scc = match op {
                        18 => (s0 as i32) < (s1 as i32),
                        20 => (s0 as i32) > (s1 as i32),
                        _ => panic!(),
                    } as u32;
                    (match self.scc != 0 {
                        true => s0,
                        false => s1,
                    }) as u32
                }
                22..=34 => {
                    let temp = match op {
                        22 => s0 & s1,
                        24 => s0 | s1,
                        26 => s0 ^ s1,
                        34 => s0 & !s1,
                        _ => panic!(),
                    };
                    self.scc = (temp != 0) as u32;
                    temp as u32
                }
                45 => ((s0 as u64) * (s1 as u64) >> 32) as u32,
                48 => match self.scc != 0 {
                    true => s0,
                    false => s1,
                },
                _ => todo_instr!(instruction),
            };
            self.write_to_sdst(sdst, ret);
        }
        // vop1
        else if instruction >> 25 == 0b0111111 {
            let s0 = self.resolve_src(instruction & 0x1ff) as u32;
            let op = (instruction >> 9) & 0xff;
            let vdst = ((instruction >> 17) & 0xff) as usize;

            if *DEBUG >= DebugLevel::INSTRUCTION {
                println!(
                    "{} src={} op={} vdst={}",
                    "VOP1".color("blue"),
                    s0,
                    op,
                    vdst,
                );
            }

            self.vec_reg[vdst] = match op {
                1 => s0,
                2 => {
                    assert!(self.exec == 1);
                    self.scalar_reg[vdst] = s0;
                    s0
                }
                5 => (s0 as i32 as f32).to_bits(),
                6 => (s0 as f32).to_bits(),
                7 => f32::from_bits(s0) as u32,
                8 => f32::from_bits(s0) as i32 as u32,
                56 => s0.reverse_bits(),
                35..=51 => {
                    let s0 = f32::from_bits(s0);
                    match op {
                        35 => {
                            let mut temp = f32::floor(s0 + 0.5);
                            if f32::floor(s0) % 2.0 != 0.0 && f32::fract(s0) == 0.5 {
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
                _ => todo_instr!(instruction),
            };
        }
        // vopd
        else if instruction >> 26 == 0b110010 {
            let instr = self.u64_instr();

            let sx = instr & 0x1ff;
            let vx = (instr >> 9) & 0xff;
            let srcx0 = self.resolve_src((sx) as u32) as u32;
            let vsrcx1 = self.vec_reg[(vx) as usize] as u32;
            let opy = (instr >> 17) & 0x1f;

            let sy = (instr >> 32) & 0x1ff;
            let vy = (instr >> 41) & 0xff;
            let opx = (instr >> 22) & 0xf;
            let srcy0 = match sy {
                255 => match sx {
                    255 => srcx0,
                    _ => self.resolve_src(sy as u32) as u32,
                },
                _ => self.resolve_src(sy as u32) as u32,
            };
            let vsrcy1 = self.vec_reg[(vy) as usize];

            let vdstx = (instr >> 56) & 0xff;
            // LSB is the opposite of VDSTX[0]
            let vdsty = ((instr >> 49) & 0x7f) << 1 | ((vdstx & 1) ^ 1);

            if *DEBUG >= DebugLevel::INSTRUCTION {
                println!(
                    "{} X=[op={}, dest={} src({sx})={}, vsrc({vx})={}] Y=[op={}, dest={}, src({sy})={}, vsrc({vy})={}]",
                    "VOPD".color("blue"),
                    opx,
                    vdstx,
                    srcx0,
                    vsrcx1,
                    opy,
                    vdsty,
                    srcy0,
                    vsrcy1,
                );
            }

            ([(opx, srcx0, vsrcx1, vdstx), (opy, srcy0, vsrcy1, vdsty)])
                .iter()
                .for_each(|(op, s0, s1, dst)| {
                    self.vec_reg[*dst as usize] = match *op {
                        0 | 1 | 3 | 4 | 5 | 6 | 10 => {
                            let s0 = f32::from_bits(*s0 as u32);
                            let s1 = f32::from_bits(*s1 as u32);
                            match *op {
                                0 => s0 * s1 + f32::from_bits(self.vec_reg[*dst as usize]),
                                1 => s0 * s1 + f32::from_bits(self.simm()),
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
                            9 => match *self.vcc != 0 {
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
            let s0 = self.resolve_src(instruction & 0x1ff) as u32;
            let s1 = self.vec_reg[((instruction >> 9) & 0xff) as usize] as u32;
            let op = (instruction >> 17) & 0xff;

            if *DEBUG >= DebugLevel::INSTRUCTION {
                println!("{} s0={} s1={} op={}", "VOPC".color("blue"), s0, s1, op);
            }

            match op {
                17 | 18 | 27 | 20 | 22 | 30 => {
                    let s0 = f32::from_bits(s0);
                    let s1 = f32::from_bits(s1);
                    self.vcc.assign(match op {
                        17 => s0 < s1,
                        18 => s0 == s1,
                        20 => s0 > s1,
                        22 => s0 >= s1,
                        27 => !(s0 > s1),
                        30 => !(s0 < s1),
                        _ => panic!(),
                    });
                }
                155 | 158 => {
                    let s0 = f32::from_bits(s0);
                    let s1 = f32::from_bits(s1);
                    self.exec = match op {
                        155 => !(s0 > s1),
                        158 => !(s0 < s1),
                        _ => panic!(),
                    } as u32;
                }
                57 | 58 | 60 | 62 => {
                    let s0 = s0 as u16;
                    let s1 = s1 as u16;

                    self.vcc.assign(match op {
                        58 => s0 == s1,
                        57 => s0 < s1,
                        60 => s0 > s1,
                        62 => s0 >= s1,
                        _ => panic!(),
                    })
                }

                52 | 65 => {
                    let s0 = s0 as i16;
                    let s1 = s1 as i16;

                    self.vcc.assign(match op {
                        52 => s0 > s1,
                        65 => s0 < s1,
                        _ => panic!(),
                    })
                }

                68 => self.vcc.assign(s0 as i32 > s1 as i32),
                74 => self.vcc.assign(s0 == s1),
                77 => self.vcc.assign(s0 != s1),
                193 => self.exec = ((s0 as i32) < (s1 as i32)) as u32,
                196 => self.exec = ((s0 as i32) > (s1 as i32)) as u32,
                202 => self.exec = (s0 == s1) as u32,
                _ => todo_instr!(instruction),
            };
        }
        // vop2
        else if instruction >> 31 == 0b0 {
            let s0 = self.resolve_src(instruction & 0x1FF) as u32;
            let s1 = self.vec_reg[((instruction >> 9) & 0xFF) as usize];
            let vdst = (instruction >> 17) & 0xFF;
            let op = (instruction >> 25) & 0x3F;

            if *DEBUG >= DebugLevel::INSTRUCTION {
                println!(
                    "{} ssrc0={} vsrc1={} vdst={} op={}",
                    "VOP2".color("blue"),
                    s0,
                    s1,
                    vdst,
                    op
                );
            }

            self.vec_reg[vdst as usize] = match op {
                1 => match *self.vcc != 0 {
                    true => s1,
                    false => s0,
                },

                3 | 4 | 8 | 16 | 43 | 44 | 45 => {
                    let s0 = f32::from_bits(s0);
                    let s1 = f32::from_bits(s1);
                    match op {
                        3 => s0 + s1,
                        4 => s0 - s1,
                        8 => s0 * s1,
                        16 => f32::max(s0, s1),
                        43 => s0 * s1 + f32::from_bits(self.vec_reg[vdst as usize]),
                        44 => s0 * f32::from_bits(self.simm()) + s1,
                        45 => s0 * s1 + f32::from_bits(self.simm()),
                        _ => panic!(),
                    }
                    .to_bits()
                }

                9 | 18 | 26 => {
                    let s0 = s0 as i32;
                    let s1 = s1 as i32;

                    (match op {
                        9 => s0 * s1,
                        18 => i32::max(s0, s1),
                        26 => s1 >> s0,
                        _ => panic!(),
                    }) as u32
                }

                11 => s0 * s1,

                24 => s1 << s0,
                29 => s0 ^ s1,
                25 => s1 >> s0,
                27 => s0 & s1,
                28 => s0 | s1,

                32 => {
                    let temp = s0 as u64 + s1 as u64 + *self.vcc as u64;
                    self.vcc.assign(if temp >= 0x100000000 { 1 } else { 0 });
                    temp as u32
                }
                33 | 34 => {
                    let temp = match op {
                        33 => s0 - s1 - *self.vcc,
                        34 => s1 - s0 - *self.vcc,
                        _ => panic!(),
                    };
                    self.vcc
                        .assign(((s1 as u64 + *self.vcc as u64) > s0 as u64) as u32);
                    temp
                }

                37 => s0 + s1,
                38 => s0 - s1,
                39 => s1 - s0,

                _ => todo_instr!(instruction),
            };
        }
        // vop3
        else if instruction >> 26 == 0b110101 {
            let instr = self.u64_instr();

            let op = (instr >> 16) & 0x3ff;
            match op {
                764 | 288 | 766 | 768 => {
                    let vdst = (instr & 0xff) as usize;
                    let sdst = ((instr >> 8) & 0x7f) as usize;
                    let s0 = self.resolve_src(((instr >> 32) & 0x1ff) as u32) as u32;
                    let s1 = self.resolve_src(((instr >> 41) & 0x1ff) as u32) as u32;
                    let s2 = self.resolve_src(((instr >> 50) & 0x1ff) as u32) as u32;
                    let omod = (instr >> 59) & 0x3;
                    let neg = (instr >> 61) & 0x7;
                    let clmp = (instr >> 15) & 0x1;
                    assert_eq!(neg, 0);
                    assert_eq!(omod, 0);
                    assert_eq!(clmp, 0);

                    if *DEBUG >= DebugLevel::INSTRUCTION {
                        println!(
                            "{} vdst={} sdst={} op={} s0={} s1={} s2={} omod={} neg={}",
                            "VOPSD".color("blue"),
                            vdst,
                            sdst,
                            op,
                            s0,
                            s1,
                            s2,
                            omod,
                            neg
                        )
                    }

                    let (ret, vcc) = match op {
                        288 => {
                            let ret = s0 as u64 + s1 as u64 + *VCC::from(s2) as u64;
                            (ret as u32, ret >= 0x100000000)
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
                        _ => todo_instr!(instruction),
                    };
                    match sdst {
                        106 => self.vcc.assign(vcc),
                        124 => {}
                        _ => self.scalar_reg[sdst] = *VCC::from(vcc as u32),
                    }
                    self.vec_reg[vdst] = ret;
                }
                _ => {
                    let vdst = (instr & 0xff) as usize;
                    let abs = ((instr >> 8) & 0x7) as usize;
                    let opsel = (instr >> 11) & 0xf;
                    let cm = (instr >> 15) & 0x1;

                    let s0 = self.resolve_src(((instr >> 32) & 0x1ff) as u32) as u32;
                    let s1 = self.resolve_src(((instr >> 41) & 0x1ff) as u32) as u32;
                    let s2 = self.resolve_src(((instr >> 50) & 0x1ff) as u32) as u32;

                    let omod = (instr >> 59) & 0x3;
                    let neg = ((instr >> 61) & 0x7) as usize;
                    assert_eq!(omod, 0);
                    assert_eq!(cm, 0);

                    if *DEBUG >= DebugLevel::INSTRUCTION {
                        println!(
                            "{} vdst={} abs={} opsel={} op={} s0={} src1={} src2={} neg=0b{:03b}",
                            "VOP3".color("blue"),
                            vdst,
                            abs,
                            opsel,
                            op,
                            s0,
                            s1,
                            s2,
                            neg
                        );
                    }

                    match op {
                        // VOPC using VOP3 encoding
                        0..=255 => {
                            let ret = match op {
                                17 | 18 | 27 | 20 | 22 | 30 | 126 | 155 => {
                                    let s0 = f32::from_bits(s0).negate(0, neg).absolute(0, abs);
                                    let s1 = f32::from_bits(s1).negate(1, neg).absolute(1, abs);
                                    match op {
                                        17 => s0 < s1,
                                        18 => s0 == s1,
                                        20 => s0 > s1,
                                        22 => s0 >= s1,
                                        27 | 155 => !(s0 > s1),
                                        30 => !(s0 < s1),
                                        126 => true,
                                        _ => panic!(),
                                    }
                                }
                                _ => {
                                    assert_eq!(neg, 0);
                                    match op {
                                        52 | 65 => {
                                            let s0 = s0 as i16;
                                            let s1 = s1 as i16;

                                            match op {
                                                52 => s0 > s1,
                                                65 => s0 < s1,
                                                _ => panic!(),
                                            }
                                        }
                                        58 | 60 => {
                                            let s0 = s0 as u16;
                                            let s1 = s1 as u16;

                                            match op {
                                                58 => s0 == s1,
                                                60 => s0 > s1,
                                                _ => panic!(),
                                            }
                                        }
                                        74 => s0 == s1,
                                        76 => s0 > s1,
                                        77 => s0 != s1,
                                        68 => s0 as i32 > s1 as i32,
                                        _ => todo_instr!(instruction),
                                    }
                                }
                            } as u32;

                            match vdst as u32 {
                                0..=SGPR_COUNT => self.scalar_reg[vdst] = ret,
                                106 => self.vcc.assign(ret),
                                126 => self.exec = ret,
                                _ => todo_instr!(instruction),
                            }
                        }
                        _ => {
                            // other VOP3 ops
                            let val64 = match op {
                                828 => {
                                    let vs1_lo = ((instr >> 41) & 0x1ff) - VGPR_COUNT as u64;
                                    let vsrc = self.vec_reg.read64(vs1_lo as usize);
                                    Some(vsrc << (s0 as u64))
                                }
                                _ => None,
                            };
                            if let Some(val) = val64 {
                                self.vec_reg.write64(vdst, val);
                                return;
                            }

                            self.vec_reg[vdst] = match op {
                                257 | 259 | 299 | 260 | 264 | 272 | 531 | 537 | 540 | 551 | 567 => {
                                    let s0 = f32::from_bits(s0).negate(0, neg).absolute(0, abs);
                                    let s1 = f32::from_bits(s1).negate(1, neg).absolute(1, abs);
                                    let s2 = f32::from_bits(s2).negate(2, neg).absolute(2, abs);
                                    match op {
                                        259 => s0 + s1,
                                        260 => s0 - s1,
                                        264 => s0 * s1,
                                        272 => f32::max(s0, s1),
                                        299 => s0 * s1 + f32::from_bits(self.vec_reg[vdst]),
                                        531 => s0 * s1 + s2,
                                        537 => f32::min(f32::min(s0, s1), s2),
                                        540 => f32::max(f32::max(s0, s1), s2),
                                        551 => s2 / s1,
                                        567 => {
                                            let ret = s0 * s1 + s2;
                                            match *self.vcc != 0 {
                                                true => 2.0_f32.powi(32) * ret,
                                                false => ret,
                                            }
                                        }
                                        // cnd_mask isn't a float alu but supports neg
                                        257 => match s2.to_bits() != 0 {
                                            true => s1,
                                            false => s0,
                                        },
                                        _ => panic!(),
                                    }
                                    .to_bits()
                                }
                                _ => {
                                    if neg != 0 {
                                        todo_instr!(instruction)
                                    }
                                    match op {
                                        522 | 541 | 529 | 814 | 826 => {
                                            let s0 = s0 as i32;
                                            let s1 = s1 as i32;
                                            let s2 = s2 as i32;
                                            (match op {
                                                522 => s0 * s1 + s2, // TODO 24 bit trunc
                                                541 => i32::max(i32::max(s0, s1), s2),
                                                529 => (s0 >> s1) & ((1 << s2) - 1),
                                                814 => ((s0 as i64) * (s1 as i64) >> 32) as i32,
                                                826 => s1 >> s0,
                                                _ => panic!(),
                                            }) as u32
                                        }

                                        771 | 772 | 773 | 824 | 825 => {
                                            let s0 = s0 as u16;
                                            let s1 = s1 as u16;
                                            (match op {
                                                771 => s0 + s1,
                                                772 => s0 - s1,
                                                773 => s0 * s1,
                                                824 => s1 << s0,
                                                825 => s1 >> s0,
                                                _ => panic!(),
                                            }) as u32
                                        }

                                        523 => s0 * s1 + s2, // TODO 24 bit trunc
                                        528 => (s0 >> s1) & ((1 << s2) - 1),
                                        576 => s0 ^ s1 ^ s2,
                                        582 => (s0 << s1) + s2,
                                        597 => s0 + s1 + s2,
                                        583 => (s0 + s1) << s2,
                                        598 => (s0 << s1) | s2,
                                        812 => s0 * s1,
                                        _ => todo_instr!(instruction),
                                    }
                                }
                            }
                        }
                    };
                }
            }
        }
        // lds
        else if instruction >> 26 == 0b110110 {
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
                    "{} offset0={} offset1={} op={} addr={} data0={} data1={} vdst={}",
                    "LDS".color("blue"),
                    offset0,
                    offset1,
                    op,
                    addr,
                    data0,
                    data1,
                    vdst
                );
            }
            let effective_addr = self.vec_reg[addr as usize] as u64 + offset0;

            match op {
                // load
                255 => (0..4).for_each(|i| {
                    self.vec_reg[vdst + i] = self.lds.read(effective_addr + 4 * i as u64);
                }),
                // store
                13 => self.lds.write(effective_addr, self.vec_reg[data0]),
                223 => (0..4).for_each(|i| {
                    self.lds
                        .write(effective_addr + 4 * i as u64, self.vec_reg[data0 + i]);
                }),
                _ => todo_instr!(instruction),
            }
        }
        // global
        else if instruction >> 26 == 0b110111 {
            let instr = self.u64_instr();
            let offset = as_signed(instr & 0x1fff, 13);
            let op = ((instr >> 18) & 0x7f) as usize;
            let addr = ((instr >> 32) & 0xff) as usize;
            let data = ((instr >> 40) & 0xff) as usize;
            let saddr = ((instr >> 48) & 0x7f) as usize;
            let vdst = ((instr >> 56) & 0xff) as usize;

            if *DEBUG >= DebugLevel::INSTRUCTION {
                println!(
                    "{} addr={} data={} saddr={} op={} offset={} vdst={}",
                    "GLOBAL".color("blue"),
                    addr,
                    data,
                    saddr,
                    op,
                    offset,
                    vdst
                );
            }

            let effective_addr = match self.resolve_src(saddr as u32) as u32 {
                0x7F | _ if saddr as u32 == NULL_SRC => {
                    self.vec_reg.read64(addr) as i64 + (offset as i64)
                }
                _ => {
                    let scalar_addr = self.scalar_reg.read64(saddr);
                    let vgpr_offset = self.vec_reg[addr];
                    scalar_addr as i64 + vgpr_offset as i64 + offset
                }
            } as u64;

            match op {
                // load
                16 => self.vec_reg[vdst] = self.gds.read::<u8>(effective_addr) as u32,
                20..=23 => (0..op - 19).for_each(|i| {
                    self.vec_reg[vdst + i] = self.gds.read(effective_addr + 4 * i as u64);
                }),
                // store
                24 => self.gds.write(effective_addr, self.vec_reg[data] as u8),
                26..=29 => (0..op - 25).for_each(|i| {
                    self.gds
                        .write(effective_addr + 4 * i as u64, self.vec_reg[data + i]);
                }),
                _ => todo_instr!(instruction),
            };
        } else {
            todo_instr!(instruction);
        }
    }

    /* ALU utils */
    fn resolve_src(&mut self, ssrc_bf: u32) -> i32 {
        match ssrc_bf {
            0..=SGPR_COUNT => self.scalar_reg[ssrc_bf as usize] as i32,
            VGPR_COUNT..=511 => self.vec_reg[(ssrc_bf - VGPR_COUNT) as usize] as i32,
            106 => *self.vcc as i32,
            126 => self.exec as i32,
            128 => 0,
            124 => NULL_SRC as i32,
            129..=192 => (ssrc_bf - 128) as i32,
            193..=208 => (ssrc_bf - 192) as i32 * -1,
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
            .find(|x| x.0 == ssrc_bf)
            .unwrap()
            .1
            .to_bits() as i32,
            255 => self.simm() as i32,
            _ => todo!("resolve_src={ssrc_bf}"),
        }
    }
    fn simm(&mut self) -> u32 {
        self.pc += 1;
        self.prg[self.pc as usize - 1]
    }

    fn write_to_sdst(&mut self, sdst_bf: u32, val: u32) {
        match sdst_bf {
            0..=SGPR_COUNT => self.scalar_reg[sdst_bf as usize] = val,
            106 => {
                self.vcc.assign(val);
                if *DEBUG >= DebugLevel::STATE {
                    println!("{} {:?}", "VCC".color("pink"), self.vcc);
                }
            }
            126 => self.exec = val,
            _ => todo!("write to sdst {}", sdst_bf),
        }
    }
}

pub fn _helper_test_cpu(wave_id: &str) -> CPU {
    let gds = BumpAllocator::new(wave_id);
    let lds = BumpAllocator::new(&format!("{wave_id}_lds"));
    CPU::new(gds, lds)
}
#[cfg(test)]
mod test_alu_utils {
    use super::*;

    #[test]
    fn test_write_to_sdst_sgpr() {
        let mut cpu = _helper_test_cpu("test_write_to_sdst_sgpr");
        cpu.write_to_sdst(10, 200);
        assert_eq!(cpu.scalar_reg[10], 200);
    }

    #[test]
    fn test_write_to_sdst_vcclo() {
        let mut cpu = _helper_test_cpu("test_write_to_sdst_sgpr");
        let val = 0b1011101011011011111011101111;
        cpu.write_to_sdst(106, val);
        assert_eq!(*cpu.vcc, 1);
    }

    #[test]
    fn test_resolve_src() {
        let mut cpu = _helper_test_cpu("test_resolve_src_negative_const");
        assert_eq!(cpu.resolve_src(129), 1);
        assert_eq!(cpu.resolve_src(192), 64);
        assert_eq!(cpu.resolve_src(193), -1);
        assert_eq!(cpu.resolve_src(208), -16);

        cpu.vec_reg[0] = 10;
        assert_eq!(cpu.resolve_src(256), 10);
    }
}

#[cfg(test)]
mod test_sop1 {
    use super::*;

    #[test]
    fn test_s_mov_b32() {
        let mut cpu = _helper_test_cpu("s_mov_b32");
        cpu.scalar_reg[15] = 42;
        cpu.interpret(&vec![0xbe82000f, END_PRG]);
        assert_eq!(cpu.scalar_reg[2], 42);
    }
}

#[cfg(test)]
mod test_sop2 {
    use super::*;

    #[test]
    fn test_s_add_u32() {
        let mut cpu = _helper_test_cpu("s_add_u32");
        cpu.scalar_reg[2] = 42;
        cpu.scalar_reg[6] = 13;
        cpu.interpret(&vec![0x80060206, END_PRG]);
        assert_eq!(cpu.scalar_reg[6], 55);
    }

    #[test]
    fn test_s_addc_u32() {
        let mut cpu = _helper_test_cpu("s_addc_u32");
        cpu.scalar_reg[7] = 42;
        cpu.scalar_reg[3] = 13;
        cpu.scc = 1;
        cpu.interpret(&vec![0x82070307, END_PRG]);
        assert_eq!(cpu.scalar_reg[7], 56);
    }

    #[test]
    fn test_s_ashr_i32() {
        let mut cpu = _helper_test_cpu("s_ashr_i32");
        cpu.scalar_reg[15] = 42;
        cpu.interpret(&vec![0x86039f0f, END_PRG]);
        assert_eq!(cpu.scalar_reg[3], 0);
    }

    #[test]
    fn test_s_lshl_b64() {
        let mut cpu = _helper_test_cpu("s_lshl_b64");
        cpu.scalar_reg[2] = 42;
        cpu.interpret(&vec![0x84828202, END_PRG]);
        assert_eq!(cpu.scalar_reg[2], 42 << 2);
    }
}

#[cfg(test)]
mod test_vopd {
    use super::*;

    #[test]
    fn test_inline_const_vopx_only() {
        let mut cpu = _helper_test_cpu("test_inline_const_vopx_only");
        cpu.vec_reg[0] = f32::to_bits(0.5);
        let constant = f32::from_bits(0x39a8b099);
        cpu.vec_reg[1] = 10;
        cpu.interpret(&vec![0xC8D000FF, 0x00000080, 0x39A8B099, END_PRG]);
        assert_eq!(f32::from_bits(cpu.vec_reg[0]), 0.5 * constant);
        assert_eq!(cpu.vec_reg[1], 0);
    }

    #[test]
    fn test_inline_const_vopy_only() {
        let mut cpu = _helper_test_cpu("test_inline_const_vopy_only");
        cpu.vec_reg[0] = 10;
        cpu.vec_reg[1] = 10;
        cpu.interpret(&vec![0xCA100080, 0x000000FF, 0x3E15F480, END_PRG]);
        assert_eq!(cpu.vec_reg[0], 0);
        assert_eq!(cpu.vec_reg[1], 0x3e15f480);
    }

    #[test]
    fn test_inline_const_shared() {
        let mut cpu = _helper_test_cpu("test_inline_const_shared");
        cpu.vec_reg[2] = f32::to_bits(2.0);
        cpu.vec_reg[3] = f32::to_bits(4.0);
        let constant = f32::from_bits(0x3e800000);
        cpu.interpret(&vec![0xC8C604FF, 0x020206FF, 0x3E800000, END_PRG]);
        assert_eq!(f32::from_bits(cpu.vec_reg[2]), 2.0 * constant);
        assert_eq!(f32::from_bits(cpu.vec_reg[3]), 4.0 * constant);
    }

    #[test]
    fn test_add_mov() {
        let mut cpu = _helper_test_cpu("add_mov");
        cpu.vec_reg[0] = f32::to_bits(10.5);
        cpu.interpret(&vec![0xC9100300, 0x00000080, END_PRG]);
        assert_eq!(f32::from_bits(cpu.vec_reg[0]), 10.5);
        assert_eq!(cpu.vec_reg[1], 0);
    }

    #[test]
    fn test_max_add() {
        let mut cpu = _helper_test_cpu("max_add");
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
        let mut cpu = _helper_test_cpu("v_mov_b32_srrc_const0");
        cpu.interpret(&vec![0x7e000280, END_PRG]);
        assert_eq!(cpu.vec_reg[0], 0);
        cpu.interpret(&vec![0x7e020280, END_PRG]);
        assert_eq!(cpu.vec_reg[1], 0);
        cpu.interpret(&vec![0x7e040280, END_PRG]);
        assert_eq!(cpu.vec_reg[2], 0);
    }

    #[test]
    fn test_v_mov_b32_srrc_register() {
        let mut cpu = _helper_test_cpu("v_mov_b32_srrc_register");
        cpu.scalar_reg[6] = 31;
        cpu.interpret(&vec![0x7e020206, END_PRG]);
        assert_eq!(cpu.vec_reg[1], 31);
    }

    fn helper_test_fexp(val: f32) -> f32 {
        let mut cpu = _helper_test_cpu("test_fexp");
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
        let mut cpu = _helper_test_cpu("test_cast_f32_i32");
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
        let mut cpu = _helper_test_cpu("test_cast_f32_u32");
        cpu.scalar_reg[4] = 2;
        cpu.interpret(&vec![0x7E000C04, END_PRG]);
        assert_eq!(cpu.vec_reg[0], 1073741824);
    }

    #[test]
    fn test_cast_u32_f32() {
        let mut cpu = _helper_test_cpu("test_cast_u32_f32");
        cpu.vec_reg[0] = 1325400062;
        cpu.interpret(&vec![0x7E000F00, END_PRG]);
        assert_eq!(cpu.vec_reg[0], 2147483392);
    }

    #[test]
    fn test_cast_i32_f32() {
        let mut cpu = _helper_test_cpu("test_cast_f32_i32");
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
        let mut cpu = _helper_test_cpu("test_v_readfirstlane_b32");
        cpu.vec_reg[0] = 2147483392;
        cpu.interpret(&vec![0x7E060500, END_PRG]);
        assert_eq!(cpu.scalar_reg[3], 2147483392);
    }
}

#[cfg(test)]
mod test_vopc {
    use super::*;

    #[test]
    fn test_v_cmp_gt_i32() {
        let mut cpu = _helper_test_cpu("test_v_cmp_gt_i32");

        cpu.vec_reg[1] = (4_i32 * -1) as u32;
        cpu.interpret(&vec![0x7c8802c1, END_PRG]);
        assert_eq!(*cpu.vcc, 1);

        cpu.vec_reg[1] = 4;
        cpu.interpret(&vec![0x7c8802c1, END_PRG]);
        assert_eq!(*cpu.vcc, 0);
    }

    #[test]
    fn test_v_cmpx_nlt_f32() {
        let mut cpu = _helper_test_cpu("test_v_cmp_gt_i32");
        cpu.vec_reg[0] = f32::to_bits(0.9);
        cpu.vec_reg[3] = f32::to_bits(0.4);
        cpu.interpret(&vec![0x7D3C0700, END_PRG]);
        assert_eq!(cpu.exec, 1);
    }
}
#[cfg(test)]
mod test_vop2 {
    use super::*;

    #[test]
    fn test_v_add_f32_e32() {
        let mut cpu = _helper_test_cpu("v_add_f32_e32");
        cpu.scalar_reg[2] = f32::to_bits(42.0);
        cpu.vec_reg[0] = f32::to_bits(1.0);
        cpu.interpret(&vec![0x06000002, END_PRG]);
        assert_eq!(f32::from_bits(cpu.vec_reg[0]), 43.0);
    }

    #[test]
    fn test_v_mul_f32_e32() {
        let mut cpu = _helper_test_cpu("v_mul_f32_e32");
        cpu.vec_reg[2] = f32::to_bits(21.0);
        cpu.vec_reg[4] = f32::to_bits(2.0);
        cpu.interpret(&vec![0x10060504, END_PRG]);
        assert_eq!(f32::from_bits(cpu.vec_reg[3]), 42.0);
    }

    #[test]
    fn test_v_ashrrev_i32() {
        let mut cpu = _helper_test_cpu("test_v_ashrrev_i32");
        cpu.vec_reg[0] = 4294967295;
        cpu.interpret(&vec![0x3402009F, END_PRG]);
        assert_eq!(cpu.vec_reg[1] as i32, -1);
    }
}

#[cfg(test)]
mod test_vop3 {
    use super::*;

    fn helper_test_vop3(id: &str, op: u32, a: f32, b: f32) -> f32 {
        let mut cpu = _helper_test_cpu(id);
        cpu.scalar_reg[0] = f32::to_bits(a);
        cpu.scalar_reg[6] = f32::to_bits(b);
        cpu.interpret(&vec![op, 0x00000006, END_PRG]);
        return f32::from_bits(cpu.vec_reg[0]);
    }

    #[test]
    fn test_v_add_f32() {
        assert_eq!(helper_test_vop3("vop3_add_f32", 0xd5030000, 0.4, 0.2), 0.6);
    }

    #[test]
    fn test_v_max_f32() {
        assert_eq!(
            helper_test_vop3("vop3_max_f32_a", 0xd5100000, 0.4, 0.2),
            0.4
        );
        assert_eq!(
            helper_test_vop3("vop3_max_f32_b", 0xd5100000, 0.2, 0.8),
            0.8
        );
    }

    #[test]
    fn test_v_mul_f32() {
        assert_eq!(
            helper_test_vop3("vop3_v_mul_f32", 0xd5080000, 0.4, 0.2),
            0.4 * 0.2
        );
    }

    #[test]
    fn test_signed_src() {
        // v0, max(s2, s2)
        let mut cpu = _helper_test_cpu("signed_src_positive");
        cpu.scalar_reg[2] = f32::to_bits(0.5);
        cpu.interpret(&vec![0xd5100000, 0x00000402, END_PRG]);
        assert_eq!(f32::from_bits(cpu.vec_reg[0]), 0.5);

        // v1, max(-s2, -s2)
        let mut cpu = _helper_test_cpu("signed_src_neg");
        cpu.scalar_reg[2] = f32::to_bits(0.5);
        cpu.interpret(&vec![0xd5100001, 0x60000402, END_PRG]);
        assert_eq!(f32::from_bits(cpu.vec_reg[1]), -0.5);
    }

    #[test]
    fn test_cnd_mask_cond_src_sgpr() {
        let mut cpu = _helper_test_cpu("test_cnd_mask_cond_src_sgpr");
        cpu.scalar_reg[3] = 30;
        cpu.interpret(&vec![0xD5010000, 0x000D0280, END_PRG]);
        assert_eq!(cpu.vec_reg[0], 1);

        cpu.scalar_reg[3] = 0;
        cpu.interpret(&vec![0xD5010000, 0x000D0280, END_PRG]);
        assert_eq!(cpu.vec_reg[0], 0);
    }

    #[test]
    fn test_cnd_mask_cond_src_vcclo() {
        let mut cpu = _helper_test_cpu("test_cnd_mask_cond_src_vcclo");
        cpu.vec_reg[2] = 20;
        cpu.vec_reg[0] = 100;
        cpu.interpret(&vec![0xD5010002, 0x41AA0102, END_PRG]);
        assert_eq!(cpu.vec_reg[2], 20);
    }

    #[test]
    fn test_v_mul_hi_i32() {
        let mut cpu = _helper_test_cpu("test_v_mul_hi_i32");
        cpu.vec_reg[2] = -2i32 as u32;
        cpu.interpret(&vec![0xD72E0003, 0x000204FF, 0x2E8BA2E9, END_PRG]);
        assert_eq!(cpu.vec_reg[3] as i32, -1);

        cpu.vec_reg[2] = 2;
        cpu.interpret(&vec![0xD72E0003, 0x000204FF, 0x2E8BA2E9, END_PRG]);
        assert_eq!(cpu.vec_reg[3], 0);
    }
}

#[cfg(test)]
mod test_smem {
    use super::*;

    fn helper_test_s_load(
        mut cpu: CPU,
        op: u32,
        offset: u32,
        data: &Vec<u32>,
        base_mem_addr: u64,
        starting_dest_sgpr: u32,
    ) {
        data.iter()
            .enumerate()
            .for_each(|(i, &v)| cpu.gds.write((base_mem_addr + (i as u64) * 4) as u64, v));
        cpu.interpret(&vec![op, offset, END_PRG]);
        data.iter()
            .enumerate()
            .for_each(|(i, &v)| assert_eq!(cpu.scalar_reg[i + (starting_dest_sgpr as usize)], v));
    }

    #[test]
    fn test_s_load_b32() {
        // no offset
        helper_test_s_load(
            _helper_test_cpu("s_load_b32_1"),
            0xf4000000,
            0xf8000000,
            &vec![42],
            0,
            0,
        );

        // positive offset
        helper_test_s_load(
            _helper_test_cpu("s_load_b32_2"),
            0xf4000000,
            0xf8000004,
            &vec![42],
            0x4,
            0,
        );
        helper_test_s_load(
            _helper_test_cpu("s_load_b32_3"),
            0xf4000000,
            0xf800000c,
            &vec![42],
            0xc,
            0,
        );

        // negative offset
        let mut cpu = _helper_test_cpu("s_load_b32_4");
        cpu.scalar_reg.write64(0, 10000);
        helper_test_s_load(cpu, 0xf4000000, 0xf81fffd8, &vec![42], 9960, 0);
    }

    #[test]
    fn test_s_load_b64() {
        let data = (0..2).collect();

        // positive offset
        helper_test_s_load(
            _helper_test_cpu("s_load_b64_1"),
            0xf4040000,
            0xf8000010,
            &data,
            0x10,
            0,
        );
        helper_test_s_load(
            _helper_test_cpu("s_load_b64_2"),
            0xf4040204,
            0xf8000268,
            &data,
            0x268,
            8,
        );

        // negative offset
        let mut cpu = _helper_test_cpu("s_load_b64_3");
        cpu.scalar_reg[2] = 612;
        cpu.scalar_reg.write64(2, 612);
        helper_test_s_load(cpu, 0xf4040301, 0xf81ffd9c, &data, 0, 12);
    }

    #[test]
    fn test_s_load_b128() {
        let data = (0..4).collect();

        // positive offset
        helper_test_s_load(
            _helper_test_cpu("s_load_b128_1"),
            0xf4080000,
            0xf8000000,
            &data,
            0,
            0,
        );

        let mut cpu = _helper_test_cpu("s_load_b128_2");
        let base_mem_addr: u64 = 0x10;
        cpu.scalar_reg.write64(6, base_mem_addr);
        helper_test_s_load(cpu, 0xf4080203, 0xf8000000, &data, base_mem_addr, 8);

        // negative offset
        let mut cpu = _helper_test_cpu("s_load_b128_3");
        cpu.scalar_reg.write64(2, 0x10);
        helper_test_s_load(cpu, 0xf4080401, 0xf81ffff0, &data, 0, 16);
    }
}

#[cfg(test)]
mod test_global {
    use super::*;

    #[test]
    fn test_store_b32() {
        let mut cpu = _helper_test_cpu("store_b32");
        cpu.interpret(&vec![0xdc6a0000, 0x00000001, END_PRG]);
        cpu.interpret(&vec![0xdc6a0000, 0x00000100, END_PRG]);
        cpu.interpret(&vec![0xdc6a0000, 0x00000002, END_PRG]);
        cpu.interpret(&vec![0xdc6a0000, 0x00000102, END_PRG]);
    }

    #[test]
    fn test_store_b96() {
        let mut cpu = _helper_test_cpu("test_store_b96");
        let val0: u32 = 10;
        let val1: u32 = 20;
        let val2: u32 = 30;
        let base = 100;
        cpu.vec_reg.write64(3, base);
        cpu.vec_reg[0] = val0;
        cpu.vec_reg[1] = val1;
        cpu.vec_reg[2] = val2;
        cpu.interpret(&vec![0xdc720000, 0x007c0003, END_PRG]);
        assert_eq!(cpu.gds.read::<u32>(base), val0);
        assert_eq!(cpu.gds.read::<u32>(base + 4), val1);
        assert_eq!(cpu.gds.read::<u32>(base + 8), val2);
    }
    #[test]
    fn test_load_b96() {
        let mut cpu = _helper_test_cpu("test_load_b96");
        let val0: u32 = 10;
        let val1: u32 = 20;
        let val2: u32 = 30;
        let base = 100;
        cpu.gds.write(base, val0);
        cpu.gds.write(base + 4, val1);
        cpu.gds.write(base + 8, val2);
        cpu.vec_reg.write64(0, base);
        cpu.interpret(&vec![0xdc5a0000, 0x007c0000, END_PRG]);
        assert_eq!(cpu.vec_reg[0], val0);
        assert_eq!(cpu.vec_reg[1], val1);
        assert_eq!(cpu.vec_reg[2], val2);
    }

    #[test]
    fn test_store_b128() {
        let mut cpu = _helper_test_cpu("test_store_b128");
        let val0: u32 = 10;
        let val1: u32 = 20;
        let val2: u32 = 30;
        let val3: u32 = 40;
        let base = 100;
        cpu.vec_reg.write64(4, base);
        cpu.vec_reg[0] = val0;
        cpu.vec_reg[1] = val1;
        cpu.vec_reg[2] = val2;
        cpu.vec_reg[3] = val3;
        cpu.interpret(&vec![0xDC760000, 0x007C0004, END_PRG]);
        assert_eq!(cpu.gds.read::<u32>(base), val0);
        assert_eq!(cpu.gds.read::<u32>(base + 4), val1);
        assert_eq!(cpu.gds.read::<u32>(base + 8), val2);
        assert_eq!(cpu.gds.read::<u32>(base + 12), val3);
    }
    #[test]
    fn test_load_b128() {
        let mut cpu = _helper_test_cpu("test_load_b128");
        let val0: u32 = 10;
        let val1: u32 = 20;
        let val2: u32 = 30;
        let val3: u32 = 40;
        let base = 100;
        cpu.gds.write(base, val0);
        cpu.gds.write(base + 4, val1);
        cpu.gds.write(base + 8, val2);
        cpu.gds.write(base + 12, val3);
        cpu.vec_reg.write64(4, base);
        cpu.interpret(&vec![0xdc5e0000, 0x047c0004, END_PRG]);
        assert_eq!(cpu.vec_reg[4], val0);
        assert_eq!(cpu.vec_reg[5], val1);
        assert_eq!(cpu.vec_reg[6], val2);
        assert_eq!(cpu.vec_reg[7], val3);
    }
}

#[cfg(test)]
mod test_real_world {
    use super::*;
    use crate::utils::parse_rdna3_file;

    fn read_array_f32(cpu: &CPU, addr: u64, sz: usize) -> Vec<f32> {
        let mut data = vec![0.0; sz];
        for i in 0..sz {
            data[i] = f32::from_bits(cpu.gds.read(addr + (i * 4) as u64));
        }
        return data;
    }

    fn to_bytes(fvec: Vec<f32>) -> Vec<u8> {
        fvec.iter()
            .map(|&v| f32::to_le_bytes(v).to_vec())
            .flatten()
            .collect()
    }
    #[test]
    fn test_add_simple() {
        let mut cpu = _helper_test_cpu("test_add_simple");
        let mut data0 = vec![0.0; 4];
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![5.0, 6.0, 7.0, 8.0];
        let expected_data0 = vec![6.0, 8.0, 10.0, 12.0];

        let data0_bytes: Vec<u8> = to_bytes(data0.clone());
        let data1_bytes: Vec<u8> = to_bytes(data1);
        let data2_bytes: Vec<u8> = to_bytes(data2);

        // malloc tinygrad style
        let data1_ptr = cpu.gds.alloc(data1_bytes.len() as u32);
        let data2_ptr = cpu.gds.alloc(data2_bytes.len() as u32);
        let data0_ptr = cpu.gds.alloc(data0_bytes.len() as u32);
        cpu.gds.write_bytes(data1_ptr, &data1_bytes);
        cpu.gds.write_bytes(data2_ptr, &data2_bytes);

        // "stack" pointers in memory
        let data0_ptr_addr: u64 = cpu.gds.alloc(24);
        let data1_ptr_addr = data0_ptr_addr + 8;
        let data2_ptr_addr = data0_ptr_addr + 16;
        cpu.gds.write(data0_ptr_addr, data0_ptr);
        cpu.gds.write(data1_ptr_addr, data1_ptr);
        cpu.gds.write(data2_ptr_addr, data2_ptr);

        // "launch" kernel
        let global_size = (4, 1, 1);
        for i in 0..global_size.0 {
            // allocate src registers
            cpu.scalar_reg.reset();
            cpu.scalar_reg.write64(0, data0_ptr_addr);
            println!("i={i}");
            cpu.scalar_reg[15] = i;
            cpu.interpret(&parse_rdna3_file("./tests/test_ops/test_add_simple.s"));
            data0[i as usize] = read_array_f32(&cpu, data0_ptr, 4)[i as usize];
        }

        assert_eq!(data0, expected_data0);
    }
}
