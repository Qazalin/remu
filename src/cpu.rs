use crate::allocator::BumpAllocator;
use crate::state::{SGPR, VGPR};
use crate::utils::{twos_complement_21bit, Colorize, DEBUG};

const SGPR_COUNT: u32 = 105;
const VGPR_COUNT: u32 = 256;
pub const END_PRG: u32 = 0xbfb00000;

pub struct CPU {
    pc: u64,
    pub allocator: BumpAllocator,
    pub scalar_reg: SGPR,
    pub vec_reg: VGPR,
    scc: u32,
    prg: Vec<u32>,
}

impl CPU {
    pub fn new() -> Self {
        return CPU {
            pc: 0,
            scc: 0,
            allocator: BumpAllocator::new(),
            scalar_reg: SGPR::new(),
            vec_reg: VGPR::new(),
            prg: vec![],
        };
    }

    pub fn interpret(&mut self, prg: &Vec<u32>) {
        self.pc = 0;
        self.prg = prg.to_vec();

        loop {
            let instruction = prg[self.pc as usize];
            self.pc += 1;

            if instruction == END_PRG {
                break;
            }

            if instruction >> 24 == 0xbf {
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
            // offset is a sign-extend immediate 21-bit constant
            let offset = twos_complement_21bit((instr >> 32) & 0x1fffff);
            let soffset = match instr & 0x7F {
                0 => 0, //  set to "NULL" to not use (offset=0).
                // the SGPR contains an unsigned byte offset (the 2 LSBs are ignored).
                val => (self.resolve_src(val as u32) & -4) as u64,
            };

            if *DEBUG >= 1 {
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
            let base_addr = self.scalar_reg.read_addr(sbase as usize);
            let effective_addr = (base_addr as i64 + offset + soffset as i64) as u64;

            match op {
                0..=2 => (0..2_usize.pow(op as u32)).for_each(|i| {
                    self.scalar_reg[sdata + i] =
                        self.allocator.read(effective_addr + (4 * i as u64));
                }),
                _ => todo!(),
            }
        }
        // sop1
        else if instruction >> 23 == 0b10_1111101 {
            let ssrc0 = self.resolve_src(instruction & 0xFF) as u32;
            let op = (instruction >> 8) & 0xFF;
            let sdst = (instruction >> 16) & 0x7F;

            if *DEBUG >= 1 {
                println!(
                    "{} ssrc0={} sdst={} op={}",
                    "SOP1".color("blue"),
                    ssrc0,
                    sdst,
                    op,
                );
            }

            match op {
                0 => self.write_to_sdst(sdst, ssrc0),
                _ => todo!(),
            }
        }
        // sop2
        else if instruction >> 30 == 0b10 {
            let ssrc0 = self.resolve_src(instruction & 0xFF);
            let ssrc1 = self.resolve_src((instruction >> 8) & 0xFF);
            let sdst = (instruction >> 16) & 0x7F;
            let op = (instruction >> 23) & 0xFF;

            if *DEBUG >= 1 {
                println!(
                    "{} ssrc0={} ssrc1={} sdst={} op={}",
                    "SOP2".color("blue"),
                    ssrc0,
                    ssrc1,
                    sdst,
                    op
                );
            }

            let tmp = match op {
                0 => (ssrc0 as u64) + (ssrc1 as u64),
                4 => (ssrc0 as u64) + (ssrc1 as u64) + (self.scc as u64),
                9 => (ssrc0 as u64) << (ssrc1 as u64 & 0x1F),
                12 => (ssrc0 >> (ssrc1 & 0x1F)) as u64,
                _ => todo!("sop2 opcode {}", op),
            };
            self.write_to_sdst(sdst, tmp as u32);
        }
        // vop1
        else if instruction >> 25 == 0b0111111 {
            let src = self.resolve_src(instruction & 0x1ff);
            let op = (instruction >> 9) & 0xff;
            let vdst = (instruction >> 17) & 0xff;

            if *DEBUG >= 1 {
                println!(
                    "{} src={} op={} vdst={}",
                    "VOP1".color("blue"),
                    src,
                    op,
                    vdst,
                );
            }

            match op {
                1 => self.vec_reg[vdst as usize] = src as u32,
                _ => todo!(),
            }
        }
        // vopd
        else if instruction >> 26 == 0b110010 {
            let instr = self.u64_instr();

            let srcx0 = self.resolve_src((instr & 0x1ff) as u32) as u64;
            let vsrcx1 = self.vec_reg[((instr >> 9) & 0xff) as usize] as u64;
            let opy = (instr >> 17) & 0x1f;

            let opx = (instr >> 22) & 0xf;
            let srcy0 = self.resolve_src(((instr >> 32) & 0x1ff) as u32) as u64;
            let vsrcy1 = self.vec_reg[((instr >> 41) & 0xff) as usize] as u64;

            let vdstx = (instr >> 56) & 0xff;
            // LSB is the opposite of VDSTX[0]
            let vdsty = ((instr >> 49) & 0x7f) << 1 | ((vdstx & 1) ^ 1);

            if *DEBUG >= 1 {
                println!(
                    "{} X=[op={}, dest={} src={}, vsrc={}] Y=[op={}, dest={}, src={}, vsrc={}]",
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

            ([[opx, srcx0, vsrcx1, vdstx], [opy, srcy0, vsrcy1, vdsty]])
                .iter()
                .for_each(|i| {
                    let s0 = f32::from_bits(i[1] as u32);
                    let s1 = f32::from_bits(i[2] as u32);
                    self.vec_reg[i[3] as usize] = match i[0] {
                        10 => f32::max(s0, s1),
                        4 => s0 + s1,
                        8 => s0,
                        _ => todo!(),
                    }
                    .to_bits();
                });
        }
        // vop2
        else if instruction >> 31 == 0b0 {
            let ssrc0 = self.resolve_src(instruction & 0x1FF);
            let vsrc1 = self.vec_reg[((instruction >> 9) & 0xFF) as usize];
            let vdst = (instruction >> 17) & 0xFF;
            let op = (instruction >> 25) & 0x3F;

            if *DEBUG >= 1 {
                println!(
                    "{} ssrc0={} vsrc1={} vdst={} op={}",
                    "VOP2".color("blue"),
                    ssrc0,
                    vsrc1,
                    vdst,
                    op
                );
            }

            let s0 = f32::from_bits(ssrc0 as u32);
            let s1 = f32::from_bits(vsrc1);

            self.vec_reg[vdst as usize] = match op {
                3 | 50 => s0 + s1,
                8 => s0 * s1,
                _ => todo!(),
            }
            .to_bits();
        }
        // vop3
        else if instruction >> 26 == 0b110101 {
            let instr = self.u64_instr();

            let vdst = instr & 0xff;
            let abs = (instr >> 8) & 0x7;
            let opsel = (instr >> 11) & 0xf;
            let op = (instr >> 16) & 0x3ff;

            let src0 = self.resolve_src(((instr >> 32) & 0x1ff) as u32);
            let src1 = self.resolve_src(((instr >> 41) & 0x1ff) as u32);
            let src2 = self.resolve_src(((instr >> 50) & 0x1ff) as u32);

            let omod = (instr >> 59) & 0x3;
            let neg = (instr >> 61) & 0x7;

            if *DEBUG >= 1 {
                println!(
                    "{} vdst={} abs={} opsel={} op={} src0={} src1={} src2={} omod={} neg=0b{:03b}",
                    "VOP3".color("blue"),
                    vdst,
                    abs,
                    opsel,
                    op,
                    src0,
                    src1,
                    src2,
                    omod,
                    neg
                );
            }

            let mut s0 = f32::from_bits(src0 as u32);
            let mut s1 = f32::from_bits(src1 as u32);

            // Negate input. TODO this could be done better
            if ((neg >> 0) & 1) == 1 {
                s0 = -s0;
            }
            if ((neg >> 1) & 1) == 1 {
                s1 = -s1;
            }

            self.vec_reg[vdst as usize] = match op {
                259 => s0 + s1,
                264 => s0 * s1,
                272 => f32::max(s0, s1),
                _ => todo!(),
            }
            .to_bits();
        }
        // global
        else if instruction >> 26 == 0b110111 {
            let instr = self.u64_instr();
            let offset = instr & 0x1fff;
            let seg = (instr >> 16) & 0x3;
            let op = (instr >> 18) & 0x7f;
            let addr = (instr >> 32) & 0xff;
            let data = (instr >> 40) & 0xff;
            let saddr = (instr >> 48) & 0x7f;
            let vdst = (instr >> 56) & 0xff;

            if *DEBUG >= 1 {
                println!(
                    "{} addr={} data={} saddr={} op={} offset={}",
                    "GLOBAL".color("blue"),
                    addr,
                    data,
                    saddr,
                    op,
                    offset
                );
            }
            assert_eq!(seg, 2, "flat and scratch arent supported");

            let effective_addr = match self.resolve_src(saddr as u32) {
                0 | 0x7F => self.vec_reg.read_addr(addr as usize).wrapping_add(offset), // SADDR is NULL or 0x7f: specifies an address
                _ => {
                    let scalar_addr = self.scalar_reg.read_addr(saddr as usize);
                    let vgpr_offset = self.vec_reg[addr as usize];
                    scalar_addr + vgpr_offset as u64 + offset
                } // SADDR is not NULL or 0x7f: specifies an offset.
            };
            let vdata = self.vec_reg[data as usize];

            match op {
                // global_load
                18 => {
                    self.vec_reg[vdst as usize] = self.allocator.read::<u16>(effective_addr) as u32
                }
                // global_store
                26 | 25 => self.allocator.write(effective_addr, vdata),
                _ => todo!(),
            }
        }
    }

    /* ALU utils */
    fn resolve_src(&self, ssrc_bf: u32) -> i32 {
        match ssrc_bf {
            0..=SGPR_COUNT => self.scalar_reg[ssrc_bf as usize] as i32,
            VGPR_COUNT..=511 => self.vec_reg[(ssrc_bf - VGPR_COUNT) as usize] as i32,
            128 => 0,
            129..=192 => (ssrc_bf - 128) as i32,
            _ => todo!(),
        }
    }
    fn write_to_sdst(&mut self, sdst_bf: u32, val: u32) {
        match sdst_bf {
            0..=SGPR_COUNT => self.scalar_reg[sdst_bf as usize] = val,
            _ => todo!("write to sdst {}", sdst_bf),
        }
    }
}

#[cfg(test)]
mod test_sop1 {
    use super::*;

    #[test]
    fn test_s_mov_b32() {
        let mut cpu = CPU::new();
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
        let mut cpu = CPU::new();
        cpu.scalar_reg[2] = 42;
        cpu.scalar_reg[6] = 13;
        cpu.interpret(&vec![0x80060206, END_PRG]);
        assert_eq!(cpu.scalar_reg[6], 55);
    }

    #[test]
    fn test_s_addc_u32() {
        let mut cpu = CPU::new();
        cpu.scalar_reg[7] = 42;
        cpu.scalar_reg[3] = 13;
        cpu.scc = 1;
        cpu.interpret(&vec![0x82070307, END_PRG]);
        assert_eq!(cpu.scalar_reg[7], 56);
    }

    #[test]
    fn test_s_ashr_i32() {
        let mut cpu = CPU::new();
        cpu.scalar_reg[15] = 42;
        cpu.interpret(&vec![0x86039f0f, END_PRG]);
        assert_eq!(cpu.scalar_reg[3], 0);
    }

    #[test]
    fn test_s_lshl_b64() {
        let mut cpu = CPU::new();
        cpu.scalar_reg[2] = 42;
        cpu.interpret(&vec![0x84828202, END_PRG]);
        assert_eq!(cpu.scalar_reg[2], 42 << 2);
    }
}

#[cfg(test)]
mod test_vopd {
    use super::*;

    #[test]
    fn test_add_mov() {
        let mut cpu = CPU::new();
        cpu.vec_reg[0] = f32::to_bits(10.5);
        cpu.interpret(&vec![0xC9100300, 0x00000080, END_PRG]);
        assert_eq!(f32::from_bits(cpu.vec_reg[0]), 10.5);
        assert_eq!(cpu.vec_reg[1], 0);
    }

    #[test]
    fn test_max_add() {
        let mut cpu = CPU::new();
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
        let mut cpu = CPU::new();
        cpu.interpret(&vec![0x7e000280, END_PRG]);
        assert_eq!(cpu.vec_reg[0], 0);
        cpu.interpret(&vec![0x7e020280, END_PRG]);
        assert_eq!(cpu.vec_reg[1], 0);
        cpu.interpret(&vec![0x7e040280, END_PRG]);
        assert_eq!(cpu.vec_reg[2], 0);
    }

    #[test]
    fn test_v_mov_b32_srrc_register() {
        let mut cpu = CPU::new();
        cpu.scalar_reg[6] = 31;
        cpu.interpret(&vec![0x7e020206, END_PRG]);
        assert_eq!(cpu.vec_reg[1], 31);
    }

    // TODO
    /*
    #[test]
    fn test_v_mov_b32_with_const() {
        let mut cpu = CPU::new();
        cpu.scalar_reg[6] = 31;
        cpu.interpret(&vec![0x7e0002ff, 0xff800000, END_PRG]);
        assert_eq!(cpu.vec_reg[1], 31);
    }
    */
}

#[cfg(test)]
mod test_vop2 {
    use super::*;

    #[test]
    fn test_v_add_f32_e32() {
        let mut cpu = CPU::new();
        cpu.scalar_reg[2] = f32::to_bits(42.0);
        cpu.vec_reg[0] = f32::to_bits(1.0);
        cpu.interpret(&vec![0x06000002, END_PRG]);
        assert_eq!(f32::from_bits(cpu.vec_reg[0]), 43.0);
    }

    #[test]
    fn test_v_mul_f32_e32() {
        let mut cpu = CPU::new();
        cpu.vec_reg[2] = f32::to_bits(21.0);
        cpu.vec_reg[4] = f32::to_bits(2.0);
        cpu.interpret(&vec![0x10060504, END_PRG]);
        assert_eq!(f32::from_bits(cpu.vec_reg[3]), 42.0);
    }
}

#[cfg(test)]
mod test_vop3 {
    use super::*;

    fn helper_test_vop3(op: u32, a: f32, b: f32) -> f32 {
        let mut cpu = CPU::new();
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
        let mut cpu = CPU::new();
        cpu.scalar_reg[2] = f32::to_bits(0.5);
        cpu.interpret(&vec![0xd5100000, 0x00000402, END_PRG]);
        assert_eq!(f32::from_bits(cpu.vec_reg[0]), 0.5);

        // v1, max(-s2, -s2)
        let mut cpu = CPU::new();
        cpu.scalar_reg[2] = f32::to_bits(0.5);
        cpu.interpret(&vec![0xd5100001, 0x60000402, END_PRG]);
        assert_eq!(f32::from_bits(cpu.vec_reg[1]), -0.5);
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
        data.iter().enumerate().for_each(|(i, &v)| {
            cpu.allocator
                .write((base_mem_addr + (i as u64) * 4) as u64, v)
        });
        cpu.interpret(&vec![op, offset, END_PRG]);
        data.iter()
            .enumerate()
            .for_each(|(i, &v)| assert_eq!(cpu.scalar_reg[i + (starting_dest_sgpr as usize)], v));
    }

    #[test]
    fn test_s_load_b32() {
        // no offset
        helper_test_s_load(CPU::new(), 0xf4000000, 0xf8000000, &vec![42], 0, 0);

        // positive offset
        helper_test_s_load(CPU::new(), 0xf4000000, 0xf8000004, &vec![42], 0x4, 0);
        helper_test_s_load(CPU::new(), 0xf4000000, 0xf800000c, &vec![42], 0xc, 0);

        // negative offset
        let mut cpu = CPU::new();
        cpu.scalar_reg.write_addr(0, 10000);
        helper_test_s_load(cpu, 0xf4000000, 0xf81fffd8, &vec![42], 9960, 0);
    }

    #[test]
    fn test_s_load_b64() {
        let data = (0..2).collect();

        // positive offset
        helper_test_s_load(CPU::new(), 0xf4040000, 0xf8000010, &data, 0x10, 0);
        helper_test_s_load(CPU::new(), 0xf4040204, 0xf8000268, &data, 0x268, 8);

        // negative offset
        let mut cpu = CPU::new();
        cpu.scalar_reg[2] = 612;
        cpu.scalar_reg.write_addr(2, 612);
        helper_test_s_load(cpu, 0xf4040301, 0xf81ffd9c, &data, 0, 12);
    }

    #[test]
    fn test_s_load_b128() {
        let data = (0..4).collect();

        // positive offset
        helper_test_s_load(CPU::new(), 0xf4080000, 0xf8000000, &data, 0, 0);

        let mut cpu = CPU::new();
        let base_mem_addr: u64 = 0x10;
        cpu.scalar_reg.write_addr(6, base_mem_addr);
        helper_test_s_load(cpu, 0xf4080203, 0xf8000000, &data, base_mem_addr, 8);

        // negative offset
        let mut cpu = CPU::new();
        cpu.scalar_reg.write_addr(2, 0x10);
        helper_test_s_load(cpu, 0xf4080401, 0xf81ffff0, &data, 0, 16);
    }
}

#[cfg(test)]
mod test_global {
    use super::*;

    #[test]
    fn test_store_b32() {
        let mut cpu = CPU::new();
        cpu.interpret(&vec![0xdc6a0000, 0x00000001, END_PRG]);
        cpu.interpret(&vec![0xdc6a0000, 0x00000100, END_PRG]);
        cpu.interpret(&vec![0xdc6a0000, 0x00000002, END_PRG]);
        cpu.interpret(&vec![0xdc6a0000, 0x00000102, END_PRG]);
    }
}

#[cfg(test)]
mod test_real_world {
    use super::*;
    use crate::utils::parse_rdna3_file;

    fn read_array(cpu: &CPU, addr: u64, sz: usize) -> Vec<u32> {
        let mut data = vec![0; sz];
        for i in 0..sz {
            data[i] = cpu.allocator.read(addr + (i * 4) as u64);
        }
        return data;
    }
    fn read_array_f32(cpu: &CPU, addr: u64, sz: usize) -> Vec<f32> {
        let mut data = vec![0.0; sz];
        for i in 0..sz {
            data[i] = f32::from_bits(cpu.allocator.read(addr + (i * 4) as u64));
        }
        return data;
    }
    fn read_array_bytes(cpu: &CPU, addr: u64, sz: usize) -> Vec<u8> {
        let mut data = vec![0; sz * 4];
        for i in 0..data.len() {
            data[i] = cpu.allocator.memory[addr as usize + i];
        }
        return data;
    }
    fn write_array(cpu: &mut CPU, addr: u64, values: Vec<f32>) {
        for i in 0..values.len() {
            cpu.allocator
                .write(addr + (i * 4) as u64, f32::to_bits(values[i]));
        }
    }

    fn to_bytes(fvec: Vec<f32>) -> Vec<u8> {
        fvec.iter()
            .map(|&v| f32::to_le_bytes(v).to_vec())
            .flatten()
            .collect()
    }
    #[test]
    fn test_add_simple() {
        let mut cpu = CPU::new();
        let mut data0 = vec![0.0; 4];
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![5.0, 6.0, 7.0, 8.0];
        let expected_data0 = vec![6.0, 8.0, 10.0, 12.0];

        let data0_bytes: Vec<u8> = to_bytes(data0.clone());
        let data1_bytes: Vec<u8> = to_bytes(data1);
        let data2_bytes: Vec<u8> = to_bytes(data2);

        // malloc tinygrad style
        let data1_ptr = cpu.allocator.alloc(data1_bytes.len() as u32);
        let data2_ptr = cpu.allocator.alloc(data2_bytes.len() as u32);
        let data0_ptr = cpu.allocator.alloc(data0_bytes.len() as u32);
        cpu.allocator.copyin(data1_ptr, &data1_bytes);
        cpu.allocator.copyin(data2_ptr, &data2_bytes);

        // "stack" pointers in memory
        let data0_ptr_addr: u64 = cpu.allocator.alloc(24);
        let data1_ptr_addr = data0_ptr_addr + 8;
        let data2_ptr_addr = data0_ptr_addr + 16;
        cpu.allocator.write(data0_ptr_addr, data0_ptr);
        cpu.allocator.write(data1_ptr_addr, data1_ptr);
        cpu.allocator.write(data2_ptr_addr, data2_ptr);

        // "launch" kernel
        let global_size = (4, 1, 1);
        for i in 0..global_size.0 {
            // allocate src registers
            cpu.scalar_reg.reset();
            cpu.scalar_reg.write_addr(0, data0_ptr_addr);
            println!("i={i}");
            cpu.scalar_reg[15] = i;
            cpu.interpret(&parse_rdna3_file("./tests/test_ops/test_add_simple.s"));
            data0[i as usize] = read_array_f32(&cpu, data0_ptr, 4)[i as usize];
        }

        assert_eq!(data0, expected_data0);
    }

    #[test]
    fn test_add_const_index() {
        let mut cpu = CPU::new();
        let mut data0 = vec![0.0; 4];
        let data1 = vec![1.0, 21.0, 3.0, 4.0];

        let data0_addr = 1000;
        write_array(&mut cpu, data0_addr, data0);
        let data1_addr = 2000;
        write_array(&mut cpu, data1_addr, data1);

        println!(
            "data0 = {:?} {:?}",
            read_array(&cpu, data0_addr, 4),
            read_array_bytes(&cpu, data0_addr, 4)
        );

        // "stack" pointers in memory
        let data0_ptr_addr = 1200;
        cpu.allocator.write(data0_ptr_addr, data0_addr);
        let data1_ptr_addr = data0_ptr_addr + 8;
        cpu.allocator.write(data1_ptr_addr, data1_addr);

        cpu.scalar_reg.write_addr(0, data0_ptr_addr);

        cpu.interpret(&parse_rdna3_file("./tests/misc/test_add_const_index.s"));

        data0 = read_array_f32(&cpu, data0_addr, 4);
        assert_eq!(data0[1], 42.0);
    }
}
