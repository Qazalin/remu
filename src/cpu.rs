#![allow(unused)]
use crate::utils::DEBUG;

const SGPR_COUNT: u32 = 105;
const VGPR_COUNT: u32 = 256;
pub const END_PRG: u32 = 0xbfb00000;

pub struct CPU {
    pc: u64,
    pub memory: Vec<u8>,
    pub scalar_reg: [u32; SGPR_COUNT as usize],
    pub vec_reg: [u32; VGPR_COUNT as usize],
    scc: u32,
}

impl CPU {
    pub fn new() -> Self {
        return CPU {
            pc: 0,
            scc: 0,
            memory: vec![0; 24 * 1_073_741_824],
            scalar_reg: [0; SGPR_COUNT as usize],
            vec_reg: [0; VGPR_COUNT as usize],
        };
    }

    pub fn read_memory_32(&self, addr_bf: u64) -> u32 {
        let addr = addr_bf as usize;
        if addr + 4 > self.memory.len() {
            panic!("Memory read out of bounds");
        }
        (self.memory[addr] as u32)
            | ((self.memory[addr + 1] as u32) << 8)
            | ((self.memory[addr + 2] as u32) << 16)
            | ((self.memory[addr + 3] as u32) << 24)
    }
    pub fn write_memory_32(&mut self, addr_bf: u64, val: u32) {
        let addr = addr_bf as usize;
        if addr + 4 > self.memory.len() {
            panic!("Memory write out of bounds");
        }
        self.memory[addr] = (val & 0xFF) as u8;
        self.memory[addr + 1] = ((val >> 8) & 0xFF) as u8;
        self.memory[addr + 2] = ((val >> 16) & 0xFF) as u8;
        self.memory[addr + 3] = ((val >> 24) & 0xFF) as u8;
    }

    pub fn interpret(&mut self, prg: &Vec<u32>) {
        self.pc = 0;

        loop {
            let instruction = &prg[self.pc as usize];
            self.pc += 1;

            if *DEBUG {
                println!("{} 0x{:08x}", self.pc, instruction);
            }

            match instruction {
                // control flow
                &END_PRG => return,
                _ if instruction >> 24 == 0xbf => {}
                // smem
                _ if instruction >> 26 == 0b111101 => {
                    let offset_info = prg[self.pc as usize] as u64;
                    let instr = offset_info << 32 | *instruction as u64;
                    // sbase has an implied LSB of zero
                    let sbase = (instr & 0x3f);
                    let sdata = (instr >> 6) & 0x7f;
                    let dlc = (instr >> 13) & 0x1;
                    let glc = (instr >> 14) & 0x1;
                    let glc = (instr >> 14) & 0x1;
                    let op = (instr >> 18) & 0xff;
                    let encoding = (instr >> 26) & 0x3f;
                    // offset is a sign-extend immediate 21-bit constant
                    let offset = ((instr >> 32) & 0x1fffff) as i64 as u64;
                    let soffset = match instr & 0x7F {
                        _ if offset == 0 => 0, // NULL
                        // the SGPR contains an unsigned byte offset (the 2 LSBs are ignored).
                        val => (self.resolve_ssrc(val as u32) & -4) as u64,
                    };

                    if *DEBUG {
                        println!("SMEM {:08X} {:08X} sbase={} sdata={} dlc={} glc={} op={} offset={} soffset={}", instruction, offset_info, sbase, sdata, dlc, glc, op, offset, soffset);
                    }

                    println!("base {:06b}", sbase);

                    let addr = (self.scalar_reg[sbase as usize] as u64) + (offset as u64) + soffset;

                    match op {
                        0..=4 => {
                            for i in 0..=2_u64.pow(op as u32) {
                                self.scalar_reg[(sdata + i) as usize] =
                                    self.read_memory_32(addr + i * 4);
                            }
                        }
                        _ => todo!("smem op {}", op),
                    }
                    panic!();

                    self.pc += 1;
                }
                0xca100080 => {
                    let mut val = prg[self.pc as usize];
                    if val < 255 {
                        val -= 128;
                        self.pc += 1;
                    } else {
                        val = prg[(self.pc + 1) as usize];
                        self.pc += 2;
                    }
                    self.vec_reg[0] = 0;
                    self.vec_reg[1] = val as u32;
                }
                // sop1
                _ if instruction >> 23 == 0b10_1111101 => {
                    let ssrc0 = self.resolve_ssrc(instruction & 0xFF);
                    let op = (instruction >> 8) & 0xFF;
                    let sdst = (instruction >> 16) & 0x7F;

                    match op {
                        0 => self.write_to_sdst(sdst, ssrc0 as u32),
                        1 => {
                            self.write_to_sdst(sdst, ssrc0 as u32);
                            self.write_to_sdst(sdst + 1, ssrc0 as u32);
                        }
                        _ => todo!("sop1 opcode {}", op),
                    }
                }
                // sop2
                _ if instruction >> 30 == 0b10 => {
                    let ssrc0 = self.resolve_ssrc(instruction & 0xFF);
                    let ssrc1 = self.resolve_ssrc((instruction >> 8) & 0xFF);
                    let sdst = (instruction >> 16) & 0x7F;
                    let op = (instruction >> 23) & 0xFF;

                    if *DEBUG {
                        println!("srcs {} {}", instruction & 0xFF, (instruction >> 8) & 0xFF);
                        println!("ssrc0={} ssrc1={} sdst={} op={}", ssrc0, ssrc1, sdst, op);
                    }

                    let tmp = match op {
                        0 => {
                            let tmp = (ssrc0 as u64) + (ssrc1 as u64);
                            self.scc = (tmp >= 0x100000000) as u32;
                            tmp as u32
                        }
                        4 => {
                            let tmp = (ssrc0 as u64) + (ssrc1 as u64) + (self.scc as u64);
                            self.scc = (tmp >= 0x100000000) as u32;
                            tmp as u32
                        }
                        9 => {
                            let tmp = ssrc0 << (ssrc1 & 0x1F);
                            self.scc = (tmp != 0) as u32;
                            tmp as u32
                        }
                        12 => {
                            let tmp = (ssrc0 >> (ssrc1 & 0x1F)) as u32;
                            self.scc = (tmp != 0) as u32;
                            tmp as u32
                        }
                        _ => todo!("sop2 opcode {}", op),
                    };
                    self.write_to_sdst(sdst, tmp);
                }
                // vop1
                _ if instruction >> 25 == 0b0111111 => {
                    let vdst = (instruction >> 17) & 0xFF;
                    let vsrc = instruction & 0x1FF;
                    self.vec_reg[vdst as usize] = self.vec_reg[vsrc as usize];
                }
                // vop2
                _ if instruction >> 31 == 0b0 => {
                    let ssrc0 = self.resolve_ssrc(instruction & 0x1FF);
                    let vsrc1 = self.vec_reg[((instruction >> 9) & 0xFF) as usize];
                    let vdst = (instruction >> 17) & 0xFF;
                    let op = (instruction >> 25) & 0x3F;

                    match op {
                        3 => {
                            self.vec_reg[vdst as usize] = (ssrc0 as f32 + vsrc1 as f32) as u32;
                        }
                        8 => {
                            self.vec_reg[vdst as usize] = (ssrc0 as f32 * vsrc1 as f32) as u32;
                        }
                        29 => {
                            self.vec_reg[vdst as usize] = (ssrc0 as u32) ^ vsrc1;
                        }
                        43 => {
                            self.vec_reg[vdst as usize] = ((ssrc0 as f32 * vsrc1 as f32)
                                + self.vec_reg[vdst as usize] as f32)
                                as u32;
                        }
                        45 => {
                            let simm32 =
                                f32::from_bits((prg[self.pc as usize] as i32).try_into().unwrap());
                            let s0 = f32::from_bits(ssrc0 as u32);
                            let s1 = f32::from_bits(vsrc1 as u32);
                            self.vec_reg[vdst as usize] = (s0 * s1 + simm32).to_bits();
                            self.pc += 1;
                        }
                        _ => todo!("vop2 opcode {}", op),
                    };
                }
                // vop3
                _ if instruction >> 26 == 0b110101 => {
                    let vdst = instruction & 0xFF;
                    let abs = (instruction >> 8) & 0x7;
                    let opsel = (instruction >> 11) & 0xf;
                    let cm = (instruction >> 15) & 0x1;
                    let op = (instruction >> 16) & 0x1ff;

                    let src_info = prg[self.pc as usize];
                    let ssrc0 = self.resolve_ssrc(src_info & 0x1ff);
                    let ssrc1 = self.resolve_ssrc((src_info >> 9) & 0x1ff);
                    let ssrc2 = (src_info >> 18) & 0x1ff;
                    let omod = (src_info >> 27) & 0x3;
                    let neg = (src_info >> 29) & 0x7;

                    match op {
                        259 => {
                            let s0 = f32::from_bits(ssrc0 as u32);
                            let s1 = f32::from_bits(ssrc1 as u32);
                            self.vec_reg[vdst as usize] = (s0 + s1).to_bits();
                        }
                        299 => {
                            let s0 = f32::from_bits(ssrc0.try_into().unwrap());
                            let s1 = f32::from_bits(ssrc1.try_into().unwrap());
                            let d0 = f32::from_bits(
                                (self.vec_reg[vdst as usize] as i32).try_into().unwrap(),
                            );
                            self.vec_reg[vdst as usize] = (s0 * s1 + d0).to_bits();
                        }
                        _ => todo!("vop3 op {op}"),
                    }

                    self.pc += 1;
                }
                // flat_scratch_global
                _ if instruction >> 26 == 0b110111 => {
                    let offset = instruction & 0x1fff;
                    let dls = (instruction >> 13) & 0x1;
                    let glc = (instruction >> 14) & 0x1;
                    let slc = (instruction >> 15) & 0x1;
                    let seg = (instruction >> 16) & 0x3;
                    let op = (instruction >> 18) & 0x7f;

                    let addr_info = prg[self.pc as usize];

                    let addr = addr_info & 0xff;
                    let data = (addr_info >> 8) & 0xff;
                    let saddr = (addr_info >> 16) & 0x7f;
                    let sve = (addr_info >> 23) & 0x1;
                    let vdst = (addr_info >> 24) & 0xff;

                    assert_eq!(seg, 2, "flat and scratch arent supported");
                    match op {
                        26 => {
                            let effective_addr = match saddr {
                                0 => {
                                    let addr_lsb = self.vec_reg[addr as usize] as u64;
                                    let addr_msb = self.vec_reg[(addr + 1) as usize] as u64;
                                    let full_addr = ((addr_msb << 32) | addr_lsb) as u64;
                                    full_addr.wrapping_add(offset as u64) // Add the offset
                                }
                                _ => todo!("address via registers not supported"),
                            };

                            let vdata = self.vec_reg[data as usize];
                            println!("{} {} {}", effective_addr, vdata, data);
                            self.write_memory_32(effective_addr, vdata);
                        }
                        _ => todo!("flat_scratch_global {}", op),
                    }

                    self.pc += 1;
                }

                _ => todo!(),
            }
        }
    }

    /* Scalar ALU utils */
    fn resolve_ssrc(&self, ssrc_bf: u32) -> i32 {
        match ssrc_bf {
            0..=SGPR_COUNT => self.scalar_reg[ssrc_bf as usize] as i32,
            VGPR_COUNT..=511 => self.vec_reg[(ssrc_bf - VGPR_COUNT) as usize] as i32,
            128 => 0,
            129..=192 => (ssrc_bf - 128) as i32,
            _ => todo!("resolve ssrc {}", ssrc_bf),
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

    #[test]
    fn test_s_mov_b64() {
        let mut cpu = CPU::new();
        cpu.interpret(&vec![0xbe920180, END_PRG]);
        assert_eq!(cpu.scalar_reg[18], 0);
        assert_eq!(cpu.scalar_reg[19], 0);
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
        assert_eq!(cpu.scc, 0);
    }

    #[test]
    fn test_s_addc_u32() {
        let mut cpu = CPU::new();
        cpu.scalar_reg[7] = 42;
        cpu.scalar_reg[3] = 13;
        cpu.scc = 1;
        cpu.interpret(&vec![0x82070307, END_PRG]);
        assert_eq!(cpu.scalar_reg[7], 56);
        assert_eq!(cpu.scc, 0);
    }

    #[test]
    fn test_s_ashr_i32() {
        let mut cpu = CPU::new();
        cpu.scalar_reg[15] = 42;
        cpu.interpret(&vec![0x86039f0f, END_PRG]);
        assert_eq!(cpu.scalar_reg[3], 0);
        assert_eq!(cpu.scc, 0);
    }

    #[test]
    fn test_s_lshl_b64() {
        let mut cpu = CPU::new();
        cpu.scalar_reg[2] = 42;
        cpu.interpret(&vec![0x84828202, END_PRG]);
        assert_eq!(cpu.scalar_reg[2], 42 << 2);
        assert_eq!(cpu.scc, 1);
    }
}

#[cfg(test)]
mod test_vop2 {
    use super::*;

    #[test]
    fn test_v_add_f32_e32() {
        let mut cpu = CPU::new();
        cpu.scalar_reg[2] = 41;
        cpu.vec_reg[0] = 1;
        cpu.interpret(&vec![0x06000002, END_PRG]);
        assert_eq!(cpu.vec_reg[0], 42);
    }

    #[test]
    fn test_v_mul_f32_e32() {
        let mut cpu = CPU::new();
        cpu.vec_reg[2] = 21;
        cpu.vec_reg[4] = 2;
        cpu.interpret(&vec![0x10060504, END_PRG]);
        assert_eq!(cpu.vec_reg[3], 42);
    }

    #[test]
    fn test_v_fmac_f32_e32() {
        let mut cpu = CPU::new();
        cpu.vec_reg[1] = 2;
        cpu.vec_reg[2] = 4;
        cpu.interpret(&vec![0x56020302, END_PRG]);
        assert_eq!(cpu.vec_reg[1], 10);
    }

    #[test]
    fn test_v_xor_b32_e32() {
        let mut cpu = CPU::new();
        cpu.vec_reg[5] = 42;
        cpu.scalar_reg[8] = 24;
        cpu.interpret(&vec![0x3a0a0a08, END_PRG]);
        assert_eq!(cpu.vec_reg[5], 50);
    }

    #[test]
    fn test_v_fmaak_f32() {
        let mut cpu = CPU::new();
        cpu.vec_reg[5] = f32::to_bits(0.42);
        cpu.scalar_reg[7] = f32::to_bits(0.24);
        cpu.interpret(&vec![0x5a100a07, f32::to_bits(0.93), END_PRG]);
        assert_eq!(f32::from_bits(cpu.vec_reg[8]), 1.0308);
    }
}

#[cfg(test)]
mod test_vop3 {
    use super::*;

    #[test]
    fn test_v_add_f32() {
        let mut cpu = CPU::new();
        cpu.scalar_reg[0] = f32::to_bits(0.4);
        cpu.scalar_reg[6] = f32::to_bits(0.2);
        cpu.interpret(&vec![0xd5030000, 0x00000006, END_PRG]);
        assert_eq!(f32::from_bits(cpu.vec_reg[0]), 0.6);
    }

    #[test]
    fn test_v_fmac_f32() {
        let mut cpu = CPU::new();
        cpu.scalar_reg[29] = f32::to_bits(0.42);
        cpu.scalar_reg[13] = f32::to_bits(0.24);
        cpu.vec_reg[0] = f32::to_bits(0.15);
        cpu.interpret(&vec![0xd52b0000, 0x00003a0d, END_PRG]);
        assert_eq!(f32::from_bits(cpu.vec_reg[0]), 0.42 * 0.24 + 0.15);
    }
}

#[cfg(test)]
mod test_smem {
    use super::*;

    fn helper_test_s_load(
        mut cpu: CPU,
        op: u32,
        offset: u32,
        data: Vec<u32>,
        base_mem_addr: u32,
        base_sgpr: u32,
    ) {
        data.iter()
            .enumerate()
            .for_each(|(i, &v)| cpu.write_memory_32((base_mem_addr + (i as u32) * 4) as u64, v));
        cpu.interpret(&vec![op, offset, END_PRG]);
        data.iter()
            .enumerate()
            .for_each(|(i, &v)| assert_eq!(cpu.scalar_reg[i + (base_sgpr as usize)], v));
    }

    #[test]
    fn test_s_load_b32() {
        helper_test_s_load(CPU::new(), 0xF4040000, 0xF8000010, vec![42], 2031616, 6);
    }

    #[test]
    fn test_s_load_b64_soffset() {
        let mut cpu = CPU::new();
        cpu.scalar_reg[16] = 22;
        helper_test_s_load(cpu, 0xf4040000, 0xf8000010, (0..=2).collect(), 2031638, 0);
    }

    #[test]
    fn test_s_load_b128() {
        helper_test_s_load(
            CPU::new(),
            0xf4080100,
            0xf8000000,
            (0..=4).collect(),
            2031616,
            4,
        )
    }

    #[test]
    fn test_s_load_b256() {
        helper_test_s_load(
            CPU::new(),
            0xf40c040d,
            0xf8000000,
            (0..=8).collect(),
            2031616,
            16,
        )
    }

    #[test]
    fn test_s_load_b512() {
        helper_test_s_load(
            CPU::new(),
            0xf410000c,
            0xf8000000,
            (0..=16).collect(),
            2031616,
            0,
        )
    }

    #[test]
    fn test_smem_offsets() {
        let mut cpu = CPU::new();
        cpu.interpret(&vec![0xf4080000, 0xf8000000, END_PRG]);
        cpu.interpret(&vec![0xf4040000, 0xf8000010, END_PRG]);
        cpu.interpret(&vec![0xf4000304, 0xf8000008, END_PRG]);
    }
}

#[cfg(test)]
mod test_flat_scratch_global {
    use super::*;

    #[test]
    fn test_global_store_b32() {
        let mut cpu = CPU::new();
        cpu.vec_reg[1] = 0xaa;
        cpu.vec_reg[2] = 0x1;
        cpu.vec_reg[0] = 0xf2;
        cpu.interpret(&vec![0xdc6a0000, 0x00000001, END_PRG]);
        assert_eq!(cpu.read_memory_32(4294967466), 0xf2);
    }
}
#[cfg(test)]
mod test_real_world {
    use super::*;

    fn helper_test_op(op: &str) -> CPU {
        let prg = crate::utils::parse_rdna3_file(&format!("./tests/test_ops/{}.s", op));
        let mut cpu = CPU::new();
        cpu.interpret(&prg);
        return cpu;
    }

    #[test]
    fn test_add_simple() {
        let cpu = helper_test_op("test_add_simple");
    }
}
