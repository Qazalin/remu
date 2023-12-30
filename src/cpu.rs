#![allow(unused)]
use crate::utils::{twos_complement_21bit, DEBUG};

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

            match instruction {
                // control flow
                &END_PRG => return,
                _ if instruction >> 24 == 0xbf => {}
                // smem
                _ if instruction >> 26 == 0b111101 => {
                    let offset_info = prg[self.pc as usize] as u64;
                    let instr = offset_info << 32 | *instruction as u64;
                    // NOTE: sbase has an implied LSB of zero
                    /**
                     * In smem reads when the address-base comes from an SGPR-pair, it's always
                     * even-aligned. s[sbase:sbase+1]
                     */
                    let sbase = (instr & 0x3f) * 2;
                    let sdata = (instr >> 6) & 0x7f;
                    let dlc = (instr >> 13) & 0x1;
                    let glc = (instr >> 14) & 0x1;
                    let glc = (instr >> 14) & 0x1;
                    let op = (instr >> 18) & 0xff;
                    let encoding = (instr >> 26) & 0x3f;
                    // offset is a sign-extend immediate 21-bit constant
                    let offset = twos_complement_21bit((instr >> 32) & 0x1fffff);
                    let soffset = match instr & 0x7F {
                        _ if offset == 0 => 0, // NULL
                        // the SGPR contains an unsigned byte offset (the 2 LSBs are ignored).
                        val => (self.resolve_ssrc(val as u32) & -4) as u64,
                    };

                    if *DEBUG >= 1 {
                        println!("SMEM {:08X} {:08X} sbase={} sdata={} dlc={} glc={} op={} offset={} soffset={}", instruction, offset_info, sbase, sdata, dlc, glc, op, offset, soffset);
                    }

                    let addr = ((self.scalar_reg[sbase as usize] as i64)
                        + offset
                        + (soffset as i64)) as u64;

                    match op {
                        0..=4 => {
                            for i in 0..2_u64.pow(op as u32) {
                                self.scalar_reg[(sdata + i) as usize] =
                                    self.read_memory_32(addr + i * 4);
                                if *DEBUG == 2 {
                                    println!(
                                        "[state] loaded the value={} from mem={} to sgpr={}",
                                        self.scalar_reg[(sdata + i) as usize],
                                        addr + i * 4,
                                        sdata + i
                                    );
                                }
                            }
                        }
                        _ => todo!("smem op {}", op),
                    }

                    self.pc += 1;
                }
                // sop1
                _ if instruction >> 23 == 0b10_1111101 => {
                    let ssrc0 = self.resolve_ssrc(instruction & 0xFF);
                    let op = (instruction >> 8) & 0xFF;
                    let sdst = (instruction >> 16) & 0x7F;

                    if *DEBUG >= 1 {
                        println!("SOP1 ssrc0={} sdst={} op={}", ssrc0, sdst, op);
                    }

                    match op {
                        0 => {
                            if *DEBUG == 2 {
                                println!(
                                "[state] writing to sdst={} the value={} (possibly from sgpr={})",
                                sdst,
                                ssrc0,
                                instruction & 0xFF
                            );
                            }
                            self.write_to_sdst(sdst, ssrc0 as u32)
                        }
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

                    if *DEBUG >= 1 {
                        println!(
                            "SOP2 ssrc0={} ssrc1={} sdst={} op={}",
                            ssrc0, ssrc1, sdst, op
                        );
                    }

                    let tmp = match op {
                        0 => {
                            if *DEBUG == 2 {
                                println!(
                                    "[state] adding the values in sgprs {} and {}",
                                    instruction & 0xfF,
                                    (instruction >> 8) & 0xFF
                                );
                            }
                            let tmp = (ssrc0 as u64) + (ssrc1 as u64);
                            self.scc = (tmp >= 0x100000000) as u32;
                            tmp as u32
                        }
                        4 => {
                            if *DEBUG == 2 {
                                println!(
                                    "[state] adding the values in sgprs {} and {}",
                                    instruction & 0xfF,
                                    (instruction >> 8) & 0xFF
                                );
                            }

                            let tmp = (ssrc0 as u64) + (ssrc1 as u64) + (self.scc as u64);
                            self.scc = (tmp >= 0x100000000) as u32;
                            tmp as u32
                        }
                        9 => {
                            if *DEBUG == 2 {
                                println!(
                                    "[state] left shift sgpr={} by {}",
                                    instruction & 0xfF,
                                    (instruction >> 8) & 0xFF
                                );
                            }
                            let tmp = ssrc0 << (ssrc1 & 0x1F);
                            self.scc = (tmp != 0) as u32;
                            tmp as u32
                        }
                        12 => {
                            if *DEBUG == 2 {
                                println!(
                                    "[state] left shift sgpr={} by {}",
                                    instruction & 0xfF,
                                    ssrc1 & 0x1F
                                );
                            }
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
                    let src = self.resolve_ssrc(instruction & 0x1ff);
                    let op = (instruction >> 9) & 0xff;
                    let vdst = (instruction >> 17) & 0xff;

                    if *DEBUG >= 1 {
                        println!("VOP1 src={} op={} vdst={}", src, op, vdst);
                    }

                    match op {
                        1 => self.vec_reg[vdst as usize] = src as u32,
                        _ => todo!(),
                    }
                }
                // vop2
                _ if instruction >> 31 == 0b0 => {
                    let ssrc0 = self.resolve_ssrc(instruction & 0x1FF);
                    let vsrc1 = self.vec_reg[((instruction >> 9) & 0xFF) as usize];
                    let vdst = (instruction >> 17) & 0xFF;
                    let op = (instruction >> 25) & 0x3F;

                    if *DEBUG >= 1 {
                        println!(
                            "VOP2 ssrc0={} vsrc1={} vdst={} op={}",
                            ssrc0, vsrc1, vdst, op
                        );
                    }

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

                    if *DEBUG >= 1 {
                        println!(
                            "VOP3 vdst={} abs={} opsel={} cm={} op={} ssrc0={} ssrc1={} ssrc2={} omod={} neg={}",
                            vdst, abs, opsel, cm, op, ssrc0, ssrc1, ssrc2, omod, neg
                        );
                    }

                    match op {
                        259 => {
                            let s0 = f32::from_bits(ssrc0 as u32);
                            let s1 = f32::from_bits(ssrc1 as u32);
                            self.vec_reg[vdst as usize] = (s0 + s1).to_bits();
                            if *DEBUG >= 2 {
                                println!("[state] store ALU of {}+{} to vec_reg[{}]", s0, s1, vdst);
                            }
                        }
                        299 => {
                            let s0 = f32::from_bits(ssrc0.try_into().unwrap());
                            let s1 = f32::from_bits(ssrc1.try_into().unwrap());
                            let d0 = f32::from_bits(
                                (self.vec_reg[vdst as usize] as i32).try_into().unwrap(),
                            );
                            self.vec_reg[vdst as usize] = (s0 * s1 + d0).to_bits();
                        }
                        _ => todo!(),
                    }

                    self.pc += 1;
                }
                // global
                _ if instruction >> 26 == 0b110111 => {
                    let addr_info = prg[self.pc as usize] as u64;
                    let instr = addr_info << 32 | *instruction as u64;

                    let offset = instr & 0x1fff;
                    let dls = (instr >> 13) & 0x1;
                    let glc = (instr >> 14) & 0x1;
                    let slc = (instr >> 15) & 0x1;
                    let seg = (instr >> 16) & 0x3;
                    let op = (instr >> 18) & 0x7f;
                    let addr = (instr >> 32) & 0xff;
                    let data = (instr >> 40) & 0xff;
                    let saddr = (instr >> 48) & 0x7f;
                    let sve = (addr_info >> 55) & 0x1;
                    let vdst = (addr_info >> 56) & 0xff;

                    if *DEBUG >= 1 {
                        println!(
                            "GLOBAL {:08X} {:08X} addr={} data={} saddr={}",
                            instruction, addr_info, addr, data, saddr
                        );
                    }

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

                            if *DEBUG >= 2 {
                                print!(
                                    "storing value={} from vector register {} to mem[{}]",
                                    vdata, data, effective_addr
                                );
                            }
                            self.write_memory_32(effective_addr, vdata);
                        }
                        _ => todo!(),
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
        data: &Vec<u32>,
        base_mem_addr: u64,
        starting_dest_sgpr: u32,
    ) {
        data.iter()
            .enumerate()
            .for_each(|(i, &v)| cpu.write_memory_32((base_mem_addr + (i as u64) * 4) as u64, v));
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
        let offset_value: i64 = -0x4;
        let mut cpu = CPU::new();
        cpu.scalar_reg[0] = 10000;
        helper_test_s_load(cpu, 0xf4000000, 0xf81fffd8, &vec![42], 19960, 0);
    }

    #[test]
    fn test_s_load_b64() {
        let data = (0..=2).collect();

        // positive offset
        helper_test_s_load(CPU::new(), 0xf4040000, 0xf8000010, &data, 0x10, 0);
        helper_test_s_load(CPU::new(), 0xf4040204, 0xf8000268, &data, 0x268, 8);

        // negative offset
        let mut cpu = CPU::new();
        cpu.scalar_reg[2] = 612;
        helper_test_s_load(cpu, 0xf4040301, 0xf81ffd9c, &data, 0, 12);
    }

    #[test]
    fn test_s_load_b128() {
        let data = (0..=4).collect();

        // positive offset
        helper_test_s_load(CPU::new(), 0xf4080000, 0xf8000000, &data, 0, 0);

        let mut cpu = CPU::new();
        let base_mem_addr: u64 = 0x10;
        cpu.scalar_reg[6] = base_mem_addr as u32;
        helper_test_s_load(cpu, 0xf4080203, 0xf8000000, &data, base_mem_addr, 8);

        // negative offset
        let mut cpu = CPU::new();
        cpu.scalar_reg[2] = 0x10;
        helper_test_s_load(cpu, 0xf4080401, 0xf81ffff0, &data, 0, 16);
    }

    #[test]
    fn test_s_load_b256() {
        let data = (0..=8).collect();

        // positive offset
        helper_test_s_load(CPU::new(), 0xf40c0000, 0xf8000000, &data, 0, 0);

        let mut cpu = CPU::new();
        let base_mem_addr: u64 = 0x55;
        cpu.scalar_reg[10] = base_mem_addr as u32;
        helper_test_s_load(cpu, 0xf40c0005, 0xf8000040, &data, base_mem_addr + 0x40, 0);

        // negative offset
        let mut cpu = CPU::new();
        let base_mem_addr: u64 = 0x55;
        cpu.scalar_reg[2] = base_mem_addr as u32;
        helper_test_s_load(cpu, 0xf40c0401, 0xf81fffd0, &data, base_mem_addr - 0x30, 16);
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

    fn helper_test_op(cpu: &mut CPU, op: &str) {
        let prg = crate::utils::parse_rdna3_file(&format!("./tests/test_ops/{}.s", op));
        cpu.interpret(&prg);
    }

    fn read_array(cpu: &CPU, addr: u64, sz: usize) -> Vec<u32> {
        let mut data = vec![0; sz];
        for i in 0..sz {
            data[i] = cpu.read_memory_32(addr + (i * 4) as u64);
        }
        return data;
    }
    fn read_array_bytes(cpu: &CPU, addr: u64, sz: usize) -> Vec<u8> {
        let mut data = vec![0; sz * 4];
        for i in 0..data.len() {
            data[i] = cpu.memory[addr as usize + i];
        }
        return data;
    }

    #[test]
    fn test_add_simple() {
        let mut cpu = CPU::new();
        let data0 = vec![0.0; 4];
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![5.0, 6.0, 7.0, 8.0];
        let expected_data0 = vec![6.0, 8.0, 10.0, 12.0];

        // allocate memory
        let data0_addr = 1000;
        for i in 0..data0.len() {
            cpu.write_memory_32(data0_addr + (i * 4) as u64, f32::to_bits(data0[i]));
        }

        let data1_addr = 2000;
        for i in 0..data1.len() {
            cpu.write_memory_32(data1_addr + (i * 4) as u64, f32::to_bits(data1[i]));
        }

        let data2_addr = 3000;
        for i in 0..data2.len() {
            cpu.write_memory_32(data2_addr + (i * 4) as u64, f32::to_bits(data2[i]));
        }

        println!("Ending memory layout:");

        println!(
            "data0 = {:?} {:?}",
            read_array(&cpu, data0_addr, 4),
            read_array_bytes(&cpu, data0_addr, 4)
        );
        println!(
            "data1 = {:?} {:?}",
            read_array(&cpu, data1_addr, 4),
            read_array_bytes(&cpu, data1_addr, 4)
        );
        println!(
            "data2 = {:?} {:?}",
            read_array(&cpu, data2_addr, 4),
            read_array_bytes(&cpu, data2_addr, 4)
        );

        // allocate src registers
        cpu.scalar_reg[6] = data1_addr as u32;
        cpu.scalar_reg[0] = data2_addr as u32;
        cpu.vec_reg[0] = data0_addr as u32;

        // "launch" kernel
        let global_size = (1, 1, 1);

        for i in 0..global_size.0 {
            cpu.scalar_reg[15] = i as u32; // TODO shouldnt this be the address of blockIdx?
            helper_test_op(&mut cpu, "test_add_simple");
        }

        for i in 0..global_size.0 {
            let val = cpu.read_memory_32(data0_addr + (i * 4) as u64);
            assert_eq!(f32::from_bits(val), expected_data0[i]);
        }
    }
}
