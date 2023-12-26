use crate::utils::DEBUG;

const SGPR_COUNT: usize = 105;
pub const END_PRG: usize = 0xbfb00000;

pub struct CPU {
    prg_counter: usize,
    pub memory: Vec<u8>,
    pub scalar_reg: [u32; SGPR_COUNT],
    pub vec_reg: [u32; 10000],
    scc: u32,
}

impl CPU {
    pub fn new() -> Self {
        return CPU {
            prg_counter: 0,
            scc: 0,
            memory: vec![0; 24 * 1_073_741_824],
            scalar_reg: [0; SGPR_COUNT],
            vec_reg: [0; 10000],
        };
    }

    pub fn read_memory_32(&self, addr: usize) -> u32 {
        if addr + 4 > self.memory.len() {
            panic!("Memory read out of bounds");
        }
        (self.memory[addr] as u32)
            | ((self.memory[addr + 1] as u32) << 8)
            | ((self.memory[addr + 2] as u32) << 16)
            | ((self.memory[addr + 3] as u32) << 24)
    }
    pub fn write_memory_32(&mut self, addr: usize, val: u32) {
        if addr + 4 > self.memory.len() {
            panic!("Memory write out of bounds");
        }
        self.memory[addr] = (val & 0xFF) as u8;
        self.memory[addr + 1] = ((val >> 8) & 0xFF) as u8;
        self.memory[addr + 2] = ((val >> 16) & 0xFF) as u8;
        self.memory[addr + 3] = ((val >> 24) & 0xFF) as u8;
    }

    pub fn interpret(&mut self, prg: &Vec<usize>) {
        self.prg_counter = 0;

        loop {
            let instruction = &prg[self.prg_counter];
            self.prg_counter += 1;

            if *DEBUG {
                println!("{} 0x{:08x}", self.prg_counter, instruction);
            }

            match instruction {
                // control flow
                &END_PRG => return,
                _ if instruction >> 24 == 0xbf => {}
                // smem
                _ if instruction >> 26 == 0b111101 => {
                    let sbase = instruction & 0x3F;
                    let sdata = (instruction >> 6) & 0x7F;
                    let dlc = (instruction >> 13) & 0x1;
                    let glc = (instruction >> 14) & 0x1;
                    let op = (instruction >> 18) & 0xFF;
                    let offset_info = prg[self.prg_counter];
                    let offset = offset_info >> 11;
                    let soffset = match offset_info & 0x7F {
                        _ if offset == 0 => 0, // NULL
                        0..=SGPR_COUNT => self.scalar_reg[offset_info & 0x7F],
                        _ => todo!("smem soffset {}", offset_info & 0x7F),
                    };

                    if *DEBUG {
                        println!(
                            "sbase={} sdata={} dlc={} glc={} op={} offset={} soffset={}",
                            sbase, sdata, dlc, glc, op, offset, soffset
                        );
                    }

                    let addr = self.scalar_reg[sbase] + (offset as u32) + soffset;

                    match op {
                        0..=4 => {
                            for i in 0..=2_usize.pow(op as u32) {
                                self.scalar_reg[sdata + i] =
                                    self.read_memory_32((addr as usize) + i * 4);
                            }
                        }
                        _ => todo!("smem op {}", op),
                    }

                    self.prg_counter += 1;
                }
                0xca100080 => {
                    let mut val = prg[self.prg_counter];
                    if val < 255 {
                        val -= 128;
                        self.prg_counter += 1;
                    } else {
                        val = prg[self.prg_counter + 1];
                        self.prg_counter += 2;
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
                _ if (0x7e000000..=0x7effffff).contains(instruction) => {
                    let vdst = (instruction >> 17) & 0xFF;
                    let vsrc = instruction & 0x1FF;
                    self.vec_reg[vdst] = self.vec_reg[vsrc];
                }
                _ => todo!(),
            }
        }
    }

    /* Scalar ALU utils */
    fn resolve_ssrc(&self, ssrc_bf: usize) -> i32 {
        match ssrc_bf {
            0..=SGPR_COUNT => self.scalar_reg[ssrc_bf] as i32,
            128 => 0,
            129..=192 => (ssrc_bf - 128) as i32,
            _ => todo!("resolve ssrc {}", ssrc_bf),
        }
    }
    fn write_to_sdst(&mut self, sdst_bf: usize, val: u32) {
        match sdst_bf {
            0..=SGPR_COUNT => self.scalar_reg[sdst_bf] = val,
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
mod test_smem {
    use super::*;

    fn helper_test_s_load(
        mut cpu: CPU,
        op: usize,
        offset: usize,
        data: Vec<u32>,
        base_mem_addr: usize,
        base_sgpr: usize,
    ) {
        data.iter()
            .enumerate()
            .for_each(|(i, &v)| cpu.write_memory_32(base_mem_addr + i * 4, v));
        cpu.interpret(&vec![op, offset, END_PRG]);
        data.iter()
            .enumerate()
            .for_each(|(i, &v)| assert_eq!(cpu.scalar_reg[i + base_sgpr], v));
    }

    #[test]
    fn test_s_load_b32() {
        helper_test_s_load(CPU::new(), 0xf4000183, 0xf8000000, vec![42], 2031616, 6);
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
