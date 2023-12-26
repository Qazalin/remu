use crate::utils::DEBUG;

const BASE_ADDRESS: u32 = 0xf8000000;

const SGPR_COUNT: usize = 105;

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
            memory: vec![0; 1_000_000],
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

    pub fn interpret(&mut self, prg: &Vec<usize>) {
        self.prg_counter = 0;

        loop {
            let instruction = &prg[self.prg_counter];
            self.prg_counter += 1;

            if *DEBUG {
                println!("{} 0x{:08x}", self.prg_counter, instruction);
            }

            match instruction {
                0xbfb00000 => return,
                0xbf850001 => {}
                // smem
                _ if instruction >> 26 == 0b111101 => {
                    let sbase = instruction & 0x3F;
                    let sdata = (instruction >> 6) & 0x7F;
                    let dlc = (instruction >> 13) & 0x1;
                    let glc = (instruction >> 14) & 0x1;
                    let op = (instruction >> 18) & 0xFF;
                    let instruction1 = prg[self.prg_counter];
                    let offset = (instruction1 & 0x1FFFFF) as u32;
                    let soffset = match instruction1 >> 25 {
                        _ if offset == 0 => 0, // NULL
                        0..=SGPR_COUNT => self.scalar_reg[instruction1 >> 25],
                        _ => todo!("smem soffset {}", instruction1 >> 25),
                    };

                    if *DEBUG {
                        println!(
                            "sbase={} sdata={} dlc={} glc={} op={} offset={} soffset={}",
                            sbase, sdata, dlc, glc, op, offset, soffset
                        );
                    }

                    let addr = self.scalar_reg[sbase] + offset + soffset;
                    match op {
                        2 => {
                            let addr = addr as usize;
                            self.scalar_reg[sdata] = self.read_memory_32(addr);
                            self.scalar_reg[sdata + 1] = self.read_memory_32(addr + 4);
                            self.scalar_reg[sdata + 2] = self.read_memory_32(addr + 8);
                            self.scalar_reg[sdata + 3] = self.read_memory_32(addr + 12);
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
                0xbf89fc07 => {}
                0xbf800000 => {}
                0xbfb60003 => self.vec_reg = [0; 10000],
                // sop1
                _ if (0xbe800080..=0xbeffffff).contains(instruction) => {
                    let sdst = (instruction >> 16) & 0x7F;
                    let _op = (instruction >> 8) & 0xFF;
                    let ssrc0 = instruction & 0xFF;
                    self.scalar_reg[sdst] = self.scalar_reg[ssrc0];
                }
                // sop2
                _ if instruction >> 30 == 0b10 => {
                    let resolve_ssrc = |ssrc_bf| match ssrc_bf {
                        0..=SGPR_COUNT => self.scalar_reg[ssrc_bf] as i32,
                        129..=192 => (ssrc_bf - 128) as i32,
                        _ => todo!("sop2 ssrc {}", ssrc_bf),
                    };
                    let ssrc0 = resolve_ssrc(instruction & 0xFF);
                    let ssrc1 = resolve_ssrc((instruction >> 8) & 0xFF);
                    let sdst = (instruction >> 16) & 0x7F;
                    let op = (instruction >> 23) & 0xFF;

                    if *DEBUG {
                        println!("srcs {} {}", instruction & 0xFF, (instruction >> 8) & 0xFF);
                        println!("ssrc0={} ssrc1={} sdst={} op={}", ssrc0, ssrc1, sdst, op);
                    }

                    match op {
                        0 => {
                            let tmp = (ssrc0 as u64) + (ssrc1 as u64);
                            self.scalar_reg[sdst] = tmp as u32;
                            self.scc = (tmp >= 0x100000000) as u32;
                        }
                        4 => {
                            let tmp = (ssrc0 as u64) + (ssrc1 as u64) + (self.scc as u64);
                            self.scalar_reg[sdst] = tmp as u32;
                            self.scc = (tmp >= 0x100000000) as u32;
                        }
                        9 => {
                            self.scalar_reg[sdst] = (ssrc0 << (ssrc1 & 0x1F)) as u32;
                            self.scc = (self.scalar_reg[sdst] != 0) as u32;
                        }
                        12 => {
                            self.scalar_reg[sdst] = (ssrc0 >> (ssrc1 & 0x1F)) as u32;
                            self.scc = (self.scalar_reg[sdst] != 0) as u32;
                        }
                        _ => todo!("sop2 opcode {}", op),
                    }
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
}

pub const END: usize = 0xbfb00000;

#[cfg(test)]
mod test_sop2 {
    use super::*;

    #[test]
    fn test_s_add_u32() {
        let mut cpu = CPU::new();
        cpu.scalar_reg[2] = 42;
        cpu.scalar_reg[6] = 13;
        cpu.interpret(&vec![0x80060206, END]);
        assert_eq!(cpu.scalar_reg[6], 55);
        assert_eq!(cpu.scc, 0);
    }

    #[test]
    fn test_s_addc_u32() {
        let mut cpu = CPU::new();
        cpu.scalar_reg[7] = 42;
        cpu.scalar_reg[3] = 13;
        cpu.scc = 1;
        cpu.interpret(&vec![0x82070307, END]);
        assert_eq!(cpu.scalar_reg[7], 56);
        assert_eq!(cpu.scc, 0);
    }

    #[test]
    fn test_s_ashr_i32() {
        let mut cpu = CPU::new();
        cpu.scalar_reg[15] = 42;
        cpu.interpret(&vec![0x86039f0f, END]);
        assert_eq!(cpu.scalar_reg[3], 0);
        assert_eq!(cpu.scc, 0);
    }

    #[test]
    fn test_s_lshl_b64() {
        let mut cpu = CPU::new();
        cpu.scalar_reg[2] = 42;
        cpu.interpret(&vec![0x84828202, END]);
        assert_eq!(cpu.scalar_reg[2], 42 << 2);
        assert_eq!(cpu.scc, 1);
    }
}

#[cfg(test)]
mod test_smem {
    use super::*;

    #[test]
    fn test_s_load_b128() {
        let mut cpu = CPU::new();
        cpu.interpret(&vec![0xf4080100, 0xf8000000, END]);
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
