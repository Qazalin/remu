use crate::utils::DEBUG;

const BASE_ADDRESS: u32 = 0xf8000000;
pub struct CPU {
    prg_counter: usize,
    pub memory: Vec<u8>,
    pub scalar_reg: [u32; 32],
    pub vec_reg: [u32; 32],
}

impl CPU {
    pub fn new() -> Self {
        return CPU {
            prg_counter: 0,
            memory: vec![0; 1_000_000],
            scalar_reg: [0; 32],
            vec_reg: [0; 32],
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
    pub fn write_memory_32(&mut self, address: usize, value: u32) {
        if address + 4 > self.memory.len() {
            panic!("Memory write out of bounds");
        }
        self.memory[address] = (value & 0xFF) as u8;
        self.memory[address + 1] = ((value >> 8) & 0xFF) as u8;
        self.memory[address + 2] = ((value >> 16) & 0xFF) as u8;
        self.memory[address + 3] = ((value >> 24) & 0xFF) as u8;
    }

    pub fn interpret(&mut self, prg: &Vec<usize>) {
        self.prg_counter = 0;

        loop {
            let op = &prg[self.prg_counter];
            self.prg_counter += 1;

            if *DEBUG {
                println!("{} 0x{:08x}", self.prg_counter, op);
            }

            match op {
                0xbfb00000 => return,
                0xbf850001 => {}
                0xf4040000 => {
                    let offset = prg[self.prg_counter] - (BASE_ADDRESS as usize);
                    self.scalar_reg[0] = self.read_memory_32(offset);
                    self.scalar_reg[1] = self.read_memory_32(offset + 4);
                    self.prg_counter += 1;
                }
                0xf4080100 => {
                    let offset = prg[self.prg_counter] - (BASE_ADDRESS as usize);
                    self.scalar_reg[0] = self.read_memory_32(offset);
                    self.scalar_reg[1] = self.read_memory_32(offset + 4);
                    self.scalar_reg[2] = self.read_memory_32(offset + 8);
                    self.scalar_reg[3] = self.read_memory_32(offset + 16);
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
                _ if (0xdc6a0000..=0xdc6affff).contains(op) => {
                    let offset = prg[self.prg_counter - 1] - 0xdc6a0000;
                    let _addr = prg[self.prg_counter + 1];
                    self.write_memory_32(offset, self.vec_reg[1]);
                    self.prg_counter += 2;
                }
                0xbf800000 => {}
                0xbfb60003 => self.vec_reg = [0; 32],
                _ => todo!(),
            }
        }
    }
}

#[cfg(test)]
mod test_ops {
    use super::*;

    #[test]
    fn test_s_endpgm() {
        let mut cpu = CPU::new();
        cpu.interpret(&vec![0xbfb00000]);
        assert_eq!(cpu.prg_counter, 1);
    }

    fn helper_test_mov(vals: &Vec<usize>, expected: usize) {
        let mut cpu = CPU::new();
        cpu.interpret(
            &vec![0xf4040000, 0xf8000000, 0xca100080]
                .iter()
                .chain(vals)
                .chain(&vec![
                    0xbf89fc07, 0xdc6a0000, 0x00000100, 0xbf800000, 0xbfb60003, 0xbfb00000,
                ])
                .map(|x| *x)
                .collect::<Vec<usize>>(),
        );
        assert_eq!(cpu.read_memory_32(0), expected as u32);
    }
    #[test]
    fn test_mov() {
        helper_test_mov(&vec![0x000000aa], 42);
        helper_test_mov(&vec![0x000000ff, 0x000000aa], 170);
        helper_test_mov(&vec![0x000000ff, 0x00000012], 18);
        helper_test_mov(&vec![0x000000ff, 0x000000ff], 255);
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
        assert_eq!(cpu.read_memory_32(0), 42);
    }
}
