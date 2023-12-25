use crate::ops::OPCODES_MAP;
use crate::utils::{print_hex, DEBUG};

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
            memory: vec![0; 256],
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

    pub fn interpret(&mut self, prg: Vec<usize>) {
        self.prg_counter = 0;

        loop {
            let op = OPCODES_MAP
                .get(&prg[self.prg_counter])
                .expect(&format!("invalid code 0x{:08x}", &prg[self.prg_counter]));
            self.prg_counter += 1;

            if *DEBUG {
                println!("{} {:?}", self.prg_counter, op);
            }

            match op.code {
                0xbfb00000 => return,
                0xbf850001 => {}
                0xf4040000 => {
                    let offset = prg[self.prg_counter] - (BASE_ADDRESS as usize);
                    let low = self.read_memory_32(offset);
                    let high = self.read_memory_32(offset + 4);
                    self.scalar_reg[0] = low;
                    self.scalar_reg[1] = high;
                    self.prg_counter += 1;
                }
                0xca100080 => {
                    let loc = prg[self.prg_counter];
                    self.vec_reg[0] = self.read_memory_32(loc);
                    self.vec_reg[1] = self.read_memory_32(loc + 4);
                    self.prg_counter += 1;
                }
                0xbf89fc07 => {}
                0xdc6a0000 => {
                    let addr = prg[self.prg_counter];
                    // let value = self.read_memory_32(addr);
                    self.prg_counter += 1;
                }
                0xbf800000 => {}
                0xbfb60003 => self.vec_reg = [0; 32],
                _ => todo!(),
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn helper_test_op(op: &str) -> CPU {
        let prg = crate::utils::parse_rdna3_file(&format!("./tests/{}.s", op));
        let mut cpu = CPU::new();
        cpu.interpret(prg);
        return cpu;
    }

    #[test]
    fn test_s_endpgm() {
        let cpu = helper_test_op("s_endpgm");
        assert_eq!(cpu.prg_counter, 1);
    }

    #[test]
    fn test_global_store() {
        helper_test_op("global_store");
    }
}
