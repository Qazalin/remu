use crate::ops::OPCODES_MAP;
use crate::utils::DEBUG;

const BASE_ADDRESS: u32 = 0xf8000000;
pub struct CPU {
    prg_counter: usize,
    pub memory: Vec<u8>,
    pub registers: [u32; 32],
}

impl CPU {
    pub fn new() -> Self {
        return CPU {
            prg_counter: 0,
            memory: vec![0; 256],
            registers: [0; 32],
        };
    }

    pub fn read_register(&self, reg: usize) -> u32 {
        self.registers[reg]
    }
    pub fn write_register(&mut self, reg: usize, value: u32) {
        self.registers[reg] = value;
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
                    self.write_register(0, low);
                    self.write_register(1, high);
                    self.prg_counter += 1;
                }
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
    fn test_kernel() {
        helper_test_op("add_2_2_0");
    }
}
