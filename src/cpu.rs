use crate::ops::OPCODES_MAP;
use crate::utils::DEBUG;

pub struct CPU {
    prg_counter: usize,
    memory: Vec<u8>,
    registers: [u32; 32],
}

impl CPU {
    pub fn new() -> Self {
        return CPU {
            prg_counter: 0,
            memory: vec![0; 256],
            registers: [0; 32],
        };
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
                0xf8000000 => {
                    let offset = prg[self.prg_counter];
                    println!("{} with offset 0x{:08x} {}", op.mnemonic, offset, offset);
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
        helper_test_op("E_4");
    }
}
