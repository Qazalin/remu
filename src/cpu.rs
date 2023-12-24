use crate::ops::OPCODES_MAP;

pub struct CPU {
    prg_counter: usize,
}

impl CPU {
    pub fn new() -> Self {
        return CPU { prg_counter: 0 };
    }

    pub fn interpret(&mut self, prg: Vec<usize>) {
        self.prg_counter = 0;

        loop {
            let op = OPCODES_MAP
                .get(&prg[self.prg_counter])
                .expect("invalid code");
            self.prg_counter += 1;

            match op.code {
                0xbfb00000 => return,
                _ => todo!(),
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn helper_test_cpu(prg: Vec<usize>) -> CPU {
        let mut cpu = CPU::new();
        cpu.interpret(prg);
        return cpu;
    }

    #[test]
    fn test_s_endpgm() {
        let cpu = helper_test_cpu([0xbfb00000].to_vec());
        assert_eq!(cpu.prg_counter, 1);
    }
}
