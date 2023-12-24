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
            let opcode = OPCODES_MAP
                .get(&prg[self.prg_counter])
                .expect("invalid code");

            match opcode.code {
                _ => todo!(),
            }
        }
    }
}
