use lazy_static::lazy_static;
use std::collections::HashMap;

pub struct OpCode {
    pub code: usize,
    pub mnemonic: &'static str,
}
impl OpCode {
    fn new(code: usize, mnemonic: &'static str) -> Self {
        OpCode { code, mnemonic }
    }
}

lazy_static! {
    pub static ref OPS: Vec<OpCode> = vec![OpCode::new(0xBFB00000, "s_endpgm"),]; // TODO tbc
    pub static ref OPCODES_MAP: HashMap<usize, &'static OpCode> = {
        let mut map = HashMap::new();
        for cpuop in &*OPS {
            map.insert(cpuop.code, cpuop);
        }
        map
    };
}
