use lazy_static::lazy_static;
use std::collections::HashMap;

#[derive(Debug)]
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
    pub static ref OPS: Vec<OpCode> = vec![
        OpCode::new(0xbfb00000, "s_endpgm"),
        OpCode::new(0xbf850001, "s_clause"),
        OpCode::new(0xf4080100, "s_load_b128"),
    ];
    pub static ref OPCODES_MAP: HashMap<usize, &'static OpCode> = {
        let mut map = HashMap::new();
        for cpuop in &*OPS {
            map.insert(cpuop.code, cpuop);
        }
        map
    };
}
