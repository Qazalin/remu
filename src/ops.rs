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
        OpCode::new(0xf4040000, "s_load_b64"),
        OpCode::new(0xca100080, "v_dual_mov_b32"),
        OpCode::new(0xbf89fc07, "s_waitcntpcc"),
        OpCode::new(0xbf89fc07, "s_waitcntpcc"),
        OpCode::new(0xdc6a0000, "global_store_b32"),
        OpCode::new(0xbf800000, "s_nop"),
        OpCode::new(0xbfb60003, "s_sendmsg"),
    ];
    pub static ref OPCODES_MAP: HashMap<usize, &'static OpCode> = {
        let mut map = HashMap::new();
        for cpuop in &*OPS {
            map.insert(cpuop.code, cpuop);
        }
        map
    };
}
