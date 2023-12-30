pub mod cpu;
pub mod utils;

use crate::cpu::CPU;

fn main() {
    let instructions = crate::utils::parse_rdna3_file("./tests/test_ops/test_add_simple.s");
    let mut cpu = CPU::new();
    cpu.interpret(&instructions);
}
