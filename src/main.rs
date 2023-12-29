#![allow(unused)]
pub mod cpu;
pub mod utils;

fn main() {
    let n0: u32 = 0xf4080000;
    let n1: u32 = 0xF8000008;

    let instr: u64 = (n1 as u64) << 32 | n0 as u64;

    let sbase = instr & 0x3f;
    let sdata = (instr >> 6) & 0x7f;
    let dlc = (instr >> 13) & 0x1;
    let glc = (instr >> 14) & 0x1;
    let glc = (instr >> 14) & 0x1;
    let op = (instr >> 18) & 0xff;
    let encoding = (instr >> 26) & 0x3f;
    let offset = (instr >> 32) & 0x1fffff;
    let soffset = (instr >> 57) & 0x7f;

    println!("{:b}", instr);
    println!("{:06b}", sbase);
    println!("{:07b}", sdata);
    println!("{:01b}", dlc);
    println!("{:01b}", glc);
    println!("{:08b}", op);
    println!("{:06b}", encoding);
    println!("{:021b}", offset);
    println!("{:07b}", soffset);
}
