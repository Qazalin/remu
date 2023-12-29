pub mod cpu;
pub mod utils;

fn main() {
    let n0: u32 = 0xf4080100;
    let n1: u32 = 0xf8000000;

    let n = (n1 as u64) << 32 | (n0 as u64);

    println!("{:b}", n);
    println!("{:b}", n0);
    println!("{:b}", n1);
}
