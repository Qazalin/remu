pub mod cpu;
pub mod utils;

fn main() {
    let sgpr_pair: u64 = 0b000011;

    let r0 = sgpr_pair & 0x7;
    let r1 = (sgpr_pair >> 3) & 0x7;
    println!("{:b}", sgpr_pair);
    println!("{:b} {}", r0, r0);
    println!("{:b} {}", r1, r1);
}
