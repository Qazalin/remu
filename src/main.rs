pub mod cpu;
pub mod utils;

fn main() {
    let value: i32 = -1;
    let mask = -4;
    println!("{:b}", value);
    println!("{:b}", value & mask);
    println!("{:b}", value as u32);
    println!("{:b}", (value as u32) & mask as u32);
    println!("{}", value as u32);
    println!("{}", (value as u32) & mask as u32);
}
