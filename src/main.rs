pub mod cpu;
pub mod utils;

fn main() {
    let sgpr_pair: i32 = 0b111111111111111111100;

    println!("{:b} {}", sgpr_pair, sgpr_pair);
}
