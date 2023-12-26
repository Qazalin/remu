pub mod cpu;
pub mod utils;

fn main() {
    let instruction: usize = 0b10000110000000111001111100001111;

    // 10 0001100 0000011 10011111 00001111

    // first 8 bits is the ssrc0
    let ssrc0 = instruction & 0xFF;
    println!("ssrc0={}", ssrc0);
    assert_eq!(ssrc0, 0b00001111);

    // next 8 bits is the ssrc1
    let ssrc1 = (instruction >> 8) & 0xFF;
    println!("ssrc1={}", ssrc1);
    assert_eq!(ssrc1, 0b10011111);

    // next 7 bits is sdst
    let sdst = (instruction >> 16) & 0x7F;
    println!("sdst={}", sdst);
    assert_eq!(sdst, 0b0000011);

    // next 8 bits is the operation
    let operation = (instruction >> 23) & 0xFF;
    println!("op={}", operation);
    assert_eq!(operation, 0b0001100);

    // last 2 bits is the opcode
    let opcode = instruction >> 30;
    println!("opcode={}", opcode);
    assert_eq!(opcode, 0b10);
}
