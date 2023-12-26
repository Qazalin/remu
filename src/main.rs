pub mod cpu;
pub mod utils;

fn main() {
    let instruction0: usize = 0b11110100000010000000000100000000;

    // 111101 00000010 000 0 0 0000100 000000
    let sbase = instruction0 & 0x3F;

    // next 7 bits
    let sdata = (instruction0 >> 6) & 0x7F;
    assert_eq!(sdata, 0b0000100);

    // next 1 bit
    let dlc = (instruction0 >> 13) & 0x1;
    assert_eq!(dlc, 0b0);

    // next 1 bit
    let glc = (instruction0 >> 14) & 0x1;
    assert_eq!(glc, 0b0);

    // next 3 bits
    let void = (instruction0 >> 15) & 0x7;
    assert_eq!(void, 0b000);

    // next 8 bits
    let op = (instruction0 >> 18) & 0xFF;
    assert_eq!(op, 2);

    // next 6 bits
    let encoding = (instruction0 >> 26) & 0x3F;
    assert_eq!(encoding, 0b111101);

    let instruction1: usize = 0b11111000000000000000000000000000;

    // first 21 bits
    let offset = instruction1 & 0x1FFFFF;
    assert_eq!(offset, 0);

    // last 7 bits
    let soffset = instruction1 & 0x7F;
    assert_eq!(soffset, 0b1111100);
}
