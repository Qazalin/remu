use once_cell::sync::Lazy;
use std::{env, fs};

pub static DEBUG: Lazy<bool> = Lazy::new(|| env::var("DEBUG").unwrap_or_default() == "1");

pub fn parse_rdna3_file(file_path: &str) -> Vec<usize> {
    let content = fs::read_to_string(file_path).unwrap();
    parse_rdna3(&content)
}

fn parse_rdna3(content: &str) -> Vec<usize> {
    let mut kernel = content.lines().skip(5);
    let name = kernel.nth(0).unwrap();
    let instructions = kernel
        .map(|line| {
            line.split_whitespace()
                .filter(|p| usize::from_str_radix(p, 16).is_ok())
                .collect::<Vec<&str>>()
        })
        .flatten()
        .map(|x| usize::from_str_radix(x, 16).unwrap())
        .collect::<Vec<usize>>();

    if *DEBUG {
        println!("{}", name);
        let hex_formatted = instructions
            .iter()
            .map(|i| format!("0x{:08x}", i))
            .collect::<Vec<String>>();
        println!("{:?}", hex_formatted);
    }
    return instructions;
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_parse_rdna3() {
        let instructions = parse_rdna3(
            "
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_4>:
	s_clause 0x1                                               // 000000001600: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001604: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000160C: F4040000 F8000010
",
        );
        assert_eq!(instructions.len(), 5);
        let hexed = instructions
            .iter()
            .map(|i| format!("0x{:08x}", i))
            .collect::<Vec<String>>();
        assert_eq!(
            hexed,
            [
                "0xbf850001",
                "0xf4080100",
                "0xf8000000",
                "0xf4040000",
                "0xf8000010"
            ]
        );
    }
}
