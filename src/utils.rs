use once_cell::sync::Lazy;
use std::{env, fs};

pub static DEBUG: Lazy<bool> = Lazy::new(|| env::var("DEBUG").unwrap_or_default() == "1");

pub fn parse_rdna3_file(file_path: &str) -> Vec<usize> {
    let content = fs::read_to_string(file_path).unwrap();
    let mut kernel = content.lines().skip(5);
    let name = kernel.nth(0).unwrap();
    let instructions = kernel
        .map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            let code_str = parts.last().unwrap().trim_start_matches("0x");
            let code = usize::from_str_radix(code_str, 16).expect("Failed to parse the opcode");
            return code;
        })
        .collect::<Vec<usize>>();

    if *DEBUG {
        println!("{} {}", file_path, name);
        let hex_formatted = instructions
            .iter()
            .map(|i| format!("0x{:08x}", i))
            .collect::<Vec<String>>();
        println!("{:?}", hex_formatted);
    }
    return instructions;
}
