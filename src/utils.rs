#![allow(unused)]
use once_cell::sync::Lazy;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::{env, fs, str};

pub static DEBUG: Lazy<i32> = Lazy::new(|| {
    env::var("DEBUG")
        .unwrap_or_default()
        .parse::<i32>()
        .unwrap_or(0)
});

pub static SGPR_INDEX: Lazy<Option<i32>> = Lazy::new(|| {
    let var = env::var("SGPR_INDEX");
    if var.is_ok() {
        return Some(var.unwrap().parse::<i32>().unwrap());
    }
    return None;
});
pub fn parse_rdna3_file(file_path: &str) -> Vec<u32> {
    let content = fs::read_to_string(file_path).unwrap();
    parse_rdna3(&content)
}

fn parse_rdna3(content: &str) -> Vec<u32> {
    let mut kernel = content.lines().skip(5);
    let _name = kernel.nth(0).unwrap();
    let instructions = kernel
        .map(|line| {
            line.split_whitespace()
                .filter(|p| u32::from_str_radix(p, 16).is_ok() && p.len() == 8)
                .collect::<Vec<&str>>()
        })
        .flatten()
        .map(|x| u32::from_str_radix(x, 16).unwrap())
        .collect::<Vec<u32>>();
    return instructions;
}

/** We use this for the smem immediate 21-bit constant OFFSET */
pub fn twos_complement_21bit(num: u64) -> i64 {
    let mut value = num;
    let is_negative = (value >> 20) & 1 != 0;
    if is_negative {
        value |= !0 << 21;
    }
    value as i64
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
	s_load_b64 s[0:1], s[0:1], null                            // 000000001600: F4040000 F8000000
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, 4               // 000000001608: CA100080 00000084
",
        );
        assert_eq!(instructions.len(), 4);
        let hexed = instructions
            .iter()
            .map(|i| format!("0x{:08x}", i))
            .collect::<Vec<String>>();
        assert_eq!(
            hexed,
            ["0xf4040000", "0xf8000000", "0xca100080", "0x00000084",]
        );
    }

    #[test]
    fn test_twos_complement_21bit() {
        assert_eq!(twos_complement_21bit(0b000000000000000101000), 40);
        assert_eq!(twos_complement_21bit(0b111111111111111011000), -40);
        assert_eq!(twos_complement_21bit(0b000000000000000000000), 0);
        assert_eq!(twos_complement_21bit(0b111111111111111111111), -1);
        assert_eq!(twos_complement_21bit(0b111000000000000000000), -262144);
        assert_eq!(twos_complement_21bit(0b000111111111111111111), 262143);
    }
}

pub trait Colorize {
    fn color(self, color: &str) -> String;
}

impl<'a> Colorize for &'a str {
    fn color(self, color: &str) -> String {
        let intensity = if color == color.to_uppercase() { 60 } else { 0 };
        let color_code = match color.to_lowercase().as_str() {
            "black" => 0,
            "red" => 1,
            "green" => 2,
            "yellow" => 3,
            "blue" => 4,
            "magenta" => 5,
            "cyan" => 6,
            _ => 7, // white
        };
        format!("\x1b[{}m{}\x1b[0m", 0 + intensity + 30 + color_code, self)
    }
}

pub fn read_asm(lib: &Vec<u8>) -> Vec<u32> {
    let mut child = Command::new("/opt/rocm/llvm/bin/llvm-objdump")
        .args(&["-d", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to execute command");

    {
        let stdin = child.stdin.as_mut().expect("Failed to open stdin");
        stdin.write_all(lib).expect("Failed to write to stdin");
    }

    let output = child.wait_with_output().expect("Failed to read stdout");
    if output.status.success() {
        let asm = String::from_utf8_lossy(&output.stdout);
        parse_rdna3(&asm.to_string())
    } else {
        let err = String::from_utf8_lossy(&output.stderr);
        panic!("Command failed with error: {}", err);
    }
}
