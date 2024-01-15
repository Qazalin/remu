#![allow(unused)]
use once_cell::sync::Lazy;
use serde_json::Value;
use std::collections::HashMap;
use std::io::{self, Read};
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::{env, fs, str};

const END_PRG: u32 = 0xbfb00000;
const WAIT_CNT_0: u32 = 0xBF89FC07;

#[derive(PartialEq, PartialOrd)]
pub enum DebugLevel {
    NONE,
    INSTRUCTION,
    MEMORY,
    STATE,
    MISC,
}

pub static DEBUG: Lazy<DebugLevel> = Lazy::new(|| {
    let var = env::var("REMU_DEBUG")
        .unwrap_or_default()
        .parse::<i32>()
        .unwrap_or(0);
    match var {
        0 => DebugLevel::NONE,
        1 => DebugLevel::INSTRUCTION,
        2 => DebugLevel::MEMORY,
        3 => DebugLevel::STATE,
        4 => DebugLevel::MISC,
        _ => panic!(),
    }
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
    parse_rdna3(&content).0
}

fn parse_rdna3(content: &str) -> (Vec<u32>, String) {
    if *DEBUG >= DebugLevel::MISC {
        println!(
            "{}",
            content
                .lines()
                .filter(|x| !x.contains("s_code_end"))
                .collect::<Vec<&str>>()
                .join("\n")
        );
    }
    let mut kernel = content.lines().skip(5);
    let name = kernel
        .nth(0)
        .unwrap()
        .split(" ")
        .nth(1)
        .unwrap()
        .replace(":", "")
        .replace("<", "")
        .replace(">", "");
    let instructions = kernel
        .map(|line| {
            line.split_whitespace()
                .filter(|p| u32::from_str_radix(p, 16).is_ok() && p.len() == 8)
                .collect::<Vec<&str>>()
        })
        .flatten()
        .map(|x| u32::from_str_radix(x, 16).unwrap())
        .collect::<Vec<u32>>();
    return (instructions, name.to_string());
}

pub fn split_asm_by_thread_syncs(instructions: &Vec<u32>) -> Vec<Vec<u32>> {
    let mut parts: Vec<Vec<u32>> = vec![];
    let mut last_idx = 0;
    instructions.iter().enumerate().for_each(|(i, x)| {
        if x == &WAIT_CNT_0 && instructions[i - 1] == WAIT_CNT_0 {
            let mut part = instructions[last_idx..=i - 2].to_vec();
            last_idx = i + 1;
            part.extend(vec![END_PRG]);
            parts.push(part);
        }
    });
    parts.push(instructions[last_idx..instructions.len()].to_vec());
    parts
}

pub fn as_signed(num: u64, bits: usize) -> i64 {
    let mut value = num;
    let is_negative = (value >> (bits - 1)) & 1 != 0;
    if is_negative {
        value |= !0 << bits;
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
        )
        .0;
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
    fn test_custom_signed_bits() {
        assert_eq!(as_signed(0b000000000000000101000, 21), 40);
        assert_eq!(as_signed(0b111111111111111011000, 21), -40);
        assert_eq!(as_signed(0b000000000000000000000, 21), 0);
        assert_eq!(as_signed(0b111111111111111111111, 21), -1);
        assert_eq!(as_signed(0b111000000000000000000, 21), -262144);
        assert_eq!(as_signed(0b000111111111111111111, 21), 262143);

        assert_eq!(as_signed(7608, 13), -584);
    }

    #[test]
    fn test_split_asm_by_thread_syncs() {
        let mut p1: Vec<u32> = vec![1, 2, 3, 4];
        let mut p2: Vec<u32> = vec![5, 6, WAIT_CNT_0, 4];
        let mut p3: Vec<u32> = vec![2, 4, 6];

        let instructions = vec![
            p1.clone(),
            vec![WAIT_CNT_0, WAIT_CNT_0],
            p2.clone(),
            vec![WAIT_CNT_0, WAIT_CNT_0],
            p3.clone(),
            vec![END_PRG],
        ]
        .concat();
        let parts = split_asm_by_thread_syncs(&instructions);

        p1.push(END_PRG);
        p2.push(END_PRG);
        p3.push(END_PRG);
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0], p1);
        assert_eq!(parts[1], p2);
        assert_eq!(parts[2], p3);
    }
}

pub trait Colorize {
    fn color(self, color: &str) -> String;
}

impl<'a> Colorize for &'a str {
    fn color(self, color: &str) -> String {
        let ansi_code = match color {
            "blue" => format!("\x1b[{};2;112;184;255m", 38),
            "jade" => format!("\x1b[{};2;39;176;139m", 38),
            "pink" => format!("\x1b[{};2;238;81;138m", 38),
            "yellow" => format!("\x1b[{};2;212;255;112m", 38),
            _ => format!("\x1b[{};2;255;255;255m", 38), // default white
        };
        format!("{}{}{}", ansi_code, self, "\x1b[0m")
    }
}

pub fn read_asm(lib: &Vec<u8>) -> (Vec<u32>, String) {
    if std::env::consts::OS == "macos" {
        let asm_map = read_json_to_hashmap("/tmp/asms.json").unwrap();
        let code = String::from_utf8(lib.to_vec()).unwrap();
        let asm = asm_map.get(&code).unwrap();
        let (base_rdna3, name) = parse_rdna3(&asm.to_string());
        let prg = match std::fs::metadata(format!("/tmp/{name}.s")) {
            Ok(_) => parse_rdna3(&fs::read_to_string(format!("/tmp/{name}.s")).unwrap()).0,
            Err(_) => base_rdna3,
        };
        return (prg, name);
    }
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

fn read_json_to_hashmap(file_path: &str) -> io::Result<HashMap<String, String>> {
    let mut file = fs::File::open(file_path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let json: Value = serde_json::from_str(&contents)?;
    if let Value::Object(map) = json {
        let hashmap: HashMap<String, String> = map
            .into_iter()
            .filter_map(|(k, v)| {
                if let Value::String(s) = v {
                    Some((k, s))
                } else {
                    None
                }
            })
            .collect();
        Ok(hashmap)
    } else {
        Err(io::Error::new(
            io::ErrorKind::Other,
            "Invalid JSON structure",
        ))
    }
}

#[macro_export]
macro_rules! todo_instr {
    ($x:expr) => {{
        let instr = format!("{:08X}", $x);
        use std::io::Write;
        use std::process::{Command, Stdio};
        Command::new("pbcopy")
            .stdin(Stdio::piped())
            .spawn()
            .and_then(|mut process| process.stdin.as_mut().unwrap().write_all(instr.as_bytes()))
            .unwrap();
        std::panic!("TODO instruction {instr}");
    }};
}
