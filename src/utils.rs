use half::f16;
use lazy_static::lazy_static;
use std::sync::Mutex;
use std::{env, str};

pub const END_PRG: u32 = 0xbfb00000;
lazy_static::lazy_static! {
    pub static ref CI: bool = env::var("CI").map(|v| v == "1").unwrap_or(false);
    pub static ref PROFILE: bool = env::var("PROFILE").map(|v| v == "1").unwrap_or(false);
    pub static ref GLOBAL_DEBUG: bool = env::var("DEBUG").map(|v| v == "1").unwrap_or(false);
}

pub fn nth(val: u32, pos: usize) -> u32 {
    return (val >> (31 - pos as u32)) & 1;
}
pub fn f16_lo(val: u32) -> f16 {
    f16::from_bits((val & 0xffff) as u16)
}
pub fn f16_hi(val: u32) -> f16 {
    f16::from_bits(((val >> 16) & 0xffff) as u16)
}

pub fn sign_ext(num: u64, bits: usize) -> i64 {
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
    fn test_custom_signed_bits() {
        assert_eq!(sign_ext(0b000000000000000101000, 21), 40);
        assert_eq!(sign_ext(0b111111111111111011000, 21), -40);
        assert_eq!(sign_ext(0b000000000000000000000, 21), 0);
        assert_eq!(sign_ext(0b111111111111111111111, 21), -1);
        assert_eq!(sign_ext(0b111000000000000000000, 21), -262144);
        assert_eq!(sign_ext(0b000111111111111111111, 21), 262143);

        assert_eq!(sign_ext(7608, 13), -584);
    }
}

pub trait Colorize {
    fn color(self, color: &str) -> String;
}
impl<'a> Colorize for &'a str {
    fn color(self, color: &str) -> String {
        let ansi_code = match color {
            "blue" => format!("\x1b[{};2;112;184;255m", 38),
            "green" => format!("\x1b[{};2;39;176;139m", 38),
            "gray" => format!("\x1b[{};2;169;169;169m", 38),
            _ => format!("\x1b[{};2;255;255;255m", 38),
        };
        format!("{}{}{}", ansi_code, self, "\x1b[0m")
    }
}

#[macro_export]
macro_rules! todo_instr {
    ($x:expr) => {{
        if std::env::var("REMU_DEBUG")
            .map(|v| v == "4")
            .unwrap_or(false)
        {
            panic!("{:08X}", $x)
        }
        Err(1)
    }};
}

#[derive(Debug)]
pub struct GlobalCounter {
    pub vgpr_used: usize,
    pub gds_ops: usize,
    pub lds_ops: usize,
    pub wmma: usize,
    pub wave_syncs: usize,
}
lazy_static! {
    pub static ref GLOBAL_COUNTER: Mutex<GlobalCounter> = Mutex::new(GlobalCounter {
        vgpr_used: 0,
        gds_ops: 0,
        lds_ops: 0,
        wmma: 0,
        wave_syncs: 0
    });
}
