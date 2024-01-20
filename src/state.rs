#![allow(unused)]
use crate::utils::{Colorize, DebugLevel, DEBUG, SGPR_INDEX};
use std::ops::{Index, IndexMut};

pub trait Register {
    fn read64(&self, idx: usize) -> u64;
    fn write64(&mut self, idx: usize, addr: u64);
    fn reset(&mut self);
}

impl<const N: usize> Register for [u32; N] {
    fn read64(&self, idx: usize) -> u64 {
        let addr_lsb = self[idx];
        let addr_msb = self[idx + 1];
        ((addr_msb as u64) << 32) | addr_lsb as u64
    }

    fn write64(&mut self, idx: usize, addr: u64) {
        self[idx] = (addr & 0xffffffff) as u32;
        self[idx + 1] = ((addr & (0xffffffff << 32)) >> 32) as u32;
    }

    fn reset(&mut self) {
        self.iter_mut().for_each(|x| *x = 0);
    }
}

#[derive(Debug)]
pub struct VCC {
    val: u32,
}

pub trait Assign<T> {
    fn assign(&mut self, val: T) {}
}
impl Assign<u32> for VCC {
    fn assign(&mut self, val: u32) {
        self.val = val & 1;
    }
}
impl Assign<bool> for VCC {
    fn assign(&mut self, val: bool) {
        self.val = val as u32;
    }
}

impl std::ops::Deref for VCC {
    type Target = u32;
    fn deref(&self) -> &Self::Target {
        &self.val
    }
}
impl From<u32> for VCC {
    fn from(val: u32) -> Self {
        let mut vcc = Self { val: 0 };
        vcc.assign(val);
        vcc
    }
}

#[cfg(test)]
mod test_state {
    use super::*;

    #[test]
    fn test_vcc() {
        let mut vcc = VCC { val: 0 };
        let val: i32 = -1;
        vcc.assign(val as u32);
        let result = 2 + 2 + *vcc;
        assert_eq!(result, 5);

        vcc.assign(0);
        let result = 2 + 2 + *vcc;
        assert_eq!(result, 4);

        vcc.assign(0b010);
        let result = 2 + 2 + *vcc;
        assert_eq!(result, 4);

        vcc.assign(0b1);
        let result = 2 + 2 + *vcc;
        assert_eq!(result, 5);

        let vcc = VCC::from(4);
        let result = 2 + 2 + *vcc;
        assert_eq!(result, 4);
    }
}
