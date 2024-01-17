#![allow(unused)]
use crate::utils::{Colorize, DebugLevel, DEBUG, SGPR_INDEX};
use std::ops::{Index, IndexMut};

pub struct RegisterGroup {
    pub values: Vec<u32>,
    name: &'static str,
    count: usize,
}
impl RegisterGroup {
    pub fn new(count: usize, name: &'static str) -> Self {
        Self {
            values: vec![0; count],
            name,
            count,
        }
    }
    pub fn read64(&self, idx: usize) -> u64 {
        let addr_lsb = self.values[idx];
        let addr_msb = self.values[idx + 1];
        ((addr_msb as u64) << 32) | addr_lsb as u64
    }
    pub fn write64(&mut self, idx: usize, addr: u64) {
        self.values[idx as usize] = (addr & 0xffffffff) as u32;
        self.values[idx as usize + 1] = ((addr & (0xffffffff << 32)) >> 32) as u32;
    }

    pub fn reset(&mut self) {
        self.values = vec![0; self.count];
    }
}

impl Index<usize> for RegisterGroup {
    type Output = u32;

    fn index(&self, index: usize) -> &Self::Output {
        if *DEBUG >= DebugLevel::STATE || Some(index as i32) == *SGPR_INDEX {
            println!("{} read {}", self.name.color("pink"), index);
        }
        &self.values[index]
    }
}

impl IndexMut<usize> for RegisterGroup {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(
            index <= self.count,
            "{} is greater than the possible register count {}",
            index,
            self.count
        );
        if *DEBUG >= DebugLevel::STATE || Some(index as i32) == *SGPR_INDEX {
            println!("{} write {}", self.name.color("pink"), index);
        }
        &mut self.values[index]
    }
}

#[derive(Debug)]
pub struct VCC {
    val: u32,
}

impl VCC {
    pub fn new() -> Self {
        Self { val: 0 }
    }
    pub fn assign(&mut self, val: u32) {
        self.val = val & 1;
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
        let mut vcc = Self::new();
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
