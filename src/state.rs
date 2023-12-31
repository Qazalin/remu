use crate::utils::{Colorize, DEBUG, SGPR_INDEX};
use std::ops::{Index, IndexMut};

const SGPR_COUNT: usize = 105;
pub struct SGPR {
    values: [u32; SGPR_COUNT],
}
impl SGPR {
    pub fn new() -> Self {
        Self {
            values: [0; SGPR_COUNT],
        }
    }
    /** read a 64bit memory address from two 32bit registers */
    pub fn read_addr(&self, idx: usize) -> u64 {
        let addr_lsb = self.values[idx];
        let addr_msb = self.values[idx + 1];
        ((addr_msb as u64) << 32) | addr_lsb as u64
    }
    /** write a 64bit memory address to two 32bit registers */
    pub fn write_addr(&mut self, idx: usize, addr: u64) {
        self.values[idx as usize] = (addr & 0xffffffff) as u32;
        self.values[idx as usize + 1] = ((addr & (0xffffffff << 32)) >> 32) as u32;
    }

    pub fn reset(&mut self) {
        self.values = [0; SGPR_COUNT]
    }
}

impl Index<usize> for SGPR {
    type Output = u32;

    fn index(&self, index: usize) -> &Self::Output {
        if *DEBUG >= 3 || Some(index as i32) == *SGPR_INDEX {
            println!("{} read {}", "[SGPR]".color("magenta"), index);
        }
        &self.values[index]
    }
}

impl IndexMut<usize> for SGPR {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if *DEBUG >= 3 || Some(index as i32) == *SGPR_INDEX {
            println!("{} write {}", "[SGPR]".color("magenta"), index);
        }
        &mut self.values[index]
    }
}

const VGPR_COUNT: usize = 256;
pub struct VGPR {
    values: [u32; VGPR_COUNT],
}
impl VGPR {
    pub fn new() -> Self {
        Self {
            values: [0; VGPR_COUNT],
        }
    }
    // TODO this is copied from SGPR
    /** read a 64bit memory address from two 32bit registers */
    pub fn read_addr(&self, idx: usize) -> u64 {
        let addr_lsb = self.values[idx];
        let addr_msb = self.values[idx + 1];
        ((addr_msb as u64) << 32) | addr_lsb as u64
    }
    /** write a 64bit memory address to two 32bit registers */
    pub fn write_addr(&mut self, idx: usize, addr: u64) {
        self.values[idx as usize] = (addr & 0xffffffff) as u32;
        self.values[idx as usize + 1] = ((addr & (0xffffffff << 32)) >> 32) as u32;
    }
}

impl Index<usize> for VGPR {
    type Output = u32;

    fn index(&self, index: usize) -> &Self::Output {
        if *DEBUG == 3 {
            println!("[VGPR] read {index}");
        }
        &self.values[index]
    }
}

impl IndexMut<usize> for VGPR {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if *DEBUG == 3 {
            println!("[VGPR] write {index}");
        }
        &mut self.values[index]
    }
}
