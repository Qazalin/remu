use crate::utils::DEBUG;
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
}

impl Index<usize> for SGPR {
    type Output = u32;

    fn index(&self, index: usize) -> &Self::Output {
        if *DEBUG == 3 {
            println!("[SGPR] read {index}");
        }
        &self.values[index]
    }
}

impl IndexMut<usize> for SGPR {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if *DEBUG == 3 {
            println!("[SGPR] write {index}");
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