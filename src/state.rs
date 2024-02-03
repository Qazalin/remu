use std::collections::HashMap;
use std::ops::{Index, IndexMut};

pub trait Register {
    fn read64(&self, idx: usize) -> u64;
    fn write64(&mut self, idx: usize, addr: u64);
}

impl<T> Register for T
where
    T: Index<usize, Output = u32> + IndexMut<usize>,
{
    fn read64(&self, idx: usize) -> u64 {
        let addr_lsb = self[idx];
        let addr_msb = self[idx + 1];
        ((addr_msb as u64) << 32) | addr_lsb as u64
    }
    fn write64(&mut self, idx: usize, addr: u64) {
        self[idx] = (addr & 0xffffffff) as u32;
        self[idx + 1] = ((addr & (0xffffffff << 32)) >> 32) as u32;
    }
}

#[derive(Clone)]
pub struct VGPR(pub HashMap<usize, [u32; 256]>);
impl Index<usize> for VGPR {
    type Output = u32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0.get(&0).unwrap()[index]
    }
}

impl IndexMut<usize> for VGPR {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0.entry(0).or_insert([0; 256])[index]
    }
}

impl VGPR {
    pub fn new() -> Self {
        let mut map = HashMap::new();
        let vals = [0; 256];
        for key in 0..32 {
            map.insert(key, vals);
        }
        VGPR(map)
    }
    pub fn get_lane(&self, lane: usize) -> [u32; 256] {
        *self.0.get(&lane).unwrap()
    }
    pub fn get_lane_mut(&mut self, lane: usize) -> &mut [u32; 256] {
        self.0.get_mut(&lane).unwrap()
    }
}

pub trait Value {
    fn mut_hi16(&mut self, val: u16);
    fn mut_lo16(&mut self, val: u16);
}
impl Value for u32 {
    fn mut_hi16(&mut self, val: u16) {
        *self = ((val as u32) << 16) | (*self as u16 as u32);
    }
    fn mut_lo16(&mut self, val: u16) {
        *self = ((((*self & (0xffff << 16)) >> 16) as u32) << 16) | val as u32;
    }
}

#[derive(Debug)]
pub struct WaveValue {
    pub value: u32,
    pub lane_id: usize,
}
impl std::ops::Deref for WaveValue {
    type Target = bool;
    fn deref(&self) -> &Self::Target {
        match (self.value >> self.lane_id) & 1 == 1 {
            true => &true,
            false => &false,
        }
    }
}
impl WaveValue {
    pub fn mut_lane(&mut self, value: bool) {
        if value {
            self.value |= 1 << self.lane_id;
        } else {
            self.value &= !(1 << self.lane_id);
        }
    }
}

#[cfg(test)]
mod test_state {
    use super::*;

    #[test]
    fn test_wave_value() {
        let val = WaveValue {
            value: 0b11000000000000011111111111101110,
            lane_id: 0,
        };
        assert!(!*val);
        let val = WaveValue {
            value: 0b11000000000000011111111111101110,
            lane_id: 31,
        };
        assert!(*val);
    }

    #[test]
    fn test_write16() {
        let mut vgpr = VGPR::new();
        vgpr[0] = 0b11100000000000001111111111111111;
        vgpr[0].mut_lo16(0b1011101111111110);
        assert_eq!(vgpr[0], 0b11100000000000001011101111111110);
    }

    #[test]
    fn test_write16hi() {
        let mut vgpr = VGPR::new();
        vgpr[0] = 0b11100000000000001111111111111111;
        vgpr[0].mut_hi16(0b1011101111111110);
        assert_eq!(vgpr[0], 0b10111011111111101111111111111111);
    }
}
