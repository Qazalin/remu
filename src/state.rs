use crate::utils::{GlobalCounter, GLOBAL_COUNTER};
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
pub struct VGPR {
    values: HashMap<usize, [u32; 256]>,
    pub default_lane: Option<usize>,
}
impl Index<usize> for VGPR {
    type Output = u32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.values.get(&self.default_lane.unwrap()).unwrap()[index]
    }
}
impl IndexMut<usize> for VGPR {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        GLOBAL_COUNTER.lock().unwrap().vgpr_used += 1;
        &mut self
            .values
            .entry(self.default_lane.unwrap())
            .or_insert([0; 256])[index]
    }
}
impl VGPR {
    pub fn new() -> Self {
        let mut values = HashMap::new();
        let vals = [0; 256];
        for key in 0..32 {
            values.insert(key, vals);
        }
        VGPR {
            values,
            default_lane: None,
        }
    }
    pub fn get_lane(&self, lane: usize) -> [u32; 256] {
        *self.values.get(&lane).unwrap()
    }
    pub fn get_lane_mut(&mut self, lane: usize) -> &mut [u32; 256] {
        self.values.get_mut(&lane).unwrap()
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

#[derive(Debug, Clone, Copy)]
pub struct WaveValue {
    pub value: u32,
    pub default_lane: Option<usize>,
    pub mutations: Option<[bool; 32]>,
}
impl WaveValue {
    pub fn new(value: u32) -> Self {
        Self {
            value,
            default_lane: None,
            mutations: None,
        }
    }
    pub fn read(&self) -> bool {
        (self.value >> self.default_lane.unwrap()) & 1 == 1
    }
    pub fn set_lane(&mut self, value: bool) {
        if self.mutations.is_none() {
            self.mutations = Some([false; 32])
        }
        self.mutations.as_mut().unwrap()[self.default_lane.unwrap()] = value;
    }
    pub fn apply_muts(&mut self) {
        self.value = 0;
        for lane in 0..32 {
            if self.mutations.unwrap()[lane] {
                self.value |= 1 << lane;
            }
        }
    }
}

#[cfg(test)]
mod test_state {
    use super::*;

    #[test]
    fn test_wave_value() {
        let mut val = WaveValue::new(0b11000000000000011111111111101110);
        val.default_lane = Some(0);
        assert!(!val.read());
        val.default_lane = Some(31);
        assert!(val.read());
    }

    #[test]
    fn test_wave_value_mutations() {
        let mut val = WaveValue::new(0b10001);
        val.default_lane = Some(0);
        val.set_lane(false);
        assert!(val.mutations.unwrap().iter().all(|x| !x));
        val.default_lane = Some(1);
        val.set_lane(true);
        assert_eq!(val.value, 0b10001);
        assert_eq!(
            val.mutations,
            Some([
                false, true, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false,
            ])
        );

        val.apply_muts();
        assert_eq!(val.value, 0b10);
    }

    #[test]
    fn test_write16() {
        let mut vgpr = VGPR::new();
        vgpr.default_lane = Some(0);
        vgpr[0] = 0b11100000000000001111111111111111;
        vgpr[0].mut_lo16(0b1011101111111110);
        assert_eq!(vgpr[0], 0b11100000000000001011101111111110);
    }

    #[test]
    fn test_write16hi() {
        let mut vgpr = VGPR::new();
        vgpr.default_lane = Some(0);
        vgpr[0] = 0b11100000000000001111111111111111;
        vgpr[0].mut_hi16(0b1011101111111110);
        assert_eq!(vgpr[0], 0b10111011111111101111111111111111);
    }

    #[test]
    fn test_vgpr() {
        let mut vgpr = VGPR::new();
        vgpr.default_lane = Some(0);
        vgpr[0] = 42;
        vgpr.default_lane = Some(10);
        vgpr[0] = 10;
        assert_eq!(vgpr.get_lane(0)[0], 42);
        assert_eq!(vgpr.get_lane(10)[0], 10);
    }
}
