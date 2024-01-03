use crate::dtype::DType;
use core::mem;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct BumpAllocator {
    pub last: u64,
    pub memory: Vec<u8>,
}

impl BumpAllocator {
    pub fn new() -> Self {
        if std::path::Path::new("/tmp/wave.bin").exists() {
            let enc = std::fs::read("/tmp/wave.bin").unwrap();
            let decoded: Self = bincode::deserialize(&enc[..]).unwrap();
            return decoded;
        }
        Self {
            last: 0,
            memory: vec![0; 24 * 1_073],
        }
    }

    pub fn alloc(&mut self, size: u32) -> u64 {
        let last_allocated = self.last;
        self.last += size as u64;
        return last_allocated + size as u64;
    }

    pub fn copyin(&mut self, addr: u64, data: &[u8]) {
        self.memory[addr as usize..addr as usize + data.len()].copy_from_slice(data);
    }

    /** "persist" memory and locations for the next library call */
    pub fn save(&self) {
        let enc = bincode::serialize(&self).unwrap();
        std::fs::write("/tmp/wave.bin", &enc[..]).unwrap();
    }

    pub fn read<D: DType>(&self, addr: u64) -> D {
        assert!(addr as usize + mem::size_of::<D>() <= self.memory.len());
        unsafe {
            let ptr = self.memory.as_ptr().offset(addr as isize) as *const D;
            *ptr
        }
    }

    pub fn write<D: DType>(&mut self, addr: u64, val: D) {
        assert!(addr as usize + mem::size_of::<D>() <= self.memory.len());
        unsafe {
            let ptr = self.memory.as_mut_ptr().offset(addr as isize) as *mut D;
            *ptr = val;
        }
    }
}

#[cfg(test)]
mod test_allocation {
    use super::*;
    #[test]
    fn test_bump_allocator() {
        // remove tmp files
        std::fs::remove_file("/tmp/wave.bin").unwrap_or(());
        let mut allocator = BumpAllocator::new();
        allocator.alloc(4);
        assert_eq!(allocator.last, 4);
        allocator.alloc(9);
        assert_eq!(allocator.last, 13);
    }

    #[test]
    fn test_persistant() {
        // remove tmp files
        std::fs::remove_file("/tmp/wave.bin").unwrap_or(());
        let mut allocator = BumpAllocator::new();
        allocator.alloc(4);
        allocator.save();
        let allocator = BumpAllocator::new();
        assert_eq!(allocator.last, 4);
    }

    #[test]
    fn test_copyin() {
        // remove tmp files
        std::fs::remove_file("/tmp/wave.bin").unwrap_or(());
        let mut allocator = BumpAllocator::new();
        allocator.alloc(4);
        allocator.copyin(0, &[0x01, 0x02, 0x03, 0x04]);
        assert_eq!(allocator.memory[0], 0x01);
        assert_eq!(allocator.memory[1], 0x02);
        assert_eq!(allocator.memory[2], 0x03);
        assert_eq!(allocator.memory[3], 0x04);
    }
}

mod test_dtype {
    use super::*;

    fn helper_test_mem<D: DType>(val: D) -> BumpAllocator {
        let mut memory = BumpAllocator::new();
        memory.write(0, val);
        return memory;
    }

    #[test]
    fn test_u8() {
        let val: u8 = 10;
        let memory = helper_test_mem(val);
        assert_eq!(memory.read::<u8>(0), val)
    }

    #[test]
    fn test_u16() {
        let val: u16 = 30000;
        let memory = helper_test_mem(val);
        assert_eq!(memory.read::<u16>(0), val);
        assert_eq!(memory.memory.get(0..2).unwrap(), vec![48, 117]);
    }

    #[test]
    fn test_u32() {
        let val: u32 = 1234567890;
        let memory = helper_test_mem(val);
        assert_eq!(memory.read::<u32>(0), val);
        assert_eq!(memory.memory.get(0..4).unwrap(), vec![210, 2, 150, 73]);
    }
}
