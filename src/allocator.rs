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
}

#[cfg(test)]
mod test {
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
