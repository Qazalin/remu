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
}
