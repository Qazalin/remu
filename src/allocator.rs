pub struct BumpAllocator {
    last: u64,
    pub memory: Vec<u8>,
}

impl BumpAllocator {
    pub fn new() -> Self {
        Self {
            last: 0,
            memory: vec![0; 24 * 1_073_741_824],
        }
    }

    pub fn alloc(&mut self, size: u32) -> u64 {
        let last_allocated = self.last;
        self.last += size as u64;
        return last_allocated + size as u64;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_bump_allocator() {
        let mut allocator = BumpAllocator::new();
        allocator.alloc(4);
        assert_eq!(allocator.last, 4);
        allocator.alloc(9);
        assert_eq!(allocator.last, 13);
    }
}
