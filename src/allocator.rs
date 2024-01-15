use crate::dtype::DType;
use crate::utils::{Colorize, DebugLevel, DEBUG};
use core::mem;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

const MAX_MEM_SIZE: usize = 1_000_000_000;
pub struct BumpAllocator {
    fp: String,
}

impl BumpAllocator {
    pub fn new(wave_id: &str) -> Self {
        let fp = format!("/tmp/{}.bin", wave_id);
        if !PathBuf::from(&fp).exists() {
            File::create(&fp).unwrap();
        }
        Self { fp }
    }

    pub fn alloc(&mut self, size: u32) -> u64 {
        let last = self.len();
        self.write_bytes(last as u64, &vec![0; size as usize]);
        return last as u64;
    }

    pub fn len(&self) -> usize {
        let file = File::open(&self.fp).unwrap();
        file.metadata().unwrap().len() as usize
    }

    pub fn read<D: DType>(&self, addr: u64) -> D {
        let bytes = self.read_bytes(addr, mem::size_of::<D>());
        unsafe {
            let ptr = bytes.as_ptr() as *const D;
            if *DEBUG >= DebugLevel::MEMORY {
                println!("{} {} {:?}", "READ".color("yellow"), addr, *ptr);
            }
            *ptr
        }
    }

    pub fn write<D: DType>(&mut self, addr: u64, val: D) {
        if *DEBUG >= DebugLevel::MEMORY {
            println!("{} {} {:?}", "WRITE".color("yellow"), addr, val);
        }
        let mut bytes = vec![0; mem::size_of::<D>()];
        unsafe {
            let ptr = bytes.as_mut_ptr() as *mut D;
            *ptr = val;
        }
        self.write_bytes(addr, &bytes)
    }

    pub fn write_bytes(&mut self, addr: u64, bytes: &[u8]) {
        assert!(self.len() + bytes.len() <= MAX_MEM_SIZE);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true) // NOTE: we dont use the builtin `append` mode since the API uses alloc
            .open(&self.fp)
            .unwrap();
        file.seek(SeekFrom::Start(addr)).unwrap();
        file.write_all(bytes).unwrap();
    }

    pub fn read_bytes(&self, addr: u64, sz: usize) -> Vec<u8> {
        assert!(sz <= MAX_MEM_SIZE);
        let mut file = File::open(&self.fp).unwrap();
        let mut bytes = vec![0; sz];

        file.seek(SeekFrom::Start(addr)).unwrap();
        file.read_exact(&mut bytes).unwrap();
        return bytes;
    }
}

#[cfg(test)]
mod test_allocation {
    use super::*;

    fn helper_test_fresh_allocator(wave_id: &str) -> BumpAllocator {
        std::fs::remove_file(format!("/tmp/{}.bin", wave_id)).unwrap_or(());
        BumpAllocator::new(wave_id)
    }

    #[test]
    fn test_bump_allocator() {
        let mut allocator = helper_test_fresh_allocator("test_bump_allocator");
        let addr = allocator.alloc(4);
        assert_eq!(addr, 0);
        let addr = allocator.alloc(9);
        assert_eq!(addr, 4);
    }

    #[test]
    fn test_persistant() {
        let mut allocator = helper_test_fresh_allocator("test_persistant");
        let addr = allocator.alloc(4);
        assert_eq!(addr, 0);
        let mut allocator = BumpAllocator::new("test_persistant");
        let addr = allocator.alloc(4);
        assert_eq!(addr, 4);
    }

    #[test]
    fn test_write_bytes() {
        let mut allocator = helper_test_fresh_allocator("test_write_bytes");
        let addr = allocator.alloc(4);
        assert_eq!(allocator.read_bytes(0, 4), [0, 0, 0, 0]);
        allocator.write_bytes(addr, &[0x01, 0x02, 0x03, 0x04]);
        assert_eq!(allocator.read_bytes(0, 4), [0x01, 0x02, 0x03, 0x04]);
    }
}

#[cfg(test)]
mod test_dtype {
    use super::*;

    #[test]
    fn test_u8() {
        /* dtypes tests */
        let mut memory = BumpAllocator::new("test_dtype_u8");
        let val: u8 = 10;
        memory.write(0, val);
        assert_eq!(memory.read::<u8>(0), val);
    }

    #[test]
    fn test_u16() {
        let mut memory = BumpAllocator::new("test_dtype_u16");
        let val: u16 = 30000;
        memory.write(0, val);
        assert_eq!(memory.read::<u16>(0), val);
        assert_eq!(memory.read_bytes(0, 2), vec![48, 117]);
    }

    #[test]
    fn test_u32() {
        let mut memory = BumpAllocator::new("test_dtype_u32");
        let val: u32 = 1234567890;
        memory.write(0, val);
        assert_eq!(memory.read::<u32>(0), val);
        assert_eq!(memory.read_bytes(0, 4), vec![210, 2, 150, 73]);
    }
}
