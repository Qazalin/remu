#[derive(Clone, Debug)]
pub struct VecDataStore {
    pub data: Vec<u8>,
}

impl VecDataStore {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
    pub fn write(&mut self, addr: usize, val: u32) {
        if addr + 4 >= self.data.len() {
            self.data.resize(self.data.len() + addr + 5, 0);
        }
        self.data[addr..addr + 4]
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| {
                *x = val.to_le_bytes()[i];
            });
    }
    pub fn write64(&mut self, addr: usize, val: u64) {
        self.write(addr, (val & 0xffffffff) as u32);
        self.write(addr + 4, ((val & (0xffffffff << 32)) >> 32) as u32);
    }
    pub fn read(&self, addr: usize) -> u32 {
        let mut bytes: [u8; 4] = [0; 4];
        bytes.copy_from_slice(&self.data[addr + 0..addr + 4]);
        u32::from_le_bytes(bytes)
    }
    pub fn read64(&mut self, addr: usize) -> u64 {
        let lsb = self.read(addr);
        let msb = self.read(addr + 4);
        ((msb as u64) << 32) | lsb as u64
    }
}
