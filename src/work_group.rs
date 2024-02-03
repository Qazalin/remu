use crate::cpu::CPU;
use crate::memory::VecDataStore;
use crate::state::{Register, VGPR};
use std::collections::HashMap;

pub struct WorkGroup<'a> {
    dispatch_dim: u32,
    id: [u32; 3],
    lds: VecDataStore,
    kernel: &'a Vec<u32>,
    kernel_args: &'a Vec<u64>,
    launch_bounds: [u32; 3],
    thread_state: HashMap<[u32; 3], (Vec<u32>, VGPR)>,
}

impl<'a> WorkGroup<'a> {
    pub fn new(
        dispatch_dim: u32,
        id: [u32; 3],
        launch_bounds: [u32; 3],
        kernel: &'a Vec<u32>,
        kernel_args: &'a Vec<u64>,
    ) -> Self {
        return Self {
            dispatch_dim,
            id,
            kernel,
            launch_bounds,
            kernel_args,
            lds: VecDataStore::new(),
            thread_state: HashMap::new(),
        };
    }

    pub fn exec_waves(&mut self) {
        let mut waves = vec![];
        let mut start = 0;
        while start < self.launch_bounds[0] {
            let end = std::cmp::min(start + 32, self.launch_bounds[0]);
            waves.push((start, end));
            start = end;
        }

        let mut barriers = vec![];
        let mut last_idx = 0;
        const WAIT_CNT_0: u32 = 0xBF89FC07;
        self.kernel.iter().enumerate().for_each(|(i, x)| {
            if (*x == WAIT_CNT_0 && self.kernel[i - 1] == WAIT_CNT_0)
                || (*x == WAIT_CNT_0 && self.kernel[i - 1] == 0xBFBD0000 && *x == WAIT_CNT_0)
            {
                let mut part = self.kernel[last_idx..=i - 2].to_vec();
                last_idx = i + 1;
                part.extend(vec![0xbfb00000]);
                barriers.push(part);
            }
        });
        barriers.push(self.kernel[last_idx..self.kernel.len()].to_vec());

        for instructions in barriers.iter() {
            for wave in waves.iter() {
                for x in wave.0..wave.1 {
                    self.exec_wave(x, instructions)
                }
            }
        }
    }

    fn exec_wave(&mut self, x: u32, instructions: &Vec<u32>) {
        for y in 0..self.launch_bounds[1] {
            for z in 0..self.launch_bounds[2] {
                let mut cpu = CPU::new(&mut self.lds);
                match self.thread_state.get(&[x, y, z]) {
                    Some(val) => {
                        cpu.scalar_reg = val.0.clone();
                        cpu.vec_reg = val.1.clone();
                    }
                    None => {
                        cpu.scalar_reg.write64(0, self.kernel_args.as_ptr() as u64);

                        match self.dispatch_dim {
                            3 => {
                                (cpu.scalar_reg[13], cpu.scalar_reg[14], cpu.scalar_reg[15]) =
                                    (self.id[0], self.id[1], self.id[2])
                            }
                            2 => {
                                (cpu.scalar_reg[14], cpu.scalar_reg[15]) = (self.id[0], self.id[1])
                            }
                            _ => cpu.scalar_reg[15] = self.id[0],
                        }

                        match (self.launch_bounds[1] != 1, self.launch_bounds[2] != 1) {
                            (false, false) => cpu.vec_reg[0] = x,
                            _ => cpu.vec_reg[0] = (z << 20) | (y << 10) | x,
                        }
                    }
                }
                cpu.interpret(instructions);
                self.thread_state
                    .insert([x, y, z], (cpu.scalar_reg, cpu.vec_reg));
            }
        }
    }
}
