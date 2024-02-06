use crate::cpu::CPU;
use crate::memory::VecDataStore;
use crate::state::{Register, VecMutation, WaveValue, VGPR};
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
        let mut blocks = vec![];
        for z in 0..self.launch_bounds[2] {
            for y in 0..self.launch_bounds[1] {
                for x in 0..self.launch_bounds[0] {
                    blocks.push([x, y, z])
                }
            }
        }
        let waves = blocks.chunks(32).map(|w| w.to_vec()).collect::<Vec<_>>();

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
                self.exec_wave(wave, instructions)
            }
        }
    }

    fn exec_wave(&mut self, threads: &Vec<[u32; 3]>, instructions: &Vec<u32>) {
        for [x, y, z] in threads.iter() {
            let mut scalar_reg = vec![0; 256];
            let mut scc = 0;
            let mut vec_reg = VGPR::new();
            let mut vcc = WaveValue::new(0);
            let mut exec = WaveValue::new(1);
            vec_reg.default_lane = Some(0);
            vcc.default_lane = Some(0);
            exec.default_lane = Some(0);
            let mut sds = VecDataStore::new();
            match self.thread_state.get(&[*x, *y, *z]) {
                Some(val) => {
                    scalar_reg = val.0.clone();
                    vec_reg = val.1.clone();
                }
                None => {
                    scalar_reg.write64(0, self.kernel_args.as_ptr() as u64);

                    match self.dispatch_dim {
                        3 => {
                            (scalar_reg[13], scalar_reg[14], scalar_reg[15]) =
                                (self.id[0], self.id[1], self.id[2])
                        }
                        2 => (scalar_reg[14], scalar_reg[15]) = (self.id[0], self.id[1]),
                        _ => scalar_reg[15] = self.id[0],
                    }

                    match (self.launch_bounds[1] != 1, self.launch_bounds[2] != 1) {
                        (false, false) => vec_reg[0] = *x,
                        _ => vec_reg[0] = (z << 20) | (y << 10) | x,
                    }
                }
            }
            let mut cpu = CPU {
                scalar_reg: &mut scalar_reg,
                scc: &mut scc,
                vec_reg: &mut vec_reg,
                vcc: &mut vcc,
                exec_mask: &mut exec,
                lds: &mut self.lds,
                sds: &mut sds,
                pc: 0,
                prg: instructions.to_vec(),
                simm: None,
                vec_mutation: VecMutation::new(),
            };

            loop {
                let instruction = instructions[cpu.pc as usize];
                cpu.pc += 1;

                if instruction == crate::utils::END_PRG {
                    break;
                }
                if instruction == 0xbfb60003 || instruction >> 20 == 0xbf8 {
                    continue;
                }

                cpu.interpret(instruction);
                if let Some(val) = cpu.vec_mutation.vcc {
                    cpu.vcc.mut_lane(0, val);
                }
                if let Some(val) = cpu.vec_mutation.exec {
                    cpu.exec_mask.mut_lane(0, val);
                }
                if let Some(_) = cpu.vec_mutation.sgpr {
                    let (idx, val) = get_sgpr_carry_out(vec![cpu.vec_mutation]);
                    cpu.scalar_reg[idx] = val.value;
                }

                cpu.simm = None;
                cpu.vec_mutation = VecMutation::new();
            }

            self.thread_state
                .insert([*x, *y, *z], (scalar_reg, vec_reg));
        }
    }
}

pub fn get_sgpr_carry_out(lane_mutations: Vec<VecMutation>) -> (usize, WaveValue) {
    let mut carry_out = WaveValue::new(0);
    lane_mutations.iter().enumerate().for_each(|(lane_id, m)| {
        carry_out.mut_lane(lane_id, m.sgpr.unwrap().1);
    });
    (lane_mutations[0].sgpr.unwrap().0, carry_out)
}

#[cfg(test)]
mod test_workgroup {
    use super::*;

    #[test]
    fn test_get_sgpr_carry_out() {
        let results = [false, true, false, false, true]
            .iter()
            .map(|x| VecMutation {
                sgpr: Some((13, *x)),
                vcc: None,
                exec: None,
            })
            .collect::<Vec<_>>();
        let sgpr = get_sgpr_carry_out(results);
        assert_eq!(sgpr.0, 13);
        assert_eq!(sgpr.1.value, 0b10010);
    }
}
