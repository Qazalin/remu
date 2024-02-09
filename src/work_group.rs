use crate::memory::VecDataStore;
use crate::state::{Register, VecMutation, WaveValue, VGPR};
use crate::thread::Thread;
use crate::utils::{Colorize, DEBUG};
use std::collections::HashMap;
use std::sync::atomic::Ordering::SeqCst;

pub struct WorkGroup<'a> {
    dispatch_dim: u32,
    id: [u32; 3],
    lds: VecDataStore,
    kernel: &'a Vec<u32>,
    kernel_args: &'a Vec<u64>,
    launch_bounds: [u32; 3],
    wave_state: HashMap<usize, (Vec<u32>, VGPR)>,
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
            wave_state: HashMap::new(),
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
            for wave in waves.iter().enumerate() {
                self.exec_wave(wave, instructions, barriers.len() > 1)
            }
        }
    }

    fn exec_wave(
        &mut self,
        (wave_id, threads): (usize, &Vec<[u32; 3]>),
        instructions: &Vec<u32>,
        save_state: bool,
    ) {
        let mut scalar_reg = match self.wave_state.get(&wave_id) {
            Some(val) => val.0.to_vec(),
            None => {
                let mut scalar_reg = vec![0; 256];
                scalar_reg.write64(0, self.kernel_args.as_ptr() as u64);
                let [gx, gy, gz] = self.id;
                match self.dispatch_dim {
                    3 => (scalar_reg[13], scalar_reg[14], scalar_reg[15]) = (gx, gy, gz),
                    2 => (scalar_reg[14], scalar_reg[15]) = (gx, gy),
                    _ => scalar_reg[15] = gx,
                }
                scalar_reg
            }
        };
        let mut scc = 0;
        let mut vec_reg = match self.wave_state.get(&wave_id) {
            Some(val) => val.1.clone(),
            _ => VGPR::new(),
        };
        let mut vcc = WaveValue::new(0);
        let mut exec = WaveValue::new(u32::MAX);

        let mut seeded_lanes = vec![];
        let mut pc = 0;
        loop {
            if instructions[pc] == crate::utils::END_PRG {
                if save_state {
                    self.wave_state
                        .insert(wave_id, (scalar_reg, vec_reg.clone()));
                }
                break;
            }
            if instructions[pc] == 0xbfb60003 || instructions[pc] >> 20 == 0xbf8 {
                pc += 1;
                continue;
            }

            let mut vec_mutations = vec![];
            for (lane_id, [x, y, z]) in threads.iter().enumerate() {
                vec_reg.default_lane = Some(lane_id);
                vcc.default_lane = Some(lane_id);
                exec.default_lane = Some(lane_id);
                if DEBUG.load(SeqCst) {
                    let lane = format!("{lane_id} {:08X} ", instructions[pc]);
                    let state = match exec.read() {
                        true => "green",
                        false => "gray",
                    };
                    print!("{:?} {:?} {}", self.id, [x, y, z], lane.color(state));
                }
                if !seeded_lanes.contains(&lane_id) && self.wave_state.get(&wave_id).is_none() {
                    match (self.launch_bounds[1] != 1, self.launch_bounds[2] != 1) {
                        (false, false) => vec_reg[0] = *x,
                        _ => vec_reg[0] = (z << 20) | (y << 10) | x,
                    }
                    seeded_lanes.push(lane_id);
                }
                let mut sds = VecDataStore::new();
                let mut thread = Thread {
                    scalar_reg: &mut scalar_reg,
                    scc: &mut scc,
                    vec_reg: &mut vec_reg,
                    vcc: &mut vcc,
                    exec: &mut exec,
                    lds: &mut self.lds,
                    sds: &mut sds,
                    pc_offset: 0,
                    stream: instructions[pc..instructions.len()].to_vec(),
                    scalar: false,
                    simm: None,
                    vec_mutation: VecMutation::new(),
                };
                thread.interpret();
                vec_mutations.push(thread.vec_mutation);
                if thread.scalar {
                    pc = ((pc as isize) + 1 + (thread.pc_offset as isize)) as usize;
                    break;
                }
                if lane_id == threads.len() - 1 {
                    pc = ((pc as isize) + 1 + (thread.pc_offset as isize)) as usize;
                }
            }

            if vec_mutations[0].vcc.is_some() {
                vcc.value = 0;
            }
            if vec_mutations[0].exec.is_some() {
                exec.value = 0;
            }
            vec_mutations.iter().enumerate().for_each(|(lane_id, m)| {
                if let Some(val) = m.vcc {
                    vcc.mut_lane(lane_id, val);
                }
                if let Some(val) = m.exec {
                    exec.mut_lane(lane_id, val);
                }
            });
            if vec_mutations[0].sgpr.is_some() {
                let (idx, val) = get_sgpr_carry_out(vec_mutations);
                scalar_reg[idx] = val.value;
            }
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
