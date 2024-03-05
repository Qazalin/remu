use crate::memory::VecDataStore;
use crate::state::{Register, WaveValue, VGPR};
use crate::thread::Thread;
use crate::utils::{Colorize, CI, DEBUG, END_PRG, GLOBAL_COUNTER};
use std::collections::HashMap;
use std::sync::atomic::Ordering::SeqCst;

struct WaveState(
    Vec<u32>,
    u32,
    VGPR,
    WaveValue,
    WaveValue,
    usize,
    HashMap<usize, VecDataStore>,
);
pub struct WorkGroup<'a> {
    dispatch_dim: u32,
    id: [u32; 3],
    lds: VecDataStore,
    kernel: &'a Vec<u32>,
    kernel_args: &'a Vec<u64>,
    launch_bounds: [u32; 3],
    wave_state: HashMap<usize, WaveState>,
}

const SYNCS: [u32; 5] = [0xBF89FC07, 0xBFBD0000, 0xBC7C0000, 0xBF890007, 0xbFB60003];
const BARRIERS: [[u32; 2]; 4] = [
    [SYNCS[0], SYNCS[0]],
    [SYNCS[0], SYNCS[1]],
    [SYNCS[0], SYNCS[2]],
    [SYNCS[3], SYNCS[1]],
];
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

        let mut syncs = 0;
        self.kernel.iter().enumerate().for_each(|(i, x)| {
            if i != 0 && BARRIERS.contains(&[*x, self.kernel[i - 1]]) {
                syncs += 1;
                GLOBAL_COUNTER.lock().unwrap().wave_syncs += 1;
            }
        });
        assert!(syncs <= 1);
        for _ in 0..=syncs {
            waves.iter().enumerate().for_each(|w| self.exec_wave(w))
        }
    }

    fn exec_wave(&mut self, (wave_id, threads): (usize, &Vec<[u32; 3]>)) {
        let wave_state = self.wave_state.get(&wave_id);
        let mut sds = match wave_state {
            Some(val) => val.6.clone(),
            None => {
                let mut sds = HashMap::new();
                for i in 0..=31 {
                    sds.insert(i, VecDataStore::new());
                }
                sds
            }
        };
        let (mut scalar_reg, mut scc, mut pc) = match wave_state {
            Some(val) => (val.0.to_vec(), val.1, val.5),
            None => {
                let mut scalar_reg = vec![0; 256];
                scalar_reg.write64(0, self.kernel_args.as_ptr() as u64);
                let [gx, gy, gz] = self.id;
                match self.dispatch_dim {
                    3 => (scalar_reg[13], scalar_reg[14], scalar_reg[15]) = (gx, gy, gz),
                    2 => (scalar_reg[14], scalar_reg[15]) = (gx, gy),
                    _ => scalar_reg[15] = gx,
                }
                (scalar_reg, 0, 0)
            }
        };
        let (mut vec_reg, mut vcc, mut exec) = match wave_state {
            Some(val) => (val.2.clone(), val.3.clone(), val.4.clone()),
            _ => (VGPR::new(), WaveValue::new(0), WaveValue::new(u32::MAX)),
        };

        let mut seeded_lanes = vec![];
        loop {
            if self.kernel[pc] == END_PRG {
                if *CI {
                    self.wave_state.insert(
                        wave_id,
                        WaveState(scalar_reg, scc, vec_reg, vcc, exec, pc, sds),
                    );
                }
                break;
            }
            if BARRIERS.contains(&[self.kernel[pc], self.kernel[pc + 1]]) && wave_state.is_none() {
                self.wave_state.insert(
                    wave_id,
                    WaveState(scalar_reg, scc, vec_reg, vcc, exec, pc, sds),
                );
                break;
            }
            if SYNCS.contains(&self.kernel[pc]) || self.kernel[pc] >> 20 == 0xbf8 || self.kernel[pc] == 0x7E000000 {
                pc += 1;
                continue;
            }

            let mut sgpr_co = None;
            for (lane_id, [x, y, z]) in threads.iter().enumerate() {
                vec_reg.default_lane = Some(lane_id);
                vcc.default_lane = Some(lane_id);
                exec.default_lane = Some(lane_id);
                if DEBUG.load(SeqCst) {
                    let lane = format!("{lane_id} {:08X} ", self.kernel[pc]);
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
                let mut thread = Thread {
                    scalar_reg: &mut scalar_reg,
                    scc: &mut scc,
                    vec_reg: &mut vec_reg,
                    vcc: &mut vcc,
                    exec: &mut exec,
                    lds: &mut self.lds,
                    sds: &mut sds.get_mut(&lane_id).unwrap(),
                    pc_offset: 0,
                    stream: self.kernel[pc..self.kernel.len()].to_vec(),
                    scalar: false,
                    simm: None,
                    sgpr_co: &mut sgpr_co,
                };
                thread.interpret();
                if thread.scalar {
                    pc = ((pc as isize) + 1 + (thread.pc_offset as isize)) as usize;
                    break;
                }
                if lane_id == threads.len() - 1 {
                    pc = ((pc as isize) + 1 + (thread.pc_offset as isize)) as usize;
                }
            }

            if vcc.mutations.is_some() {
                vcc.apply_muts();
                vcc.mutations = None;
            }
            if exec.mutations.is_some() {
                exec.apply_muts();
                exec.mutations = None;
            }
            if let Some((idx, mut wv)) = sgpr_co {
                wv.apply_muts();
                scalar_reg[idx] = wv.value;
                sgpr_co = None;
            }
        }
    }
}

#[cfg(test)]
mod test_workgroup {
    use super::*;

    #[test]
    fn test_wave_value_state_vcc() {
        let kernel = vec![
            0xBEEA00FF,
            0b11111111111111111111111111111111, // initial vcc state
            0x7E140282,
            0x7C94010A, // cmp blockDim.x == 2
            END_PRG,
        ];
        let args = vec![];
        let mut wg = WorkGroup::new(1, [0, 0, 0], [3, 1, 1], &kernel, &args);
        wg.exec_waves();
        let w0 = wg.wave_state.get(&0).unwrap();
        assert_eq!(w0.3.value, 0b100);
    }

    #[test]
    fn test_wave_value_state_exec() {
        let kernel = vec![
            0xBEFE00FF,
            0b11111111111111111111111111111111,
            0x7E140282,
            0x7D9C010A, // cmpx blockDim.x <= 2
            END_PRG,
        ];
        let args = vec![];
        let mut wg = WorkGroup::new(1, [0, 0, 0], [4, 1, 1], &kernel, &args);
        wg.exec_waves();
        let w0 = wg.wave_state.get(&0).unwrap();
        assert_eq!(w0.4.value, 0b0111);
    }

    #[test]
    fn test_wave_value_sgpr_co() {
        DEBUG.store(true, SeqCst);
        let kernel = vec![
            0xBE8D00FF,
            0x7FFFFFFF,
            0x7E1402FF,
            u32::MAX,
            0xD7000D0A,
            0x0002010A,
            END_PRG,
        ];
        let args = vec![];
        let mut wg = WorkGroup::new(1, [0, 0, 0], [5, 1, 1], &kernel, &args);
        wg.exec_waves();
        let w0 = wg.wave_state.get(&0).unwrap();
        assert_eq!(w0.0[13], 0b11110);
    }
}
