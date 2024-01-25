use crate::cpu::{CPU, SGPR_COUNT};
use crate::state::{Assign, Register, VGPR};
use crate::utils::{Colorize, DebugLevel, DEBUG, PROGRESS};
use std::collections::HashMap;
use std::os::raw::{c_char, c_void};
mod alu_modifiers;
mod cpu;
mod dtype;
mod state;
mod utils;

#[no_mangle]
pub extern "C" fn hipModuleLaunchKernel(
    lib: *const c_char,
    lib_sz: u32,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    block_dim_x: u32,
    block_dim_y: u32,
    block_dim_z: u32,
    _shared_mem_bytes: u32,
    _stream: *const *const c_void,
    _kernel_params: *const *const c_void,
    args_len: u32,
    args: *const *const c_void,
) {
    let mut lib_bytes: Vec<u8> = Vec::new();
    unsafe {
        for i in 0..lib_sz {
            lib_bytes.push(*lib.offset(i as isize) as u8);
        }
    }

    let mut kernel_args: Vec<u64> = Vec::new();
    unsafe {
        for i in 0..args_len {
            let ptr = *args.offset(i as isize);
            kernel_args.push(ptr as u64);
        }
    }

    let (prg, function_name) = &utils::read_asm(&lib_bytes);

    if *DEBUG >= DebugLevel::NONE {
        println!(
            "[remu] launching kernel {function_name} with global_size {} {} {} local_size {} {} {} args {:?}",
            grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z, kernel_args
        );
    }

    let prg = utils::split_asm_by_thread_syncs(&prg);
    let total = grid_dim_x * grid_dim_y * grid_dim_z * block_dim_x * block_dim_y * block_dim_z;
    let pb = match *PROGRESS != 0 {
        true => {
            let pb = indicatif::ProgressBar::new(total as u64);
            pb.set_style(
                indicatif::ProgressStyle::with_template(
                    "[{elapsed_precise}] [{wide_bar:.green/white}] {pos:>7}/{len:7} {msg}",
                )
                .unwrap()
                .progress_chars("#>-"),
            );
            Some(pb)
        }
        false => None,
    };

    for gx in 0..grid_dim_x {
        for gy in 0..grid_dim_y {
            for gz in 0..grid_dim_z {
                let mut thread_registers = HashMap::new();
                let mut lds = Vec::new();
                for prg in prg.iter() {
                    for tx in 0..block_dim_x {
                        for ty in 0..block_dim_y {
                            for tz in 0..block_dim_z {
                                launch_thread(
                                    [gx, gy, gz],
                                    [tx, ty, tz],
                                    [grid_dim_x, grid_dim_y, grid_dim_z],
                                    [block_dim_x, block_dim_y, block_dim_z],
                                    kernel_args.as_ptr() as u64,
                                    &prg,
                                    &mut lds,
                                    &mut thread_registers,
                                );
                                if let Some(ref _pb) = pb {
                                    _pb.inc(1);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn launch_thread(
    grid_id: [u32; 3],
    thread_id: [u32; 3],
    global_size: [u32; 3],
    local_size: [u32; 3],
    stack_ptr: u64,
    prg: &Vec<u32>,
    lds: &mut Vec<u8>,
    thread_registers: &mut HashMap<[u32; 3], ([u32; SGPR_COUNT], VGPR, u32)>,
) {
    if *DEBUG >= DebugLevel::NONE {
        println!(
            "{}={:?}, {}={:?}",
            "block".color("jade"),
            grid_id,
            "thread".color("jade"),
            thread_id
        );
    }
    let mut cpu = CPU::new(lds);
    match thread_registers.get(&thread_id) {
        Some(val) => {
            cpu.scalar_reg = val.0.clone();
            cpu.vec_reg = val.1.clone();
            cpu.vcc.assign(val.2);
        }
        None => {
            cpu.scalar_reg.write64(0, stack_ptr);

            match (global_size[1] != 1, global_size[2] != 1) {
                (true, true) => {
                    (cpu.scalar_reg[13], cpu.scalar_reg[14], cpu.scalar_reg[15]) =
                        (grid_id[0], grid_id[1], grid_id[2])
                }
                (true, false) => {
                    (cpu.scalar_reg[14], cpu.scalar_reg[15]) = (grid_id[0], grid_id[1])
                }
                _ => cpu.scalar_reg[15] = grid_id[0],
            }

            match (local_size[1] != 1, local_size[2] != 1) {
                (false, false) => cpu.vec_reg[0] = thread_id[0],
                _ => cpu.vec_reg[0] = (thread_id[2] << 20) | (thread_id[1] << 10) | (thread_id[0]),
            }
        }
    }
    cpu.interpret(prg);
    thread_registers.insert(thread_id, (cpu.scalar_reg, cpu.vec_reg, *cpu.vcc));
}
