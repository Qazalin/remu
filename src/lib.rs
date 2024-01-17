use crate::allocator::BumpAllocator;
use crate::cpu::CPU;
use crate::utils::{Colorize, DebugLevel, DEBUG};
use std::collections::HashMap;
use std::os::raw::{c_char, c_void};
mod allocator;
mod alu_modifiers;
mod cpu;
mod dtype;
mod state;
mod utils;

const WAVE_ID: &str = "wave_id";

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

    let mut gds = BumpAllocator::new(WAVE_ID);
    let stack_ptr = gds.alloc(kernel_args.len() as u32 * 8);
    kernel_args.iter().enumerate().for_each(|(i, x)| {
        gds.write_bytes(stack_ptr + i as u64 * 8, &x.to_le_bytes());
    });

    if *DEBUG >= DebugLevel::NONE {
        println!(
            "[remu] launching kernel {function_name} with global_size {} {} {} local_size {} {} {} args {:?}",
            grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z, kernel_args
        );
    }

    let prg = utils::split_asm_by_thread_syncs(&prg);
    for gx in 0..grid_dim_x {
        for gy in 0..grid_dim_y {
            for gz in 0..grid_dim_z {
                let mut thread_registers = HashMap::<[u32; 3], (Vec<u32>, Vec<u32>, u32)>::new();
                for prg in prg.iter() {
                    for tx in 0..block_dim_x {
                        for ty in 0..block_dim_y {
                            for tz in 0..block_dim_z {
                                launch_thread(
                                    [gx, gy, gz],
                                    [tx, ty, tz],
                                    [grid_dim_x, grid_dim_y, grid_dim_z],
                                    [block_dim_x, block_dim_y, block_dim_z],
                                    stack_ptr,
                                    &prg,
                                    &mut thread_registers,
                                );
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
    thread_registers: &mut HashMap<[u32; 3], (Vec<u32>, Vec<u32>, u32)>,
) {
    if *DEBUG >= DebugLevel::MISC {
        println!(
            "{}={:?}, {}={:?}",
            "block".color("jade"),
            grid_id,
            "thread".color("jade"),
            thread_id
        );
    }
    let lds = BumpAllocator::new(&format!("{WAVE_ID}_lds_{:?}", grid_id));
    let gds = BumpAllocator::new(WAVE_ID);
    let mut cpu = CPU::new(gds, lds);

    match thread_registers.get(&thread_id) {
        Some(val) => {
            cpu.scalar_reg.values = val.0.clone();
            cpu.vec_reg.values = val.1.clone();
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
    thread_registers.insert(
        thread_id,
        (cpu.scalar_reg.values, cpu.vec_reg.values, *cpu.vcc),
    );
}

#[no_mangle]
pub extern "C" fn hipMalloc(ptr: *mut c_void, size: u32) {
    let mut gds = BumpAllocator::new(WAVE_ID);

    unsafe {
        let data_ptr = ptr as *mut u64;
        *data_ptr = gds.alloc(size);
    }

    if *DEBUG >= DebugLevel::MISC {
        println!("[remu] hipMalloc({})", size);
    }
}

#[no_mangle]
pub extern "C" fn hipMemcpy(dest: *const c_void, src: *const c_void, size: u32, mode: u32) {
    let mut gds = BumpAllocator::new(WAVE_ID);

    match mode {
        1 => {
            let bytes =
                unsafe { std::slice::from_raw_parts(src as *const u8, size as usize) }.to_vec();
            gds.write_bytes(dest as u64, &bytes);
        }
        2 => {
            let bytes = &gds.read_bytes(src as u64, size as usize);
            unsafe {
                let dest = dest as *mut u8;
                std::slice::from_raw_parts_mut(dest, bytes.len()).copy_from_slice(&bytes);
            }
        }
        _ => panic!("invalid mode"),
    }
}

#[no_mangle]
pub extern "C" fn hipFree(_ptr: u64) {}
