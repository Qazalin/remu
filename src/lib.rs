use crate::allocator::BumpAllocator;
use crate::cpu::CPU;
use crate::utils::{Colorize, DebugLevel, DEBUG};
use std::collections::HashMap;
use std::os::raw::{c_char, c_void};
mod allocator;
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
    if *DEBUG >= DebugLevel::MISC {
        println!(
            "[remu] launching kernel {function_name} with global_size {} {} {} local_size {} {} {} args {:?}",
            grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z, kernel_args
        );
    }

    let mut gds = BumpAllocator::new(WAVE_ID);
    let stack_ptr = gds.alloc(kernel_args.len() as u32 * 8);
    kernel_args.iter().enumerate().for_each(|(i, x)| {
        gds.write_bytes(stack_ptr + i as u64 * 8, &x.to_le_bytes());
    });

    let prg = utils::split_asm_by_thread_syncs(&prg);
    for i in 0..grid_dim_x {
        let mut thread_registers = HashMap::<u32, [Vec<u32>; 2]>::new();
        for prg in prg.iter() {
            for j in 0..block_dim_x {
                let lds = BumpAllocator::new(&format!("{WAVE_ID}_lds{i}"));
                let gds = BumpAllocator::new(WAVE_ID);
                let mut cpu = CPU::new(gds, lds);
                if *DEBUG >= DebugLevel::MISC {
                    println!(
                        "{}={}, {}={}",
                        "blockIdx.x".color("jade"),
                        i,
                        "threadIdx.x".color("jade"),
                        j
                    );
                }

                match thread_registers.get(&j) {
                    Some(val) => {
                        cpu.scalar_reg.values = val[0].clone();
                        cpu.vec_reg.values = val[1].clone();
                    }
                    None => {
                        cpu.scalar_reg.write_addr(0, stack_ptr);
                        cpu.scalar_reg[15] = i;
                        cpu.vec_reg[0] = j;
                    }
                }

                cpu.interpret(&prg);
                thread_registers.insert(j, [cpu.scalar_reg.values, cpu.vec_reg.values]);
            }
        }
    }
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
