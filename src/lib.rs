use crate::cpu::CPU;
use crate::utils::DEBUG;
use std::os::raw::{c_uchar, c_void};
mod allocator;
mod cpu;
mod state;
mod utils;

#[no_mangle]
pub extern "C" fn hipModuleLaunchKernel(
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
    let mut kernel_args: Vec<u64> = Vec::new();
    unsafe {
        for i in 0..args_len {
            let ptr = *args.offset(i as isize);
            kernel_args.push(ptr as u64);
        }
    }

    if *DEBUG >= 1 {
        println!(
            "[remu] launching kernel with global_size {} {} {} local_size {} {} {} {} args {:?}",
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            block_dim_x,
            block_dim_y,
            block_dim_z,
            args_len,
            kernel_args
        );
    }

    let mut cpu = CPU::new();
    let prg: Vec<u32> = vec![]; // TODO this should come from the lib arg
}

#[no_mangle]
pub extern "C" fn hipMalloc(ptr: *mut c_void, size: u32) {
    let mut cpu = CPU::new();

    unsafe {
        let data_ptr = ptr as *mut u64;
        *data_ptr = cpu.allocator.alloc(size);
    }

    if *DEBUG >= 1 {
        println!("[remu] hipMalloc({})", size);
    }

    cpu.allocator.save();
}

#[no_mangle]
pub extern "C" fn hipMemcpy(dest: *const c_void, src: *const c_uchar, size: u32, mode: u32) {
    match mode {
        1 => {
            println!("copyin {}", size);

            let bytes =
                unsafe { std::slice::from_raw_parts(src as *const u8, size as usize) }.to_vec();

            let as_fp: Vec<f32> = bytes
                .chunks(4) // Process 4 bytes at a time
                .filter_map(|chunk| {
                    if let Ok(array) = chunk.try_into() {
                        Some(f32::from_le_bytes(array)) // Convert each chunk to f32 directly
                    } else {
                        None // Skip incomplete chunks
                    }
                })
                .collect();
            println!("recieved data = {:?}", as_fp);
        }
        2 => {
            println!("copyout {}", size);
        }
        _ => panic!("invalid mode"),
    }
}
