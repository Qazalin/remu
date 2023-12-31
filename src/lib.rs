use crate::cpu::CPU;
use crate::utils::DEBUG;
use std::os::raw::c_void;
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
    args: *const *const c_void,
) {
    let mut kernel_args: Vec<u64> = Vec::new();
    unsafe {
        for i in 0..5 {
            let ptr = *args.offset(i as isize);
            kernel_args.push(ptr as u64);
        }
    }

    if *DEBUG >= 1 {
        println!(
            "[remu] launching kernel with global_size {} {} {} local_size {} {} {} args {:?}",
            grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z, kernel_args
        );
    }

    let mut cpu = CPU::new();
    let prg: Vec<u32> = vec![]; // TODO this should come from the lib arg

    let bufs_ptr = kernel_args[1] as *const u64;
    let bufs_size_ptr = kernel_args[3] as *const usize;

    let bufs_size: usize;
    unsafe {
        bufs_size = *bufs_size_ptr;
    }

    let bufs_vec: Vec<u64>;
    unsafe {
        bufs_vec = Vec::from_raw_parts(bufs_ptr as *mut u64, bufs_size, bufs_size);
    }

    println!("values={:?}", bufs_vec);
}
