mod cpu;
mod state;
mod utils;

#[no_mangle]
pub extern "C" fn hipModuleLaunchKernel(
    f: u32,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    block_dim_x: u32,
    block_dim_y: u32,
    block_dim_z: u32,
    shared_mem_bytes: u32,
    stream: u64,
    kernel_params: u64,
    extra: u64,
) {
    println!("hello from rust!");
}
