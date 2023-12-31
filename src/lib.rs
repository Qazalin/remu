use std::os::raw::c_void;
mod cpu;
mod state;
mod utils;

#[no_mangle]
pub extern "C" fn hipModuleLaunchKernel(params: *const *const c_void, len: usize) {
    let mut kernel_params: Vec<u64> = Vec::new();

    unsafe {
        for i in 0..len {
            let ptr = *params.offset(i as isize);
            kernel_params.push(ptr as u64);
        }
    }

    println!("kernel_params: {:?}", kernel_params);
}
