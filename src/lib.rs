use crate::utils::{GLOBAL_COUNTER, PROFILE};
use crate::work_group::WorkGroup;
use std::os::raw::c_char;
use std::slice;
mod dtype;
mod memory;
mod state;
mod thread;
mod utils;
mod work_group;

#[no_mangle]
pub extern "C" fn run_asm(
    lib: *const c_char,
    lib_sz: u32,
    gx: u32,
    gy: u32,
    gz: u32,
    lx: u32,
    ly: u32,
    lz: u32,
    args_ptr: *const u64,
) {
    if lib.is_null() || (lib_sz % 4) != 0 {
        panic!("Pointer is null or length is not properly aligned to 4 bytes");
    }

    let raw_asm;
    unsafe {
        raw_asm = slice::from_raw_parts(lib as *const u32, (lib_sz / 4) as usize);
    }
    let kernel = raw_asm.to_vec();

    if *PROFILE {
        println!(
            "[remu] launching kernel with global_size {gx} {gy} {gz} local_size {lx} {ly} {lz}"
        );
    }

    let dispatch_dim = match (gy != 1, gz != 1) {
        (true, true) => 3,
        (true, false) => 2,
        _ => 1,
    };
    for gx in 0..gx {
        for gy in 0..gy {
            for gz in 0..gz {
                WorkGroup::new(dispatch_dim, [gx, gy, gz], [lx, ly, lz], &kernel, args_ptr)
                    .exec_waves();
            }
        }
    }
    if *PROFILE {
        println!("{:?}", GLOBAL_COUNTER.lock().unwrap());
    }
}
