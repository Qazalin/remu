#![allow(unused)]
use crate::utils::{DEBUG, GLOBAL_COUNTER, PROFILE};
use crate::work_group::WorkGroup;
use std::os::raw::{c_char, c_void};
use std::sync::atomic::Ordering::SeqCst;
mod dtype;
mod memory;
mod state;
mod thread;
mod utils;
mod work_group;

#[no_mangle]
pub extern "C" fn hipModuleLaunchKernel(
    lib: *const c_char,
    lib_sz: u32,
    gx: u32,
    gy: u32,
    gz: u32,
    lx: u32,
    ly: u32,
    lz: u32,
    _shared_mem_bytes: u32,
    _stream: *const *const c_void,
    _kernel_params: *const *const c_void,
    args_len: u32,
    launch_args: *const *const c_void,
) {
    let mut lib_bytes: Vec<u8> = Vec::new();
    unsafe {
        for i in 0..lib_sz {
            lib_bytes.push(*lib.offset(i as isize) as u8);
        }
    }
    let mut args: Vec<u64> = Vec::new();
    unsafe {
        for i in 0..args_len {
            let ptr = *launch_args.offset(i as isize);
            args.push(ptr as u64);
        }
    }

    let (kernel, function_name) = utils::read_asm(&lib_bytes);
    if *PROFILE {
        println!("[remu] launching kernel {function_name} with global_size {gx} {gy} {gz} local_size {lx} {ly} {lz} args {:?}", args);
    }

    let dispatch_dim = match (gy != 1, gz != 1) {
        (true, true) => 3,
        (true, false) => 2,
        _ => 1,
    };
    for gx in 0..gx {
        for gy in 0..gy {
            for gz in 0..gz {
                WorkGroup::new(dispatch_dim, [gx, gy, gz], [lx, ly, lz], &kernel, &args)
                    .exec_waves();
            }
        }
    }
    if *PROFILE {
        println!("{:?}", GLOBAL_COUNTER.lock().unwrap());
    }
}
