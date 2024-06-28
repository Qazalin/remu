use crate::utils::{GLOBAL_COUNTER, OSX, PROFILE};
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
) -> i32 {
    let kernel = match *OSX {
        true => {
            let mut lib_bytes: Vec<u8> = Vec::with_capacity(lib_sz as usize);
            unsafe {
                lib_bytes.extend((0..lib_sz).map(|i| *lib.offset(i as isize) as u8));
            }
            let (kernel, function_name) = utils::read_asm(&lib_bytes);
            println!(
                "[remu] launching kernel {function_name} with global_size {gx} {gy} {gz} local_size {lx} {ly} {lz}"
            );
            kernel
        }
        false => {
            if lib.is_null() || (lib_sz % 4) != 0 {
                panic!("Pointer is null or length is not properly aligned to 4 bytes");
            }
            unsafe { slice::from_raw_parts(lib as *const u32, (lib_sz / 4) as usize).to_vec() }
        }
    };
    let dispatch_dim = match (gy != 1, gz != 1) {
        (true, true) => 3,
        (true, false) => 2,
        _ => 1,
    };
    for gx in 0..gx {
        for gy in 0..gy {
            for gz in 0..gz {
                let mut wg =
                    WorkGroup::new(dispatch_dim, [gx, gy, gz], [lx, ly, lz], &kernel, args_ptr);
                if let Err(err) = wg.exec_waves() {
                    return err;
                }
            }
        }
    }
    if *PROFILE {
        println!("{:?}", GLOBAL_COUNTER.lock().unwrap());
    }
    0
}
