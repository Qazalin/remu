use crate::cpu::CPU;
use crate::utils::DEBUG;
use std::os::raw::{c_char, c_void};
mod allocator;
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
    let prg = utils::read_asm(&lib_bytes);

    let stack_ptr = cpu.allocator.alloc(kernel_args.len() as u32 * 8);
    kernel_args.iter().enumerate().for_each(|(i, x)| {
        cpu.allocator
            .copyin(stack_ptr + i as u64 * 8, &x.to_le_bytes());
    });

    for i in 0..grid_dim_x {
        cpu.scalar_reg.reset();
        cpu.scalar_reg.write_addr(0, stack_ptr);
        cpu.scalar_reg[15] = i;
        cpu.interpret(&prg);
    }

    cpu.allocator.save();
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
pub extern "C" fn hipMemcpy(dest: *const c_void, src: *const c_void, size: u32, mode: u32) {
    let mut cpu = CPU::new();

    match mode {
        1 => {
            let bytes =
                unsafe { std::slice::from_raw_parts(src as *const u8, size as usize) }.to_vec();
            cpu.allocator.copyin(dest as u64, &bytes);
        }
        2 => {
            let bytes = &cpu.allocator.memory[src as usize..src as usize + size as usize];
            unsafe {
                let dest = dest as *mut u8;
                std::slice::from_raw_parts_mut(dest, bytes.len()).copy_from_slice(&bytes);
            }
        }
        _ => panic!("invalid mode"),
    }

    cpu.allocator.save();
}
