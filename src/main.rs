use anyhow::{anyhow, Result};
use clap::Parser;
use std::fs;
use utils::GLOBAL_DEBUG;
use work_group::WorkGroup;
mod dtype;
mod memory;
mod state;
mod thread;
mod utils;
mod work_group;

#[derive(Parser, Debug)]
pub struct Cli {
    #[arg(short, help = "path to the RDNA3 disassembly", long)]
    fp: String,
    #[arg(short, help = "global size in gx,gy,gz order", long)]
    global_size: String,
    #[arg(short, help = "local size in lx,ly,lz order", long)]
    local_size: String,
    #[arg(short, help = "comma seperated size of each buffer arg", long)]
    bufs: String,
    // TODO:
    // #[arg(short, help = "kernels args", long)]
    // args: String,
}

fn parse_size(sz: &String) -> [u32; 3] {
    let vec_val = sz
        .splitn(3, ",")
        .map(|s| s.parse().unwrap())
        .collect::<Vec<_>>();
    [vec_val[0], vec_val[1], vec_val[2]]
}

fn main() -> Result<()> {
    let args = Cli::parse();
    let (asm, name) = utils::parse_rdna3(&fs::read_to_string(&args.fp)?);
    if asm.len() == 0 {
        return Err(anyhow!("file {} contains no assembly", args.fp));
    }
    let (global_size, local_size) = (parse_size(&args.global_size), parse_size(&args.local_size));
    if *GLOBAL_DEBUG {
        println!("[remu] launching kernel {name} with global_size {global_size:?} local_size {local_size:?}");
    }

    let dispatch_dim = match (global_size[1] != 1, global_size[2] != 1) {
        (true, true) => 3,
        (true, false) => 2,
        _ => 1,
    };

    let mut kernel_args = vec![];
    args.bufs.split(",").for_each(|s| {
        let size = s.parse::<usize>().unwrap();
        let val: Vec<u8> = vec![0; size];
        kernel_args.extend(val);
    });

    println!("{:?}", kernel_args);

    for gx in 0..global_size[0] {
        for gy in 0..global_size[1] {
            for gz in 0..global_size[2] {
                WorkGroup::new(
                    dispatch_dim,
                    [gx, gy, gz],
                    local_size,
                    &asm,
                    kernel_args.as_ptr(),
                )
                .exec_waves();
            }
        }
    }
    println!("{:?}", local_size);
    Ok(())
}
