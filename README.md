This is a minimal RDNA3 emulator built for and tested against [tinygrad kernels](https://github.com/tinygrad/tinygrad).

## Run locally

#### Usage

1. Install the latest version
```sh
curl -s https://api.github.com/repos/Qazalin/remu/releases/latest | \
    jq -r '.assets[] | select(.name == "libremu.so").browser_download_url' | \
    xargs curl -L -o /usr/local/lib/libremu.so
```

2. Run tinygrad with `MOCKGPU=1 AMD=1`.


## Build from source

`cargo build --release --lib ` creates the binary [used in tinygrad](https://github.com/tinygrad/tinygrad/tree/master/extra/mockgpu/amd). The tinygrad integration only works in linux.


`cargo build --release --bin remu-cli` builds a minimal version of remu you can run independent of tinygrad on any platform.

## Limitations

Does not implement all RDNA3 instructions.
