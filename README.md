This is a minimal RDNA3 emulator built for and tested against [tinygrad kernels](https://github.com/tinygrad/tinygrad).

## Run locally


#### Requirements
- [ROCM toolkit](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/) - for compiling the kernels


#### Usage

1. Install the latest version
```sh
curl -s https://api.github.com/repos/Qazalin/remu/releases/latest | \
    jq -r '.assets[] | select(.name == "libremu.so").browser_download_url' | \
    xargs curl -L -o /usr/local/lib/libremu.so
```

2. Run tinygrad with `HIP=1 HIPCPU=1`. Use `REMU_DEBUG=1` to see logs from the emulator.

## Limitations

Does not implement all RDNA3 instructions.
