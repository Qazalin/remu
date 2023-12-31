import ctypes

remu = ctypes.CDLL("./target/release/libremu.dylib")

hipModuleLaunchKernel = remu.hipModuleLaunchKernel
hipModuleLaunchKernel.restype = None
hipModuleLaunchKernel.argtypes = [
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t
]

my_arr = ctypes.c_void_p * 5
my_arr = my_arr(0x45, 0x10, 3, 4, 5)
remu.hipModuleLaunchKernel(my_arr, 5)
