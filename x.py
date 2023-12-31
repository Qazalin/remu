import ctypes

remu = ctypes.CDLL("./target/release/libremu.dylib")

remu.say_hi.argtypes = []
remu.say_hi.restype = None

remu.say_hi()
