import struct, math
c = 3.1415926535897931 / 2
cb = struct.unpack("I", struct.pack("f", c))[0]
r = math.pi / 2
rb = struct.unpack("I", struct.pack("f", r))[0]
print(f"C (float)(PI/2): 0x{cb:08x}")
print(f"Rust FRAC_PI_2:   0x{rb:08x}")
print(f"match = {cb == rb}")
