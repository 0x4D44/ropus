import struct, math
# Compute pi4 = (float)(M_PI*M_PI*M_PI*M_PI) the two ways

# (A) M_PI = 3.141592653 (short form, if math.h has NOT defined M_PI first)
MPI_short = 3.141592653
pi4_short = MPI_short ** 4
pi4_short_f32 = struct.unpack("f", struct.pack("f", pi4_short))[0]
bits_a = struct.unpack("I", struct.pack("f", pi4_short_f32))[0]

# (B) M_PI from math.h (most glibc): 3.14159265358979323846
MPI_full = 3.14159265358979323846
pi4_full = MPI_full ** 4
pi4_full_f32 = struct.unpack("f", struct.pack("f", pi4_full))[0]
bits_b = struct.unpack("I", struct.pack("f", pi4_full_f32))[0]

# (C) Rust: core::f32::consts::PI (f32) ** 4, computed on f32 throughout
# Actually let's check what the port does:
# const PI4: f32 = (core::f32::consts::PI as f32) * (core::f32::consts::PI as f32)
#                  * (core::f32::consts::PI as f32) * (core::f32::consts::PI as f32);
# This does 4 f32 multiplies.
PI_f32 = struct.unpack("f", struct.pack("f", math.pi))[0]
# Simulate const f32 compile-time: (PI*PI) = a, (PI*PI) = b, a*b - compile time does exact as-float
# Rust const fn for const float arithmetic - actually this is runtime-only... Actually it's const-fn in Rust.
# Rust *const* float arithmetic is actually IEEE-754 single-precision at *compile time*.
def mul_f32(a, b):
    r = a * b
    return struct.unpack("f", struct.pack("f", r))[0]
step1 = mul_f32(PI_f32, PI_f32)
step2 = mul_f32(PI_f32, PI_f32)
step3 = mul_f32(step1, step2)  # This matches the port's structure (PI*PI) * (PI*PI)
bits_c = struct.unpack("I", struct.pack("f", step3))[0]

print(f"(A) Short M_PI^4 -> f32:  0x{bits_a:08x}  ({pi4_short_f32})")
print(f"(B) Full M_PI^4 -> f32:   0x{bits_b:08x}  ({pi4_full_f32})")
print(f"(C) Rust PI_f32^4 (f32 ops): 0x{bits_c:08x}  ({step3})")
print()
print(f"A == B: {bits_a == bits_b}")
print(f"B == C: {bits_b == bits_c}")
print(f"A == C: {bits_a == bits_c}")
