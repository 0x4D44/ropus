/* Minimal arch.h stub for conformance tests.
 *
 * The reference `test_opus_api.c` includes `arch.h` for `opus_int32` /
 * `opus_uint32` typedefs. The real `arch.h` lives in `reference/celt/` and
 * drags in a lot of encoder/decoder internal machinery we don't need. This
 * stub forwards the type definitions via `opus_types.h` plus a few small
 * helper macros that the reference test sources reference directly.
 *
 * The `ARCH_H` header guard is deliberately the same as the real
 * `reference/celt/arch.h`. Any test source that also pulls in the real
 * arch.h (via `float_cast.h`, `mathops.h`, or `mapping_matrix.c`) will see
 * this stub first (because `tests/conformance/include/` precedes
 * `reference/celt/` on the include path) and the real one becomes a no-op.
 *
 * Macros added for Piece B (projection test + mapping_matrix.c):
 *   - `opus_res`, `INT16TORES`, `RES2INT16`, `INT24TORES`, `RES2INT24`
 *   - fixed-point helpers (SAT16, PSHR32, SHL32, EXTEND32, ADD32, SHR32)
 *   - `align` (4-byte alignment up-round)
 *   - `celt_assert` (no-op; ENABLE_ASSERTIONS is off)
 *   - `opus_val16` / `opus_val32` / `opus_val64` typedefs
 * Values match the `FIXED_POINT=1`, `ENABLE_RES24` off build target.
 */
#ifndef ARCH_H
#define ARCH_H

#include "opus_types.h"
#include "opus_defines.h"
#include <stdlib.h> /* abort() for celt_fatal */

/* Integer min/max helpers (see reference/celt/arch.h:104-105).
 * `test_opus_encode.c:264` uses IMIN; IMAX is added for symmetry since it
 * shares the same one-line definition and costs nothing. */
#define IMIN(a,b) ((a) < (b) ? (a) : (b))
#define IMAX(a,b) ((a) > (b) ? (a) : (b))

/* ---- Numeric typedefs (match reference/celt/arch.h:140-164 under
 *      FIXED_POINT, ENABLE_RES24 off). ---- */
typedef opus_int16 opus_val16;
typedef opus_int32 opus_val32;
typedef long long  opus_val64;
typedef opus_int64 opus_int64_t; /* not strictly needed; kept for symmetry */
typedef opus_val16 opus_res;

/* ---- Fixed-point arithmetic primitives (reference/celt/fixed_generic.h,
 *      stripped to the surface needed by mapping_matrix.c). ---- */
#define EXTRACT16(x)       ((opus_val16)(x))
#define EXTEND32(x)        ((opus_val32)(x))
#define SHR32(a,shift)     ((a) >> (shift))
#define SHL32(a,shift)     ((opus_int32)((opus_uint32)(a) << (shift)))
#define ADD32(a,b)         ((opus_val32)(a) + (opus_val32)(b))
#define PSHR32(a,shift)    (SHR32((a) + ((EXTEND32(1) << (shift)) >> 1), (shift)))

/* SAT16 / SATURATE16: saturating casts to opus_int16 (reference/celt/arch.h:230,
 * reference/celt/fixed_generic.h:136). SAT16 is the static inline form;
 * SATURATE16 is the macro form used by `mapping_matrix.c`. */
static __inline opus_int16 SAT16(opus_int32 x) {
    return x > 32767 ? 32767 : x < -32768 ? -32768 : (opus_int16)x;
}
#define SATURATE16(x) (EXTRACT16((x) > 32767 ? 32767 : (x) < -32768 ? -32768 : (x)))
#define SATURATE(x,a) (((x) > (a) ? (a) : (x) < -(a) ? -(a) : (x)))

/* ---- opus_res conversions (reference/celt/arch.h:164-177 under
 *      FIXED_POINT, ENABLE_RES24 off). ---- */
#define RES_SHIFT    0
#define RES2INT16(a) (a)
#define INT16TORES(a) (a)
#define RES2INT24(a) SHL32(EXTEND32(a), 8)
#define INT24TORES(a) SAT16(PSHR32(a, 8))

/* ---- Alignment (reference/celt/arch.h:221). Rounds up to 4-byte bound. ---- */
#define ALIGN 4
#define align(a) ((((a) + ALIGN - 1) / ALIGN) * ALIGN)

/* ---- Assertions (reference/celt/arch.h:47 — with ENABLE_ASSERTIONS off
 *      celt_assert is a no-op). Conformance compiles without the macro. ---- */
#define celt_assert(cond)  ((void)0)
#define celt_assert2(cond, message) ((void)0)
#define celt_sig_assert(cond) ((void)0)

#endif
