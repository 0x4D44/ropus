/* Minimal arch.h stub for conformance tests.
 *
 * The reference `test_opus_api.c` includes `arch.h` for `opus_int32` /
 * `opus_uint32` typedefs. The real `arch.h` lives in `reference/celt/` and
 * drags in a lot of encoder/decoder internal machinery we don't need. This
 * stub forwards the type definitions via `opus_types.h` plus a few small
 * helper macros that the reference test sources reference directly.
 */
#ifndef ARCH_H_STUB
#define ARCH_H_STUB

#include "opus_types.h"
#include "opus_defines.h"

/* Integer min/max helpers (see reference/celt/arch.h:104-105).
 * `test_opus_encode.c:264` uses IMIN; IMAX is added for symmetry since it
 * shares the same one-line definition and costs nothing. */
#define IMIN(a,b) ((a) < (b) ? (a) : (b))
#define IMAX(a,b) ((a) > (b) ? (a) : (b))

#endif
