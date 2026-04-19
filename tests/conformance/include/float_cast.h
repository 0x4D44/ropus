/* Minimal float_cast.h stub for conformance tests.
 *
 * The real `float_cast.h` in `reference/celt/` pulls in SSE intrinsics via
 * `<intrin.h>` on MSVC targeting x64, which in turn pulls in xmmintrin.h /
 * mmintrin.h / setjmp.h. Under the `cc` crate's default command line those
 * headers fail to parse (missing target-specific predefines).
 *
 * `test_opus_projection.c` and `mapping_matrix.c` include this header but
 * — when built with `FIXED_POINT=1` and `DISABLE_FLOAT_API=1` (both set in
 * `build.rs`) — neither file calls `float2int` or `FLOAT2INT16`. A
 * placeholder header that simply claims the guard is sufficient.
 *
 * Same header-guard trick as `arch.h`: claiming `FLOAT_CAST_H` makes any
 * subsequent transitive include of the real header a no-op.
 */
#ifndef FLOAT_CAST_H
#define FLOAT_CAST_H

#include "arch.h"

#endif
