/* Minimal mathops.h stub for conformance tests.
 *
 * `test_opus_projection.c` `#include "mathops.h"` but only uses `floor` and
 * `sqrt` from `<math.h>`. The real `mathops.h` lives in `reference/celt/`
 * and drags in `entcode.h` (entropy coder) and `os_support.h` (allocator
 * wrappers); we vendor a stub that just pulls in `<math.h>` and our
 * already-vendored `os_support.h`, so test code that calls `opus_alloc` /
 * `opus_free` (and expects those to be inlined from `os_support.h` via
 * the real `mathops.h`) resolves correctly.
 *
 * Same header-guard trick as `arch.h`: claiming `MATHOPS_H` makes any
 * subsequent transitive include of the real header a no-op.
 */
#ifndef MATHOPS_H
#define MATHOPS_H

#include "arch.h"
#include "os_support.h"
#include <math.h>

#endif
