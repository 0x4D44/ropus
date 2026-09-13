/* Keep the deep-PLC DNN oracle on the scalar vector implementation. */

#include "config.h"
#include "vec.h"

#ifndef NO_OPTIMIZATIONS
#error "deep-PLC harness requires the scalar dnn/vec.h branch"
#endif

void ropus_test_dnn_scalar_kernel_guard(void) {}
