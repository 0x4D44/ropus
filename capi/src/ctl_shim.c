/* CTL varargs shim.
 *
 * The Opus C API exposes `opus_encoder_ctl` and `opus_decoder_ctl` as
 * C-variadic functions. We avoid Rust's unstable `c_variadic` feature by
 * doing all `va_arg` unpacking here, then calling strongly-typed Rust entry
 * points (`mdopus_*_ctl_*`). Each request kind dispatches to one typed call;
 * the default arm returns OPUS_UNIMPLEMENTED so drift surfaces as a test
 * failure rather than silent UB.
 *
 * Phase 1 only needs OPUS_RESET_STATE; later phases add getters/setters.
 */

#include <stdarg.h>
#include <stdint.h>

/* Mirror the public error/request codes from opus_defines.h. The values
 * are part of the stable ABI so we pin them here rather than including
 * the header (which would drag in the whole libopus header set). */
#define OPUS_OK                0
#define OPUS_BAD_ARG          -1
#define OPUS_UNIMPLEMENTED    -5

#define OPUS_RESET_STATE       4028

/* Forward-declarations of the typed Rust entry points. These symbols are
 * defined with `#[no_mangle] extern "C"` in `ctl.rs`. */
typedef struct OpusEncoder OpusEncoder;
typedef struct OpusDecoder OpusDecoder;

extern int mdopus_encoder_ctl_reset(OpusEncoder *st);
extern int mdopus_decoder_ctl_reset(OpusDecoder *st);

int opus_encoder_ctl(OpusEncoder *st, int request, ...)
{
    va_list ap;
    int ret;

    if (st == (void *)0) {
        return OPUS_BAD_ARG;
    }

    va_start(ap, request);
    switch (request) {
        case OPUS_RESET_STATE:
            ret = mdopus_encoder_ctl_reset(st);
            break;

        default:
            ret = OPUS_UNIMPLEMENTED;
            break;
    }
    va_end(ap);
    return ret;
}

int opus_decoder_ctl(OpusDecoder *st, int request, ...)
{
    va_list ap;
    int ret;

    if (st == (void *)0) {
        return OPUS_BAD_ARG;
    }

    va_start(ap, request);
    switch (request) {
        case OPUS_RESET_STATE:
            ret = mdopus_decoder_ctl_reset(st);
            break;

        default:
            ret = OPUS_UNIMPLEMENTED;
            break;
    }
    va_end(ap);
    return ret;
}
