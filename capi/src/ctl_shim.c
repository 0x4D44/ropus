/* CTL varargs shim.
 *
 * The Opus C API exposes `opus_encoder_ctl` and `opus_decoder_ctl` as
 * C-variadic functions. We avoid Rust's unstable `c_variadic` feature by
 * doing all `va_arg` unpacking here, then calling strongly-typed Rust entry
 * points (`mdopus_*_ctl_*`). Each request kind dispatches to one typed call;
 * the default arm returns OPUS_UNIMPLEMENTED so drift surfaces as a test
 * failure rather than silent UB.
 *
 * Phase 2 adds decoder int getters/setters and the uint32 getter needed
 * by `test_opus_decode.c` (FINAL_RANGE / LAST_PACKET_DURATION). Encoder
 * CTLs beyond RESET_STATE land in phase 4.
 */

#include <stdarg.h>
#include <stdint.h>

/* Mirror the public error/request codes from opus_defines.h. The values
 * are part of the stable ABI so we pin them here rather than including
 * the header (which would drag in the whole libopus header set). */
#define OPUS_OK                0
#define OPUS_BAD_ARG          -1
#define OPUS_UNIMPLEMENTED    -5

#define OPUS_RESET_STATE                        4028

/* Generic/decoder request codes. Values from opus_defines.h. */
#define OPUS_GET_BANDWIDTH_REQUEST              4009
#define OPUS_GET_SAMPLE_RATE_REQUEST            4029
#define OPUS_GET_FINAL_RANGE_REQUEST            4031
#define OPUS_GET_PITCH_REQUEST                  4033
#define OPUS_SET_GAIN_REQUEST                   4034
#define OPUS_GET_GAIN_REQUEST                   4045  /* yes, 4045 per header */
#define OPUS_GET_LAST_PACKET_DURATION_REQUEST   4039
#define OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST 4046
#define OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST 4047

typedef uint32_t opus_uint32;

/* Forward-declarations of the typed Rust entry points. These symbols are
 * defined with `#[no_mangle] extern "C"` in `ctl.rs`. */
typedef struct OpusEncoder OpusEncoder;
typedef struct OpusDecoder OpusDecoder;

extern int mdopus_encoder_ctl_reset(OpusEncoder *st);
extern int mdopus_decoder_ctl_reset(OpusDecoder *st);
extern int mdopus_decoder_ctl_set_int(OpusDecoder *st, int request, int value);
extern int mdopus_decoder_ctl_get_int(OpusDecoder *st, int request, int *out);
extern int mdopus_decoder_ctl_get_uint32(OpusDecoder *st, int request, opus_uint32 *out);

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

        /* int setters — add new codes here. */
        case OPUS_SET_GAIN_REQUEST:
        case OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST:
            ret = mdopus_decoder_ctl_set_int(st, request, va_arg(ap, int));
            break;

        /* int getters. */
        case OPUS_GET_BANDWIDTH_REQUEST:
        case OPUS_GET_SAMPLE_RATE_REQUEST:
        case OPUS_GET_PITCH_REQUEST:
        case OPUS_GET_GAIN_REQUEST:
        case OPUS_GET_LAST_PACKET_DURATION_REQUEST:
        case OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST:
            ret = mdopus_decoder_ctl_get_int(st, request, va_arg(ap, int *));
            break;

        /* uint32 getters (only final-range right now). */
        case OPUS_GET_FINAL_RANGE_REQUEST:
            ret = mdopus_decoder_ctl_get_uint32(st, request, va_arg(ap, opus_uint32 *));
            break;

        default:
            ret = OPUS_UNIMPLEMENTED;
            break;
    }
    va_end(ap);
    return ret;
}
