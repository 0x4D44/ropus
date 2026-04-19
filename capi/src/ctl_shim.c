/* CTL varargs shim.
 *
 * The Opus C API exposes `opus_encoder_ctl`, `opus_decoder_ctl`,
 * `opus_multistream_encoder_ctl`, and `opus_multistream_decoder_ctl` as
 * C-variadic functions. We avoid Rust's unstable `c_variadic` feature by
 * doing all `va_arg` unpacking here and calling strongly-typed Rust entry
 * points. Each request kind dispatches to one typed call; the default arm
 * returns OPUS_UNIMPLEMENTED so drift surfaces as a test failure rather than
 * silent UB.
 */

#include <stdarg.h>
#include <stdint.h>

/* Error and request codes (values are part of the stable public ABI, defined
 * in opus_defines.h). We pin them rather than #include the full headers. */
#define OPUS_OK                0
#define OPUS_BAD_ARG          -1
#define OPUS_UNIMPLEMENTED    -5

#define OPUS_RESET_STATE                           4028

#define OPUS_SET_APPLICATION_REQUEST               4000
#define OPUS_GET_APPLICATION_REQUEST               4001
#define OPUS_SET_BITRATE_REQUEST                   4002
#define OPUS_GET_BITRATE_REQUEST                   4003
#define OPUS_SET_MAX_BANDWIDTH_REQUEST             4004
#define OPUS_GET_MAX_BANDWIDTH_REQUEST             4005
#define OPUS_SET_VBR_REQUEST                       4006
#define OPUS_GET_VBR_REQUEST                       4007
#define OPUS_SET_BANDWIDTH_REQUEST                 4008
#define OPUS_GET_BANDWIDTH_REQUEST                 4009
#define OPUS_SET_COMPLEXITY_REQUEST                4010
#define OPUS_GET_COMPLEXITY_REQUEST                4011
#define OPUS_SET_INBAND_FEC_REQUEST                4012
#define OPUS_GET_INBAND_FEC_REQUEST                4013
#define OPUS_SET_PACKET_LOSS_PERC_REQUEST          4014
#define OPUS_GET_PACKET_LOSS_PERC_REQUEST          4015
#define OPUS_SET_DTX_REQUEST                       4016
#define OPUS_GET_DTX_REQUEST                       4017
#define OPUS_SET_VBR_CONSTRAINT_REQUEST            4020
#define OPUS_GET_VBR_CONSTRAINT_REQUEST            4021
#define OPUS_SET_FORCE_CHANNELS_REQUEST            4022
#define OPUS_GET_FORCE_CHANNELS_REQUEST            4023
#define OPUS_SET_SIGNAL_REQUEST                    4024
#define OPUS_GET_SIGNAL_REQUEST                    4025
#define OPUS_GET_LOOKAHEAD_REQUEST                 4027
#define OPUS_GET_SAMPLE_RATE_REQUEST               4029
#define OPUS_GET_FINAL_RANGE_REQUEST               4031
#define OPUS_GET_PITCH_REQUEST                     4033
#define OPUS_SET_GAIN_REQUEST                      4034
#define OPUS_GET_GAIN_REQUEST                      4045  /* yes, 4045 per header */
#define OPUS_SET_LSB_DEPTH_REQUEST                 4036
#define OPUS_GET_LSB_DEPTH_REQUEST                 4037
#define OPUS_GET_LAST_PACKET_DURATION_REQUEST      4039
#define OPUS_SET_EXPERT_FRAME_DURATION_REQUEST     4040
#define OPUS_GET_EXPERT_FRAME_DURATION_REQUEST     4041
#define OPUS_SET_PREDICTION_DISABLED_REQUEST       4042
#define OPUS_GET_PREDICTION_DISABLED_REQUEST       4043
#define OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST  4046
#define OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST  4047
#define OPUS_GET_IN_DTX_REQUEST                    4049
#define OPUS_SET_IGNORE_EXTENSIONS_REQUEST         4058
#define OPUS_GET_IGNORE_EXTENSIONS_REQUEST         4059

/* DNN weight-loading CTL (opus_defines.h:174). Takes two args:
 * const unsigned char *data, opus_int32 len. Writes-only (xiph's GET
 * counterpart at 4053 is commented out). */
#define OPUS_SET_DNN_BLOB_REQUEST                  4052

#define OPUS_MULTISTREAM_GET_ENCODER_STATE_REQUEST 5120
#define OPUS_MULTISTREAM_GET_DECODER_STATE_REQUEST 5122

/* From celt/celt.h — multistream-internal CTLs forwarded into CELT.
 * LFE takes an int, ENERGY_MASK takes an int pointer (celt_glog *). */
#define OPUS_SET_LFE_REQUEST                       10024
#define OPUS_SET_ENERGY_MASK_REQUEST               10026

/* From opus_private.h:172 — private encoder CTL used by test_opus_encode.c
 * (MODE_SILK_ONLY | MODE_HYBRID | MODE_CELT_ONLY | OPUS_AUTO). */
#define OPUS_SET_FORCE_MODE_REQUEST                11002
/* Semi-obsolete voice/music ratio hint (opus_private.h), `-1..=100`. */
#define OPUS_SET_VOICE_RATIO_REQUEST               11018
#define OPUS_GET_VOICE_RATIO_REQUEST               11019

/* Projection-specific CTLs (opus_projection.h). Live on the ENCODER CTL
 * surface (the reference source and test_opus_projection.c both call
 * opus_projection_encoder_ctl with these). */
#define OPUS_PROJECTION_GET_DEMIXING_MATRIX_GAIN_REQUEST  6001
#define OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST  6003
#define OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST       6005

typedef uint32_t opus_uint32;

typedef struct OpusEncoder OpusEncoder;
typedef struct OpusDecoder OpusDecoder;
typedef struct OpusMSEncoder OpusMSEncoder;
typedef struct OpusMSDecoder OpusMSDecoder;
typedef struct OpusProjectionEncoder OpusProjectionEncoder;
typedef struct OpusProjectionDecoder OpusProjectionDecoder;

/* Rust-side typed dispatchers (see ctl.rs). */
extern int mdopus_encoder_ctl_reset(OpusEncoder *st);
extern int mdopus_encoder_ctl_set_int(OpusEncoder *st, int request, int value);
extern int mdopus_encoder_ctl_get_int(OpusEncoder *st, int request, int *out);
extern int mdopus_encoder_ctl_get_uint32(OpusEncoder *st, int request, opus_uint32 *out);
extern int mdopus_encoder_ctl_set_energy_mask(OpusEncoder *st, const int *ptr);

extern int mdopus_decoder_ctl_reset(OpusDecoder *st);
extern int mdopus_decoder_ctl_set_int(OpusDecoder *st, int request, int value);
extern int mdopus_decoder_ctl_get_int(OpusDecoder *st, int request, int *out);
extern int mdopus_decoder_ctl_get_uint32(OpusDecoder *st, int request, opus_uint32 *out);
extern int mdopus_decoder_ctl_set_dnn_blob(OpusDecoder *st, const unsigned char *data, int len);

extern int mdopus_ms_encoder_ctl_reset(OpusMSEncoder *st);
extern int mdopus_ms_encoder_ctl_set_int(OpusMSEncoder *st, int request, int value);
extern int mdopus_ms_encoder_ctl_get_int(OpusMSEncoder *st, int request, int *out);
extern int mdopus_ms_encoder_ctl_get_uint32(OpusMSEncoder *st, int request, opus_uint32 *out);
extern int mdopus_ms_encoder_ctl_get_encoder_state(OpusMSEncoder *st, int stream_id, OpusEncoder **out);
extern int mdopus_ms_encoder_ctl_set_energy_mask(OpusMSEncoder *st, const int *ptr);

extern int mdopus_ms_decoder_ctl_reset(OpusMSDecoder *st);
extern int mdopus_ms_decoder_ctl_set_int(OpusMSDecoder *st, int request, int value);
extern int mdopus_ms_decoder_ctl_get_int(OpusMSDecoder *st, int request, int *out);
extern int mdopus_ms_decoder_ctl_get_uint32(OpusMSDecoder *st, int request, opus_uint32 *out);
extern int mdopus_ms_decoder_ctl_get_decoder_state(OpusMSDecoder *st, int stream_id, OpusDecoder **out);

extern int mdopus_proj_encoder_ctl_reset(OpusProjectionEncoder *st);
extern int mdopus_proj_encoder_ctl_set_int(OpusProjectionEncoder *st, int request, int value);
extern int mdopus_proj_encoder_ctl_get_int(OpusProjectionEncoder *st, int request, int *out);
extern int mdopus_proj_encoder_ctl_get_uint32(OpusProjectionEncoder *st, int request, opus_uint32 *out);
extern int mdopus_proj_encoder_ctl_get_demixing_matrix(OpusProjectionEncoder *st, unsigned char *out, int size);

extern int mdopus_proj_decoder_ctl_reset(OpusProjectionDecoder *st);
extern int mdopus_proj_decoder_ctl_set_int(OpusProjectionDecoder *st, int request, int value);
extern int mdopus_proj_decoder_ctl_get_int(OpusProjectionDecoder *st, int request, int *out);
extern int mdopus_proj_decoder_ctl_get_uint32(OpusProjectionDecoder *st, int request, opus_uint32 *out);

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

        case OPUS_SET_APPLICATION_REQUEST:
        case OPUS_SET_BITRATE_REQUEST:
        case OPUS_SET_MAX_BANDWIDTH_REQUEST:
        case OPUS_SET_VBR_REQUEST:
        case OPUS_SET_BANDWIDTH_REQUEST:
        case OPUS_SET_COMPLEXITY_REQUEST:
        case OPUS_SET_INBAND_FEC_REQUEST:
        case OPUS_SET_PACKET_LOSS_PERC_REQUEST:
        case OPUS_SET_DTX_REQUEST:
        case OPUS_SET_VBR_CONSTRAINT_REQUEST:
        case OPUS_SET_FORCE_CHANNELS_REQUEST:
        case OPUS_SET_SIGNAL_REQUEST:
        case OPUS_SET_LSB_DEPTH_REQUEST:
        case OPUS_SET_EXPERT_FRAME_DURATION_REQUEST:
        case OPUS_SET_PREDICTION_DISABLED_REQUEST:
        case OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST:
        case OPUS_SET_FORCE_MODE_REQUEST:
        case OPUS_SET_VOICE_RATIO_REQUEST:
        case OPUS_SET_LFE_REQUEST:
            ret = mdopus_encoder_ctl_set_int(st, request, va_arg(ap, int));
            break;

        case OPUS_GET_APPLICATION_REQUEST:
        case OPUS_GET_BITRATE_REQUEST:
        case OPUS_GET_MAX_BANDWIDTH_REQUEST:
        case OPUS_GET_VBR_REQUEST:
        case OPUS_GET_BANDWIDTH_REQUEST:
        case OPUS_GET_COMPLEXITY_REQUEST:
        case OPUS_GET_INBAND_FEC_REQUEST:
        case OPUS_GET_PACKET_LOSS_PERC_REQUEST:
        case OPUS_GET_DTX_REQUEST:
        case OPUS_GET_VBR_CONSTRAINT_REQUEST:
        case OPUS_GET_FORCE_CHANNELS_REQUEST:
        case OPUS_GET_SIGNAL_REQUEST:
        case OPUS_GET_LOOKAHEAD_REQUEST:
        case OPUS_GET_SAMPLE_RATE_REQUEST:
        case OPUS_GET_LSB_DEPTH_REQUEST:
        case OPUS_GET_EXPERT_FRAME_DURATION_REQUEST:
        case OPUS_GET_PREDICTION_DISABLED_REQUEST:
        case OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST:
        case OPUS_GET_VOICE_RATIO_REQUEST:
        case OPUS_GET_IN_DTX_REQUEST: {
            int *out = va_arg(ap, int *);
            if (out == (void *)0) {
                ret = OPUS_BAD_ARG;
            } else {
                ret = mdopus_encoder_ctl_get_int(st, request, out);
            }
            break;
        }

        case OPUS_GET_FINAL_RANGE_REQUEST: {
            opus_uint32 *out = va_arg(ap, opus_uint32 *);
            if (out == (void *)0) {
                ret = OPUS_BAD_ARG;
            } else {
                ret = mdopus_encoder_ctl_get_uint32(st, request, out);
            }
            break;
        }

        /* Multistream-internal: energy mask is a pointer in the C ABI
         * (celt_glog *, which is `int` in the fixed-point build). The
         * reference API does not include a length — callers in
         * opus_multistream_encoder.c size the buffer as `21 * channels`
         * (L993/L1008). Rust reads the channel count from the encoder
         * state and sizes the slice internally, so stereo (42) and mono
         * (21) sub-encoders are handled correctly. */
        case OPUS_SET_ENERGY_MASK_REQUEST: {
            const int *ptr = va_arg(ap, const int *);
            ret = mdopus_encoder_ctl_set_energy_mask(st, ptr);
            break;
        }

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

        case OPUS_SET_GAIN_REQUEST:
        case OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST:
        case OPUS_SET_IGNORE_EXTENSIONS_REQUEST:
        case OPUS_SET_COMPLEXITY_REQUEST:
            ret = mdopus_decoder_ctl_set_int(st, request, va_arg(ap, int));
            break;

        case OPUS_GET_BANDWIDTH_REQUEST:
        case OPUS_GET_SAMPLE_RATE_REQUEST:
        case OPUS_GET_PITCH_REQUEST:
        case OPUS_GET_GAIN_REQUEST:
        case OPUS_GET_LAST_PACKET_DURATION_REQUEST:
        case OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST:
        case OPUS_GET_IGNORE_EXTENSIONS_REQUEST:
        case OPUS_GET_COMPLEXITY_REQUEST: {
            int *out = va_arg(ap, int *);
            if (out == (void *)0) {
                ret = OPUS_BAD_ARG;
            } else {
                ret = mdopus_decoder_ctl_get_int(st, request, out);
            }
            break;
        }

        case OPUS_GET_FINAL_RANGE_REQUEST: {
            opus_uint32 *out = va_arg(ap, opus_uint32 *);
            if (out == (void *)0) {
                ret = OPUS_BAD_ARG;
            } else {
                ret = mdopus_decoder_ctl_get_uint32(st, request, out);
            }
            break;
        }

        /* DNN blob: two args (pointer + length). Mirrors xiph's
         * `opus_decoder.c:1218` gated on USE_WEIGHTS_FILE + ENABLE_DEEP_PLC —
         * we always route the CTL; the Rust side's `set_dnn_blob` parses the
         * blob and dispatches named sections to `LPCNetPLCState::load_model`
         * (real weight-name dispatch ported in Stage 7b.1.5). */
        case OPUS_SET_DNN_BLOB_REQUEST: {
            const unsigned char *data = va_arg(ap, const unsigned char *);
            int len = va_arg(ap, int);
            ret = mdopus_decoder_ctl_set_dnn_blob(st, data, len);
            break;
        }

        default:
            ret = OPUS_UNIMPLEMENTED;
            break;
    }
    va_end(ap);
    return ret;
}

int opus_multistream_encoder_ctl(OpusMSEncoder *st, int request, ...)
{
    va_list ap;
    int ret;

    if (st == (void *)0) {
        return OPUS_BAD_ARG;
    }

    va_start(ap, request);
    switch (request) {
        case OPUS_RESET_STATE:
            ret = mdopus_ms_encoder_ctl_reset(st);
            break;

        case OPUS_SET_APPLICATION_REQUEST:
        case OPUS_SET_BITRATE_REQUEST:
        case OPUS_SET_MAX_BANDWIDTH_REQUEST:
        case OPUS_SET_VBR_REQUEST:
        case OPUS_SET_BANDWIDTH_REQUEST:
        case OPUS_SET_COMPLEXITY_REQUEST:
        case OPUS_SET_INBAND_FEC_REQUEST:
        case OPUS_SET_PACKET_LOSS_PERC_REQUEST:
        case OPUS_SET_DTX_REQUEST:
        case OPUS_SET_VBR_CONSTRAINT_REQUEST:
        case OPUS_SET_FORCE_CHANNELS_REQUEST:
        case OPUS_SET_SIGNAL_REQUEST:
        case OPUS_SET_LSB_DEPTH_REQUEST:
        case OPUS_SET_EXPERT_FRAME_DURATION_REQUEST:
        case OPUS_SET_PREDICTION_DISABLED_REQUEST:
        case OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST:
        case OPUS_SET_FORCE_MODE_REQUEST:
        case OPUS_SET_LFE_REQUEST:
            ret = mdopus_ms_encoder_ctl_set_int(st, request, va_arg(ap, int));
            break;

        case OPUS_GET_APPLICATION_REQUEST:
        case OPUS_GET_BITRATE_REQUEST:
        case OPUS_GET_MAX_BANDWIDTH_REQUEST:
        case OPUS_GET_VBR_REQUEST:
        case OPUS_GET_BANDWIDTH_REQUEST:
        case OPUS_GET_COMPLEXITY_REQUEST:
        case OPUS_GET_INBAND_FEC_REQUEST:
        case OPUS_GET_PACKET_LOSS_PERC_REQUEST:
        case OPUS_GET_DTX_REQUEST:
        case OPUS_GET_VBR_CONSTRAINT_REQUEST:
        case OPUS_GET_FORCE_CHANNELS_REQUEST:
        case OPUS_GET_SIGNAL_REQUEST:
        case OPUS_GET_LOOKAHEAD_REQUEST:
        case OPUS_GET_SAMPLE_RATE_REQUEST:
        case OPUS_GET_LSB_DEPTH_REQUEST:
        case OPUS_GET_EXPERT_FRAME_DURATION_REQUEST:
        case OPUS_GET_PREDICTION_DISABLED_REQUEST:
        case OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST: {
            int *out = va_arg(ap, int *);
            if (out == (void *)0) {
                ret = OPUS_BAD_ARG;
            } else {
                ret = mdopus_ms_encoder_ctl_get_int(st, request, out);
            }
            break;
        }

        case OPUS_GET_FINAL_RANGE_REQUEST: {
            opus_uint32 *out = va_arg(ap, opus_uint32 *);
            if (out == (void *)0) {
                ret = OPUS_BAD_ARG;
            } else {
                ret = mdopus_ms_encoder_ctl_get_uint32(st, request, out);
            }
            break;
        }

        case OPUS_MULTISTREAM_GET_ENCODER_STATE_REQUEST: {
            int stream_id = va_arg(ap, int);
            OpusEncoder **out = va_arg(ap, OpusEncoder **);
            if (out == (void *)0) {
                ret = OPUS_BAD_ARG;
            } else {
                ret = mdopus_ms_encoder_ctl_get_encoder_state(st, stream_id, out);
            }
            break;
        }

        /* MS-level energy mask: Rust sizes the slice as
         * `21 * layout.nb_channels` internally and fans out per sub-encoder
         * using layout-aware slicing (matches encode-time distribution in
         * `opus_multistream_encoder.c` L989-1014). */
        case OPUS_SET_ENERGY_MASK_REQUEST: {
            const int *ptr = va_arg(ap, const int *);
            ret = mdopus_ms_encoder_ctl_set_energy_mask(st, ptr);
            break;
        }

        default:
            ret = OPUS_UNIMPLEMENTED;
            break;
    }
    va_end(ap);
    return ret;
}

int opus_multistream_decoder_ctl(OpusMSDecoder *st, int request, ...)
{
    va_list ap;
    int ret;

    if (st == (void *)0) {
        return OPUS_BAD_ARG;
    }

    va_start(ap, request);
    switch (request) {
        case OPUS_RESET_STATE:
            ret = mdopus_ms_decoder_ctl_reset(st);
            break;

        case OPUS_SET_GAIN_REQUEST:
        case OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST:
            ret = mdopus_ms_decoder_ctl_set_int(st, request, va_arg(ap, int));
            break;

        case OPUS_GET_BANDWIDTH_REQUEST:
        case OPUS_GET_SAMPLE_RATE_REQUEST:
        case OPUS_GET_GAIN_REQUEST:
        case OPUS_GET_LAST_PACKET_DURATION_REQUEST:
        case OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST: {
            int *out = va_arg(ap, int *);
            if (out == (void *)0) {
                ret = OPUS_BAD_ARG;
            } else {
                ret = mdopus_ms_decoder_ctl_get_int(st, request, out);
            }
            break;
        }

        case OPUS_GET_FINAL_RANGE_REQUEST: {
            opus_uint32 *out = va_arg(ap, opus_uint32 *);
            if (out == (void *)0) {
                ret = OPUS_BAD_ARG;
            } else {
                ret = mdopus_ms_decoder_ctl_get_uint32(st, request, out);
            }
            break;
        }

        case OPUS_MULTISTREAM_GET_DECODER_STATE_REQUEST: {
            int stream_id = va_arg(ap, int);
            OpusDecoder **out = va_arg(ap, OpusDecoder **);
            if (out == (void *)0) {
                ret = OPUS_BAD_ARG;
            } else {
                ret = mdopus_ms_decoder_ctl_get_decoder_state(st, stream_id, out);
            }
            break;
        }

        default:
            ret = OPUS_UNIMPLEMENTED;
            break;
    }
    va_end(ap);
    return ret;
}

int opus_projection_encoder_ctl(OpusProjectionEncoder *st, int request, ...)
{
    va_list ap;
    int ret;

    if (st == (void *)0) {
        return OPUS_BAD_ARG;
    }

    va_start(ap, request);
    switch (request) {
        case OPUS_RESET_STATE:
            ret = mdopus_proj_encoder_ctl_reset(st);
            break;

        /* int-value setters exposed on projection encoder. Surface is
         * deliberately narrow: conformance exercises OPUS_SET_BITRATE
         * plus the three projection getters below. Other common int
         * setters (complexity, vbr) are routed for symmetry with the
         * existing MS encoder plumbing. */
        case OPUS_SET_BITRATE_REQUEST:
        case OPUS_SET_COMPLEXITY_REQUEST:
        case OPUS_SET_VBR_REQUEST:
            ret = mdopus_proj_encoder_ctl_set_int(st, request, va_arg(ap, int));
            break;

        case OPUS_GET_BITRATE_REQUEST:
        case OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST:
        case OPUS_PROJECTION_GET_DEMIXING_MATRIX_GAIN_REQUEST: {
            int *out = va_arg(ap, int *);
            if (out == (void *)0) {
                ret = OPUS_BAD_ARG;
            } else {
                ret = mdopus_proj_encoder_ctl_get_int(st, request, out);
            }
            break;
        }

        case OPUS_GET_FINAL_RANGE_REQUEST: {
            opus_uint32 *out = va_arg(ap, opus_uint32 *);
            if (out == (void *)0) {
                ret = OPUS_BAD_ARG;
            } else {
                ret = mdopus_proj_encoder_ctl_get_uint32(st, request, out);
            }
            break;
        }

        /* OPUS_PROJECTION_GET_DEMIXING_MATRIX(ptr, size) — two-argument CTL.
         * `ptr` is an `unsigned char *` writable buffer, `size` is an
         * `opus_int32` advertising its length. Rust-side copies exactly
         * `size` bytes and returns OPUS_BAD_ARG if the size differs from
         * `get_demixing_matrix_size()`. */
        case OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST: {
            unsigned char *out = va_arg(ap, unsigned char *);
            int size = va_arg(ap, int);
            ret = mdopus_proj_encoder_ctl_get_demixing_matrix(st, out, size);
            break;
        }

        default:
            ret = OPUS_UNIMPLEMENTED;
            break;
    }
    va_end(ap);
    return ret;
}

int opus_projection_decoder_ctl(OpusProjectionDecoder *st, int request, ...)
{
    va_list ap;
    int ret;

    if (st == (void *)0) {
        return OPUS_BAD_ARG;
    }

    va_start(ap, request);
    switch (request) {
        case OPUS_RESET_STATE:
            ret = mdopus_proj_decoder_ctl_reset(st);
            break;

        case OPUS_SET_GAIN_REQUEST:
            ret = mdopus_proj_decoder_ctl_set_int(st, request, va_arg(ap, int));
            break;

        case OPUS_GET_SAMPLE_RATE_REQUEST:
        case OPUS_GET_GAIN_REQUEST: {
            int *out = va_arg(ap, int *);
            if (out == (void *)0) {
                ret = OPUS_BAD_ARG;
            } else {
                ret = mdopus_proj_decoder_ctl_get_int(st, request, out);
            }
            break;
        }

        case OPUS_GET_FINAL_RANGE_REQUEST: {
            opus_uint32 *out = va_arg(ap, opus_uint32 *);
            if (out == (void *)0) {
                ret = OPUS_BAD_ARG;
            } else {
                ret = mdopus_proj_decoder_ctl_get_uint32(st, request, out);
            }
            break;
        }

        default:
            ret = OPUS_UNIMPLEMENTED;
            break;
    }
    va_end(ap);
    return ret;
}
