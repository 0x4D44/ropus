/*
 * Stage 7b.3 diagnostic peek into C-reference internals.
 *
 * The tier-2 harness links the xiph C ref as a static library in float mode
 * with ENABLE_DEEP_PLC=1. These pure-read getter functions let the Rust-side
 * diagnostic code extract state from a live C OpusDecoder opaque pointer so
 * we can compare C state against ropus state at loss/recovery boundaries.
 *
 * This file is compiled into the same static lib as the C ref (in
 * harness-deep-plc/build.rs), so it sees the private opus_decoder internals
 * via the same header include path. The reference tree itself stays read-only.
 */

#include "opus_types.h"
#include "opus_defines.h"
#include "arch.h"
#include "modes.h"
#include "structs.h" /* silk_decoder_state + silk_PLC_struct */

/*
 * silk_decoder super-struct (from dec_API.c:47-56). Not exposed in a public
 * header, so we replicate the prefix. Only the `channel_state[0]` prefix is
 * needed — the sStereo / counts come after DECODER_NUM_CHANNELS entries.
 *
 * DECODER_NUM_CHANNELS = 2 (define_stream_structs.h). Must match xiph's
 * configure to keep sizeof(channel_state) in agreement.
 */

/* ---- CELT private layout ------------------------------------------------- */
/*
 * Mirror of `struct OpusCustomDecoder` from celt_decoder.c:87 — only the prefix
 * up to `_decode_mem[1]`. We don't use this to write, only to compute offsets
 * to the variable trailing arrays. Must match celt_decoder.c byte-for-byte.
 */

typedef struct PeekCeltHdr {
    const OpusCustomMode *mode;
    int overlap;
    int channels;
    int stream_channels;
    int downsample;
    int start;
    int end;
    int signalling;
    int disable_inv;
    int complexity;
    int arch;

    opus_uint32 rng;
    int error;
    int last_pitch_index;
    int loss_duration;
    int plc_duration;
    int last_frame_type;
    int skip_plc;
    int postfilter_period;
    int postfilter_period_old;
    opus_val16 postfilter_gain;
    opus_val16 postfilter_gain_old;
    int postfilter_tapset;
    int postfilter_tapset_old;
    int prefilter_and_fold;

    celt_sig preemph_memD[2];

#ifdef ENABLE_DEEP_PLC
    /* PLC_UPDATE_SAMPLES = PLC_UPDATE_FRAMES * FRAME_SIZE = 4 * 160 = 640
     * (celt_decoder.c:81-82, FRAME_SIZE defined in dnn/freq.h). */
    opus_int16 plc_pcm[640 /* PLC_UPDATE_SAMPLES */];
    int plc_fill;
    float plc_preemphasis_mem;
#endif

    celt_sig _decode_mem[1];
} PeekCeltHdr;

/*
 * OpusDecoder shape from src/opus_decoder.c:65. Only the prefix is fixed;
 * we just need celt_dec_offset and silk_dec_offset — the first two ints.
 */

int peek_opus_celt_offset(const void *st) {
    const int *p = (const int *)st;
    return p[0]; /* celt_dec_offset */
}

int peek_opus_silk_offset(const void *st) {
    const int *p = (const int *)st;
    return p[1]; /* silk_dec_offset */
}

/* ---- CELT state getters -------------------------------------------------- */

int peek_celt_channels(const void *celt_st) {
    const PeekCeltHdr *h = (const PeekCeltHdr *)celt_st;
    return h->channels;
}

int peek_celt_overlap(const void *celt_st) {
    const PeekCeltHdr *h = (const PeekCeltHdr *)celt_st;
    return h->overlap;
}

/* `celt_sig` is float in float-mode. */
int peek_celt_decode_mem(const void *celt_st, int offset, int count, float *out) {
    const PeekCeltHdr *h = (const PeekCeltHdr *)celt_st;
    const celt_sig *buf = h->_decode_mem;
    for (int i = 0; i < count; i++) {
        out[i] = (float)buf[offset + i];
    }
    return count;
}

#define PEEK_DECODE_BUFFER_SIZE 2048

static const celt_glog *peek_celt_trailing_base(const PeekCeltHdr *h) {
    int per_ch = PEEK_DECODE_BUFFER_SIZE + h->overlap;
    int total = per_ch * h->channels;
    return (const celt_glog *)(h->_decode_mem + total);
}

int peek_celt_nb_ebands(const void *celt_st) {
    const PeekCeltHdr *h = (const PeekCeltHdr *)celt_st;
    return h->mode->nbEBands;
}

int peek_celt_old_band_e(const void *celt_st, int offset, int count, float *out) {
    const PeekCeltHdr *h = (const PeekCeltHdr *)celt_st;
    const celt_glog *base = peek_celt_trailing_base(h);
    for (int i = 0; i < count; i++) {
        out[i] = (float)base[offset + i];
    }
    return count;
}

int peek_celt_old_log_e(const void *celt_st, int offset, int count, float *out) {
    const PeekCeltHdr *h = (const PeekCeltHdr *)celt_st;
    int nb = h->mode->nbEBands;
    const celt_glog *base = peek_celt_trailing_base(h) + 2 * nb;
    for (int i = 0; i < count; i++) {
        out[i] = (float)base[offset + i];
    }
    return count;
}

int peek_celt_old_log_e2(const void *celt_st, int offset, int count, float *out) {
    const PeekCeltHdr *h = (const PeekCeltHdr *)celt_st;
    int nb = h->mode->nbEBands;
    const celt_glog *base = peek_celt_trailing_base(h) + 4 * nb;
    for (int i = 0; i < count; i++) {
        out[i] = (float)base[offset + i];
    }
    return count;
}

int peek_celt_background_log_e(const void *celt_st, int offset, int count, float *out) {
    const PeekCeltHdr *h = (const PeekCeltHdr *)celt_st;
    int nb = h->mode->nbEBands;
    const celt_glog *base = peek_celt_trailing_base(h) + 6 * nb;
    for (int i = 0; i < count; i++) {
        out[i] = (float)base[offset + i];
    }
    return count;
}

/* ---- SILK state getters -------------------------------------------------- */
/*
 * The `silk_decoder` super-struct (dec_API.c:47-56) has `channel_state` as
 * its first field — so the silk_dec pointer we get via opus's silk_dec_offset
 * is already the start of channel_state[0].
 */

/* Returns decoder sample rate in kHz. Useful to see when fs_kHz drifts. */
int peek_silk_fs_khz(const void *silk_dec) {
    const silk_decoder_state *s = (const silk_decoder_state *)silk_dec;
    return s->fs_kHz;
}

/* Last-frame global gain (SILK_DECODER_STATE_RESET_START = prev_gain_Q16). */
opus_int32 peek_silk_prev_gain_q16(const void *silk_dec) {
    const silk_decoder_state *s = (const silk_decoder_state *)silk_dec;
    return s->prev_gain_Q16;
}

/* Copy out the LPC synthesis history buffer. MAX_LPC_ORDER = 16 entries. */
int peek_silk_sLPC_Q14(const void *silk_dec, opus_int32 *out, int max_count) {
    const silk_decoder_state *s = (const silk_decoder_state *)silk_dec;
    int n = max_count < MAX_LPC_ORDER ? max_count : MAX_LPC_ORDER;
    for (int i = 0; i < n; i++) out[i] = s->sLPC_Q14_buf[i];
    return n;
}

/* PLC sub-struct: prevGain_Q16[2] — per-subframe smoothed gain. */
int peek_silk_plc_prev_gain(const void *silk_dec, opus_int32 *out) {
    const silk_decoder_state *s = (const silk_decoder_state *)silk_dec;
    out[0] = s->sPLC.prevGain_Q16[0];
    out[1] = s->sPLC.prevGain_Q16[1];
    return 2;
}

opus_int32 peek_silk_plc_pitch_l_q8(const void *silk_dec) {
    const silk_decoder_state *s = (const silk_decoder_state *)silk_dec;
    return s->sPLC.pitchL_Q8;
}

opus_int32 peek_silk_plc_rand_scale_q14(const void *silk_dec) {
    const silk_decoder_state *s = (const silk_decoder_state *)silk_dec;
    return (opus_int32)s->sPLC.randScale_Q14;
}

int peek_silk_plc_last_frame_lost(const void *silk_dec) {
    const silk_decoder_state *s = (const silk_decoder_state *)silk_dec;
    return s->sPLC.last_frame_lost;
}

int peek_silk_plc_fs_khz(const void *silk_dec) {
    const silk_decoder_state *s = (const silk_decoder_state *)silk_dec;
    return s->sPLC.fs_kHz;
}

/* Copy `count` samples from the SILK output buffer. outBuf has length
 * MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH. */
int peek_silk_out_buf(const void *silk_dec, int offset, int count, opus_int16 *out) {
    const silk_decoder_state *s = (const silk_decoder_state *)silk_dec;
    for (int i = 0; i < count; i++) out[i] = s->outBuf[offset + i];
    return count;
}

int peek_silk_ltp_mem_length(const void *silk_dec) {
    const silk_decoder_state *s = (const silk_decoder_state *)silk_dec;
    return s->ltp_mem_length;
}

int peek_silk_frame_length(const void *silk_dec) {
    const silk_decoder_state *s = (const silk_decoder_state *)silk_dec;
    return s->frame_length;
}

/* ---- Top-level wrappers: take OpusDecoder*, find CELT/SILK, call inner --- */

static const void *celt_from_opus(const void *opus_st) {
    const char *base = (const char *)opus_st;
    int off = peek_opus_celt_offset(opus_st);
    return (const void *)(base + off);
}

static const void *silk_from_opus(const void *opus_st) {
    const char *base = (const char *)opus_st;
    int off = peek_opus_silk_offset(opus_st);
    return (const void *)(base + off);
}

int peek_decode_mem(const void *opus_st, int offset, int count, float *out) {
    return peek_celt_decode_mem(celt_from_opus(opus_st), offset, count, out);
}

int peek_decode_mem_stride(const void *opus_st) {
    const void *c = celt_from_opus(opus_st);
    return PEEK_DECODE_BUFFER_SIZE + peek_celt_overlap(c);
}

int peek_old_band_e(const void *opus_st, int offset, int count, float *out) {
    return peek_celt_old_band_e(celt_from_opus(opus_st), offset, count, out);
}

int peek_old_log_e(const void *opus_st, int offset, int count, float *out) {
    return peek_celt_old_log_e(celt_from_opus(opus_st), offset, count, out);
}

int peek_background_log_e(const void *opus_st, int offset, int count, float *out) {
    return peek_celt_background_log_e(celt_from_opus(opus_st), offset, count, out);
}

int peek_nb_ebands(const void *opus_st) {
    return peek_celt_nb_ebands(celt_from_opus(opus_st));
}

/* SILK wrappers */

int peek_silk_fs_khz_top(const void *opus_st) {
    return peek_silk_fs_khz(silk_from_opus(opus_st));
}
opus_int32 peek_silk_prev_gain(const void *opus_st) {
    return peek_silk_prev_gain_q16(silk_from_opus(opus_st));
}
int peek_silk_s_lpc_q14(const void *opus_st, opus_int32 *out, int max_count) {
    return peek_silk_sLPC_Q14(silk_from_opus(opus_st), out, max_count);
}
int peek_silk_plc_prev_gain_top(const void *opus_st, opus_int32 *out) {
    return peek_silk_plc_prev_gain(silk_from_opus(opus_st), out);
}
opus_int32 peek_silk_plc_pitch(const void *opus_st) {
    return peek_silk_plc_pitch_l_q8(silk_from_opus(opus_st));
}
opus_int32 peek_silk_plc_rand_scale(const void *opus_st) {
    return peek_silk_plc_rand_scale_q14(silk_from_opus(opus_st));
}
int peek_silk_plc_last_lost(const void *opus_st) {
    return peek_silk_plc_last_frame_lost(silk_from_opus(opus_st));
}
int peek_silk_plc_fs(const void *opus_st) {
    return peek_silk_plc_fs_khz(silk_from_opus(opus_st));
}
int peek_silk_outbuf(const void *opus_st, int offset, int count, opus_int16 *out) {
    return peek_silk_out_buf(silk_from_opus(opus_st), offset, count, out);
}
int peek_silk_ltpmem(const void *opus_st) {
    return peek_silk_ltp_mem_length(silk_from_opus(opus_st));
}
int peek_silk_framelen(const void *opus_st) {
    return peek_silk_frame_length(silk_from_opus(opus_st));
}
