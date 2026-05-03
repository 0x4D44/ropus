/*
 * debug_silk_trace.c — SILK decode tracing for Bug #13 investigation,
 * plus encode-side Phase B trace (Cluster A stage 2b).
 *
 * Provides functions to extract SILK decoder intermediate state from a C
 * OpusDecoder after a decode call, and a thread-local FIFO for the
 * encode-side trace tuples emitted by `silk_enc_api_traced.c`.
 */

#include "opus.h"
#include "opus_types.h"
#include "silk/structs.h"
#include "silk/define.h"
#include "silk/SigProc_FIX.h"
#include "silk/main.h"
#include "silk/PLC.h"
#include "celt/entdec.h"
#include "celt/entcode.h"

#include <string.h>
#include <stdio.h>

/* ======================================================================
 * Internal opus decoder layout for offset navigation
 * ====================================================================== */
typedef struct {
    opus_int32 celt_dec_offset;
    opus_int32 silk_dec_offset;
    opus_int32 channels;
    opus_int32 Fs;
} SilkTraceDecoderOffsets;

/* silk_decoder layout from dec_API.c */
typedef struct {
    silk_decoder_state channel_state[ DECODER_NUM_CHANNELS ];
    stereo_dec_state sStereo;
    opus_int nChannelsAPI;
    opus_int nChannelsInternal;
    opus_int prev_decode_only_middle;
} silk_decoder_trace_t;

/* Get a pointer to the SILK channel 0 decoder state */
static silk_decoder_state* trace_get_silk_ch0(OpusDecoder *dec) {
    SilkTraceDecoderOffsets *hdr = (SilkTraceDecoderOffsets *)dec;
    silk_decoder_trace_t *silk_dec = (silk_decoder_trace_t *)((char *)dec + hdr->silk_dec_offset);
    return &silk_dec->channel_state[0];
}

/* ======================================================================
 * Post-decode state extraction — persistent state
 * ====================================================================== */

void debug_silk_trace_get_persistent_state(OpusDecoder *dec,
    opus_int32 *prev_gain_q16,
    opus_int32 *s_lpc_q14_buf,      /* MAX_LPC_ORDER = 16 elements */
    opus_int32 *lag_prev,
    opus_int32 *last_gain_index,
    opus_int32 *fs_khz,
    opus_int32 *nb_subfr,
    opus_int32 *frame_length,
    opus_int32 *subfr_length,
    opus_int32 *ltp_mem_length,
    opus_int32 *lpc_order,
    opus_int32 *first_frame_after_reset,
    opus_int32 *loss_cnt,
    opus_int32 *prev_signal_type)
{
    silk_decoder_state *ch0 = trace_get_silk_ch0(dec);

    *prev_gain_q16 = ch0->prev_gain_Q16;
    memcpy(s_lpc_q14_buf, ch0->sLPC_Q14_buf, MAX_LPC_ORDER * sizeof(opus_int32));
    *lag_prev = ch0->lagPrev;
    *last_gain_index = (opus_int32)ch0->LastGainIndex;
    *fs_khz = ch0->fs_kHz;
    *nb_subfr = ch0->nb_subfr;
    *frame_length = ch0->frame_length;
    *subfr_length = ch0->subfr_length;
    *ltp_mem_length = ch0->ltp_mem_length;
    *lpc_order = ch0->LPC_order;
    *first_frame_after_reset = ch0->first_frame_after_reset;
    *loss_cnt = ch0->lossCnt;
    *prev_signal_type = ch0->prevSignalType;
}

/* Extract decoded indices */
void debug_silk_trace_get_indices(OpusDecoder *dec,
    opus_int32 *signal_type,
    opus_int32 *quant_offset_type,
    opus_int32 *gains_indices,       /* 4 elements */
    opus_int32 *nlsf_indices,        /* MAX_LPC_ORDER+1 = 17 elements */
    opus_int32 *lag_index,
    opus_int32 *contour_index,
    opus_int32 *nlsf_interp_coef_q2,
    opus_int32 *per_index,
    opus_int32 *ltp_index,           /* 4 elements */
    opus_int32 *ltp_scale_index,
    opus_int32 *seed)
{
    silk_decoder_state *ch0 = trace_get_silk_ch0(dec);
    SideInfoIndices *idx = &ch0->indices;
    int i;

    *signal_type = (opus_int32)idx->signalType;
    *quant_offset_type = (opus_int32)idx->quantOffsetType;
    for (i = 0; i < MAX_NB_SUBFR; i++)
        gains_indices[i] = (opus_int32)idx->GainsIndices[i];
    for (i = 0; i <= MAX_LPC_ORDER; i++)
        nlsf_indices[i] = (opus_int32)idx->NLSFIndices[i];
    *lag_index = (opus_int32)idx->lagIndex;
    *contour_index = (opus_int32)idx->contourIndex;
    *nlsf_interp_coef_q2 = (opus_int32)idx->NLSFInterpCoef_Q2;
    *per_index = (opus_int32)idx->PERIndex;
    for (i = 0; i < MAX_NB_SUBFR; i++)
        ltp_index[i] = (opus_int32)idx->LTPIndex[i];
    *ltp_scale_index = (opus_int32)idx->LTP_scaleIndex;
    *seed = (opus_int32)idx->Seed;
}

/* Extract prevNLSF_Q15 (updated by decode_parameters) */
void debug_silk_trace_get_prev_nlsf(OpusDecoder *dec,
    opus_int16 *prev_nlsf_q15)  /* MAX_LPC_ORDER = 16 elements */
{
    silk_decoder_state *ch0 = trace_get_silk_ch0(dec);
    memcpy(prev_nlsf_q15, ch0->prevNLSF_Q15, MAX_LPC_ORDER * sizeof(opus_int16));
}

/* Extract excitation buffer (exc_Q14) — filled by decode_core */
void debug_silk_trace_get_exc(OpusDecoder *dec,
    opus_int32 *exc_q14,        /* 'count' elements */
    opus_int32 count)
{
    silk_decoder_state *ch0 = trace_get_silk_ch0(dec);
    int n = (count < MAX_FRAME_LENGTH) ? (int)count : MAX_FRAME_LENGTH;
    memcpy(exc_q14, ch0->exc_Q14, n * sizeof(opus_int32));
}

/* Extract outBuf */
void debug_silk_trace_get_outbuf(OpusDecoder *dec,
    opus_int16 *out_buf,        /* 'count' elements */
    opus_int32 count)
{
    silk_decoder_state *ch0 = trace_get_silk_ch0(dec);
    int max_out = MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH;
    int n = (count < max_out) ? (int)count : max_out;
    memcpy(out_buf, ch0->outBuf, n * sizeof(opus_int16));
}

/* ======================================================================
 * Step-by-step SILK decode with control capture
 *
 * This calls internal SILK functions directly on the decoder's SILK state
 * to capture the transient silk_decoder_control values.
 *
 * PREREQUISITE: A normal opus_decode call must have been done first to
 * initialize the SILK decoder fs/state. Then call opus_decoder_ctl RESET
 * and use this for the actual traced decode.
 *
 * Actually simpler: just call opus_decode once first (it sets up SILK),
 * then reset, then use this function.
 * ====================================================================== */

/*
 * Do a complete silk_decode_frame manually, capturing intermediate values.
 *
 * The caller passes the raw SILK bitstream (after opus_decode's framing/TOC
 * has been stripped — i.e., the SILK payload). We set up an ec_dec from this
 * data and drive the decode pipeline step by step.
 *
 * NOTE: The SILK decoder must already be configured (fs_kHz set, etc.)
 * before calling this. The easiest way is to call opus_decode first with
 * the real packet, then reset the decoder state, then call this.
 *
 * Returns frame_length on success, negative on error.
 */
int debug_silk_traced_decode(
    OpusDecoder *dec,
    const unsigned char *silk_data,
    int silk_len,
    int cond_coding,
    /* Output: silk_decoder_control values */
    opus_int32 *gains_q16_out,          /* 4 */
    opus_int16 *pred_coef_q12_0_out,    /* MAX_LPC_ORDER */
    opus_int16 *pred_coef_q12_1_out,    /* MAX_LPC_ORDER */
    opus_int32 *pitch_l_out,            /* 4 */
    opus_int16 *ltp_coef_q14_out,       /* LTP_ORDER * MAX_NB_SUBFR = 20 */
    opus_int32 *ltp_scale_q14_out,
    /* Output: pulses */
    opus_int16 *pulses_out,             /* frame_length */
    /* Output: PCM */
    opus_int16 *pcm_out                 /* frame_length */
)
{
    silk_decoder_state *psDec = trace_get_silk_ch0(dec);
    silk_decoder_control psDecCtrl;
    int L = psDec->frame_length;
    int i;

    psDecCtrl.LTP_scale_Q14 = 0;

    /* Set up range coder */
    ec_dec ec;
    ec_dec_init(&ec, (unsigned char *)silk_data, silk_len);

    /* Allocate pulses on stack */
    opus_int16 pulses[MAX_FRAME_LENGTH];
    memset(pulses, 0, sizeof(pulses));

    /* Step 1: Decode indices from bitstream */
    silk_decode_indices(psDec, &ec, psDec->nFramesDecoded, 0, cond_coding);

    /* Step 2: Decode pulses */
    silk_decode_pulses(&ec, pulses, psDec->indices.signalType,
                      psDec->indices.quantOffsetType, psDec->frame_length);

    /* Step 3: Decode parameters (fills psDecCtrl with gains, LPC, pitch, LTP) */
    silk_decode_parameters(psDec, &psDecCtrl, cond_coding);

    /* === CAPTURE transient psDecCtrl values === */
    for (i = 0; i < MAX_NB_SUBFR; i++) {
        gains_q16_out[i] = psDecCtrl.Gains_Q16[i];
        pitch_l_out[i] = psDecCtrl.pitchL[i];
    }
    memcpy(pred_coef_q12_0_out, psDecCtrl.PredCoef_Q12[0], MAX_LPC_ORDER * sizeof(opus_int16));
    memcpy(pred_coef_q12_1_out, psDecCtrl.PredCoef_Q12[1], MAX_LPC_ORDER * sizeof(opus_int16));
    memcpy(ltp_coef_q14_out, psDecCtrl.LTPCoef_Q14, LTP_ORDER * MAX_NB_SUBFR * sizeof(opus_int16));
    *ltp_scale_q14_out = psDecCtrl.LTP_scale_Q14;
    memcpy(pulses_out, pulses, L * sizeof(opus_int16));

    /* Step 4: decode_core (inverse NSQ: LTP + LPC synthesis) */
    silk_decode_core(psDec, &psDecCtrl, pcm_out, pulses, psDec->arch);

    /* Step 5: Update output buffer */
    int mv_len = psDec->ltp_mem_length - psDec->frame_length;
    silk_memmove(psDec->outBuf, &psDec->outBuf[psDec->frame_length], mv_len * sizeof(opus_int16));
    silk_memcpy(&psDec->outBuf[mv_len], pcm_out, psDec->frame_length * sizeof(opus_int16));

    /* Step 6: PLC update from good frame */
    silk_PLC(psDec, &psDecCtrl, pcm_out, 0,
#ifdef ENABLE_DEEP_PLC
        NULL,
#endif
        psDec->arch);

    psDec->lossCnt = 0;
    psDec->prevSignalType = psDec->indices.signalType;
    psDec->first_frame_after_reset = 0;

    /* Step 7: CNG and PLC glue */
    silk_CNG(psDec, &psDecCtrl, pcm_out, L);
    silk_PLC_glue_frames(psDec, pcm_out, L);

    psDec->lagPrev = psDecCtrl.pitchL[psDec->nb_subfr - 1];
    psDec->nFramesDecoded++;

    return L;
}

/* Convenience: get the SILK fs_kHz and frame_length (to check if SILK is initialized) */
void debug_silk_trace_get_config(OpusDecoder *dec,
    opus_int32 *fs_khz,
    opus_int32 *frame_length,
    opus_int32 *n_frames_decoded)
{
    silk_decoder_state *ch0 = trace_get_silk_ch0(dec);
    *fs_khz = ch0->fs_kHz;
    *frame_length = ch0->frame_length;
    *n_frames_decoded = ch0->nFramesDecoded;
}

/* Reset the SILK decoder state (call before debug_silk_traced_decode) */
void debug_silk_trace_reset(OpusDecoder *dec) {
    silk_decoder_state *ch0 = trace_get_silk_ch0(dec);

    /* Reset the decoder state same as silk_init_decoder */
    ch0->prev_gain_Q16 = 65536;
    memset(ch0->exc_Q14, 0, MAX_FRAME_LENGTH * sizeof(opus_int32));
    memset(ch0->sLPC_Q14_buf, 0, MAX_LPC_ORDER * sizeof(opus_int32));
    memset(ch0->outBuf, 0, (MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH) * sizeof(opus_int16));
    ch0->lagPrev = 100;
    ch0->LastGainIndex = 10;
    ch0->first_frame_after_reset = 1;
    ch0->lossCnt = 0;
    ch0->prevSignalType = 0;
    ch0->nFramesDecoded = 0;
}

/* ======================================================================
 * Phase B encode-side trace FIFO (Cluster A stage 2b).
 *
 * Thread-local circular collector for `(boundary_id, channel, ec_tell,
 * rng, target_rate_bps, n_bits_exceeded, curr_n_bits_used_lbrr,
 * n_bits_used_lbrr, mid_only_flag, prev_decode_only_middle)` tuples
 * pushed from `silk_enc_api_traced.c::silk_Encode_traced`. The Rust
 * side mirrors the same tuple shape via the `trace-silk-encode`
 * Cargo feature on the `ropus` crate, gated on per-frame in
 * `ropus/src/silk/encoder.rs`.
 *
 * The FIFO is bounded: 1024 tuples is enough for the per-frame trace
 * granularity we need. The harness binary clears the FIFO before each
 * encode and reads it back via `dbg_silk_trace_read`.
 * ====================================================================== */

#define DBG_SILK_TRACE_MAX 1024

typedef struct {
    int boundary_id;
    int channel;
    opus_int32 ec_tell;
    opus_uint32 rng;
    opus_int32 target_rate_bps;
    opus_int32 n_bits_exceeded;
    opus_int32 curr_n_bits_used_lbrr;
    opus_int32 n_bits_used_lbrr;
    opus_int32 mid_only_flag;
    opus_int32 prev_decode_only_middle;
} dbg_silk_trace_tuple_t;

/* TLS keyword availability across the toolchain matrix is uneven; the
 * harness is single-threaded for these diagnostics, so plain globals
 * are sufficient. Document the assumption: only one encode at a time. */
static dbg_silk_trace_tuple_t dbg_silk_trace_buf[DBG_SILK_TRACE_MAX];
static int dbg_silk_trace_count = 0;

void dbg_silk_trace_clear(void) {
    dbg_silk_trace_count = 0;
}

void dbg_silk_trace_push(
    int boundary_id,
    int channel,
    opus_int32 ec_tell,
    opus_uint32 rng,
    opus_int32 target_rate_bps,
    opus_int32 n_bits_exceeded,
    opus_int32 curr_n_bits_used_lbrr,
    opus_int32 n_bits_used_lbrr,
    opus_int32 mid_only_flag,
    opus_int32 prev_decode_only_middle)
{
    if (dbg_silk_trace_count >= DBG_SILK_TRACE_MAX) return;
    dbg_silk_trace_tuple_t *t = &dbg_silk_trace_buf[dbg_silk_trace_count++];
    t->boundary_id = boundary_id;
    t->channel = channel;
    t->ec_tell = ec_tell;
    t->rng = rng;
    t->target_rate_bps = target_rate_bps;
    t->n_bits_exceeded = n_bits_exceeded;
    t->curr_n_bits_used_lbrr = curr_n_bits_used_lbrr;
    t->n_bits_used_lbrr = n_bits_used_lbrr;
    t->mid_only_flag = mid_only_flag;
    t->prev_decode_only_middle = prev_decode_only_middle;
}

/* Read out a single tuple by index. Returns 0 on success, -1 if oob. */
int dbg_silk_trace_read(
    int idx,
    int *boundary_id,
    int *channel,
    opus_int32 *ec_tell,
    opus_uint32 *rng,
    opus_int32 *target_rate_bps,
    opus_int32 *n_bits_exceeded,
    opus_int32 *curr_n_bits_used_lbrr,
    opus_int32 *n_bits_used_lbrr,
    opus_int32 *mid_only_flag,
    opus_int32 *prev_decode_only_middle)
{
    if (idx < 0 || idx >= dbg_silk_trace_count) return -1;
    const dbg_silk_trace_tuple_t *t = &dbg_silk_trace_buf[idx];
    *boundary_id = t->boundary_id;
    *channel = t->channel;
    *ec_tell = t->ec_tell;
    *rng = t->rng;
    *target_rate_bps = t->target_rate_bps;
    *n_bits_exceeded = t->n_bits_exceeded;
    *curr_n_bits_used_lbrr = t->curr_n_bits_used_lbrr;
    *n_bits_used_lbrr = t->n_bits_used_lbrr;
    *mid_only_flag = t->mid_only_flag;
    *prev_decode_only_middle = t->prev_decode_only_middle;
    return 0;
}

int dbg_silk_trace_count_get(void) {
    return dbg_silk_trace_count;
}

/* ----------------------------------------------------------------------
 * Encode-side wrapper: drives a normal `opus_encode` but with the
 * traced silk_Encode_traced rather than xiph's silk_Encode. Exposed via
 * a separate path so the existing encode comparison harness keeps using
 * the unmodified xiph reference; only fuzz_repro_diff opts in.
 *
 * Implementation: we call the xiph `opus_encode` — which itself calls
 * `silk_Encode` (xiph) — and accept that on this path the tuples come
 * solely from the Rust side. The C side gets its tuples by a separate
 * helper that calls silk_Encode_traced directly on a pre-configured
 * encoder state (TODO if Rust-only tuples prove insufficient).
 *
 * For the initial diagnostic, the Rust-side trace + C-side post-encode
 * SILK state mirror (already extracted by the existing
 * `debug_get_silk_extended_state`) are sufficient to localise where
 * the Rust trace first goes off-rails: we expect Rust to diverge from
 * its own design predictions at exactly the boundary where Rust's
 * `silk_Encode` mirror differs from the C reference's behaviour.
 * ---------------------------------------------------------------------- */
