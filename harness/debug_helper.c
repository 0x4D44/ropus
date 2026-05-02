/* Debug helpers for extracting internal state from C reference encoder/decoder. */

#include "opus.h"
#include "opus_types.h"
#include "silk/fixed/structs_FIX.h"
#include "silk/SigProc_FIX.h"
#include "celt/celt.h"
#include "celt/modes.h"
#include "celt/mathops.h"
#include "celt/vq.h"
#include "celt/mdct.h"
#include "celt/cpu_support.h"
#include "src/analysis.h"
#include "opus_private.h" /* for struct OpusMSEncoder + align() helper */

#include <stdio.h>
#include <stddef.h>

/* Internal opus_encoder layout - first two i32 fields are offsets */
typedef struct {
    opus_int32 celt_enc_offset;
    opus_int32 silk_enc_offset;
} OpusEncoderOffsets;

/* Internal opus_decoder layout - first two i32 fields are offsets */
typedef struct {
    opus_int32 celt_dec_offset;
    opus_int32 silk_dec_offset;
    opus_int32 channels;
    opus_int32 Fs;
} OpusDecoderOffsets;

/* Forward declarations for SILK functions we want to test */
extern opus_int32 silk_lin2log(const opus_int32 inLin);
extern void silk_gains_quant(
    opus_int8 ind[], opus_int32 gain_Q16[], opus_int8 *prev_ind,
    const opus_int conditional, const opus_int nb_subfr);

void debug_test_gains_quant(void) {
    opus_int32 gains[4] = {115456, 115456, 115456, 115456};
    opus_int8 ind[4] = {0};
    opus_int8 prev_ind = 10;

    fprintf(stderr, "[C TEST] silk_lin2log(115456)=%d\n", silk_lin2log(115456));

    silk_gains_quant(ind, gains, &prev_ind, 0, 4);
    fprintf(stderr, "[C TEST] gains_quant result: ind=[%d, %d, %d, %d] final_prev=%d\n",
        (int)ind[0], (int)ind[1], (int)ind[2], (int)ind[3], (int)prev_ind);
    fprintf(stderr, "[C TEST] gains_q16_after=[%d, %d, %d, %d]\n",
        gains[0], gains[1], gains[2], gains[3]);
}

/* ======================================================================
 * CELT decoder trace: dump oldBandE after opus_decode
 * ======================================================================
 *
 * Replicate the CELTDecoder struct layout (fixed-point, no QEXT, no DEEP_PLC)
 * so we can compute the offset of oldBandE without modifying the C reference.
 */
struct CELTDecoder_trace {
    const OpusCustomMode *mode;
    int overlap;
    int channels;
    int stream_channels;
    int downsample;
    int start, end;
    int signalling;
    int disable_inv;
    int complexity;
    int arch;
    /* DECODER_RESET_START */
    opus_uint32 rng;
    int error;
    int last_pitch_index;
    int loss_duration;
    int plc_duration;
    int last_frame_type;
    int skip_plc;
    int postfilter_period;
    int postfilter_period_old;
    opus_int16 postfilter_gain;
    opus_int16 postfilter_gain_old;
    int postfilter_tapset;
    int postfilter_tapset_old;
    int prefilter_and_fold;
    opus_int32 preemph_memD[2];
    opus_int32 decode_mem[1]; /* variable-length */
};

void debug_dump_celt_decoder_bands(OpusDecoder *opus_dec) {
    OpusDecoderOffsets *hdr = (OpusDecoderOffsets *)opus_dec;
    struct CELTDecoder_trace *celt =
        (struct CELTDecoder_trace *)((char *)opus_dec + hdr->celt_dec_offset);

    int nbEBands = celt->mode->nbEBands;
    int overlap = celt->overlap;
    int CC = hdr->channels;
    int decode_buffer_size = 2048; /* DEC_PITCH_BUF_SIZE, non-QEXT */

    /* oldBandE follows _decode_mem, same formula as celt_decoder.c line 1183 */
    opus_int32 *oldBandE = celt->decode_mem + (decode_buffer_size + overlap) * CC;

    fprintf(stderr, "[C  DEC] oldBandE[0..10]:");
    {
        int i;
        for (i = 0; i < 10 && i < 2 * nbEBands; i++) {
            fprintf(stderr, " %d", (int)oldBandE[i]);
        }
    }
    fprintf(stderr, "\n");
}

void debug_dump_silk_indices(OpusEncoder *enc) {
    OpusEncoderOffsets *offsets = (OpusEncoderOffsets *)enc;
    silk_encoder *silk = (silk_encoder *)((char *)enc + offsets->silk_enc_offset);
    silk_encoder_state *state = &silk->state_Fxx[0].sCmn;

    fprintf(stderr, "[C SILK] speech_activity_Q8=%d signal_type=%d quant_offset=%d\n",
        state->speech_activity_Q8,
        (int)state->indices.signalType,
        (int)state->indices.quantOffsetType);
    fprintf(stderr, "[C SILK] gains_indices=[%d, %d, %d, %d] nb_subfr=%d\n",
        (int)state->indices.GainsIndices[0],
        (int)state->indices.GainsIndices[1],
        (int)state->indices.GainsIndices[2],
        (int)state->indices.GainsIndices[3],
        state->nb_subfr);
    fprintf(stderr, "[C SILK] nlsf_indices=[%d",
        (int)state->indices.NLSFIndices[0]);
    {
        int i;
        for (i = 1; i <= 10; i++) {
            fprintf(stderr, ", %d", (int)state->indices.NLSFIndices[i]);
        }
    }
    fprintf(stderr, "]\n");
    fprintf(stderr, "[C SILK] VAD_flags=[%d] LBRR_flag=%d inDTX=%d\n",
        (int)state->VAD_flags[0],
        (int)state->LBRR_flag,
        state->inDTX);
    fprintf(stderr, "[C SILK] seed=%d n_frames_encoded=%d\n",
        (int)state->indices.Seed,
        state->nFramesEncoded);
}

/* INSTRUMENT: Get SILK encoder LBRR and rate control state for comparison */
void debug_get_silk_lbrr_state(OpusEncoder *enc,
    opus_int32 *n_bits_used_lbrr,
    opus_int32 *lbrr_flag,
    opus_int32 *n_bits_exceeded,
    opus_int32 *signal_type,
    opus_int32 *quant_offset_type,
    opus_int32 *gains_indices,       /* 4 elements */
    opus_int32 *lag_index,
    opus_int32 *contour_index,
    opus_int32 *seed,
    opus_int32 *ltp_scale_index,
    opus_int32 *nlsf_interp_coef)
{
    OpusEncoderOffsets *hdr = (OpusEncoderOffsets *)enc;
    silk_encoder *silk = (silk_encoder *)((char *)enc + hdr->silk_enc_offset);
    silk_encoder_state *st = &silk->state_Fxx[0].sCmn;

    *n_bits_used_lbrr = silk->nBitsUsedLBRR;
    *lbrr_flag = st->LBRR_flag;
    *n_bits_exceeded = silk->nBitsExceeded;
    *signal_type = (opus_int32)st->indices.signalType;
    *quant_offset_type = (opus_int32)st->indices.quantOffsetType;
    gains_indices[0] = (opus_int32)st->indices.GainsIndices[0];
    gains_indices[1] = (opus_int32)st->indices.GainsIndices[1];
    gains_indices[2] = (opus_int32)st->indices.GainsIndices[2];
    gains_indices[3] = (opus_int32)st->indices.GainsIndices[3];
    *lag_index = (opus_int32)st->indices.lagIndex;
    *contour_index = (opus_int32)st->indices.contourIndex;
    *seed = (opus_int32)st->indices.Seed;
    *ltp_scale_index = (opus_int32)st->indices.LTP_scaleIndex;
    *nlsf_interp_coef = (opus_int32)st->indices.NLSFInterpCoef_Q2;
}

/* INSTRUMENT: Get SILK encoder NLSF indices, pulses, and additional state */
void debug_get_silk_nlsf_and_pulses(OpusEncoder *enc,
    opus_int32 *nlsf_indices,       /* MAX_LPC_ORDER+1 = 17 elements */
    opus_int32 *ltp_indices,        /* MAX_NB_SUBFR = 4 elements */
    opus_int32 *per_index,
    opus_int32 *prev_signal_type,
    opus_int32 *prev_lag,
    opus_int32 *frame_counter,
    opus_int32 *ec_prev_lag_index,
    opus_int32 *ec_prev_signal_type,
    opus_int32 *first_frame_after_reset,
    opus_int32 *controlled_since_last_payload,
    opus_int32 *pulses_sum)         /* sum of abs(pulses) for quick comparison */
{
    OpusEncoderOffsets *hdr = (OpusEncoderOffsets *)enc;
    silk_encoder *silk = (silk_encoder *)((char *)enc + hdr->silk_enc_offset);
    silk_encoder_state *st = &silk->state_Fxx[0].sCmn;
    int i;

    for (i = 0; i <= MAX_LPC_ORDER && i < 17; i++) {
        nlsf_indices[i] = (opus_int32)st->indices.NLSFIndices[i];
    }
    for (i = 0; i < MAX_NB_SUBFR && i < 4; i++) {
        ltp_indices[i] = (opus_int32)st->indices.LTPIndex[i];
    }
    *per_index = (opus_int32)st->indices.PERIndex;
    *prev_signal_type = st->prevSignalType;
    *prev_lag = st->prevLag;
    *frame_counter = st->frameCounter;
    *ec_prev_lag_index = (opus_int32)st->ec_prevLagIndex;
    *ec_prev_signal_type = st->ec_prevSignalType;
    *first_frame_after_reset = st->first_frame_after_reset;
    *controlled_since_last_payload = st->controlled_since_last_payload;

    /* Sum of absolute pulse values for quick comparison */
    opus_int32 sum = 0;
    for (i = 0; i < st->frame_length; i++) {
        opus_int32 p = (opus_int32)st->pulses[i];
        sum += (p < 0) ? -p : p;
    }
    *pulses_sum = sum;
}

/* INSTRUMENT: Get CELT encoder delayed_intra and loss_rate for comparison.
 * We replicate the CELTEncoder struct layout to reach these fields. */
struct CELTEncoder_trace {
    const OpusCustomMode *mode;   /* pointer: 8 bytes on x86_64 */
    int channels;
    int stream_channels;
    int force_intra;
    int clip;
    int disable_pf;
    int complexity;
    int upsample;
    int start, end;
    opus_int32 bitrate;
    int vbr;
    int signalling;
    int constrained_vbr;
    int loss_rate;
    int lsb_depth;
    int lfe;
    int disable_inv;
    int arch;
    /* ENCODER_RESET_START */
    opus_uint32 rng;
    int spread_decision;
    opus_int32 delayedIntra;   /* opus_val32 = opus_int32 in fixed-point */
    int tonal_average;
    int lastCodedBands;
    int hf_average;
    int tapset_decision;
    int prefilter_period;
    opus_int16 prefilter_gain;
    int prefilter_tapset;
    int consec_transient;
    AnalysisInfo analysis;
    SILKInfo silk_info;
    opus_val32 preemph_memE[2];
    opus_val32 preemph_memD[2];
    opus_int32 vbr_reservoir;
    opus_int32 vbr_drift;
    opus_int32 vbr_offset;
    opus_int32 vbr_count;
    opus_val32 overlap_max;
    opus_val16 stereo_saving;
    int intensity;
    celt_glog *energy_mask;
    celt_glog spec_avg;
};

void debug_get_celt_encoder_state(OpusEncoder *enc,
    opus_int32 *delayed_intra,
    opus_int32 *loss_rate,
    opus_int32 *prefilter_period,
    opus_int32 *prefilter_gain,
    opus_int32 *prefilter_tapset,
    opus_int32 *force_intra,
    opus_int32 *spread_decision,
    opus_int32 *tonal_average,
    opus_int32 *last_coded_bands,
    opus_int32 *consec_transient)
{
    OpusEncoderOffsets *hdr = (OpusEncoderOffsets *)enc;
    struct CELTEncoder_trace *celt =
        (struct CELTEncoder_trace *)((char *)enc + hdr->celt_enc_offset);

    *delayed_intra = celt->delayedIntra;
    *loss_rate = celt->loss_rate;
    *prefilter_period = celt->prefilter_period;
    *prefilter_gain = (opus_int32)celt->prefilter_gain;
    *prefilter_tapset = celt->prefilter_tapset;
    *force_intra = celt->force_intra;
    *spread_decision = celt->spread_decision;
    *tonal_average = celt->tonal_average;
    *last_coded_bands = celt->lastCodedBands;
    *consec_transient = celt->consec_transient;
}

/* INSTRUMENT: Get extended CELT encoder state for cross-codec bit-exactness
 * investigation. Covers the post-reset long-running accumulators that are
 * candidates for sub-ULP drift between C and Rust. */
void debug_get_celt_encoder_state_ext(OpusEncoder *enc,
    opus_int32 *stereo_saving,   /* int16_t in C — promoted to i32 for FFI */
    opus_int32 *hf_average,
    opus_int32 *spec_avg,
    opus_int32 *intensity,
    opus_int32 *overlap_max,
    opus_int32 *vbr_reservoir,
    opus_int32 *vbr_drift,
    opus_int32 *vbr_offset,
    opus_int32 *vbr_count,
    opus_int32 *preemph_memE_0,
    opus_int32 *preemph_memE_1,
    opus_int32 *preemph_memD_0,
    opus_int32 *preemph_memD_1)
{
    OpusEncoderOffsets *hdr = (OpusEncoderOffsets *)enc;
    struct CELTEncoder_trace *celt =
        (struct CELTEncoder_trace *)((char *)enc + hdr->celt_enc_offset);

    *stereo_saving = (opus_int32)celt->stereo_saving;
    *hf_average = celt->hf_average;
    *spec_avg = celt->spec_avg;
    *intensity = celt->intensity;
    *overlap_max = celt->overlap_max;
    *vbr_reservoir = celt->vbr_reservoir;
    *vbr_drift = celt->vbr_drift;
    *vbr_offset = celt->vbr_offset;
    *vbr_count = celt->vbr_count;
    *preemph_memE_0 = celt->preemph_memE[0];
    *preemph_memE_1 = celt->preemph_memE[1];
    *preemph_memD_0 = celt->preemph_memD[0];
    *preemph_memD_1 = celt->preemph_memD[1];
}

/* ======================================================================
 * Math function comparison helpers
 * ====================================================================== */

opus_int32 debug_c_celt_sqrt32(opus_int32 x) {
    return celt_sqrt32(x);
}

opus_int32 debug_c_celt_atan2p_norm(opus_int32 y, opus_int32 x) {
    return celt_atan2p_norm(y, x);
}

opus_int32 debug_c_celt_atan_norm(opus_int32 x) {
    return celt_atan_norm(x);
}

opus_int32 debug_c_frac_div32(opus_int32 a, opus_int32 b) {
    return frac_div32(a, b);
}

opus_int32 debug_c_stereo_itheta(const opus_int32 *X, const opus_int32 *Y,
                                  int stereo, int N) {
    return stereo_itheta(X, Y, stereo, N, 0);
}

opus_int32 debug_c_celt_inner_prod_norm_shift(const opus_int32 *x,
                                               const opus_int32 *y, int len) {
    return celt_inner_prod_norm_shift(x, y, len, 0);
}

opus_int32 debug_c_celt_rsqrt_norm32(opus_int32 x) {
    return celt_rsqrt_norm32(x);
}

opus_int32 debug_c_celt_rsqrt_norm(opus_int32 x) {
    return (opus_int32)celt_rsqrt_norm(x);
}

/* celt_rcp_norm16 has external linkage but is not declared in mathops.h */
extern opus_val16 celt_rcp_norm16(opus_val16 x);

opus_int32 debug_c_celt_rcp_norm16(opus_int32 x) {
    return (opus_int32)celt_rcp_norm16((opus_int16)x);
}

opus_int32 debug_c_celt_rcp_norm32(opus_int32 x) {
    return celt_rcp_norm32(x);
}

opus_int32 debug_c_celt_rcp(opus_int32 x) {
    return celt_rcp(x);
}

opus_int32 debug_c_normalise_residual_g(opus_int32 ryy, opus_int32 gain) {
    /* Replicate the gain computation from normalise_residual */
    int k = celt_ilog2(ryy) >> 1;
    opus_int32 t = VSHR32(ryy, 2*(k-7)-15);
    opus_int32 g = MULT32_32_Q31(celt_rsqrt_norm32(t), gain);
    return g;
}

/* Compute normalise_bands for a single band, matching C exactly. */
void debug_c_normalise_band(opus_int32 *X_out, const opus_int32 *freq,
                             opus_int32 bandE, int start_j, int end_j) {
    int j;
    int shift;
    opus_int32 E, g;
    E = bandE;
    if (E < 10) E += 1 /* EPSILON */;
    shift = 30 - celt_zlog2(E);
    E = SHL32(E, shift);
    g = celt_rcp_norm32(E);
    for (j = start_j; j < end_j; j++) {
        X_out[j - start_j] = PSHR32(MULT32_32_Q31(g, SHL32(freq[j], shift)), 30-24);
    }
}

int debug_c_opus_fast_int64(void) {
    return OPUS_FAST_INT64;
}

/* Call unquant_coarse_energy + unquant_fine_energy on given data,
 * with the provided old_e_bands state, and return the result.
 * This allows comparing C and Rust decode of a specific packet. */
extern void unquant_coarse_energy(const CELTMode *m, int start, int end,
    opus_int32 *oldEBands, int intra, ec_dec *dec, int C, int LM);
extern void unquant_fine_energy(const CELTMode *m, int start, int end,
    opus_int32 *oldEBands, int *prev_quant, int *extra_quant, ec_dec *dec, int C);
extern void unquant_energy_finalise(const CELTMode *m, int start, int end,
    opus_int32 *oldEBands, int *extra_quant, int *fine_priority,
    int bits_left, ec_dec *dec, int C);

/* Trace version of ec_laplace_decode for debugging */
extern int ec_laplace_decode(ec_dec *dec, unsigned fs, int decay);

static int traced_ec_laplace_decode(ec_dec *dec, unsigned fs, int decay, int band) {
    int tell_before = ec_tell(dec);
    if (band < 3) {
        fprintf(stderr, "[C lap-detail] rng=%08x val=%08x ext=%08x fs=%u\n",
            dec->rng, dec->val, dec->ext, fs);
    }
    int val = ec_laplace_decode(dec, fs, decay);
    int tell_after = ec_tell(dec);
    if (band < 3) {
        fprintf(stderr, "[C laplace] band=%d fs=%u decay=%d -> val=%d tell=%d->%d (consumed %d) post_rng=%08x\n",
            band, fs, decay, val, tell_before, tell_after, tell_after - tell_before, dec->rng);
    }
    return val;
}

void debug_c_decode_energy(
    const unsigned char *data, int len,
    opus_int32 *old_bands_inout, /* 2*21 = 42 i32 values, modified in place */
    int *fine_quant_out,         /* 21 i32 values output */
    int cc, int lm
) {
    const CELTMode *mode = opus_custom_mode_create(48000, 960, NULL);
    int start = 0;
    int end = 21;
    int nb_ebands = 21;

    ec_dec dec;
    ec_dec_init(&dec, (unsigned char *)data, len);

    /* Skip silence bit (tell starts at 1 for Opus CELT-only) */
    /* Actually, the dec starts at tell=0, the Opus layer reads 1 bit before */
    /* For this test, we need to read the same bits as the decoder would */

    int total_bits = len * 8;
    int tell = ec_tell(&dec);

    /* Mono consistency */
    if (cc == 1) {
        int i;
        for (i = 0; i < nb_ebands; i++) {
            if (old_bands_inout[i] < old_bands_inout[nb_ebands + i])
                old_bands_inout[i] = old_bands_inout[nb_ebands + i];
        }
    }

    /* Silence */
    int silence = 0;
    if (tell >= total_bits) {
        silence = 1;
    } else if (tell == 1) {
        silence = ec_dec_bit_logp(&dec, 15);
    }
    if (silence) tell = total_bits;

    /* Postfilter */
    int pf_pitch = 0, pf_gain = 0, pf_tapset = 0;
    if (start == 0 && tell + 16 <= total_bits) {
        int has_pf = ec_dec_bit_logp(&dec, 1);
        if (has_pf) {
            int octave = ec_dec_uint(&dec, 6);
            pf_pitch = (16 << octave) + ec_dec_bits(&dec, 4+octave) - 1;
            int qg = ec_dec_bits(&dec, 3);
            pf_gain = (3072) * (qg+1); /* QCONST16(0.09375, 15) = 3072 */
            tell = ec_tell(&dec);
            if (tell + 2 <= total_bits) {
                unsigned char tapset_icdf[3] = {2, 1, 0};
                pf_tapset = ec_dec_icdf(&dec, tapset_icdf, 2);
            }
        }
        tell = ec_tell(&dec);
    }

    /* Transient */
    int is_transient = 0;
    if (lm > 0 && tell + 3 <= total_bits) {
        is_transient = ec_dec_bit_logp(&dec, 3);
        tell = ec_tell(&dec);
    }

    /* Intra energy */
    int intra = 0;
    if (tell + 3 <= total_bits) {
        intra = ec_dec_bit_logp(&dec, 3);
    }

    fprintf(stderr, "[C energy-test] silence=%d pf_pitch=%d pf_gain=%d is_trans=%d intra=%d tell=%d\n",
        silence, pf_pitch, pf_gain, is_transient, intra, ec_tell(&dec));

    /* Call unquant_coarse_energy from the actual C reference */
    unquant_coarse_energy(mode, start, end, old_bands_inout, intra, &dec, cc, lm);
    fprintf(stderr, "[C energy-test] after REAL coarse: band0=%d tell=%d nbits=%d rng=%08x\n",
        old_bands_inout[0], ec_tell(&dec), dec.nbits_total, dec.rng);

    /* SKIP the replication below since we used the real function */
    if (0) {
    /* Coarse energy - use traced laplace decode */
    {
        /* Replicate unquant_coarse_energy with tracing */
        /* Use actual e_prob_model from quant_bands.c */
        static const unsigned char e_prob_model_all[4][2][42] = {
            {{72,127,65,129,66,128,65,128,64,128,62,128,64,128,64,128,92,78,92,79,92,78,90,79,116,41,115,40,114,40,132,26,132,26,145,17,161,12,176,10,177,11},
             {24,179,48,138,54,135,54,132,53,134,56,133,55,132,55,132,61,114,70,96,74,88,75,88,87,74,89,66,91,67,100,59,108,50,120,40,122,37,97,43,78,50}},
            {{83,78,84,81,88,75,86,74,87,71,90,73,93,74,93,74,109,40,114,36,117,34,117,34,143,17,145,18,146,19,162,12,165,10,178,7,189,6,190,8,177,9},
             {23,178,54,115,63,102,66,98,69,99,74,89,71,91,73,91,78,89,86,80,92,66,93,64,102,59,103,60,104,60,117,52,123,44,138,35,133,31,97,38,77,45}},
            {{61,90,93,60,105,42,107,41,110,45,116,38,113,38,112,38,124,26,132,27,136,19,140,20,155,14,159,16,158,18,170,13,177,10,187,8,192,6,175,9,159,10},
             {21,178,59,110,71,86,75,85,84,83,91,66,88,73,87,72,92,75,98,72,105,58,107,54,115,52,114,55,112,56,129,51,132,40,150,33,140,29,98,35,77,42}},
            {{42,121,96,66,108,43,111,40,117,44,123,32,120,36,119,33,127,33,134,34,139,21,147,23,152,20,158,25,154,26,166,21,173,16,184,13,184,10,150,13,139,15},
             {22,178,63,114,74,82,84,83,92,82,103,62,96,72,96,67,101,73,107,72,113,55,118,52,125,52,118,52,117,55,135,49,137,39,157,32,145,29,97,33,77,40}}
        };
        const unsigned char *prob_model = e_prob_model_all[lm][intra ? 1 : 0];
        static const opus_int16 pred_coef_arr[4] = {29440, 26112, 21248, 16384};
        static const opus_int16 beta_coef_arr[4] = {22528, 12288, 6144, 4915};
        opus_int16 coef = intra ? 0 : pred_coef_arr[lm];
        opus_int16 beta = intra ? 15360 : beta_coef_arr[lm];
        opus_val64 prev[2] = {0, 0};
        int budget = len * 8;
        int i, c2;

        if (intra) { coef = 0; beta = 15360; /* beta_intra */ }

        for (i = start; i < end; i++) {
            c2 = 0;
            do {
                int qi;
                opus_int32 q, tmp;
                int tell2 = ec_tell(&dec);
                if (budget - tell2 >= 15) {
                    int pi = 2 * (i < 20 ? i : 20);
                    qi = traced_ec_laplace_decode(&dec,
                        prob_model[pi] << 7, prob_model[pi+1] << 6, i);
                } else if (budget - tell2 >= 2) {
                    unsigned char small_icdf[] = {2, 1, 0};
                    qi = ec_dec_icdf(&dec, small_icdf, 2);
                    qi = (qi>>1) ^ -(qi&1);
                } else if (budget - tell2 >= 1) {
                    qi = -ec_dec_bit_logp(&dec, 1);
                } else {
                    qi = -1;
                }
                q = SHL32(EXTEND32(qi), 24);
                old_bands_inout[i + c2*21] = MAX32(-GCONST(9.0), old_bands_inout[i + c2*21]);
                tmp = MULT16_32_Q15(coef, old_bands_inout[i + c2*21]) + prev[c2] + q;
                tmp = MIN32(GCONST(28.0), MAX32(-GCONST(28.0), tmp));
                old_bands_inout[i + c2*21] = tmp;
                prev[c2] = prev[c2] + q - MULT16_32_Q15(beta, q);
            } while (++c2 < cc);
        }
        fprintf(stderr, "[C energy-test] after coarse: band0=%d tell=%d nbits=%d\n", old_bands_inout[0], ec_tell(&dec), dec.nbits_total);
    } } /* end if(0) */

    /* Now do fine energy too */
    {
        /* Run allocation */
        int cap_arr[21];
        extern void init_caps(const CELTMode *mode, int *cap, int LM, int C);
        init_caps(mode, cap_arr, lm, cc);

        int offsets_arr[21] = {0};
        int dynalloc_logp = 6;
        int total_bits_local = len * 8;
        total_bits_local <<= 3; /* BITRES */
        int tell_frac = ec_tell_frac(&dec);

        int i;
        for (i = start; i < end; i++) {
            int width = cc * ((mode->eBands[i+1] - mode->eBands[i])) << lm;
            int quanta = width << 3;
            if (6 << 3 > quanta) quanta = 6 << 3;
            if (width < quanta) quanta = width;
            int dynalloc_loop_logp = dynalloc_logp;
            int boost = 0;
            while (tell_frac + (dynalloc_loop_logp << 3) < total_bits_local && boost < cap_arr[i]) {
                int flag = ec_dec_bit_logp(&dec, dynalloc_loop_logp);
                tell_frac = ec_tell_frac(&dec);
                if (!flag) break;
                boost += quanta;
                total_bits_local -= quanta;
                dynalloc_loop_logp = 1;
            }
            offsets_arr[i] = boost;
            if (boost > 0) {
                dynalloc_logp = dynalloc_logp > 2 ? dynalloc_logp - 1 : 2;
            }
        }

        /* Alloc trim */
        unsigned char trim_icdf[11] = {126, 124, 119, 109, 87, 41, 19, 9, 4, 2, 0};
        int alloc_trim;
        if (ec_tell_frac(&dec) + (6<<3) <= (len*8)<<3) {
            alloc_trim = ec_dec_icdf(&dec, trim_icdf, 7);
        } else {
            alloc_trim = 5;
        }

        /* Anti-collapse reservation */
        int anti_collapse_rsv = 0;
        int bits_pre = ((len*8)<<3) - ec_tell_frac(&dec) - 1;
        if (0 /* is_transient */ && lm >= 2 && bits_pre >= (lm+2) << 3) {
            anti_collapse_rsv = 1 << 3;
        }
        int bits = bits_pre - anti_collapse_rsv;

        extern int clt_compute_allocation(const CELTMode *m, int start, int end,
            const int *offsets, const int *cap, int alloc_trim,
            int *intensity, int *dual_stereo, opus_int32 total,
            opus_int32 *balance, int *pulses, int *ebits, int *fine_priority,
            int C, int LM, ec_dec *ec, int encode, int prev, int signalBandwidth);

        int pulses_arr[21], fine_quant_arr[21], fine_prio_arr[21];
        int intensity_out = 0, dual_out = 0;
        opus_int32 balance_out = 0;
        int coded_bands = clt_compute_allocation(mode, start, end,
            offsets_arr, cap_arr, alloc_trim,
            &intensity_out, &dual_out, bits, &balance_out,
            pulses_arr, fine_quant_arr, fine_prio_arr,
            cc, lm, &dec, 0, 0, -1);

        fprintf(stderr, "[C energy-test] allocation: fine_quant[0..5]=%d %d %d %d %d coded_bands=%d\n",
            fine_quant_arr[0], fine_quant_arr[1], fine_quant_arr[2], fine_quant_arr[3], fine_quant_arr[4], coded_bands);

        /* Fine energy */
        unquant_fine_energy(mode, start, end, old_bands_inout, NULL, fine_quant_arr, &dec, cc);
        fprintf(stderr, "[C energy-test] after fine: band0=%d tell=%d\n", old_bands_inout[0], ec_tell(&dec));

        /* Finalise */
        unquant_energy_finalise(mode, start, end, old_bands_inout, fine_quant_arr, fine_prio_arr,
            len*8 - ec_tell(&dec), &dec, cc);
        fprintf(stderr, "[C energy-test] after finalise: band0=%d tell=%d\n", old_bands_inout[0], ec_tell(&dec));

        /* Output fine_quant for comparison */
        {
            int j;
            for (j = 0; j < 21 && j < 5; j++) {
                fine_quant_out[j] = fine_quant_arr[j];
            }
        }
    }

    /* For fine energy, we'd need the allocation -- skip for now, just compare coarse */
}

/* Return the preemph_memD from the CELT decoder */
void debug_get_celt_preemph_mem(OpusDecoder *opus_dec, opus_int32 *out_mem) {
    OpusDecoderOffsets *hdr = (OpusDecoderOffsets *)opus_dec;
    struct CELTDecoder_trace *celt =
        (struct CELTDecoder_trace *)((char *)opus_dec + hdr->celt_dec_offset);
    out_mem[0] = celt->preemph_memD[0];
    out_mem[1] = celt->preemph_memD[1];
}

/* Return decode_mem samples from the CELT decoder.
 * offset: sample offset within channel 0's decode_mem
 * count: number of samples to return
 * out: output buffer */
void debug_get_celt_decode_mem(OpusDecoder *opus_dec, int offset, int count, opus_int32 *out) {
    OpusDecoderOffsets *hdr = (OpusDecoderOffsets *)opus_dec;
    struct CELTDecoder_trace *celt =
        (struct CELTDecoder_trace *)((char *)opus_dec + hdr->celt_dec_offset);
    int i;
    for (i = 0; i < count; i++) {
        out[i] = celt->decode_mem[offset + i];
    }
}

/* Return the full oldBandE array from the CELT decoder.
 * out_buf must have room for 2*nbEBands i32 values.
 * Returns nbEBands (caller knows total length = 2*nbEBands). */
int debug_get_celt_old_band_e(OpusDecoder *opus_dec, opus_int32 *out_buf, int max_len) {
    OpusDecoderOffsets *hdr = (OpusDecoderOffsets *)opus_dec;
    struct CELTDecoder_trace *celt =
        (struct CELTDecoder_trace *)((char *)opus_dec + hdr->celt_dec_offset);

    int nbEBands = celt->mode->nbEBands;
    int overlap = celt->overlap;
    int CC = hdr->channels;
    int decode_buffer_size = 2048;

    /* Sanity check: verify known field values */
    if (overlap != 120) {
        fprintf(stderr, "[C BUG] overlap=%d (expected 120), struct offset may be wrong!\n", overlap);
    }
    if (nbEBands != 21) {
        fprintf(stderr, "[C BUG] nbEBands=%d (expected 21), mode pointer may be wrong!\n", nbEBands);
    }

    opus_int32 *oldBandE = celt->decode_mem + (decode_buffer_size + overlap) * CC;
    fprintf(stderr, "[C DEC] rng=%08x overlap=%d ch=%d nbEBands=%d oldBandE[0]=%d\n",
        celt->rng, overlap, CC, nbEBands, oldBandE[0]);
    int total = 2 * nbEBands;
    if (total > max_len) total = max_len;

    int i;
    for (i = 0; i < total; i++) {
        out_buf[i] = oldBandE[i];
    }
    return nbEBands;
}

/* Return the postfilter state from the CELT decoder */
void debug_get_celt_postfilter(OpusDecoder *opus_dec,
                                opus_int32 *period, opus_int32 *period_old,
                                opus_int32 *gain, opus_int32 *gain_old,
                                opus_int32 *tapset, opus_int32 *tapset_old) {
    OpusDecoderOffsets *hdr = (OpusDecoderOffsets *)opus_dec;
    struct CELTDecoder_trace *celt =
        (struct CELTDecoder_trace *)((char *)opus_dec + hdr->celt_dec_offset);

    *period = celt->postfilter_period;
    *period_old = celt->postfilter_period_old;
    *gain = celt->postfilter_gain;
    *gain_old = celt->postfilter_gain_old;
    *tapset = celt->postfilter_tapset;
    *tapset_old = celt->postfilter_tapset_old;
}

/* Return the oldLogE / oldLogE2 arrays from the CELT decoder.
 * Each is 2*nbEBands long. */
int debug_get_celt_old_log_e(OpusDecoder *opus_dec, opus_int32 *out_log_e, opus_int32 *out_log_e2, int max_len) {
    OpusDecoderOffsets *hdr = (OpusDecoderOffsets *)opus_dec;
    struct CELTDecoder_trace *celt =
        (struct CELTDecoder_trace *)((char *)opus_dec + hdr->celt_dec_offset);

    int nbEBands = celt->mode->nbEBands;
    int overlap = celt->overlap;
    int CC = hdr->channels;
    int decode_buffer_size = 2048;

    /* Layout: decode_mem | oldBandE | oldLogE | oldLogE2 | backgroundLogE */
    opus_int32 *oldBandE  = celt->decode_mem + (decode_buffer_size + overlap) * CC;
    opus_int32 *oldLogE   = oldBandE + 2 * nbEBands;
    opus_int32 *oldLogE2  = oldLogE + 2 * nbEBands;

    int total = 2 * nbEBands;
    if (total > max_len) total = max_len;

    int i;
    for (i = 0; i < total; i++) {
        out_log_e[i] = oldLogE[i];
        out_log_e2[i] = oldLogE2[i];
    }
    return nbEBands;
}

/* Extract encoder internal state for comparison.
 * Requires knowing the struct layout from opus_encoder.c.
 *
 * Must exactly match `struct OpusEncoder` from `reference/src/opus_encoder.c`
 * under our build config (FIXED_POINT, !DISABLE_FLOAT_API, !ENABLE_DRED,
 * !ENABLE_QEXT). Any mismatch shifts every field offset and produces
 * garbage reads for callers that reach past the early pre-reset fields. */
typedef struct {
    opus_int32 celt_enc_offset;
    opus_int32 silk_enc_offset;
    silk_EncControlStruct silk_mode;
    int application;
    int channels;
    int delay_compensation;
    int force_channels;
    int signal_type;
    int user_bandwidth;
    int max_bandwidth;
    int user_forced_mode;
    int voice_ratio;
    opus_int32 Fs;
    int use_vbr;
    int vbr_constraint;
    int variable_duration;
    opus_int32 bitrate_bps;
    opus_int32 user_bitrate_bps;
    int lsb_depth;
    int encoder_buffer;
    int lfe;
    int arch;
    int use_dtx;
    int fec_config;
    /* Float analysis (FLOAT_API enabled). Matches reference/src/opus_encoder.c
     * lines 104-106: `#ifndef DISABLE_FLOAT_API ... TonalityAnalysisState
     * analysis; #endif`. Must appear BEFORE OPUS_ENCODER_RESET_START. */
    TonalityAnalysisState analysis;
    /* OPUS_ENCODER_RESET_START */
    int stream_channels;
    opus_int16 hybrid_stereo_width_Q14;
    opus_int32 variable_HP_smth2_Q15;
    opus_int16 prev_HB_gain;
    opus_int32 hp_mem[4];
    int mode;
    int prev_mode;
    int prev_channels;
    int prev_framesize;
    int bandwidth;
    int auto_bandwidth;
    int silk_bw_switch;
    int first;
    celt_glog *energy_masking;
    /* StereoWidthState width_mem: XX, XY, YY (i32); smoothed_width, max_follower (i16). */
    opus_val32 width_mem_XX;
    opus_val32 width_mem_XY;
    opus_val32 width_mem_YY;
    opus_val16 width_mem_smoothed_width;
    opus_val16 width_mem_max_follower;
    int detected_bandwidth;
} OpusEncoderTrace;

/* INSTRUMENT: Return the Opus-level width_mem accumulators plus
 * hybrid_stereo_width_Q14 for bit-exactness diagnostics. */
void debug_get_opus_stereo_state(OpusEncoder *enc,
    opus_int32 *hybrid_stereo_width_q14,
    opus_int32 *width_xx,
    opus_int32 *width_xy,
    opus_int32 *width_yy,
    opus_int32 *width_smoothed,
    opus_int32 *width_max_follower,
    opus_int32 *detected_bandwidth,
    opus_int32 *mode,
    opus_int32 *prev_mode,
    opus_int32 *bandwidth)
{
    OpusEncoderTrace *st = (OpusEncoderTrace *)enc;
    *hybrid_stereo_width_q14 = (opus_int32)st->hybrid_stereo_width_Q14;
    *width_xx = st->width_mem_XX;
    *width_xy = st->width_mem_XY;
    *width_yy = st->width_mem_YY;
    *width_smoothed = (opus_int32)st->width_mem_smoothed_width;
    *width_max_follower = (opus_int32)st->width_mem_max_follower;
    *detected_bandwidth = st->detected_bandwidth;
    *mode = st->mode;
    *prev_mode = st->prev_mode;
    *bandwidth = st->bandwidth;
}

void debug_get_encoder_hp_state(OpusEncoder *enc,
    opus_int32 *hp_mem_out,
    opus_int32 *variable_hp_smth2,
    opus_int32 *mode_out,
    opus_int32 *stream_channels_out,
    opus_int32 *bandwidth_out)
{
    /* Use byte offsets from the original struct definition in opus_encoder.c.
     * The OpusEncoder is opaque, but we know its layout. Use sizeof checks. */
    OpusEncoderTrace *st = (OpusEncoderTrace *)enc;

    /* Verify our struct matches by checking known values */
    fprintf(stderr, "[debug] sizeof(silk_EncControlStruct)=%d sizeof(OpusEncoderTrace)=%d\n",
        (int)sizeof(silk_EncControlStruct), (int)sizeof(OpusEncoderTrace));
    fprintf(stderr, "[debug] app=%d channels=%d Fs=%d enc_buf=%d\n",
        st->application, st->channels, st->Fs, st->encoder_buffer);
    fprintf(stderr, "[debug] stream_ch=%d hw_Q14=%d hp_smth2=%d prev_hb=%d\n",
        st->stream_channels, (int)st->hybrid_stereo_width_Q14,
        st->variable_HP_smth2_Q15, (int)st->prev_HB_gain);
    fprintf(stderr, "[debug] mode=%d prev_mode=%d\n", st->mode, st->prev_mode);

    int i;
    /* Scan bytes around expected location to find the right offset */
    opus_int32 *raw = (opus_int32 *)enc;
    int total_ints = (int)sizeof(OpusEncoderTrace) / 4;
    fprintf(stderr, "[debug] scanning for hp_smth2 (expect ~2903 after frame 0):\n");
    /* Print the int32 values around the expected location */
    int start_field = (int)(((char*)&st->variable_HP_smth2_Q15 - (char*)st) / 4);
    for (i = start_field - 3; i <= start_field + 5 && i < total_ints; i++) {
        if (i >= 0) {
            fprintf(stderr, "  raw[%d] = %d (0x%08x)%s\n", i, raw[i], (unsigned)raw[i],
                i == start_field ? " <-- variable_HP_smth2_Q15" : "");
        }
    }

    for (i = 0; i < 4; i++) hp_mem_out[i] = st->hp_mem[i];
    *variable_hp_smth2 = st->variable_HP_smth2_Q15;
    *mode_out = st->mode;
    *stream_channels_out = st->stream_channels;
    *bandwidth_out = 0;
}

/* Run the C hp_cutoff biquad on stereo data and return the result.
 * This lets us compare the Rust hp_cutoff output with the C output exactly. */
void debug_c_hp_cutoff_stereo(
    const opus_int16 *in,
    opus_int32 cutoff_Hz,
    opus_int16 *out,
    opus_int32 *hp_mem,  /* 4 elements, modified in place */
    int len,
    opus_int32 Fs
) {
    opus_int32 B_Q28[3], A_Q28[2];
    opus_int32 Fc_Q19, r_Q28, r_Q22;

    Fc_Q19 = silk_DIV32_16(silk_SMULBB(SILK_FIX_CONST(1.5 * 3.14159 / 1000, 19), cutoff_Hz), Fs/1000);
    r_Q28 = SILK_FIX_CONST(1.0, 28) - silk_MUL(SILK_FIX_CONST(0.92, 9), Fc_Q19);

    B_Q28[0] = r_Q28;
    B_Q28[1] = silk_LSHIFT(-r_Q28, 1);
    B_Q28[2] = r_Q28;

    r_Q22 = silk_RSHIFT(r_Q28, 6);
    A_Q28[0] = silk_SMULWW(r_Q22, silk_SMULWW(Fc_Q19, Fc_Q19) - SILK_FIX_CONST(2.0, 22));
    A_Q28[1] = silk_SMULWW(r_Q22, r_Q22);

    /* Use the stride2 function that the real C code uses for stereo */
    silk_biquad_alt_stride2(in, B_Q28, A_Q28, hp_mem, out, len, 0);
}

/* Run C silk_stereo_LR_to_MS and return the mid/side outputs for comparison.
 * x1_in/x2_in: L/R input (frame_length samples each)
 * mid_out/side_out: output buffers (frame_length+2 samples each)
 * The first 2 elements of mid_out/side_out are the overlap (sMid/sSide state).
 */
void debug_c_stereo_lr_to_ms(
    const opus_int16 *x1_in,
    const opus_int16 *x2_in,
    opus_int16 *mid_out,
    opus_int16 *side_out,
    opus_int8 *pred_ix_out,   /* 6 bytes: ix[2][3] */
    opus_int8 *mid_only_out,
    opus_int32 *mid_side_rates_out, /* 2 ints */
    opus_int32 total_rate_bps,
    opus_int prev_speech_act_Q8,
    int to_mono,
    int fs_kHz,
    int frame_length)
{
    stereo_enc_state state;
    opus_int16 x1_buf[500]; /* frame_length + 2 extra for overlap */
    opus_int16 x2_buf[500];
    opus_int8 ix[2][3];
    opus_int8 mid_only_flag;
    opus_int32 mid_side_rates[2];
    int i;

    silk_memset(&state, 0, sizeof(state));

    /* Set up inputBuf layout: [overlap0, overlap1, frame_data...] */
    x1_buf[0] = 0; x1_buf[1] = 0;
    x2_buf[0] = 0; x2_buf[1] = 0;
    for (i = 0; i < frame_length; i++) {
        x1_buf[i + 2] = x1_in[i];
        x2_buf[i + 2] = x2_in[i];
    }

    silk_stereo_LR_to_MS(&state, &x1_buf[2], &x2_buf[2], ix, &mid_only_flag,
        mid_side_rates, total_rate_bps, prev_speech_act_Q8, to_mono, fs_kHz, frame_length);

    /* After LR_to_MS: x1_buf[0..2] = sMid (from state), x1_buf[2..fl+2] = mid signal */
    /* x2_buf[1..fl+1] = side signal (written at x2[n-1] for n=0..fl) */
    for (i = 0; i < frame_length + 2; i++) {
        mid_out[i] = x1_buf[i];
    }
    /* Side starts at x2_buf[1] */
    for (i = 0; i < frame_length + 1; i++) {
        side_out[i] = x2_buf[i];
    }

    for (i = 0; i < 6; i++) pred_ix_out[i] = ((opus_int8*)ix)[i];
    *mid_only_out = mid_only_flag;
    mid_side_rates_out[0] = mid_side_rates[0];
    mid_side_rates_out[1] = mid_side_rates[1];
}

/* Wrapper around C MDCT backward for direct comparison.
 * Uses the 48kHz mode (N=1920, shortMdctSize=120, overlap=120, maxLM=3).
 * in: frequency-domain input (N2=N/2 >> shift values, strided)
 * out: time-domain output buffer (N = 1920>>shift values)
 * overlap_buf: initial overlap data to place in out[0..overlap] before MDCT
 * overlap: overlap size (120)
 * shift: 0..3
 * stride: B (1 for long blocks, M for short)
 */
extern CELTMode *opus_custom_mode_create(opus_int32 Fs, int frame_size, int *error);

void debug_clt_mdct_backward(
    const opus_int32 *in,
    opus_int32 *out,
    const opus_int32 *overlap_buf,
    int overlap,
    int shift,
    int stride,
    int n_mdct)
{
    const CELTMode *mode = opus_custom_mode_create(48000, 960, NULL);
    int i;

    /* Copy overlap data into the output buffer before calling MDCT backward */
    for (i = 0; i < overlap; i++) {
        out[i] = overlap_buf[i];
    }
    /* Zero the rest of the output buffer */
    for (i = overlap; i < n_mdct; i++) {
        out[i] = 0;
    }

    clt_mdct_backward(&mode->mdct, (kiss_fft_scalar *)in, out,
                       mode->window, overlap, shift, stride, 0);
}

/* Dump the C encoder's SILK stereo state to stderr.
 * Called from Rust harness after each opus_encode frame. */
void debug_dump_silk_stereo(OpusEncoder *enc)
{
    OpusEncoderOffsets *hdr = (OpusEncoderOffsets *)enc;
    silk_encoder *psEnc = (silk_encoder *)((char *)enc + hdr->silk_enc_offset);
    stereo_enc_state *st = &psEnc->sStereo;
    int n = psEnc->state_Fxx[0].sCmn.nFramesEncoded;
    fprintf(stderr, "[C_STEREO] nFramesEncoded=%d pred_ix=[%d,%d,%d],[%d,%d,%d] mid_only=%d "
        "sMid=[%d,%d] sSide=[%d,%d] width_prev=%d smth_width=%d "
        "pred_prev=[%d,%d] nChannelsInt=%d\n",
        n,
        (int)st->predIx[0][0][0], (int)st->predIx[0][0][1], (int)st->predIx[0][0][2],
        (int)st->predIx[0][1][0], (int)st->predIx[0][1][1], (int)st->predIx[0][1][2],
        (int)st->mid_only_flags[0],
        (int)st->sMid[0], (int)st->sMid[1],
        (int)st->sSide[0], (int)st->sSide[1],
        (int)st->width_prev_Q14, (int)st->smth_width_Q14,
        (int)st->pred_prev_Q13[0], (int)st->pred_prev_Q13[1],
        psEnc->nChannelsInternal);
}

/* Extract key SILK encoder internal state for comparison with Rust. */
void debug_get_silk_state(OpusEncoder *enc,
    opus_int32 *fs_khz,
    opus_int32 *frame_length,
    opus_int32 *nb_subfr,
    opus_int32 *input_buf_ix,
    opus_int32 *n_frames_per_packet,
    opus_int32 *packet_size_ms,
    opus_int32 *first_frame_after_reset,
    opus_int32 *controlled_since_last_payload,
    opus_int32 *prefill_flag,
    opus_int32 *n_frames_encoded,
    opus_int32 *speech_activity_q8,
    opus_int32 *signal_type,
    opus_int32 *input_quality_bands_q15)
{
    OpusEncoderOffsets *hdr = (OpusEncoderOffsets *)enc;
    silk_encoder *psEnc = (silk_encoder *)((char *)enc + hdr->silk_enc_offset);
    silk_encoder_state *st = &psEnc->state_Fxx[0].sCmn;

    *fs_khz = st->fs_kHz;
    *frame_length = st->frame_length;
    *nb_subfr = st->nb_subfr;
    *input_buf_ix = st->inputBufIx;
    *n_frames_per_packet = st->nFramesPerPacket;
    *packet_size_ms = st->PacketSize_ms;
    *first_frame_after_reset = st->first_frame_after_reset;
    *controlled_since_last_payload = st->controlled_since_last_payload;
    *prefill_flag = st->prefillFlag;
    *n_frames_encoded = st->nFramesEncoded;
    *speech_activity_q8 = st->speech_activity_Q8;
    *signal_type = st->indices.signalType;
    *input_quality_bands_q15 = st->input_quality_bands_Q15[0];
}

/* Local copy of silk_decoder layout from dec_API.c */
typedef struct {
    silk_decoder_state channel_state[ DECODER_NUM_CHANNELS ];
    stereo_dec_state sStereo;
    opus_int nChannelsAPI;
    opus_int nChannelsInternal;
    opus_int prev_decode_only_middle;
} silk_decoder_local;

/* Dump SILK decoder PLC state for comparison with Rust */
void debug_dump_silk_plc_state(OpusDecoder *dec,
    opus_int16 *rand_scale_q14,
    opus_int32 *rand_seed,
    opus_int32 *pitch_l_q8,
    opus_int32 *loss_cnt,
    opus_int32 *prev_signal_type)
{
    OpusDecoderOffsets *hdr = (OpusDecoderOffsets *)dec;
    silk_decoder_local *silk_dec = (silk_decoder_local *)((char *)dec + hdr->silk_dec_offset);
    silk_decoder_state *ch0 = &silk_dec->channel_state[0];

    *rand_scale_q14 = ch0->sPLC.randScale_Q14;
    *rand_seed = ch0->sPLC.rand_seed;
    *pitch_l_q8 = ch0->sPLC.pitchL_Q8;
    *loss_cnt = ch0->lossCnt;
    *prev_signal_type = ch0->prevSignalType;
    /* Also print outBuf, sLPC_Q14_buf, exc_Q14 for comparison */
    fprintf(stderr, "INSTRUMENT C_EXTRA frame=? outbuf[0..4]=[%d,%d,%d,%d] lpc_buf[0..4]=[%d,%d,%d,%d] exc[0..4]=[%d,%d,%d,%d]\n",
        ch0->outBuf[0], ch0->outBuf[1], ch0->outBuf[2], ch0->outBuf[3],
        ch0->sLPC_Q14_buf[0], ch0->sLPC_Q14_buf[1], ch0->sLPC_Q14_buf[2], ch0->sLPC_Q14_buf[3],
        ch0->exc_Q14[0], ch0->exc_Q14[1], ch0->exc_Q14[2], ch0->exc_Q14[3]);
}

/* Extract SILK encoder inter-frame state for multiframe divergence debugging. */
void debug_get_silk_interframe_state(OpusEncoder *enc, int channel,
    opus_int32 *last_gain_index,
    opus_int32 *prev_gain_q16,
    opus_int32 *variable_hp_smth1_q15,
    opus_int32 *variable_hp_smth2_q15,
    opus_int32 *harm_shape_gain_smth,
    opus_int32 *tilt_smth,
    opus_int32 *prev_signal_type,
    opus_int32 *prev_lag,
    opus_int32 *ec_prev_lag_index,
    opus_int32 *ec_prev_signal_type,
    opus_int16 *prev_nlsfq_q15,  /* 16 elements */
    /* stereo state (shared, not per-channel) */
    opus_int32 *stereo_width_prev_q14,
    opus_int32 *stereo_smth_width_q14,
    opus_int32 *stereo_pred_prev_q13_0,
    opus_int32 *stereo_pred_prev_q13_1,
    opus_int32 *n_bits_exceeded)
{
    OpusEncoderOffsets *hdr = (OpusEncoderOffsets *)enc;
    silk_encoder *psEnc = (silk_encoder *)((char *)enc + hdr->silk_enc_offset);
    silk_encoder_state_FIX *ch = &psEnc->state_Fxx[channel];
    silk_encoder_state *st = &ch->sCmn;

    *last_gain_index = (opus_int32)ch->sShape.LastGainIndex;
    *prev_gain_q16 = st->sNSQ.prev_gain_Q16;
    *variable_hp_smth1_q15 = st->variable_HP_smth1_Q15;
    *variable_hp_smth2_q15 = st->variable_HP_smth2_Q15;
    *harm_shape_gain_smth = ch->sShape.HarmShapeGain_smth_Q16;
    *tilt_smth = ch->sShape.Tilt_smth_Q16;
    *prev_signal_type = st->prevSignalType;
    *prev_lag = st->prevLag;
    *ec_prev_lag_index = (opus_int32)st->ec_prevLagIndex;
    *ec_prev_signal_type = st->ec_prevSignalType;

    int i;
    for (i = 0; i < 16; i++) {
        prev_nlsfq_q15[i] = st->prev_NLSFq_Q15[i];
    }

    /* Stereo state is shared */
    *stereo_width_prev_q14 = (opus_int32)psEnc->sStereo.width_prev_Q14;
    *stereo_smth_width_q14 = (opus_int32)psEnc->sStereo.smth_width_Q14;
    *stereo_pred_prev_q13_0 = (opus_int32)psEnc->sStereo.pred_prev_Q13[0];
    *stereo_pred_prev_q13_1 = (opus_int32)psEnc->sStereo.pred_prev_Q13[1];
    *n_bits_exceeded = psEnc->nBitsExceeded;
}

/* Extract NLSF indices from SILK encoder state */
void debug_get_silk_nlsf_indices(OpusEncoder *enc, int channel,
    opus_int8 *nlsf_indices,   /* MAX_LPC_ORDER + 1 = 17 elements */
    opus_int32 *predict_lpc_order,
    opus_int32 *signal_type,
    opus_int32 *nlsf_interp_coef_q2)
{
    OpusEncoderOffsets *hdr = (OpusEncoderOffsets *)enc;
    silk_encoder *psEnc = (silk_encoder *)((char *)enc + hdr->silk_enc_offset);
    silk_encoder_state_FIX *ch = &psEnc->state_Fxx[channel];
    silk_encoder_state *st = &ch->sCmn;

    int i;
    for (i = 0; i < 17; i++) {
        nlsf_indices[i] = st->indices.NLSFIndices[i];
    }
    *predict_lpc_order = st->predictLPCOrder;
    *signal_type = st->indices.signalType;
    *nlsf_interp_coef_q2 = (opus_int32)st->indices.NLSFInterpCoef_Q2;
}

/* Hash the input buffer (x_buf) for quick divergence detection */
void debug_get_silk_xbuf_hash(OpusEncoder *enc, int channel,
    opus_int32 *hash_out,
    opus_int32 *buf_len)
{
    OpusEncoderOffsets *hdr = (OpusEncoderOffsets *)enc;
    silk_encoder *psEnc = (silk_encoder *)((char *)enc + hdr->silk_enc_offset);
    silk_encoder_state_FIX *ch = &psEnc->state_Fxx[channel];
    silk_encoder_state *st = &ch->sCmn;

    /* inputBuf is MAX_FRAME_LENGTH + 2 = 322 elements */
    int total_len = MAX_FRAME_LENGTH + 2;
    *buf_len = total_len;

    /* Simple hash of the input buffer */
    opus_int32 h = 0;
    const opus_int16 *buf = st->inputBuf;
    int j;
    for (j = 0; j < total_len; j++) {
        h = h * 31 + (opus_int32)buf[j];
    }
    *hash_out = h;
}

/* Extended SILK encoder inter-frame state dump — covers the "untracked"
 * state that the bug E/F/G/H investigation needs. */
void debug_get_silk_extended_state(OpusEncoder *enc, int channel,
    /* NSQ state */
    opus_int32 *nsq_rand_seed,
    opus_int32 *nsq_slf_ar_shp_q14,
    opus_int32 *nsq_lag_prev,
    opus_int32 *nsq_sdiff_shp_q14,
    opus_int32 *nsq_sltp_buf_idx,
    opus_int32 *nsq_sltp_shp_buf_idx,
    opus_int32 *nsq_rewhite_flag,
    opus_int32 *nsq_slpc_q14,      /* 16 entries */
    opus_int32 *nsq_sar2_q14,      /* 16 entries (MAX_SHAPE_LPC_ORDER) */
    opus_int32 *nsq_sltp_shp_q14,  /* 32 entries */
    /* VAD state */
    opus_int32 *vad_hp_state,
    opus_int32 *vad_counter,
    opus_int32 *vad_noise_level_bias, /* 4 entries (VAD_N_BANDS) */
    opus_int32 *vad_ana_state,        /* 2 entries */
    opus_int32 *vad_ana_state1,       /* 2 entries */
    opus_int32 *vad_ana_state2,       /* 2 entries */
    opus_int32 *vad_nrg_ratio_smth_q8,/* 4 entries */
    opus_int32 *vad_nl,               /* 4 entries */
    opus_int32 *vad_inv_nl,           /* 4 entries */
    /* Common state */
    opus_int32 *sum_log_gain_q7,
    opus_int32 *in_hp_state,          /* 2 entries */
    opus_int32 *input_tilt_q15,
    opus_int32 *input_quality_bands_q15, /* 4 entries */
    opus_int32 *frame_counter,
    opus_int32 *no_speech_counter,
    /* LP state */
    opus_int32 *lp_in_lp_state,       /* 2 entries */
    opus_int32 *lp_transition_frame_no,
    /* Shape state */
    opus_int32 *shape_harm_boost_smth,
    /* x_buf hash */
    opus_int32 *x_buf_hash,
    opus_int32 *ltp_corr_q15,
    opus_int32 *res_nrg_smth)
{
    OpusEncoderOffsets *hdr = (OpusEncoderOffsets *)enc;
    silk_encoder *psEnc = (silk_encoder *)((char *)enc + hdr->silk_enc_offset);
    silk_encoder_state_FIX *ch = &psEnc->state_Fxx[channel];
    silk_encoder_state *st = &ch->sCmn;

    /* ----- NSQ ----- */
    *nsq_rand_seed = st->sNSQ.rand_seed;
    *nsq_slf_ar_shp_q14 = st->sNSQ.sLF_AR_shp_Q14;
    *nsq_lag_prev = st->sNSQ.lagPrev;
    *nsq_sdiff_shp_q14 = st->sNSQ.sDiff_shp_Q14;
    *nsq_sltp_buf_idx = st->sNSQ.sLTP_buf_idx;
    *nsq_sltp_shp_buf_idx = st->sNSQ.sLTP_shp_buf_idx;
    *nsq_rewhite_flag = st->sNSQ.rewhite_flag;
    {
        int i;
        for (i = 0; i < 16; i++) {
            nsq_slpc_q14[i] = st->sNSQ.sLPC_Q14[i];
        }
        for (i = 0; i < 16; i++) {
            nsq_sar2_q14[i] = st->sNSQ.sAR2_Q14[i];
        }
        for (i = 0; i < 32; i++) {
            nsq_sltp_shp_q14[i] = st->sNSQ.sLTP_shp_Q14[i];
        }
    }

    /* ----- VAD ----- */
    *vad_hp_state = (opus_int32)st->sVAD.HPstate;
    *vad_counter = st->sVAD.counter;
    {
        int i;
        for (i = 0; i < VAD_N_BANDS && i < 4; i++) {
            vad_noise_level_bias[i] = st->sVAD.NoiseLevelBias[i];
            vad_nrg_ratio_smth_q8[i] = st->sVAD.NrgRatioSmth_Q8[i];
            vad_nl[i] = st->sVAD.NL[i];
            vad_inv_nl[i] = st->sVAD.inv_NL[i];
        }
        for (i = 0; i < 2; i++) {
            vad_ana_state[i] = st->sVAD.AnaState[i];
            vad_ana_state1[i] = st->sVAD.AnaState1[i];
            vad_ana_state2[i] = st->sVAD.AnaState2[i];
        }
    }

    /* ----- Common ----- */
    *sum_log_gain_q7 = st->sum_log_gain_Q7;
    in_hp_state[0] = st->In_HP_State[0];
    in_hp_state[1] = st->In_HP_State[1];
    *input_tilt_q15 = st->input_tilt_Q15;
    {
        int i;
        for (i = 0; i < VAD_N_BANDS && i < 4; i++) {
            input_quality_bands_q15[i] = st->input_quality_bands_Q15[i];
        }
    }
    *frame_counter = st->frameCounter;
    *no_speech_counter = st->noSpeechCounter;

    /* ----- LP state ----- */
    lp_in_lp_state[0] = st->sLP.In_LP_State[0];
    lp_in_lp_state[1] = st->sLP.In_LP_State[1];
    *lp_transition_frame_no = st->sLP.transition_frame_no;

    /* ----- Shape ----- */
    *shape_harm_boost_smth = ch->sShape.HarmBoost_smth_Q16;

    /* ----- x_buf hash ----- */
    {
        int x_buf_len = 2 * MAX_FRAME_LENGTH + LA_SHAPE_MAX;
        opus_int32 h = 0;
        int j;
        for (j = 0; j < x_buf_len; j++) {
            h = h * 31 + (opus_int32)ch->x_buf[j];
        }
        *x_buf_hash = h;
    }

    *ltp_corr_q15 = ch->LTPCorr_Q15;
    *res_nrg_smth = ch->resNrgSmth;
}

/* ----------------------------------------------------------------------- *
 * Top-level Opus encoder state for the encoder-state-accumulation
 * diagnostic (Cluster A, Findings #7 + #8). Exposes the post-setter,
 * pre-encode values that the public CTL setters mutate, so the harness
 * can detect ms_set_*-vs-set_* asymmetries in the Rust port.
 *
 * The C OpusEncoder layout (reference/src/opus_encoder.c:76) under this
 * harness's config (FIXED_POINT, no DISABLE_FLOAT_API, no ENABLE_DRED,
 * no ENABLE_QEXT) is:
 *   [0]  int          celt_enc_offset
 *   [4]  int          silk_enc_offset
 *   [8]  silk_EncControlStruct silk_mode  (no preceding DRED block)
 *        int application; int channels; int delay_compensation;
 *        int force_channels; int signal_type; int user_bandwidth;
 *        int max_bandwidth; int user_forced_mode; int voice_ratio;
 *        opus_int32 Fs; int use_vbr; int vbr_constraint;
 *        int variable_duration; opus_int32 bitrate_bps;
 *        opus_int32 user_bitrate_bps; int lsb_depth; int encoder_buffer;
 *        int lfe; int arch; int use_dtx; int fec_config;
 *        TonalityAnalysisState analysis; ... (we never read past
 *        fec_config so the analysis layout is not needed).
 */
typedef struct {
    int          celt_enc_offset;
    int          silk_enc_offset;
    silk_EncControlStruct silk_mode;
    int          application;
    int          channels;
    int          delay_compensation;
    int          force_channels;
    int          signal_type;
    int          user_bandwidth;
    int          max_bandwidth;
    int          user_forced_mode;
    int          voice_ratio;
    opus_int32   Fs;
    int          use_vbr;
    int          vbr_constraint;
    int          variable_duration;
    opus_int32   bitrate_bps;
    opus_int32   user_bitrate_bps;
    int          lsb_depth;
    int          encoder_buffer;
    int          lfe;
    int          arch;
    int          use_dtx;
    int          fec_config;
} OpusEncoderTopLevel;

void debug_get_opus_silk_mode_state(OpusEncoder *enc,
    opus_int32 *sm_use_in_band_fec,
    opus_int32 *sm_use_cbr,
    opus_int32 *sm_use_dtx,
    opus_int32 *sm_lbrr_coded,
    opus_int32 *sm_complexity,
    opus_int32 *sm_packet_loss_percentage,
    opus_int32 *sm_bit_rate,
    opus_int32 *sm_payload_size_ms,
    opus_int32 *sm_n_channels_internal,
    opus_int32 *sm_max_internal_sample_rate,
    opus_int32 *sm_min_internal_sample_rate,
    opus_int32 *sm_desired_internal_sample_rate,
    opus_int32 *use_vbr,
    opus_int32 *vbr_constraint,
    opus_int32 *use_dtx,
    opus_int32 *fec_config,
    opus_int32 *user_bitrate_bps,
    opus_int32 *bitrate_bps,
    opus_int32 *force_channels,
    opus_int32 *signal_type,
    opus_int32 *lsb_depth,
    opus_int32 *lfe,
    opus_int32 *application)
{
    OpusEncoderTopLevel *top = (OpusEncoderTopLevel *)enc;

    *sm_use_in_band_fec              = top->silk_mode.useInBandFEC;
    *sm_use_cbr                      = top->silk_mode.useCBR;
    *sm_use_dtx                      = top->silk_mode.useDTX;
    *sm_lbrr_coded                   = top->silk_mode.LBRR_coded;
    *sm_complexity                   = top->silk_mode.complexity;
    *sm_packet_loss_percentage       = top->silk_mode.packetLossPercentage;
    *sm_bit_rate                     = top->silk_mode.bitRate;
    *sm_payload_size_ms              = top->silk_mode.payloadSize_ms;
    *sm_n_channels_internal          = top->silk_mode.nChannelsInternal;
    *sm_max_internal_sample_rate     = top->silk_mode.maxInternalSampleRate;
    *sm_min_internal_sample_rate     = top->silk_mode.minInternalSampleRate;
    *sm_desired_internal_sample_rate = top->silk_mode.desiredInternalSampleRate;

    *use_vbr          = top->use_vbr;
    *vbr_constraint   = top->vbr_constraint;
    *use_dtx          = top->use_dtx;
    *fec_config       = top->fec_config;
    *user_bitrate_bps = top->user_bitrate_bps;
    *bitrate_bps      = top->bitrate_bps;
    *force_channels   = top->force_channels;
    *signal_type      = top->signal_type;
    *lsb_depth        = top->lsb_depth;
    *lfe              = top->lfe;
    *application      = top->application;
}

/* ----------------------------------------------------------------------- *
 * Multistream wrapper: return the pointer to a sub-encoder within an
 * OpusMSEncoder, so callers can `debug_get_opus_silk_mode_state` it.
 * Layout per reference/src/opus_multistream_encoder.c:78-84.
 *
 * Returns NULL if `stream_id` is out of range.
 */
extern int opus_encoder_get_size(int channels);

OpusEncoder *debug_get_inner_opus_encoder(struct OpusMSEncoder *ms, int stream_id)
{
    if (ms == NULL || stream_id < 0) {
        return NULL;
    }
    int nb_streams = ms->layout.nb_streams;
    int coupled_streams = ms->layout.nb_coupled_streams;
    if (stream_id >= nb_streams) {
        return NULL;
    }
    int coupled_size = opus_encoder_get_size(2);
    int mono_size = opus_encoder_get_size(1);
    char *ptr = (char *)ms + align((int)sizeof(*ms));
    int i;
    for (i = 0; i < stream_id; i++) {
        if (i < coupled_streams) {
            ptr += align(coupled_size);
        } else {
            ptr += align(mono_size);
        }
    }
    return (OpusEncoder *)ptr;
}

