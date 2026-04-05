/* Debug helpers for extracting internal state from C reference encoder/decoder. */

#include "opus.h"
#include "opus_types.h"
#include "silk/fixed/structs_FIX.h"
#include "celt/celt.h"
#include "celt/modes.h"
#include "celt/mathops.h"
#include "celt/vq.h"

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

opus_int32 debug_c_celt_cos_norm32(opus_int32 x) {
    return celt_cos_norm32(x);
}

opus_int32 debug_c_celt_rsqrt_norm32(opus_int32 x) {
    return celt_rsqrt_norm32(x);
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
