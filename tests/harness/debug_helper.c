/* Debug helpers for extracting internal state from C reference encoder/decoder. */

#include "opus.h"
#include "opus_types.h"
#include "silk/fixed/structs_FIX.h"
#include "silk/SigProc_FIX.h"
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
 * Requires knowing the struct layout from opus_encoder.c. */
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
    /* OPUS_ENCODER_RESET_START */
    int stream_channels;
    opus_int16 hybrid_stereo_width_Q14;
    opus_int32 variable_HP_smth2_Q15;
    opus_int16 prev_HB_gain;
    opus_int32 hp_mem[4];
    int mode;
    int prev_mode;
} OpusEncoderTrace;

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
