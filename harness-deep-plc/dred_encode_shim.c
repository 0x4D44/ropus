/* Stage 8.6 C shim: expose the xiph C DRED encoder-side pipeline through
 * a flat C API so the Rust differential test can compare byte-exact DRED
 * payload output without replicating the `DREDEnc` layout across FFI.
 *
 * Two entry points wrap the C reference's normal flow:
 *   1. `ropus_test_dredenc_new` — allocate + initialise the `DREDEnc`
 *      struct, load weights from the compile-time embedded tables
 *      (`rdovaeenc_arrays`, `pitchdnn_arrays`), and return an opaque
 *      pointer. Mirrors `dred_encoder_init + dred_encoder_load_model` in
 *      the `!USE_WEIGHTS_FILE` path.
 *   2. `ropus_test_dred_compute_latents` / `..._encode_silk_frame` —
 *      delegate to the same-named xiph functions at `arch = 0`.
 *
 * The test passes an `activity_mem` buffer it owns, so there's no need
 * to wire up a real VAD — a "always active" mem of all-1s forces the
 * normal code path.
 */

#include <stdlib.h>
#include <string.h>

#include "dred_encoder.h"
#include "dred_decoder.h"
#include "dred_config.h"
#include "dred_rdovae_constants.h"
#include "lpcnet.h"

/* Defined by `dred_rdovae_enc_data.c`. */
extern const WeightArray rdovaeenc_arrays[];
/* Defined by `pitchdnn_data.c` (via lpcnet_encoder_init fallback). */

void *ropus_test_dredenc_new(int fs, int channels) {
    DREDEnc *enc = (DREDEnc *)calloc(1, sizeof(*enc));
    if (!enc) return NULL;
    dred_encoder_init(enc, fs, channels);
    /* In `!USE_WEIGHTS_FILE` mode, `dred_encoder_init` already initialises
     * RDOVAE + pitchdnn from compile-time tables; `loaded` is 1 on
     * success. No explicit load_model call needed for the harness. */
    if (!enc->loaded) {
        free(enc);
        return NULL;
    }
    return enc;
}

void ropus_test_dredenc_free(void *enc) {
    free(enc);
}

/* Report input_buffer_fill after init (for parity-check assertions in the
 * Rust test). */
int ropus_test_dredenc_input_buffer_fill(const void *enc) {
    return ((const DREDEnc *)enc)->input_buffer_fill;
}

void ropus_test_dred_compute_latents(
    void *enc,
    const float *pcm,
    int frame_size,
    int extra_delay
) {
    dred_compute_latents((DREDEnc *)enc, pcm, frame_size, extra_delay, 0);
}

int ropus_test_dred_encode_silk_frame(
    void *enc,
    unsigned char *buf,
    int max_chunks,
    int max_bytes,
    int q0,
    int dQ,
    int qmax,
    unsigned char *activity_mem
) {
    return dred_encode_silk_frame(
        (DREDEnc *)enc,
        buf,
        max_chunks,
        max_bytes,
        q0,
        dQ,
        qmax,
        activity_mem,
        0
    );
}

/* Expose the internal state the Rust test needs to compare after
 * `compute_latents` (for diagnosing where divergence starts). */
int ropus_test_dredenc_latents_buffer_fill(const void *enc) {
    return ((const DREDEnc *)enc)->latents_buffer_fill;
}

int ropus_test_dredenc_dred_offset(const void *enc) {
    return ((const DREDEnc *)enc)->dred_offset;
}

int ropus_test_dredenc_latent_offset(const void *enc) {
    return ((const DREDEnc *)enc)->latent_offset;
}

/* Copy the first `n` floats of the latents buffer out to `dst`. */
void ropus_test_dredenc_copy_latents(const void *enc, float *dst, int n) {
    const DREDEnc *e = (const DREDEnc *)enc;
    memcpy(dst, e->latents_buffer, n * sizeof(float));
}

/* Copy the first `n` floats of the state buffer out to `dst`. */
void ropus_test_dredenc_copy_state(const void *enc, float *dst, int n) {
    const DREDEnc *e = (const DREDEnc *)enc;
    memcpy(dst, e->state_buffer, n * sizeof(float));
}

/* Copy the first `n` floats of the 16 kHz input buffer — useful for
 * isolating whether divergence starts in the resampler (input_buffer
 * differs), the LPCNet feature extractor (input_buffer matches but
 * features differ), or the RDOVAE encoder. */
void ropus_test_dredenc_copy_input_buffer(const void *enc, float *dst, int n) {
    const DREDEnc *e = (const DREDEnc *)enc;
    memcpy(dst, e->input_buffer, n * sizeof(float));
}

/* Copy the 9-wide resampler memory. Diverging `resample_mem` immediately
 * after one `convert_to_16k` call means the filter itself drifts; a
 * diverging `input_buffer` but matching `resample_mem` means the filter
 * matches and the divergence is in the output scatter. */
void ropus_test_dredenc_copy_resample_mem(const void *enc, float *dst, int n) {
    const DREDEnc *e = (const DREDEnc *)enc;
    memcpy(dst, e->resample_mem, n * sizeof(float));
}

/* Copy the latest LPCNet feature vector out of the encoder's embedded
 * state (`lpcnet_enc_state.features[0..n]`). Diagnostic hook for
 * isolating feature-extractor drift vs RDOVAE drift. The C features
 * array is 36 floats wide (NB_TOTAL_FEATURES); DRED reads 20 of them. */
void ropus_test_dredenc_copy_lpcnet_features(const void *enc, float *dst, int n) {
    const DREDEnc *e = (const DREDEnc *)enc;
    memcpy(dst, e->lpcnet_enc_state.features, n * sizeof(float));
}

/* ========================================================================
 * Stage 8.7: payload-dump + C decoder entrypoints.
 * ========================================================================
 * The Rust C-differential test supplies synthetic latents + state arrays,
 * wants the C encoder to produce a byte buffer from them, then wants to
 * decode that same buffer with both C and Rust `dred_ec_decode` and
 * cross-check. These helpers poke buffers directly into the DREDEnc
 * struct without running the resample/LPCNet/RDOVAE upstream, so the
 * test covers only the range-coder + quantiser logic (which is what 8.7
 * owns).
 */

/* Overwrite the encoder's `state_buffer` with user-supplied floats.
 * Only the first DRED_STATE_DIM floats are populated (the chunk being
 * emitted starts at offset 0 when `latent_offset == 0`). */
void ropus_test_dredenc_set_state_buffer(void *enc, const float *src, int n) {
    DREDEnc *e = (DREDEnc *)enc;
    memcpy(e->state_buffer, src, n * sizeof(float));
}

/* Overwrite the encoder's `latents_buffer` with user-supplied floats.
 * Caller owns the layout (chunk `i` lives at `2 * i * DRED_LATENT_DIM`
 * per the encoder's interleave). */
void ropus_test_dredenc_set_latents_buffer(void *enc, const float *src, int n) {
    DREDEnc *e = (DREDEnc *)enc;
    memcpy(e->latents_buffer, src, n * sizeof(float));
}

/* Mutators for the non-RDOVAE bookkeeping that `dred_encode_silk_frame`
 * reads. The test sets these to known values (usually zeros + a chosen
 * `latents_buffer_fill`) rather than letting `dred_compute_latents`
 * fill them in. */
void ropus_test_dredenc_set_bookkeeping(
    void *enc,
    int latent_offset,
    int latents_buffer_fill,
    int dred_offset,
    int last_extra_dred_offset
) {
    DREDEnc *e = (DREDEnc *)enc;
    e->latent_offset = latent_offset;
    e->latents_buffer_fill = latents_buffer_fill;
    e->dred_offset = dred_offset;
    e->last_extra_dred_offset = last_extra_dred_offset;
}

/* Run the C `dred_ec_decode` on a byte buffer and copy the resulting
 * state + latents out to caller-owned slices. Returns `dec->nb_latents`
 * (same as the C function). */
int ropus_test_dred_ec_decode(
    const unsigned char *bytes,
    int num_bytes,
    int min_feature_frames,
    int dred_frame_offset,
    float *out_state,           /* DRED_STATE_DIM floats */
    float *out_latents,         /* (DRED_NUM_REDUNDANCY_FRAMES/2) * (DRED_LATENT_DIM+1) floats */
    int *out_nb_latents,
    int *out_process_stage,
    int *out_dred_offset
) {
    OpusDRED dred;
    memset(&dred, 0, sizeof(dred));
    int ret = dred_ec_decode(&dred, bytes, num_bytes, min_feature_frames, dred_frame_offset);
    if (out_state) {
        memcpy(out_state, dred.state, DRED_STATE_DIM * sizeof(float));
    }
    if (out_latents) {
        memcpy(
            out_latents,
            dred.latents,
            (DRED_NUM_REDUNDANCY_FRAMES / 2) * (DRED_LATENT_DIM + 1) * sizeof(float)
        );
    }
    if (out_nb_latents) *out_nb_latents = dred.nb_latents;
    if (out_process_stage) *out_process_stage = dred.process_stage;
    if (out_dred_offset) *out_dred_offset = dred.dred_offset;
    return ret;
}

/* ========================================================================
 * Stage 8.8: full C encoder shim.
 * ========================================================================
 * Drives a complete `opus_encoder_create + opus_encoder_ctl(OPUS_SET_DRED_
 * DURATION) + opus_encode` flow so the Rust integration test can feed the
 * resulting packets to the Rust `OpusDREDDecoder::parse` and assert format-
 * level compatibility. Compile-time weights are used (`!USE_WEIGHTS_FILE` in
 * our `config.h`), so no explicit `OPUS_SET_DNN_BLOB` call is needed.
 */

#include "opus.h"

/* Stage-5 extension: parameterise the DRED encoder factory so the new
 * Tier-1 differential test can hit configurations where
 * `compute_dred_bitrate` returns a non-zero `dred_bitrate_bps`. The
 * extra knobs map 1:1 onto Opus CTLs:
 *   - bitrate_bps    -> OPUS_SET_BITRATE
 *   - use_inband_fec -> OPUS_SET_INBAND_FEC (0 or 1)
 *   - loss_perc      -> OPUS_SET_PACKET_LOSS_PERC
 *   - use_vbr        -> OPUS_SET_VBR (0 or 1; pass `-1` to leave default)
 *
 * Complexity stays fixed at 5 to match the existing call sites.
 */
void *ropus_test_c_encoder_new_ex(
    int fs,
    int channels,
    int application,
    int dred_duration,
    int bitrate_bps,
    int use_inband_fec,
    int loss_perc,
    int use_vbr
) {
    int err = 0;
    OpusEncoder *enc = opus_encoder_create(fs, channels, application, &err);
    if (err != OPUS_OK || !enc) return NULL;
    opus_encoder_ctl(enc, OPUS_SET_BITRATE(bitrate_bps));
    opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY(5));
    /* Default (!USE_WEIGHTS_FILE) doesn't auto-load the FEC blob, but
     * `dred_encoder_init` wired into `opus_encoder_init` does the compile-
     * time load for us. */
    opus_encoder_ctl(enc, OPUS_SET_PACKET_LOSS_PERC(loss_perc));
    opus_encoder_ctl(enc, OPUS_SET_INBAND_FEC(use_inband_fec));
    if (use_vbr == 0 || use_vbr == 1) {
        opus_encoder_ctl(enc, OPUS_SET_VBR(use_vbr));
    }
    if (dred_duration > 0) {
        int r = opus_encoder_ctl(enc, OPUS_SET_DRED_DURATION(dred_duration));
        if (r != OPUS_OK) {
            opus_encoder_destroy(enc);
            return NULL;
        }
        /* Verify the CTL took. */
        int check_dur = -1;
        opus_encoder_ctl(enc, OPUS_GET_DRED_DURATION(&check_dur));
        if (check_dur != dred_duration) {
            opus_encoder_destroy(enc);
            return NULL;
        }
    }
    return enc;
}

/* Backwards-compatible factory using the original hard-coded knobs
 * (32 kbps, complexity 5, FEC=1, loss=20, default VBR). Existing tests
 * (`dred_integrated_encode.rs`, `dred_dtx_first_frame_diff.rs`,
 * `dred_bitrate_plumbing_diff.rs`) keep calling this; the new
 * Stage-5 differential test uses `ropus_test_c_encoder_new_ex` to
 * exercise non-zero `dred_bitrate_bps` configurations.
 */
void *ropus_test_c_encoder_new(int fs, int channels, int application, int dred_duration) {
    return ropus_test_c_encoder_new_ex(
        fs, channels, application, dred_duration,
        /* bitrate_bps    */ 32000,
        /* use_inband_fec */ 1,
        /* loss_perc      */ 20,
        /* use_vbr        */ -1   /* sentinel: leave VBR at default */
    );
}

void ropus_test_c_encoder_free(void *enc) {
    if (enc) opus_encoder_destroy((OpusEncoder *)enc);
}

int ropus_test_c_encoder_encode(
    void *enc,
    const opus_int16 *pcm,
    int frame_size,
    unsigned char *data,
    int max_data_bytes
) {
    return opus_encode((OpusEncoder *)enc, pcm, frame_size, data, max_data_bytes);
}

/* Parse a Rust-emitted packet with the C `opus_dred_parse` path and report
 * whether a DRED extension was found + populated. `*out_nb_latents` and
 * `*out_process_stage` mirror the `OpusDRED` fields after parse. Returns
 * the number of samples available to be decoded via DRED (C's
 * `opus_dred_parse` return value), or a negative error code. */
int ropus_test_c_dred_parse(
    const unsigned char *data,
    int len,
    int max_dred_samples,
    int sampling_rate,
    int *out_nb_latents,
    int *out_process_stage,
    int *out_dred_offset
) {
    OpusDREDDecoder *dec = opus_dred_decoder_create(NULL);
    if (!dec) return OPUS_ALLOC_FAIL;
    OpusDRED *dred = opus_dred_alloc(NULL);
    if (!dred) {
        opus_dred_decoder_destroy(dec);
        return OPUS_ALLOC_FAIL;
    }
    /* `opus_dred_alloc` returns uninitialised memory — zero it so the
     * `nb_latents`/`dred_offset` fields report zero on "no extension"
     * rather than stack garbage. `opus_dred_parse` sets `process_stage`
     * to -1 unconditionally so that one field is fine either way. */
    memset(dred, 0, opus_dred_get_size());
    int ret = opus_dred_parse(dec, dred, data, len, max_dred_samples, sampling_rate, NULL, 0);
    if (out_nb_latents) *out_nb_latents = dred->nb_latents;
    if (out_process_stage) *out_process_stage = dred->process_stage;
    if (out_dred_offset) *out_dred_offset = dred->dred_offset;
    opus_dred_free(dred);
    opus_dred_decoder_destroy(dec);
    return ret;
}

/* ========================================================================
 * Stage-5 (apply-feedback): direct FFI scalar fixture for the DRED bitrate
 * helpers.
 * ========================================================================
 *
 * Pivot from "transitive-through-encode" Tier-1 validation to direct
 * scalar comparison. Both `compute_dred_bitrate` and
 * `estimate_dred_bitrate` are `static` in `opus_encoder.c` so the linker
 * can't see them. To expose them across FFI without touching vendor
 * source we copy the two function bodies verbatim into this shim file
 * (and the supporting `dred_bits_table` constant), then wrap them with
 * `extern`-visible `ropus_c_*` entry points.
 *
 * The wrappers do NOT take an `OpusEncoder *` — `compute_dred_bitrate`
 * only reads four scalar fields off the encoder (`silk_mode.useInBandFEC`,
 * `silk_mode.packetLossPercentage`, `Fs`, `dred_duration`) and writes
 * four scalar fields back (`dred_q0`, `dred_dQ`, `dred_qmax`,
 * `dred_target_chunks`). Reproducing those reads/writes as plain `int`
 * parameters keeps the FFI surface flat and matches the Rust port's
 * `compute_dred_bitrate(enc, bitrate_bps, frame_size)` interface.
 *
 * VERBATIM COPY of `reference/src/opus_encoder.c` lines 668-730. If the
 * vendor source ever updates either function or the bits table, MIRROR
 * those changes here byte-for-byte; otherwise the FFI fixture silently
 * stops being a true differential. The top of each verbatim block flags
 * the source line range so future grep / diff catches drift.
 *
 * Headers: ecintrin.h provides `EC_ILOG`; arch.h provides `IMIN`,
 * `IMAX`, `MIN16`; celt.h provides `bitrate_to_bits` / `bits_to_bitrate`
 * (static inlines so they don't need a separate object); dred_coding.h
 * provides `compute_quantizer`. dred_config.h is already included above
 * for the DRED_NUM_REDUNDANCY_FRAMES / DRED_EXPERIMENTAL_BYTES values
 * the verbatim copies need.
 */

#include <math.h>

#include "arch.h"
#include "ecintrin.h"
#include "entcode.h"  /* For ec_ilog when EC_CLZ isn't available. */
#include "celt.h"     /* bitrate_to_bits, bits_to_bitrate. */
#include "dred_coding.h"  /* compute_quantizer. */

/* VERBATIM COPY of opus_encoder.c:668. Keep in sync with vendor source. */
static const float ropus_c_dred_bits_table[16] = {73.2f, 68.1f, 62.5f, 57.0f, 51.5f, 45.7f, 39.9f, 32.4f, 26.4f, 20.4f, 16.3f, 13.f, 9.3f, 8.2f, 7.2f, 6.4f};

/* VERBATIM COPY of opus_encoder.c:669-685 (`estimate_dred_bitrate`).
 * Only differences: function name prefixed with `ropus_c_`, and the
 * `dred_bits_table` reference renamed to `ropus_c_dred_bits_table` so
 * we don't collide with the static one in `opus_encoder.o` at link
 * time (the C TU keeps its own private copy; ours lives next to it). */
static int ropus_c_estimate_dred_bitrate_impl(int q0, int dQ, int qmax, int duration, opus_int32 target_bits, int *target_chunks) {
   int dred_chunks;
   int i;
   float bits;
   /* Signaling DRED costs 3 bytes. */
   bits = 8*(3+DRED_EXPERIMENTAL_BYTES);
   /* Approximation for the size of the IS. */
   bits += 50.f+ropus_c_dred_bits_table[q0];
   dred_chunks = IMIN((duration+5)/4, DRED_NUM_REDUNDANCY_FRAMES/2);
   if (target_chunks != NULL) *target_chunks = 0;
   for (i=0;i<dred_chunks;i++) {
      int q = compute_quantizer(q0, dQ, qmax, i);
      bits += ropus_c_dred_bits_table[q];
      if (target_chunks != NULL && bits < target_bits) *target_chunks = i+1;
   }
   return (int)floor(.5f+bits);
}

/* VERBATIM COPY of opus_encoder.c:687-730 (`compute_dred_bitrate`),
 * with the `OpusEncoder *st` parameter expanded into the four scalar
 * fields it actually reads (`useInBandFEC`, `packetLossPercentage`, `Fs`,
 * `dred_duration`) and the four scalar fields it actually writes
 * (`dred_q0`, `dred_dQ`, `dred_qmax`, `dred_target_chunks`) lifted to
 * out-pointers. The body of the function is unchanged — operand order,
 * floating constants, integer ladders all preserved. */
static opus_int32 ropus_c_compute_dred_bitrate_impl(
    int use_in_band_fec,
    int packet_loss_perc,
    opus_int32 Fs,
    int dred_duration,
    opus_int32 bitrate_bps,
    int frame_size,
    int *out_q0,
    int *out_dQ,
    int *out_qmax,
    int *out_target_chunks
) {
   float dred_frac;
   int bitrate_offset;
   opus_int32 dred_bitrate;
   opus_int32 target_dred_bitrate;
   int target_chunks;
   opus_int32 max_dred_bits;
   int q0, dQ, qmax;
   if (use_in_band_fec) {
      dred_frac = MIN16(.7f, 3.f*packet_loss_perc/100.f);
      bitrate_offset = 20000;
   } else {
      if (packet_loss_perc > 5) {
         dred_frac = MIN16(.8f, .55f + packet_loss_perc/100.f);
      } else {
         dred_frac = 12*packet_loss_perc/100.f;
      }
      bitrate_offset = 12000;
   }
   /* Account for the fact that longer packets require less redundancy. */
   dred_frac = dred_frac/(dred_frac + (1-dred_frac)*(frame_size*50.f)/Fs);
   /* Approximate fit based on a few experiments. Could probably be improved. */
   q0 = IMIN(15, IMAX(4, 51 - 3*EC_ILOG(IMAX(1, bitrate_bps-bitrate_offset))));
   dQ = bitrate_bps-bitrate_offset > 36000 ? 3 : 5;
   qmax = 15;
   target_dred_bitrate = IMAX(0, (int)(dred_frac*(bitrate_bps-bitrate_offset)));
   if (dred_duration > 0) {
      opus_int32 target_bits = bitrate_to_bits(target_dred_bitrate, Fs, frame_size);
      max_dred_bits = ropus_c_estimate_dred_bitrate_impl(q0, dQ, qmax, dred_duration, target_bits, &target_chunks);
   } else {
      max_dred_bits = 0;
      target_chunks=0;
   }
   dred_bitrate = IMIN(target_dred_bitrate, bits_to_bitrate(max_dred_bits, Fs, frame_size));
   /* If we can't afford enough bits, don't bother with DRED at all. */
   if (target_chunks < 2)
      dred_bitrate = 0;
   if (out_q0) *out_q0 = q0;
   if (out_dQ) *out_dQ = dQ;
   if (out_qmax) *out_qmax = qmax;
   if (out_target_chunks) *out_target_chunks = target_chunks;
   return dred_bitrate;
}

/* extern "C" wrappers — these are what the Rust FFI calls. The `_impl`
 * suffix on the verbatim copies isolates them so future maintainers
 * don't accidentally diverge them while editing the public surface. */
int ropus_c_estimate_dred_bitrate(
    int q0,
    int dQ,
    int qmax,
    int duration,
    int target_bits,
    int *target_chunks
) {
    return ropus_c_estimate_dred_bitrate_impl(q0, dQ, qmax, duration, target_bits, target_chunks);
}

int ropus_c_compute_dred_bitrate(
    int use_in_band_fec,
    int packet_loss_perc,
    int Fs,
    int dred_duration,
    int bitrate_bps,
    int frame_size,
    int *out_q0,
    int *out_dQ,
    int *out_qmax,
    int *out_target_chunks
) {
    return (int)ropus_c_compute_dred_bitrate_impl(
        use_in_band_fec,
        packet_loss_perc,
        (opus_int32)Fs,
        dred_duration,
        (opus_int32)bitrate_bps,
        frame_size,
        out_q0, out_dQ, out_qmax, out_target_chunks
    );
}
