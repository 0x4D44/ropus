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

void *ropus_test_c_encoder_new(int fs, int channels, int application, int dred_duration) {
    int err = 0;
    OpusEncoder *enc = opus_encoder_create(fs, channels, application, &err);
    if (err != OPUS_OK || !enc) return NULL;
    /* Bump bitrate so there's room for DRED + audio. */
    opus_encoder_ctl(enc, OPUS_SET_BITRATE(32000));
    opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY(5));
    /* Default (!USE_WEIGHTS_FILE) doesn't auto-load the FEC blob, but
     * `dred_encoder_init` wired into `opus_encoder_init` does the compile-
     * time load for us. Enable DRED + loss perc so the encoder allocates
     * a non-trivial redundancy budget. */
    opus_encoder_ctl(enc, OPUS_SET_PACKET_LOSS_PERC(20));
    opus_encoder_ctl(enc, OPUS_SET_INBAND_FEC(1));
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
