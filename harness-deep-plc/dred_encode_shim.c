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
#include "dred_config.h"
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
