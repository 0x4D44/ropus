/* Minimal no-op DRED stubs for opus_demo.
 *
 * Scope: only symbols actually link-referenced by opus_demo.c's decode path.
 * Verified by grep against reference/src/opus_demo.c, the 7 DRED entry points
 * below are the full set the binary depends on:
 *   opus_dred_decoder_create, opus_dred_alloc, opus_dred_decoder_ctl,
 *   opus_dred_parse, opus_decoder_dred_decode24, opus_dred_free,
 *   opus_dred_decoder_destroy.
 * The other DRED symbols declared in opus.h are never referenced by
 * opus_demo (nor by opus_compare.c), so they aren't stubbed here.
 *
 * Our stubs behave as follows:
 *  - `_create` / `_alloc` — return NULL with `*error = OPUS_UNIMPLEMENTED`.
 *    opus_demo doesn't treat NULL as fatal (see lines 921-922 and the
 *    subsequent `if (dred_dec)` guard at line 926 plus the
 *    `if (dred_input > 0)` guard at line 1148).
 *  - `_parse` — returns 0. opus_demo then sees `dred_input = 0` and skips
 *    the `opus_decoder_dred_decode24` call, falling through to the normal
 *    `opus_decode24(NULL, 0, ...)` PLC path.
 *  - `_destroy` / `_free` — no-op (NULL-safe).
 *
 * Inertness caveat:
 *   These stubs are INERT on RFC 8251 test vectors only. A malformed or
 *   DRED-carrying bitstream that reaches `opus_dred_parse` with len>0 will
 *   hit `OPUS_UNIMPLEMENTED` and silently skip PLC; a valid DRED stream
 *   would see NULL-return pathways that the test harness surfaces as decode
 *   failure, not as "stub was hit". Do not extend the stubs — port real
 *   DRED if that path matters.
 */

#include <stddef.h>
#include "opus.h"

/* OPUS_UNIMPLEMENTED = -5 per opus_defines.h. */
#define OPUS_STUB_UNIMPLEMENTED (-5)

OpusDREDDecoder *opus_dred_decoder_create(int *error) {
    if (error) *error = OPUS_STUB_UNIMPLEMENTED;
    return NULL;
}

void opus_dred_decoder_destroy(OpusDREDDecoder *dec) {
    (void)dec;
}

int opus_dred_decoder_ctl(OpusDREDDecoder *dred_dec, int request, ...) {
    (void)dred_dec;
    (void)request;
    return OPUS_STUB_UNIMPLEMENTED;
}

OpusDRED *opus_dred_alloc(int *error) {
    if (error) *error = OPUS_STUB_UNIMPLEMENTED;
    return NULL;
}

void opus_dred_free(OpusDRED *dec) {
    (void)dec;
}

int opus_dred_parse(OpusDREDDecoder *dred_dec, OpusDRED *dred,
                    const unsigned char *data, opus_int32 len,
                    opus_int32 max_dred_samples, opus_int32 sampling_rate,
                    int *dred_end, int defer_processing) {
    (void)dred_dec;
    (void)dred;
    (void)data;
    (void)len;
    (void)max_dred_samples;
    (void)sampling_rate;
    (void)defer_processing;
    if (dred_end) *dred_end = 0;
    return 0;
}

int opus_decoder_dred_decode24(OpusDecoder *st, const OpusDRED *dred,
                               opus_int32 dred_offset, opus_int32 *pcm,
                               opus_int32 frame_size) {
    (void)st;
    (void)dred;
    (void)dred_offset;
    (void)pcm;
    (void)frame_size;
    return OPUS_STUB_UNIMPLEMENTED;
}
