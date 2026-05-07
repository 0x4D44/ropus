/* Differential-test thunk for `burg_cepstral_analysis` (reference/dnn/freq.c:183).
 *
 * `burg_cepstral_analysis` is already a non-static public symbol in the C
 * reference (declared in `dnn/freq.h:56`), so technically the test could call
 * it directly. The thunk exists only to provide a stable C-side entry point
 * with the harness-conventional `ropus_test_*` name and to keep the FFI
 * surface uniform with the other diff thunks (`ropus_test_dred_*`,
 * `peek_*`).
 *
 * The function has no internal state — it computes the 2 * NB_BANDS = 36
 * cepstral output from a single FRAME_SIZE = 160 sample buffer and is
 * deterministic frame-by-frame. No priming required. See HLD §4 in
 * `wrk_docs/2026.05.07 - HLD - burg-cepstrum-pow-fix.md`.
 */

#include "freq.h"

void ropus_test_burg_cepstral_analysis(const float *x, float *ceps) {
    burg_cepstral_analysis(ceps, x);
}
