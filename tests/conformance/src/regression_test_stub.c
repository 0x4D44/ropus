/* Stub implementation of `regression_test(void)` for `test_opus_encode.c`.
 *
 * Background: upstream `test_opus_encode` links both `test_opus_encode.c`
 * (which calls `regression_test()`) and `opus_encode_regressions.c` (which
 * defines it). The regressions file reproduces historical crashes by
 * driving `opus_multistream_surround_encoder_create` with odd channel
 * counts (1, 192, 255...) and `opus_projection_ambisonics_encoder_create`
 * for ambisonics configs.
 *
 * Neither surround-mode encoding (`OpusMSEncoder::new_surround` with
 * non-trivial mapping families) nor ambisonic projection is validated
 * in ropus — per the `publish-as-crate` HLD's non-goals, both are out of
 * scope. Linking `opus_encode_regressions.c` would drag in those paths
 * and likely wedge on the first surround `encode()` call.
 *
 * Compromise: provide a no-op `regression_test()` so `test_opus_encode.c`
 * links and the main test body (which already stresses the single-stream
 * encoder + 2-channel dual-mono multistream thoroughly) still runs. The
 * cost is losing regression-fixture coverage for those specific historical
 * crashes; those live upstream of our scope and would require porting
 * surround analysis + projection before we can check against them.
 *
 * Noted as a deviation from brief in the phase-4 report so the supervisor
 * can decide whether to scope surround/projection into a later phase.
 */
#include <stdio.h>

void regression_test(void)
{
    fprintf(stderr,
        "  [mdopus conformance] regression_test() stubbed — "
        "surround + projection regressions are out of scope.\n");
}
