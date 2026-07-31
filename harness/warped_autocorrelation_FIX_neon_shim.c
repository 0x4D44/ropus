#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "main_FIX.h"

/*
 * The pinned Opus reference's AArch64 implementation loses precision in
 * warped_autocorrelation_FIX_neon. Keep the presumed-NEON selector intact,
 * but route this one known-divergent routine through its scalar oracle.
 *
 * Upstream tracks the correction in xiph/opus#473. Remove this shim when the
 * pinned reference includes that fix and OPUS_CHECK_ASM passes this path.
 */
void silk_warped_autocorrelation_FIX_neon(
    opus_int32 *corr,
    opus_int *scale,
    const opus_int16 *input,
    const opus_int warping_Q16,
    const opus_int length,
    const opus_int order
)
{
    silk_warped_autocorrelation_FIX_c(
        corr, scale, input, warping_Q16, length, order
    );
}
