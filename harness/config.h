/* Minimal config.h for building opus reference from the cc crate.
   Platform-independent, fixed-point, no SIMD. */

#ifndef CONFIG_H
#define CONFIG_H

/* Use fixed-point arithmetic for bit-exact reproducibility */
#define FIXED_POINT 1

/* Stack allocation strategy: alloca on Windows, VLAs elsewhere */
#ifdef _WIN32
# define USE_ALLOCA 1
#else
# define VAR_ARRAYS 1
#endif

/* We have C99 lrintf */
#define HAVE_LRINTF 1

/* Float analysis API is enabled on both sides. The Rust port is under
   way (Stage 6 of the 2026-04-19 deferred-work closeout); while that
   is in flight, encode-comparison tests that exercise the analysis
   path will diverge. That divergence is expected and is the visible
   signal that Stage 6 is needed. Do NOT re-gate analysis by adding
   DISABLE_FLOAT_API back unless the closeout is abandoned. */

/* Enable x86 SIMD (SSE/SSE2/SSE4.1) with runtime detection */
#define OPUS_HAVE_RTCD 1
#define OPUS_X86_MAY_HAVE_SSE 1
#define OPUS_X86_MAY_HAVE_SSE2 1
#define OPUS_X86_MAY_HAVE_SSE4_1 1
#define CPU_INFO_BY_C 1

/* DNN / ML features.
 *
 * ENABLE_DEEP_PLC stays OFF here. Upstream xiph explicitly prohibits the
 * combination with fixed-point: `reference/configure.ac:973` aborts
 * autotools with `AC_MSG_ERROR([--enable-fixed-point cannot be used with
 * --enable-deep-plc, --enable-dred, and --enable-osce.])`. It's not a CI
 * gap — it's an enforced incompatibility.
 *
 * The Stage 7a follow-up (see wrk_journals 2026-04-19) investigated a
 * dual-cc::Build workaround ("Option C": DNN sources compiled float,
 * core compiled fixed). The spike found that DNN's `freq.c:245,262`
 * calls `opus_fft(&kfft, ...)` which macro-expands to `opus_fft_c(...)`
 * defined in `celt/kiss_fft.c`. If that function is compiled with
 * FIXED_POINT=1 while `kfft` is defined in a float-compiled
 * `dnn/lpcnet_tables.c`, the struct layouts disagree
 * (`kiss_fft_state` has an extra `scale_shift` under FIXED_POINT,
 * and `kiss_fft_scalar` changes from `float` to `opus_int32`).
 * Cross-TU ABI mismatch → silent memory corruption. Every DNN source
 * transitively includes `kiss_fft.h` via `freq.h`, so the FIXED_POINT
 * view cannot be scoped out. Option C is NOT viable; a cleaner
 * workaround (e.g. vendored patched `lpcnet_tables.c` + float-mode
 * harness, or a second harness binary) is deferred to Stage 7b. */
/* #undef ENABLE_DEEP_PLC */
/* #undef ENABLE_DRED */
/* #undef ENABLE_OSCE */

/* Package info */
#define PACKAGE_VERSION "1.5.2-harness"

#endif /* CONFIG_H */
