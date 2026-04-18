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

/* Disable float analysis API for bit-exact fixed-point comparison.
   The Rust implementation doesn't have the analysis module, so we need
   to disable it in C to get an apples-to-apples comparison. */
#define DISABLE_FLOAT_API 1

/* Enable x86 SIMD (SSE/SSE2/SSE4.1) with runtime detection */
#define OPUS_HAVE_RTCD 1
#define OPUS_X86_MAY_HAVE_SSE 1
#define OPUS_X86_MAY_HAVE_SSE2 1
#define OPUS_X86_MAY_HAVE_SSE4_1 1
#define CPU_INFO_BY_C 1

/* Disable DNN/ML features (not needed for core codec comparison) */
/* #undef ENABLE_DEEP_PLC */
/* #undef ENABLE_DRED */
/* #undef ENABLE_OSCE */

/* Package info */
#define PACKAGE_VERSION "1.5.2-harness"

#endif /* CONFIG_H */
