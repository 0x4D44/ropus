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

/* Disable all SIMD / arch-specific code */
/* (no OPUS_X86_MAY_HAVE_*, no OPUS_ARM_*, etc.) */

/* Disable DNN/ML features (not needed for core codec comparison) */
/* #undef ENABLE_DEEP_PLC */
/* #undef ENABLE_DRED */
/* #undef ENABLE_OSCE */

/* Package info */
#define PACKAGE_VERSION "1.5.2-harness"

#endif /* CONFIG_H */
