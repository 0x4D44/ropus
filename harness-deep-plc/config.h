/* Float-mode config.h for the sibling DEEP_PLC harness.
 *
 * The main harness (`harness/config.h`) is fixed-point; Xiph upstream explicitly
 * forbids `--enable-fixed-point` + `--enable-deep-plc`
 * (`reference/configure.ac:973` AC_MSG_ERROR). Stage 7b.2 therefore ships a
 * second harness crate (`harness-deep-plc/`) compiled in FLOAT mode with
 * DEEP_PLC enabled, so we can link the full xiph neural PLC against the
 * C reference and compare to ropus's Rust port via an SNR (tier-2) oracle.
 */

#ifndef CONFIG_H
#define CONFIG_H

/* Float-mode (NO FIXED_POINT define). DEEP_PLC is only supported here. */

/* Stack allocation strategy: alloca on Windows, VLAs elsewhere */
#ifdef _WIN32
# define USE_ALLOCA 1
#else
# define VAR_ARRAYS 1
#endif

/* We have C99 lrintf */
#define HAVE_LRINTF 1

/* Keep SIMD off in this harness — scalar only, same choice as the sibling
 * fixed-point harness. Avoids pulling in x86/ARM source files and keeps
 * C/Rust comparison focused on the scalar reference path. */

/* Enable DEEP_PLC and its companion flags. DRED + OSCE sources are linked
 * transitively but their top-level features are NOT enabled — Stage 8 scope. */
#define ENABLE_DEEP_PLC 1

/* Stage 8.8: enable the encoder-side DRED wiring in `opus_encoder.c` so the
 * `OPUS_SET_DRED_DURATION` CTL and the `dred_compute_latents` /
 * `dred_encode_silk_frame` path become live on the C side. Lets the Stage 8.8
 * integration test drive a full C encoder with DRED on and feed its packets
 * to the Rust decoder (and vice versa). */
#define ENABLE_DRED 1

/* Use the built-in compile-time weights from `reference/dnn/*_data.c` —
 * NOT a runtime weights file. This matches xiph's default configure path
 * and means `opus_decoder_create` auto-initialises DEEP_PLC without any
 * `OPUS_SET_DNN_BLOB` call. */
/* #undef USE_WEIGHTS_FILE */

/* Package info */
#define PACKAGE_VERSION "1.5.2-harness-deep-plc"

#endif /* CONFIG_H */
