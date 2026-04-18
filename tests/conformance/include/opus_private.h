/* Minimal opus_private.h stub for conformance tests.
 *
 * `reference/tests/test_opus_encode.c:47` and
 * `reference/tests/test_opus_extensions.c:43` both do
 *     #include "../src/opus_private.h"
 * which would otherwise resolve (relative to the source file) to
 * `reference/src/opus_private.h` — the real upstream header, which
 * drags in arch.h, celt.h, and hundreds of declarations we don't need.
 *
 * We force-include THIS file before any other header (via `/FI` on
 * MSVC / `-include` on GCC-clang), which claims the `OPUS_PRIVATE_H`
 * header guard and makes the subsequent real include a no-op.
 *
 * Contents are the minimal subset of `reference/src/opus_private.h`
 * that the in-scope conformance tests reference:
 *   - `MODE_SILK_ONLY`/`MODE_HYBRID`/`MODE_CELT_ONLY`,
 *     `OPUS_SET_FORCE_MODE` CTL (for test_opus_encode.c)
 *   - `opus_extension_data` struct, the five
 *     `opus_packet_extensions_*` functions, `opus_packet_parse_impl`,
 *     `opus_repacketizer_out_range_impl`, and an `OpusRepacketizer`
 *     struct layout with `nb_frames` at the correct byte offset
 *     (for test_opus_extensions.c).
 *
 * The `OpusRepacketizer` layout here is NOT the upstream layout — it
 * mirrors our capi handle so `rp.nb_frames` reads the mirrored slot.
 * See `capi/src/repacketizer.rs` `OpusRepacketizerHandle`. The other
 * fields upstream exposes (`toc`, `frames[48]`, `len[48]`, …) are not
 * accessed by `test_opus_extensions.c`, so they remain padding.
 */
#ifndef OPUS_PRIVATE_H
#define OPUS_PRIVATE_H

/* Upstream `opus_private.h` does `#include "opus.h"` at its top, which
 * transitively brings in opus_defines.h, opus_types.h, and the public
 * typedef `OpusRepacketizer`. `test_opus_extensions.c` relies on that
 * chain (it includes no other opus header). Preserve the behaviour. */
#include "opus.h"
#include "opus_multistream.h"
#include "opus_defines.h"
#include "opus_types.h"
/* test_opus_encode.c uses `IMIN` (line 264) but never explicitly includes
 * arch.h — it assumes the real opus_private.h drags it in. Our stub has to
 * preserve that transitive include so the test body still resolves. */
#include "arch.h"

#define MODE_SILK_ONLY   1000
#define MODE_HYBRID      1001
#define MODE_CELT_ONLY   1002

#define OPUS_SET_FORCE_MODE_REQUEST   11002
#define OPUS_SET_FORCE_MODE(x)        OPUS_SET_FORCE_MODE_REQUEST, opus_check_int(x)

/* ---------------------- Packet extensions ------------------------ */

typedef struct {
   int id;
   int frame;
   const unsigned char *data;
   opus_int32 len;
} opus_extension_data;

int opus_packet_parse_impl(const unsigned char *data, opus_int32 len,
      int self_delimited, unsigned char *out_toc,
      const unsigned char *frames[48], opus_int16 size[48],
      int *payload_offset, opus_int32 *packet_offset,
      const unsigned char **padding, opus_int32 *padding_len);

opus_int32 opus_packet_extensions_count(const unsigned char *data,
 opus_int32 len, int nb_frames);

opus_int32 opus_packet_extensions_count_ext(const unsigned char *data,
 opus_int32 len, opus_int32 *nb_frame_exts, int nb_frames);

opus_int32 opus_packet_extensions_parse(const unsigned char *data,
 opus_int32 len, opus_extension_data *extensions, opus_int32 *nb_extensions,
 int nb_frames);

opus_int32 opus_packet_extensions_parse_ext(const unsigned char *data,
 opus_int32 len, opus_extension_data *extensions, opus_int32 *nb_extensions,
 const opus_int32 *nb_frame_exts, int nb_frames);

opus_int32 opus_packet_extensions_generate(unsigned char *data, opus_int32 len,
 const opus_extension_data *extensions, opus_int32 nb_extensions,
 int nb_frames, int pad);

/* ---------------------- Repacketizer internals ------------------- */

/* Layout mirror of `capi::OpusRepacketizerHandle`. `nb_frames` at byte
 * offset 24 must match the Rust-side mirror field — `static_assert` in
 * the capi crate pins the offset, and drift surfaces here as a test
 * misread of `rp.nb_frames`. Upstream `struct OpusRepacketizer` layout
 * is intentionally NOT reproduced; `test_opus_extensions.c` reads only
 * `nb_frames`. */
struct OpusRepacketizer {
   unsigned char _prefix[24];   /* magic (8) + inner ptr (8) + generation (8) */
   int nb_frames;                /* offset 24 */
   unsigned char _pad[36];       /* total size = 64 */
};

opus_int32 opus_repacketizer_out_range_impl(OpusRepacketizer *rp, int begin, int end,
      unsigned char *data, opus_int32 maxlen, int self_delimited, int pad,
      const opus_extension_data *extensions, int nb_extensions);

#endif /* OPUS_PRIVATE_H */
