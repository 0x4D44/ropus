/* ropus_fb2k.h — stable C ABI between the foobar2000 C++ shim and the
 * ropus-fb2k Rust staticlib. Hand-written; reviewed; not generated.
 *
 * License: BSD-3-Clause (matches the rest of the ropus workspace).
 */

#ifndef ROPUS_FB2K_H
#define ROPUS_FB2K_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RopusFb2kReader RopusFb2kReader;

/* ----- IO callbacks supplied by the C++ side ------------------------------
 *
 * All callbacks receive the opaque `ctx` the caller registered in RopusFb2kIo.
 * The C++ shim wraps fb2k's file::ptr and abort_callback behind these; the
 * Rust side never touches the filesystem directly.
 *
 * - `read`:  read up to `n` bytes into `buf`. Return bytes read (>=0), or -1
 *            on I/O error. Return 0 only at EOF. Partial reads are fine —
 *            the Rust side loops until satisfied or EOF.
 *            Must be non-null; a null `read` returns BAD_ARG.
 * - `seek`:  seek to absolute byte offset. Return 0 on success, -1 on error.
 * - `size`:  write total stream length in bytes to *out_size. Return 0 on
 *            success, -1 if unknown (e.g. a live HTTP stream — we then
 *            disable seeking and report total_samples = 0).
 * - `abort`: return non-zero if the user has requested cancellation.
 *            Called frequently; must be cheap.
 */
typedef int64_t (*RopusFb2kReadFn) (void* ctx, uint8_t* buf, size_t n);
typedef int     (*RopusFb2kSeekFn) (void* ctx, uint64_t abs_offset);
typedef int     (*RopusFb2kSizeFn) (void* ctx, uint64_t* out_size);
typedef int     (*RopusFb2kAbortFn)(void* ctx);

typedef struct {
    void*              ctx;
    RopusFb2kReadFn    read;
    RopusFb2kSeekFn    seek;   /* may be NULL for unseekable streams */
    RopusFb2kSizeFn    size;   /* may be NULL if size is unknown */
    RopusFb2kAbortFn   abort;  /* may be NULL -> never aborts */
} RopusFb2kIo;

/* ----- Open flags --------------------------------------------------------- */

/* Info-only: read just enough (first two pages + reverse-scan for last
 * granule) to populate RopusFb2kInfo + tags. Skips the full seek-index
 * build. This is the fast path for fb2k's `input_open_info_read` during
 * library scans. decode_next / seek will still work but will build the
 * seek index lazily on first seek. */
#define ROPUS_FB2K_OPEN_INFO_ONLY  (1u << 0)

typedef struct {
    uint32_t sample_rate;     /* always 48000 for our scope */
    uint8_t  channels;        /* 1 or 2 */
    uint16_t pre_skip;        /* 48 kHz samples; Rust trims these on first decode */
    uint64_t total_samples;   /* per-channel, post-pre_skip; 0 if unknown */
    int32_t  nominal_bitrate; /* bits/s; -1 if unknown */

    /* ReplayGain, extracted from OpusTags at open time. In dB relative to the
     * ReplayGain reference (i.e. already converted from R128's -23 LUFS target
     * by adding +5 dB). NaN means "not present". */
    float rg_track_gain;
    float rg_album_gain;
    float rg_track_peak;      /* linear, NaN if absent */
    float rg_album_peak;
} RopusFb2kInfo;

/* ----- API ---------------------------------------------------------------- *
 *
 * Status convention: 0 = OK, negative = error code.
 * Errors:
 *   -1 BAD_ARG         caller-supplied arguments failed validation
 *   -2 IO              read/seek/size callback reported failure
 *   -3 ABORTED         caller's abort callback signalled cancellation
 *   -4 INVALID_STREAM  stream is malformed or parser rejected its contents
 *   -5 UNSUPPORTED     stream is well-formed but uses an unsupported feature
 *   -6 INTERNAL        panic or impossible state reached inside the Rust
 *                      backend; the call's return value may be a sentinel
 *                      (see `ropus_fb2k_last_error_code` for the unified code)
 * Error codes are stable; human messages via ropus_fb2k_last_error().
 */

RopusFb2kReader* ropus_fb2k_open(const RopusFb2kIo* io, uint32_t flags);
void              ropus_fb2k_close(RopusFb2kReader*);

int ropus_fb2k_get_info(RopusFb2kReader*, RopusFb2kInfo* out);

/* Tag enumeration. Keys are uppercased (per vorbis_comment convention);
 * values are UTF-8, null-terminated, valid only for the duration of the
 * callback. METADATA_BLOCK_PICTURE is filtered out (see scope). */
typedef void (*RopusFb2kTagCb)(void* ctx, const char* key, const char* value);
int ropus_fb2k_read_tags(RopusFb2kReader*, RopusFb2kTagCb cb, void* ctx);

/* Decode next packet into caller buffer (interleaved float, 48 kHz).
 * `max_samples_per_ch` must be >= 5760 (120 ms, the longest Opus frame).
 * Returns samples-per-channel decoded, 0 on EOF, negative on error.
 *
 * `out_bytes_consumed` is required (non-null; a null pointer returns
 * `ROPUS_FB2K_BAD_ARG`). On a successful decode (samples > 0),
 * `*out_bytes_consumed` is set to the sum of encoded packet payload bytes
 * that produced the returned samples. Packets fully discarded as
 * post-seek pre-roll (RFC 7845 §4.2) do NOT contribute — they map to
 * zero caller-visible samples, so charging them to the bitrate signal
 * would skew it (the C++ shim divides bytes by kept-sample count).
 * On EOF (return 0), `*out_bytes_consumed` is set to 0.
 * On error (negative return), it is not written. */
int ropus_fb2k_decode_next(RopusFb2kReader*,
                           float*    out_interleaved,
                           size_t    max_samples_per_ch,
                           uint64_t* out_bytes_consumed);

/* Seek to per-channel sample position (48 kHz). Clamped to [0, total_samples].
 * Post-seek, the next `decode_next` silently discards >= 80 ms per RFC 7845
 * sec. 4.2 before returning real audio; callers do not see the pre-roll. */
int ropus_fb2k_seek(RopusFb2kReader*, uint64_t sample_pos);

/* Thread-local last-error string for the calling thread. Never NULL. */
const char* ropus_fb2k_last_error(void);

/* Thread-local last-error code. Returns the status code (negative) associated
 * with the most recent failed call on this thread, or 0 if the last relevant
 * call succeeded. Paired with `ropus_fb2k_last_error()` for message text. */
int ropus_fb2k_last_error_code(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROPUS_FB2K_H */
