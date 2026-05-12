// input_ropus_opus.cpp — real decode path for .opus files, backed by the
// Rust `ropus` codec via the staticlib-linked ropus_fb2k C ABI.
//
// Layout:
//  - IoCtx holds fb2k's file::ptr + a non-owning abort_callback* and a
//    std::exception_ptr slot used to marshal C++ exceptions across the
//    Rust FFI boundary (Rust sees a -1 return; C++ rethrows afterwards).
//  - io_read_cb / io_seek_cb / io_size_cb / io_abort_cb are the C callbacks
//    registered in RopusFb2kIo. Each catches any fb2k exception, stashes
//    it in IoCtx::pending, and returns an error sentinel.
//  - throw_from_ropus_error() rethrows a pending fb2k exception if present,
//    otherwise maps ropus_fb2k_last_error_code() to the appropriate SDK
//    exception per HLD §5.4.
//
// Registration shape unchanged from M4: inherit from input_stubs, implement
// input_singletrack_impl, register via input_singletrack_factory_t<>.

#include <SDK/foobar2000.h>

#include "ropus_fb2k.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <stdexcept>

namespace {

// IoCtx — per-reader bridge between fb2k's file::ptr / abort_callback and
// the plain-C RopusFb2kIo callbacks. The file pointer is refcounted; the
// abort pointer is non-owning and refreshed before each FFI call (fb2k
// hands us a fresh abort_callback& per decode_run / decode_seek).
struct IoCtx {
    service_ptr_t<file> file;
    abort_callback*     abort = nullptr;
    std::exception_ptr  pending;
};

// All 5 callbacks below are strict C-linkage to match the function-pointer
// types in RopusFb2kIo (declared inside extern "C" {} in ropus_fb2k.h). On
// MSVC-x64 C++ and C linkage happen to share a single calling convention so
// plain C++-linkage statics would work, but that's a coincidence of this
// target; the extern "C" wrapper makes it portable and ISO-conformant. Note
// that extern "C" functions are still allowed to be file-static.
extern "C" {

static int64_t io_read_cb(void* ctx, uint8_t* buf, size_t n) {
    IoCtx* io = static_cast<IoCtx*>(ctx);
    try {
        t_size got = io->file->read(buf, n, *io->abort);
        // Defensive clamp against adversarial file subclasses that could in
        // theory return got > INT64_MAX on a 64-bit t_size. Impossible for
        // any real fb2k file implementation, but the cast would otherwise
        // be implementation-defined.
        return got <= static_cast<t_size>(INT64_MAX)
            ? static_cast<int64_t>(got)
            : INT64_MAX;
    } catch (...) {
        io->pending = std::current_exception();
        return -1;
    }
}

static int io_seek_cb(void* ctx, uint64_t abs_offset) {
    IoCtx* io = static_cast<IoCtx*>(ctx);
    try {
        io->file->seek(static_cast<t_filesize>(abs_offset), *io->abort);
        return 0;
    } catch (...) {
        io->pending = std::current_exception();
        return -1;
    }
}

static int io_size_cb(void* ctx, uint64_t* out_size) {
    IoCtx* io = static_cast<IoCtx*>(ctx);
    try {
        t_filesize sz = io->file->get_size(*io->abort);
        if (sz == filesize_invalid) return -1;
        *out_size = static_cast<uint64_t>(sz);
        return 0;
    } catch (...) {
        io->pending = std::current_exception();
        return -1;
    }
}

static int io_abort_cb(void* ctx) {
    if (ctx == nullptr) return 1;  // treat null as "aborted" — safest default
    IoCtx* io = static_cast<IoCtx*>(ctx);
    if (io->abort == nullptr) return 0;
    try {
        return io->abort->is_aborting() ? 1 : 0;
    } catch (...) {
        // abort_callback::is_aborting is virtual and not declared noexcept
        // (abort_callback.h:20). A custom subclass could throw. Returning 1
        // (abort) is the safer default than allowing the exception to unwind
        // into Rust (UB per extern "C" fn ABI). We don't stash pending here
        // because Rust's abort channel doesn't round-trip a message.
        return 1;
    }
}

// Tag callback target: Rust invokes once per OpusTags entry. Key/value
// pointers are only valid for the duration of this call — file_info's
// meta_add / meta_add_value copy them internally, so this is safe.
static void tag_cb(void* ctx, const char* key, const char* value) {
    if (ctx == nullptr || key == nullptr || value == nullptr) return;
    file_info* info = static_cast<file_info*>(ctx);

    // Filter the synthetic VENDOR tag. The OpusTags vendor string is header
    // metadata (encoder identification), not a user-editable vorbis comment;
    // surfacing it through file_info would mislead users into thinking it's
    // a real tag. The Rust side emits it for completeness; we drop it here.
    if (stricmp_utf8(key, "VENDOR") == 0) return;

    // Multi-value tags: the vorbis-comment spec allows the same key to
    // appear multiple times (e.g. two ARTIST entries for a collaboration).
    // meta_add would store each as a *separate* meta entry, and fb2k's
    // meta_get_count_by_name(key) would then return 1 regardless of how
    // many are actually present. Instead, on the second+ occurrence of a
    // key, append the value to the existing entry via meta_add_value so
    // meta_get_count_by_name reports the right count.
    t_size idx = info->meta_find(key);
    if (idx == SIZE_MAX) {
        info->meta_add(key, value);
    } else {
        info->meta_add_value(idx, value);
    }
}

} // extern "C"

class input_ropus_opus : public input_stubs {
public:
    ~input_ropus_opus() {
        if (m_reader != nullptr) {
            ropus_fb2k_close(m_reader);
            m_reader = nullptr;
        }
    }

    // --- input_singletrack_impl surface ------------------------------------

    void open(service_ptr_t<file> p_filehint,
              const char* p_path,
              t_input_open_reason p_reason,
              abort_callback& p_abort) {
        // Defend against re-open: release any prior reader before starting.
        if (m_reader != nullptr) {
            ropus_fb2k_close(m_reader);
            m_reader = nullptr;
        }

        // We do not support tagging. Throwing here causes fb2k to fall
        // through to the next handler for info_write requests.
        if (p_reason == input_open_info_write) {
            throw exception_tagging_unsupported();
        }

        // Resolve the file handle (no-op when hint is non-null).
        m_file = p_filehint;
        input_open_file_helper(m_file, p_path, p_reason, p_abort);

        // Bind the IO context. abort is refreshed per-call below.
        m_io_ctx.file    = m_file;
        m_io_ctx.abort   = &p_abort;
        m_io_ctx.pending = nullptr;

        RopusFb2kIo io = {};
        io.ctx   = &m_io_ctx;
        io.read  = &io_read_cb;
        io.seek  = m_file->can_seek() ? &io_seek_cb : nullptr;
        io.size  = &io_size_cb;
        io.abort = &io_abort_cb;

        const uint32_t flags = (p_reason == input_open_info_read)
                                   ? ROPUS_FB2K_OPEN_INFO_ONLY
                                   : 0u;

        m_reader = ropus_fb2k_open(&io, flags);
        if (m_reader == nullptr) {
            throw_from_ropus_error(/*during_open=*/true);
        }

        // Cache RopusFb2kInfo; needed by get_info + decode_can_seek.
        // Note: m_info.pre_skip is consumed transparently by
        // ropus_fb2k_decode_next (RFC 7845 §4.2); the C++ side never reads
        // or forwards it.
        const int rc = ropus_fb2k_get_info(m_reader, &m_info);
        if (rc != 0) {
            throw_from_ropus_error(/*during_open=*/true);
        }
    }

    void get_info(file_info& p_info, abort_callback& p_abort) {
        (void)p_abort;

        p_info.reset();

        // Standard numeric info.
        p_info.info_set_int("samplerate", static_cast<t_int64>(m_info.sample_rate));
        p_info.info_set_int("channels",   static_cast<t_int64>(m_info.channels));
        p_info.info_set("codec", "Opus");
        p_info.info_set("encoding", "lossy");

        if (m_info.total_samples > 0 && m_info.sample_rate > 0) {
            p_info.set_length(
                static_cast<double>(m_info.total_samples) /
                static_cast<double>(m_info.sample_rate));
        }

        if (m_info.nominal_bitrate > 0) {
            // info_set_bitrate asserts > 0 and takes kbps.
            const t_int64 kbps = (m_info.nominal_bitrate + 500) / 1000;
            if (kbps > 0) p_info.info_set_bitrate(kbps);
        }

        // ReplayGain. NaN in RopusFb2kInfo means "tag absent".
        replaygain_info rg = p_info.get_replaygain();
        if (!std::isnan(m_info.rg_track_gain)) rg.m_track_gain = m_info.rg_track_gain;
        if (!std::isnan(m_info.rg_album_gain)) rg.m_album_gain = m_info.rg_album_gain;
        if (!std::isnan(m_info.rg_track_peak)) rg.m_track_peak = m_info.rg_track_peak;
        if (!std::isnan(m_info.rg_album_peak)) rg.m_album_peak = m_info.rg_album_peak;
        p_info.set_replaygain(rg);

        // Vorbis-comment style tags via the Rust-side callback. file_info's
        // meta_add copies the key/value strings, so it's safe for those
        // pointers to go out of scope once the callback returns.
        if (m_reader != nullptr) {
            m_io_ctx.abort   = &p_abort;
            m_io_ctx.pending = nullptr;
            const int rc = ropus_fb2k_read_tags(m_reader, &tag_cb, &p_info);
            if (rc != 0) {
                throw_from_ropus_error(/*during_open=*/false);
            }
        }
    }

    t_filestats2 get_stats2(uint32_t f, abort_callback& p_abort) {
        return m_file->get_stats2_(f, p_abort);
    }

    void decode_initialize(unsigned p_flags, abort_callback& p_abort) {
        (void)p_flags;
        FB2K_console_formatter() << "[ropus] decode_initialize flags=" << p_flags;
        // SDK contract (input.h:73-74): decode_initialize resets playback
        // position to the beginning and may be called more than once. Rust's
        // reader retains state across decode_next calls, so we must explicitly
        // seek to 0 here to reset it. Also refresh the abort pointer and clear
        // any stale pending exception from a prior FFI call.
        m_io_ctx.abort   = &p_abort;
        m_io_ctx.pending = nullptr;
        if (m_reader != nullptr) {
            const int rc = ropus_fb2k_seek(m_reader, 0);
            if (rc != 0) throw_from_ropus_error(/*during_open=*/false);
        }

        // Seed the live-bitrate EWMA with the file-wide average so the status
        // bar starts at a sensible value rather than ramping from zero. If the
        // nominal bitrate is unknown (-1), leave at 0.0 — decode_run's
        // cold-start branch will seed directly from the first real packet.
        // Reset m_reported_kbps so the next dynamic-info call re-asserts the
        // value (matters across decode_initialize replays).
        m_smoothed_bps  = m_info.nominal_bitrate > 0
                              ? static_cast<double>(m_info.nominal_bitrate)
                              : 0.0;
        m_reported_kbps = -1;
    }

    bool decode_run(audio_chunk& p_chunk, abort_callback& p_abort) {
        if (m_reader == nullptr) return false;

        m_io_ctx.abort   = &p_abort;
        m_io_ctx.pending = nullptr;

        uint64_t bytes_consumed = 0;
        const int samples = ropus_fb2k_decode_next(
            m_reader,
            m_decode_buf.data(),
            MAX_SAMPLES_PER_CHANNEL,
            &bytes_consumed);

        if (samples < 0) {
            throw_from_ropus_error(/*during_open=*/false);
        }
        if (samples == 0) {
            return false; // EOF
        }

        const unsigned channels = m_info.channels > 0 ? m_info.channels : 1;
        const unsigned srate    = m_info.sample_rate != 0 ? m_info.sample_rate : 48000;

        // Update the live-bitrate EWMA from this call's bytes-consumed signal.
        // Smoothing parameters and display logic live here on the C++ side
        // (HLD §4.1): the Rust reader stays UI-agnostic, the shim translates.
        {
            const double inst_bps = static_cast<double>(bytes_consumed) * 8.0 *
                                    static_cast<double>(srate) /
                                    static_cast<double>(samples);
            if (m_smoothed_bps <= 0.0) {
                // Cold start (unseekable stream / nominal_bitrate < 0). Seed
                // directly from the first real packet rather than ramping from
                // zero, which would otherwise flash 3 → 6 → 9 kbps in the
                // status bar over the first second of playback.
                m_smoothed_bps = inst_bps;
            } else {
                constexpr double ALPHA = 0.02;   // ~1 s @ 20 ms packets
                m_smoothed_bps = ALPHA * inst_bps + (1.0 - ALPHA) * m_smoothed_bps;
            }
        }

        // set_data_32 takes float* directly — audio_sample is double on x64,
        // and it performs the float→double conversion internally (with
        // g_guess_channel_config(channels) for the channel mask).
        p_chunk.set_data_32(
            m_decode_buf.data(),
            static_cast<size_t>(samples),
            channels,
            srate);
        return true;
    }

    // Called by fb2k after every successful decode_run (input.h:99-103), on
    // the same thread — no dirty flag or atomics needed. We round the
    // smoothed bps to integer kbps and report it only when it changes,
    // matching fb2k's display resolution. Returning false leaves the
    // existing readout untouched.
    bool decode_get_dynamic_info(file_info& p_out, double& p_timestamp_delta) {
        const int kbps = static_cast<int>((m_smoothed_bps + 500.0) / 1000.0);
        if (kbps == m_reported_kbps || kbps <= 0) return false;
        p_out.info_set_bitrate(kbps);
        p_timestamp_delta = 0.0;   // applies as of the chunk we just delivered
        m_reported_kbps   = kbps;
        return true;
    }

    void decode_seek(double p_seconds, abort_callback& p_abort) {
        if (m_reader == nullptr) return;
        m_file->ensure_seekable();

        m_io_ctx.abort   = &p_abort;
        m_io_ctx.pending = nullptr;

        // Guard against NaN/Inf before the uint64_t cast. `x < 0.0` is false
        // for NaN, and `static_cast<uint64_t>(NaN)` / `(Inf)` are UB by the
        // standard. Clamp to 0 on anything non-finite as well as negatives.
        if (!std::isfinite(p_seconds) || p_seconds < 0.0) p_seconds = 0.0;
        const double rate = m_info.sample_rate != 0 ? m_info.sample_rate : 48000.0;
        const uint64_t target = static_cast<uint64_t>(p_seconds * rate + 0.5);

        const int rc = ropus_fb2k_seek(m_reader, target);
        if (rc != 0) {
            throw_from_ropus_error(/*during_open=*/false);
        }
    }

    bool decode_can_seek() {
        return m_file.is_valid() && m_file->can_seek() && m_info.total_samples > 0;
    }

    void retag(const file_info& /*p_info*/, abort_callback& /*p_abort*/) {
        throw exception_tagging_unsupported();
    }

    void remove_tags(abort_callback& /*p_abort*/) {
        throw exception_tagging_unsupported();
    }

    // --- static factory hooks ---------------------------------------------

    static bool g_is_our_content_type(const char* p_content_type) {
        // fb2k asks by MIME type for network streams. Match audio/opus only;
        // audio/ogg is shared with Vorbis/FLAC/Speex-in-Ogg and claiming it
        // would hijack unrelated streams. Extension match (.opus) covers
        // all file-path dispatch.
        return stricmp_utf8(p_content_type, "audio/opus") == 0;
    }

    static bool g_is_our_path(const char* /*p_path*/, const char* p_extension) {
        return stricmp_utf8(p_extension, "opus") == 0;
    }

    static const char* g_get_name() {
        return "ropus (Rust Opus decoder)";
    }

    static GUID g_get_guid() {
        // {6CA98269-9907-4C04-AE11-937D6E1933B4}
        // Generated fresh for this component so we never collide with any
        // other input_entry the user has installed. DO NOT reuse this GUID.
        static constexpr GUID guid = {
            0x6ca98269, 0x9907, 0x4c04,
            { 0xae, 0x11, 0x93, 0x7d, 0x6e, 0x19, 0x33, 0xb4 }
        };
        return guid;
    }

    // Explicit per HLD §5.3 — new installs land at the top of the decoder
    // priority list so we beat the built-in libopus entry for .opus files.
    static bool g_is_low_merit() { return false; }

private:
    // Longest Opus frame is 120 ms = 5760 samples @ 48 kHz, per RFC 6716 §2.
    // Stereo worst case → 5760 × 2 floats in a single decode_next call.
    static constexpr size_t MAX_SAMPLES_PER_CHANNEL = 5760;

    [[noreturn]] void throw_from_ropus_error(bool during_open) {
        // 1. A fb2k exception captured in a callback wins over the Rust
        //    status string: rethrowing preserves the original type
        //    (exception_aborted, exception_io_data, ...).
        if (m_io_ctx.pending) {
            std::exception_ptr e;
            std::swap(e, m_io_ctx.pending);
            std::rethrow_exception(e);
        }

        // 2. Fall through to Rust's last-error code + message. Mapping per
        //    HLD §5.4 (canonical):
        //      -1 BAD_ARG        → exception_io_data (programmer error; shouldn't reach users)
        //      -2 IO             → exception_io_data (IO call returned -1)
        //      -3 ABORTED        → exception_aborted (user cancelled)
        //      -4 INVALID_STREAM → exception_io_data (corrupt / truncated)
        //      -5 UNSUPPORTED    → exception_io_unsupported_format (feature we don't handle; fb2k falls through)
        //      -6 INTERNAL       → exception_io_data (Rust panic caught; surface as corrupt)
        //
        // A previous iteration split -4 to `exception_io_unsupported_format`
        // during open to trigger fb2k's next-input fall-through. That was
        // wrong: for `.opus` we're the only decoder, and claiming a corrupt
        // `.opus` is an "unsupported format" is semantically false (the
        // format IS opus, it's just bad). HLD §5.4 gets this right with
        // unconditional exception_io_data for -4. The `during_open`
        // parameter is retained in the signature for call-site readability
        // but is no longer load-bearing for the mapping.
        (void)during_open;

        const int code       = ropus_fb2k_last_error_code();
        const char* const msg = ropus_fb2k_last_error();

        switch (code) {
        case -3: // ABORTED — user cancellation.
            throw exception_aborted();
        case -5: // UNSUPPORTED — tells fb2k to fall through to next input.
            throw exception_io_unsupported_format(msg);
        default: // -1 BAD_ARG, -2 IO, -4 INVALID_STREAM, -6 INTERNAL, unexpected.
            throw exception_io_data(msg);
        }
    }

    service_ptr_t<file> m_file;
    IoCtx               m_io_ctx;
    RopusFb2kReader*    m_reader = nullptr;
    RopusFb2kInfo       m_info   = {};
    // Live-VBR readout state (HLD §4.4). EWMA in bits/sec, plus the last
    // integer kbps we pushed to fb2k via decode_get_dynamic_info. Seeded in
    // decode_initialize, updated in decode_run.
    double              m_smoothed_bps  = 0.0;
    int                 m_reported_kbps = -1;
    std::array<float, MAX_SAMPLES_PER_CHANNEL * 2> m_decode_buf{};
};

static input_singletrack_factory_t<input_ropus_opus> g_input_ropus_opus_factory;

} // anonymous namespace

// Register .opus in fb2k's Open-File dialog filters. Independent of the
// input_entry g_is_our_path / g_is_our_content_type dispatch above, which
// drives actual decoder routing.
DECLARE_FILE_TYPE_EX("opus", "Opus audio file", "Opus audio files");
