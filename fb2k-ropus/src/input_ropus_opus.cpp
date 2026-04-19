// input_ropus_opus.cpp — M4 scaffold for the .opus input.
//
// This is a deliberately inert skeleton: each method logs its name and
// either returns a safe zero/false or throws exception_io_unsupported_format
// so fb2k can route requests through to the next decoder in priority order.
// No Rust FFI, no Ogg demux, no audio. M5 replaces the bodies with calls
// into the ropus_fb2k staticlib linked into this DLL (see CMakeLists.txt).
//
// Registration shape follows sdk/foobar2000/foo_sample/input_raw.cpp:
// inherit from `input_stubs` (supplies stub implementations of the rarely-
// used input_decoder_v4 methods), implement the `input_singletrack_impl`
// surface, and register via `input_singletrack_factory_t<>`.

#include <SDK/foobar2000.h>

namespace {

class input_ropus_opus : public input_stubs {
public:
    // --- input_singletrack_impl surface ------------------------------------

    void open(service_ptr_t<file> p_filehint,
              const char* p_path,
              t_input_open_reason p_reason,
              abort_callback& p_abort) {
        FB2K_console_formatter() << "[ropus] open reason=" << static_cast<int>(p_reason)
                                 << " path=" << (p_path ? p_path : "(null)");

        // We do not support tagging. Throwing here causes fb2k to fall
        // through to the next handler for info_write requests, while still
        // allowing info_read and decode attempts through our pipeline.
        if (p_reason == input_open_info_write) {
            throw exception_tagging_unsupported();
        }

        // Ensure we have an open file handle for the M5 pipeline to use.
        // `input_open_file_helper` is a no-op when p_filehint is already
        // non-null. We stash it even though M4 never reads from it — this
        // keeps the lifetime story identical to the eventual M5 wiring.
        m_file = p_filehint;
        input_open_file_helper(m_file, p_path, p_reason, p_abort);
    }

    void get_info(file_info& p_info, abort_callback& p_abort) {
        (void)p_abort;
        FB2K_console_formatter() << "[ropus] get_info";

        // Empty info until the Rust reader is wired in. Returning zeros
        // for sample_rate / channels is legal here — fb2k will refuse to
        // play, which is exactly the failure mode we want for M4.
        p_info.reset();
    }

    t_filestats2 get_stats2(uint32_t f, abort_callback& p_abort) {
        FB2K_console_formatter() << "[ropus] get_stats2 flags=" << f;
        return m_file->get_stats2_(f, p_abort);
    }

    void decode_initialize(unsigned p_flags, abort_callback& p_abort) {
        (void)p_flags;
        (void)p_abort;
        FB2K_console_formatter() << "[ropus] decode_initialize flags=" << p_flags;
        // Throwing from a zero-body decode_initialize would leave
        // foobar2000 with no graceful recovery; the decode_run stub
        // returning EOF is enough to indicate "nothing to play".
    }

    bool decode_run(audio_chunk& p_chunk, abort_callback& p_abort) {
        (void)p_chunk;
        (void)p_abort;
        FB2K_console_formatter() << "[ropus] decode_run (stub → EOF)";
        return false; // immediate EOF; no samples produced
    }

    void decode_seek(double p_seconds, abort_callback& p_abort) {
        (void)p_seconds;
        (void)p_abort;
        FB2K_console_formatter() << "[ropus] decode_seek t=" << p_seconds;
    }

    bool decode_can_seek() {
        FB2K_console_formatter() << "[ropus] decode_can_seek → false";
        return false;
    }

    void retag(const file_info& p_info, abort_callback& p_abort) {
        (void)p_info;
        (void)p_abort;
        throw exception_tagging_unsupported();
    }

    void remove_tags(abort_callback& p_abort) {
        (void)p_abort;
        throw exception_tagging_unsupported();
    }

    // --- static factory hooks ---------------------------------------------

    static bool g_is_our_content_type(const char* p_content_type) {
        // fb2k asks by MIME type for network streams. Match audio/opus only;
        // audio/ogg is shared with Vorbis/FLAC/Speex-in-Ogg and claiming it
        // while decode_run returns EOF would silently break those streams.
        // Extension match (.opus) continues to cover all file-path dispatch.
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
        // other input_entry the user has installed. DO NOT reuse this GUID
        // in other components.
        static constexpr GUID guid = {
            0x6ca98269, 0x9907, 0x4c04,
            { 0xae, 0x11, 0x93, 0x7d, 0x6e, 0x19, 0x33, 0xb4 }
        };
        return guid;
    }

    // g_is_low_merit() defaults to false via input_stubs, which is what we
    // want (§5.3 of the HLD — we go above foobar2000's built-in libopus
    // entry on first install). We do NOT override it in M4; the priority
    // tweak is formalised in M5 once the decoder is real. Leaving it at
    // the default here is deliberate.

    // g_get_preferences_guid() defaults to pfc::guid_null via input_stubs,
    // which means no Preferences page. Correct for M4.

private:
    service_ptr_t<file> m_file;
};

static input_singletrack_factory_t<input_ropus_opus> g_input_ropus_opus_factory;

} // anonymous namespace

// Register .opus in fb2k's Open-File dialog filters. Independent of the
// input_entry g_is_our_path / g_is_our_content_type dispatch above, which
// drives actual decoder routing.
DECLARE_FILE_TYPE_EX("opus", "Opus audio file", "Opus audio files");
