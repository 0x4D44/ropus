//! Shared integration-test helpers for the `ropus-fb2k` C ABI.
//!
//! Every integration test drives the public C entry points (not the old
//! `open_rust` rlib helper, which was dropped in an earlier review). These
//! helpers keep the individual test cases short by pushing fixture
//! construction and the mem-backed `RopusFb2kIo` boilerplate into one place.
//!
//! `MemIo` stores all state in a `Box` whose pointer is passed as the `ctx`
//! field on `RopusFb2kIo`, so multiple tests (and multiple threads) can run
//! in parallel without a global Mutex. The test owns the `MemIo`; the FFI
//! handle only borrows from it through the raw pointer we hand back.

#![allow(dead_code)] // each test pulls different helpers from this file.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;
use std::sync::{Mutex, OnceLock};

use ogg::writing::{PacketWriteEndInfo, PacketWriter};
use ropus::{Application, Channels, Encoder};

use ropus_fb2k::{RopusFb2kIo, RopusFb2kReader};

// ---------------------------------------------------------------------------
// Fixture construction
// ---------------------------------------------------------------------------

/// Ogg serial we use for fixtures. Same constant `ropus-cli` uses, which is
/// convenient if a failing fixture ever gets dumped to disk and inspected.
pub const FIXTURE_STREAM_SERIAL: u32 = 0xC0DE_C0DE;

/// 20 ms stereo silence, valid OpusHead + empty OpusTags + one audio packet.
/// Cached via `OnceLock` because encoder init is ~20 ms per call.
pub fn minimal_opus_fixture() -> &'static [u8] {
    static CACHE: OnceLock<Vec<u8>> = OnceLock::new();
    CACHE.get_or_init(|| build_opus_fixture("ropus-fb2k-test", &[]))
}

/// Same as `minimal_opus_fixture` but with a real comment block. Used by
/// `open_parses_tags` so a silent-drop bug in the parser can't pass.
pub fn opus_fixture_with_artist_alice() -> &'static [u8] {
    static CACHE: OnceLock<Vec<u8>> = OnceLock::new();
    CACHE.get_or_init(|| build_opus_fixture("ropus-fb2k-test", &[("ARTIST", "Alice")]))
}

/// Build an Ogg Opus file whose OpusTags block carries the given vendor
/// and `(KEY, VALUE)` comments (each as a single `KEY=VALUE` entry).
pub fn build_opus_fixture(vendor: &str, comments: &[(&str, &str)]) -> Vec<u8> {
    build_opus_fixture_with_audio_packets(vendor, comments, 1, None)
}

/// Build an Ogg Opus file with a configurable number of 20 ms silence
/// packets. Every packet has granule stepping of 960 (20 ms × 48 kHz), and
/// the final packet carries the end-of-stream flag. `recorded_pre_skip`
/// lets the caller override `OpusHead::pre_skip` — pass `None` to use the
/// encoder's natural lookahead.
///
/// Used by the decode / duration integration tests
/// (`decode_wiring_matches_direct_ogg_path`, `info_populates_*`) that need
/// a deterministic multi-packet stream with a known duration.
pub fn build_opus_fixture_with_audio_packets(
    vendor: &str,
    comments: &[(&str, &str)],
    audio_packets: usize,
    recorded_pre_skip: Option<u16>,
) -> Vec<u8> {
    build_opus_fixture_audio_source(vendor, comments, audio_packets, recorded_pre_skip, |_| {
        vec![0i16; 960 * 2]
    })
}

/// Like `build_opus_fixture_with_audio_packets` but calls `pcm_for` to
/// synthesise each 20 ms stereo packet's PCM input. The closure receives
/// the packet index (0-based) so it can emit a time-varying signal — used
/// by the seek-tolerance test, which needs a continuous sine distinguishable
/// at every 20 ms step.
pub fn build_opus_fixture_audio_source(
    vendor: &str,
    comments: &[(&str, &str)],
    audio_packets: usize,
    recorded_pre_skip: Option<u16>,
    mut pcm_for: impl FnMut(usize) -> Vec<i16>,
) -> Vec<u8> {
    assert!(audio_packets >= 1, "need at least one audio packet");

    let mut encoder = Encoder::builder(48_000, Channels::Stereo, Application::Audio)
        .build()
        .expect("encoder builds");
    let pre_skip = recorded_pre_skip.unwrap_or_else(|| {
        u16::try_from(encoder.lookahead()).expect("lookahead fits u16")
    });

    let mut encoded: Vec<Vec<u8>> = Vec::with_capacity(audio_packets);
    for i in 0..audio_packets {
        let pcm = pcm_for(i);
        assert_eq!(pcm.len(), 960 * 2, "pcm_for must return a 20 ms stereo frame");
        let mut packet = vec![0u8; 4000];
        let n = encoder.encode(&pcm, &mut packet).expect("encode pcm");
        packet.truncate(n);
        encoded.push(packet);
    }

    let mut out = Vec::with_capacity(4096);
    let mut writer = PacketWriter::new(&mut out);

    writer
        .write_packet(
            build_opus_head(2, 48_000, pre_skip),
            FIXTURE_STREAM_SERIAL,
            PacketWriteEndInfo::EndPage,
            0,
        )
        .expect("write OpusHead page");

    writer
        .write_packet(
            build_opus_tags(vendor, comments),
            FIXTURE_STREAM_SERIAL,
            PacketWriteEndInfo::EndPage,
            0,
        )
        .expect("write OpusTags page");

    for (idx, packet) in encoded.into_iter().enumerate() {
        let is_last = idx + 1 == audio_packets;
        let end = if is_last {
            PacketWriteEndInfo::EndStream
        } else {
            PacketWriteEndInfo::NormalPacket
        };
        let absgp = ((idx + 1) as u64) * 960;
        writer
            .write_packet(packet, FIXTURE_STREAM_SERIAL, end, absgp)
            .expect("write audio page");
    }

    drop(writer);
    out
}

/// Hand-roll an `OpusHead` packet body.
pub fn build_opus_head(channels: u8, input_sample_rate: u32, pre_skip: u16) -> Vec<u8> {
    let mut h = Vec::with_capacity(19);
    h.extend_from_slice(b"OpusHead");
    h.push(1);
    h.push(channels);
    h.extend_from_slice(&pre_skip.to_le_bytes());
    h.extend_from_slice(&input_sample_rate.to_le_bytes());
    h.extend_from_slice(&0i16.to_le_bytes());
    h.push(0);
    h
}

/// Hand-roll an `OpusTags` packet body with the given comments.
pub fn build_opus_tags(vendor: &str, comments: &[(&str, &str)]) -> Vec<u8> {
    let vb = vendor.as_bytes();
    let mut t = Vec::with_capacity(8 + 4 + vb.len() + 4);
    t.extend_from_slice(b"OpusTags");
    t.extend_from_slice(&(vb.len() as u32).to_le_bytes());
    t.extend_from_slice(vb);
    t.extend_from_slice(&(comments.len() as u32).to_le_bytes());
    for (k, v) in comments {
        let entry = format!("{k}={v}");
        t.extend_from_slice(&(entry.len() as u32).to_le_bytes());
        t.extend_from_slice(entry.as_bytes());
    }
    t
}

/// Build an Ogg Opus file whose `OpusHead` claims channel_mapping=1 (family
/// 1 / surround). The reader must reject this as UNSUPPORTED.
pub fn surround_family_fixture() -> Vec<u8> {
    let mut out = Vec::with_capacity(512);
    let mut writer = PacketWriter::new(&mut out);
    let mut head = build_opus_head(2, 48_000, 312);
    head[18] = 1;
    writer
        .write_packet(
            head,
            FIXTURE_STREAM_SERIAL,
            PacketWriteEndInfo::EndPage,
            0,
        )
        .unwrap();
    writer
        .write_packet(
            build_opus_tags("dummy", &[]),
            FIXTURE_STREAM_SERIAL,
            PacketWriteEndInfo::EndStream,
            0,
        )
        .unwrap();
    drop(writer);
    out
}

// ---------------------------------------------------------------------------
// MemIo: an in-memory `RopusFb2kIo` whose state lives in a Box
// ---------------------------------------------------------------------------

/// Mutable state carried by the `ctx` pointer. Wrapped in `Mutex` so the
/// `&` we get in the callbacks is enough to mutate — the alternative would
/// be `UnsafeCell`-per-field, which is uglier.
pub struct MemState {
    pub bytes: Vec<u8>,
    pub pos: usize,
    pub abort: bool,
    /// If set, `read` will count down and then flip `abort` to true on the
    /// `(n+1)`-th call. Used by `abort_halts_open` to exercise the race
    /// where an abort trips mid-stream rather than at time zero.
    pub abort_after_n_reads: Option<usize>,
    /// Running total of `read` callback invocations. Drives the
    /// `open_uses_bounded_reads` assertion — we want the open fast path
    /// bounded by a small constant, not O(file_size / buffer_size).
    pub read_calls: usize,
    /// Running total of `abort` callback invocations. Used by
    /// `decode_propagates_abort` to prove non-vacuousness — if
    /// `decode_next` ever returned without polling the IO layer at all,
    /// neither `read_calls` nor `abort_calls` would move.
    pub abort_calls: usize,
}

/// Owns the `MemState` box and hands out a `RopusFb2kIo` pointing at it.
/// Drop this struct *after* closing any handle opened from it — the C
/// side holds a raw pointer that would otherwise dangle.
pub struct MemIo {
    state: Box<Mutex<MemState>>,
    /// Whether the `seek` callback is wired. Tests that exercise the
    /// unseekable path (`info_without_seek_returns_zero_total_samples`)
    /// flip this off via `without_seek()`.
    has_seek: bool,
}

impl MemIo {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self {
            state: Box::new(Mutex::new(MemState {
                bytes,
                pos: 0,
                abort: false,
                abort_after_n_reads: None,
                read_calls: 0,
                abort_calls: 0,
            })),
            has_seek: true,
        }
    }

    pub fn with_abort_flag(self) -> Self {
        self.state.lock().unwrap().abort = true;
        self
    }

    pub fn with_abort_after(self, n: usize) -> Self {
        self.state.lock().unwrap().abort_after_n_reads = Some(n);
        self
    }

    /// Flip the abort flag after `open` has already succeeded, so the next
    /// `decode_next` is the call that trips. Used by
    /// `decode_propagates_abort` because the reverse-scan inside open has a
    /// variable read-count that would make `with_abort_after(n)` flaky.
    pub fn set_aborting(&self) {
        self.state.lock().unwrap().abort = true;
    }

    /// Pretend this is an unseekable stream (e.g. live HTTP). Drops the
    /// `seek` callback from the published `RopusFb2kIo`; the Rust side
    /// must fall back to the degraded (zero-duration) path.
    pub fn without_seek(mut self) -> Self {
        self.has_seek = false;
        self
    }

    /// Snapshot the running `read` callback count. Used by bounded-read
    /// tests to assert the open fast path doesn't walk the file.
    pub fn read_calls(&self) -> usize {
        self.state.lock().unwrap().read_calls
    }

    /// Snapshot the running `abort` callback count. Used as a
    /// non-vacuousness proof by the abort tests: if a decode call
    /// returned without polling the IO layer at all, neither this nor
    /// `read_calls` would move.
    pub fn abort_calls(&self) -> usize {
        self.state.lock().unwrap().abort_calls
    }

    /// Build a `RopusFb2kIo` pointing at this `MemIo`'s state. The returned
    /// struct is `Copy` — duplicating it is cheap — but its `ctx` points
    /// into `self`, so `self` must outlive any reader opened with it.
    pub fn io(&self) -> RopusFb2kIo {
        RopusFb2kIo {
            ctx: &*self.state as *const Mutex<MemState> as *mut c_void,
            read: Some(mem_read),
            seek: if self.has_seek { Some(mem_seek) } else { None },
            size: Some(mem_size),
            abort: Some(mem_abort),
        }
    }
}

extern "C" fn mem_read(ctx: *mut c_void, out: *mut u8, n: usize) -> i64 {
    // SAFETY: `ctx` was initialised by `MemIo::io()` from a valid
    // `&Mutex<MemState>` reference that outlives this call.
    let m = unsafe { &*(ctx as *const Mutex<MemState>) };
    let mut st = m.lock().unwrap();

    st.read_calls += 1;

    // If the caller asked for mid-stream abort, tick the counter first —
    // the `check_abort` inside `CallbackReader` will then trip on the
    // following call.
    if let Some(ref mut remaining) = st.abort_after_n_reads {
        if *remaining == 0 {
            st.abort = true;
        } else {
            *remaining -= 1;
        }
    }

    let pos = st.pos;
    let remaining = st.bytes.len().saturating_sub(pos);
    let take = remaining.min(n);
    if take > 0 {
        // SAFETY: `out` has capacity for at least `n >= take` bytes per the
        // header contract; we only read `take` bytes from the owned Vec.
        unsafe { ptr::copy_nonoverlapping(st.bytes[pos..].as_ptr(), out, take) };
    }
    st.pos += take;
    take as i64
}

extern "C" fn mem_seek(ctx: *mut c_void, off: u64) -> c_int {
    let m = unsafe { &*(ctx as *const Mutex<MemState>) };
    let mut st = m.lock().unwrap();
    if off > st.bytes.len() as u64 {
        return -1;
    }
    st.pos = off as usize;
    0
}

extern "C" fn mem_size(ctx: *mut c_void, out: *mut u64) -> c_int {
    let m = unsafe { &*(ctx as *const Mutex<MemState>) };
    let st = m.lock().unwrap();
    // SAFETY: header contract: caller supplies a writable u64 destination.
    unsafe { *out = st.bytes.len() as u64 };
    0
}

extern "C" fn mem_abort(ctx: *mut c_void) -> c_int {
    let m = unsafe { &*(ctx as *const Mutex<MemState>) };
    let mut st = m.lock().unwrap();
    st.abort_calls += 1;
    if st.abort { 1 } else { 0 }
}

// ---------------------------------------------------------------------------
// Convenience entry points
// ---------------------------------------------------------------------------

/// Open an Ogg Opus stream from an owned byte vector. Returns the Rust
/// `MemIo` (keeping `ctx` alive) plus the raw handle. Caller must close
/// the handle via `ropus_fb2k::ropus_fb2k_close` before dropping `MemIo`.
///
/// Returns `(io, handle)`. `handle` is null on failure; inspect
/// `ropus_fb2k_last_error[_code]()` for details.
pub fn open_from_bytes(bytes: Vec<u8>) -> (MemIo, *mut RopusFb2kReader) {
    let io = MemIo::new(bytes);
    let fb2k_io = io.io();
    let handle = unsafe { ropus_fb2k::ropus_fb2k_open(&fb2k_io, 0) };
    (io, handle)
}

/// Variant that passes the info-only open flag.
pub fn open_from_bytes_info_only(bytes: Vec<u8>) -> (MemIo, *mut RopusFb2kReader) {
    let io = MemIo::new(bytes);
    let fb2k_io = io.io();
    let handle = unsafe {
        ropus_fb2k::ropus_fb2k_open(&fb2k_io, ropus_fb2k::ROPUS_FB2K_OPEN_INFO_ONLY)
    };
    (io, handle)
}

/// Open a stream whose underlying IO advertises no `seek` callback, mimicking
/// a live HTTP source. `open` must still succeed (HLD §4.3 permits zero
/// duration); `total_samples` / `nominal_bitrate` should reflect the absence.
pub fn open_from_bytes_without_seek(
    bytes: Vec<u8>,
    flags: u32,
) -> (MemIo, *mut RopusFb2kReader) {
    let io = MemIo::new(bytes).without_seek();
    let fb2k_io = io.io();
    let handle = unsafe { ropus_fb2k::ropus_fb2k_open(&fb2k_io, flags) };
    (io, handle)
}

// ---------------------------------------------------------------------------
// Tag callback collector
// ---------------------------------------------------------------------------

/// Call `ropus_fb2k_read_tags` on the given handle and return the collected
/// `(key, value)` pairs (in emission order). Assumes the caller owns the
/// handle for the duration of the call.
pub fn read_tags_collect(reader: *mut RopusFb2kReader) -> Vec<(String, String)> {
    // We pass a `&mut Vec` as `ctx`; the callback pushes into it. That way
    // the capture lifetime is entirely scoped to this function — no thread
    // locals, no globals.
    let mut caps: Vec<(String, String)> = Vec::new();
    let rc = unsafe {
        ropus_fb2k::ropus_fb2k_read_tags(
            reader,
            Some(tag_collect_cb),
            &mut caps as *mut Vec<(String, String)> as *mut c_void,
        )
    };
    assert_eq!(rc, 0, "read_tags returned {rc}");
    caps
}

extern "C" fn tag_collect_cb(ctx: *mut c_void, key: *const c_char, value: *const c_char) {
    // SAFETY: `ctx` was initialised above to point at a `Vec<(String,String)>`;
    // `key` and `value` are valid NUL-terminated C strings per the ABI contract.
    let caps = unsafe { &mut *(ctx as *mut Vec<(String, String)>) };
    let k = unsafe { CStr::from_ptr(key) }.to_string_lossy().into_owned();
    let v = unsafe { CStr::from_ptr(value) }.to_string_lossy().into_owned();
    caps.push((k, v));
}

/// Read the thread-local last-error string as a Rust `String`. Never panics.
pub fn last_error_string() -> String {
    let p = unsafe { ropus_fb2k::ropus_fb2k_last_error() };
    if p.is_null() {
        return String::new();
    }
    // SAFETY: the crate guarantees the pointer is a valid NUL-terminated
    // UTF-8 C string for the duration of this call.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}
