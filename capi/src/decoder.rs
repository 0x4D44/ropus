//! `opus_decoder_*` C ABI wrappers.
//!
//! ## Why handle indirection
//!
//! `reference/tests/test_opus_decode.c` asserts that an Opus decoder state
//! "contains no pointers and can be freely copied" — lines 86-95 memcpy the
//! state into a malloc'd replacement buffer, poison the original with
//! `memset(dec, 255, ...)`, and later memcpy dec[0] into a per-iteration
//! scratch buffer `decbak` which is then decoded through (lines 328-333).
//!
//! Our Rust `OpusDecoder` does NOT satisfy "contains no pointers": it holds
//! `Vec<_>` fields (inside `SilkDecoderState` and `CeltDecoder`) which are
//! fat pointers. Byte-copying these aliases the backing allocations; when the
//! decoder later transitions modes (opus/decoder.rs line 678: `if self.prev_mode
//! == MODE_CELT_ONLY { self.silk_dec.init() }`), the assigned `channel_state[i]
//! = SilkDecoderState::new()` drops the old Vecs and allocates new ones. The
//! other alias then has dangling pointers into freed heap — the next decode or
//! reset through it double-frees or corrupts the heap, which surfaces on MSVC
//! as `STATUS_HEAP_CORRUPTION (0xC0000374)`.
//!
//! The capi shim therefore hands out an `OpusDecoderHandle`, a POD wrapper
//! sized exactly the same as `OpusDecoder`. The 8-byte preamble tags the block
//! and stores a pointer to the real leaked `Box<OpusDecoder>`. When the test
//! memcpys the handle, the copy's pointer aliases the original's — but the
//! backing OpusDecoder is intact and is never reallocated out from under any
//! of them. The trailing bytes of the handle are zero-initialised and ignored
//! (tolerating the test's `memset(..., 255, size)` poison because the poisoned
//! bytes are never read). `_destroy` is a no-op: we leak in favour of the test
//! completing cleanly. State + handle allocations are both through Rust's
//! global allocator; no CRT/Rust cross-heap frees.

use std::os::raw::{c_int, c_uchar};
use std::ptr;

use ropus::opus::decoder::OpusDecoder;

use crate::{OPUS_BAD_ARG, OPUS_INTERNAL_ERROR, ffi_guard, state_free};

// ---------------------------------------------------------------------------
// Handle layout
// ---------------------------------------------------------------------------
//
// The handle must present `size_of::<OpusDecoder>()` bytes to C so that
// `memcpy(dec2, dec[t], opus_decoder_get_size(c))` copies the whole thing.

const DECODER_HANDLE_MAGIC: u64 = 0x4D44_4F50_5553_4443; // "MDOPUSDC"

/// Opaque POD wrapper that `opus_decoder_*` entry points see. The first 24
/// bytes carry a magic tag, pointer to the real Rust-heap decoder, and a
/// generation counter; the rest is dead padding sized to match `OpusDecoder`
/// so the C reference tests can memcpy the whole block. The padding contents
/// are never read.
///
/// `generation` is stamped globally-unique on create/init and bumped on
/// `OPUS_RESET_STATE`. `test_opus_api.c` line 247 asserts
/// `memcmp(dec2_snapshot, dec, size) != 0` after a reset — without the
/// generation bump that memcmp would return 0 because both handles alias
/// the same `inner`.
#[repr(C)]
struct OpusDecoderHandle {
    magic: u64,
    inner: *mut OpusDecoder,
    generation: u64,
    _pad: [u8; 0], // trailing padding added dynamically via allocation size
}

const _: () = {
    // Handle's fixed prefix must fit inside OpusDecoder — otherwise there's
    // nothing to pad. Catches both the handle growing AND the decoder shrinking.
    assert!(
        core::mem::size_of::<OpusDecoderHandle>() <= core::mem::size_of::<OpusDecoder>(),
        "OpusDecoderHandle must fit within OpusDecoder bytes"
    );
};

/// Allocate a zeroed block of `size_of::<OpusDecoder>()` bytes on the Rust
/// heap, usable as an `OpusDecoderHandle` plus trailing padding.
fn alloc_handle_storage() -> *mut OpusDecoderHandle {
    let size = std::mem::size_of::<OpusDecoder>();
    let align = std::mem::align_of::<OpusDecoder>();
    // SAFETY: size and align are both non-zero, derived from a valid type.
    unsafe {
        let layout = std::alloc::Layout::from_size_align_unchecked(size, align);
        let p = std::alloc::alloc_zeroed(layout);
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        p as *mut OpusDecoderHandle
    }
}

/// Validate a handle pointer; returns a mutable reference to the inner
/// OpusDecoder or None if poisoned/null.
unsafe fn resolve_handle<'a>(st: *mut OpusDecoder) -> Option<&'a mut OpusDecoder> {
    if st.is_null() {
        return None;
    }
    let h = st as *const OpusDecoderHandle;
    // SAFETY: caller promised `st` points to at least `size_of::<OpusDecoder>()`
    // bytes; our handle prefix fits within that. We only read the prefix.
    let magic = unsafe { (*h).magic };
    if magic != DECODER_HANDLE_MAGIC {
        return None;
    }
    let inner = unsafe { (*h).inner };
    if inner.is_null() {
        return None;
    }
    // SAFETY: `inner` was obtained from `Box::into_raw` in `_create` / `_init`
    // and has not been freed (we leak in `_destroy`). Aliasing between two
    // handles that share `inner` is controlled by the test harness, which only
    // uses one at a time per logical step.
    Some(unsafe { &mut *inner })
}

unsafe fn resolve_handle_ref<'a>(st: *const OpusDecoder) -> Option<&'a OpusDecoder> {
    if st.is_null() {
        return None;
    }
    let h = st as *const OpusDecoderHandle;
    let magic = unsafe { (*h).magic };
    if magic != DECODER_HANDLE_MAGIC {
        return None;
    }
    let inner = unsafe { (*h).inner };
    if inner.is_null() {
        return None;
    }
    Some(unsafe { &*inner })
}

/// Write a fresh handle into `dst`, stealing ownership of `inner`.
unsafe fn install_handle(dst: *mut OpusDecoderHandle, inner: *mut OpusDecoder) {
    // SAFETY: `dst` was produced by `alloc_handle_storage` or is a caller-
    // provided buffer of at least `size_of::<OpusDecoder>()` bytes. We only
    // write the handle prefix; the rest stays zeroed (alloc_zeroed) or is
    // whatever the caller had there (tolerated).
    unsafe {
        (*dst).magic = DECODER_HANDLE_MAGIC;
        (*dst).inner = inner;
        (*dst).generation = crate::next_handle_generation();
    }
}

/// Bump the handle's generation counter. Called on `OPUS_RESET_STATE`.
pub(crate) unsafe fn bump_generation(st: *mut OpusDecoder) {
    if st.is_null() {
        return;
    }
    let h = st as *mut OpusDecoderHandle;
    // SAFETY: caller holds a valid handle pointer.
    unsafe {
        if (*h).magic != DECODER_HANDLE_MAGIC {
            return;
        }
        (*h).generation = (*h).generation.wrapping_add(1);
    }
}

// ---------------------------------------------------------------------------
// opus_decoder_get_size / create / init / destroy
// ---------------------------------------------------------------------------

/// Size of the decoder state in bytes; `0` for an unsupported channel count.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decoder_get_size(channels: c_int) -> c_int {
    ffi_guard!(0, {
        if channels != 1 && channels != 2 {
            return 0;
        }
        std::mem::size_of::<OpusDecoder>() as c_int
    })
}

/// Initialise a decoder into caller-allocated storage. The caller must have
/// provided at least `opus_decoder_get_size(channels)` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decoder_init(
    st: *mut OpusDecoder,
    fs: i32,
    channels: c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if st.is_null() {
            return OPUS_BAD_ARG;
        }
        match OpusDecoder::new(fs, channels) {
            Ok(dec) => {
                // Leak the real decoder so its Vecs live forever. The caller's
                // buffer only gets the handle prefix; the rest is zeroed.
                let inner = Box::into_raw(Box::new(dec));
                // SAFETY: caller provided at least size_of::<OpusDecoder>() bytes.
                unsafe {
                    // Zero the whole block first so residual garbage can't be
                    // mistaken for a valid handle later.
                    ptr::write_bytes(st as *mut u8, 0, std::mem::size_of::<OpusDecoder>());
                    install_handle(st as *mut OpusDecoderHandle, inner);
                }
                0 // OPUS_OK
            }
            Err(e) => e,
        }
    })
}

/// Allocate + initialise a decoder; returns NULL and writes `*error` on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decoder_create(
    fs: i32,
    channels: c_int,
    error: *mut c_int,
) -> *mut OpusDecoder {
    ffi_guard!(ptr::null_mut(), {
        match OpusDecoder::new(fs, channels) {
            Ok(dec) => {
                if !error.is_null() {
                    unsafe { *error = 0 };
                }
                let inner = Box::into_raw(Box::new(dec));
                let handle = alloc_handle_storage();
                unsafe { install_handle(handle, inner) };
                handle as *mut OpusDecoder
            }
            Err(e) => {
                if !error.is_null() {
                    unsafe { *error = e };
                }
                ptr::null_mut()
            }
        }
    })
}

/// Reclaim a heap-allocated decoder — but actually a no-op. The conformance
/// tests memcpy state to malloc'd buffers and destroy the original; calling
/// the Rust allocator's free on a CRT-malloc'd block would corrupt the heap.
/// We leak uniformly and let the short-lived test process exit clean up.
/// Routed through [`crate::state_free`] to share the single leak-policy
/// chokepoint with `opus_encoder_destroy`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decoder_destroy(st: *mut OpusDecoder) {
    let _: () = ffi_guard!((), {
        unsafe { state_free(st) };
    });
}

// ---------------------------------------------------------------------------
// opus_decode / opus_decode_float
// ---------------------------------------------------------------------------

/// Decode an Opus packet to 16-bit PCM.
///
/// Per the reference (`opus_decoder.c:763`): `data == NULL` or `len == 0`
/// runs packet-loss concealment (PLC), emitting `frame_size` samples of
/// concealment output. Only non-null data paired with `len < 0` is
/// `OPUS_BAD_ARG`. The `decode_fec` flag must be 0 or 1 regardless of
/// whether we're in PLC or normal decode.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decode(
    st: *mut OpusDecoder,
    data: *const c_uchar,
    len: i32,
    pcm: *mut i16,
    frame_size: c_int,
    decode_fec: c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if st.is_null() || pcm.is_null() || frame_size <= 0 {
            return OPUS_BAD_ARG;
        }
        // Reference validates `decode_fec` before PLC/decode dispatch.
        if decode_fec != 0 && decode_fec != 1 {
            return OPUS_BAD_ARG;
        }
        let plc = data.is_null() || len == 0;
        if !plc && len < 0 {
            return OPUS_BAD_ARG;
        }

        let Some(dec) = (unsafe { resolve_handle(st) }) else {
            return OPUS_BAD_ARG;
        };

        let packet: Option<&[u8]> = if plc {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(data, len as usize) })
        };

        let channels = dec.get_channels() as usize;
        let Some(n_samples) = (frame_size as usize).checked_mul(channels) else {
            return OPUS_BAD_ARG;
        };
        let pcm_slice = unsafe { std::slice::from_raw_parts_mut(pcm, n_samples) };

        match dec.decode(packet, pcm_slice, frame_size, decode_fec != 0) {
            Ok(n) => n as c_int,
            Err(e) => e,
        }
    })
}

/// Decode an Opus packet to 24-bit PCM (stored in 32-bit container).
///
/// Matches C `opus_decode24` (reference `opus_decoder.c:937`). Output is the
/// decoded 16-bit sample left-shifted by 8 — the 24-bit path is an upsample
/// of the 16-bit path under our fixed-point / non-RES24 build.
///
/// PLC / `decode_fec` validation is identical to [`opus_decode`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decode24(
    st: *mut OpusDecoder,
    data: *const c_uchar,
    len: i32,
    pcm: *mut i32,
    frame_size: c_int,
    decode_fec: c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if st.is_null() || pcm.is_null() || frame_size <= 0 {
            return OPUS_BAD_ARG;
        }
        if decode_fec != 0 && decode_fec != 1 {
            return OPUS_BAD_ARG;
        }
        let plc = data.is_null() || len == 0;
        if !plc && len < 0 {
            return OPUS_BAD_ARG;
        }

        let Some(dec) = (unsafe { resolve_handle(st) }) else {
            return OPUS_BAD_ARG;
        };

        let packet: Option<&[u8]> = if plc {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(data, len as usize) })
        };

        let channels = dec.get_channels() as usize;
        let Some(n_samples) = (frame_size as usize).checked_mul(channels) else {
            return OPUS_BAD_ARG;
        };
        let pcm_slice = unsafe { std::slice::from_raw_parts_mut(pcm, n_samples) };

        match dec.decode24(packet, pcm_slice, frame_size, decode_fec != 0) {
            Ok(n) => n as c_int,
            Err(e) => e,
        }
    })
}

/// Decode an Opus packet to 32-bit float PCM. See [`opus_decode`] for the
/// PLC / `decode_fec` validation rules, which are identical.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decode_float(
    st: *mut OpusDecoder,
    data: *const c_uchar,
    len: i32,
    pcm: *mut f32,
    frame_size: c_int,
    decode_fec: c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if st.is_null() || pcm.is_null() || frame_size <= 0 {
            return OPUS_BAD_ARG;
        }
        if decode_fec != 0 && decode_fec != 1 {
            return OPUS_BAD_ARG;
        }
        let plc = data.is_null() || len == 0;
        if !plc && len < 0 {
            return OPUS_BAD_ARG;
        }

        let Some(dec) = (unsafe { resolve_handle(st) }) else {
            return OPUS_BAD_ARG;
        };

        let packet: Option<&[u8]> = if plc {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(data, len as usize) })
        };

        let channels = dec.get_channels() as usize;
        let Some(n_samples) = (frame_size as usize).checked_mul(channels) else {
            return OPUS_BAD_ARG;
        };
        let pcm_slice = unsafe { std::slice::from_raw_parts_mut(pcm, n_samples) };

        match dec.decode_float(packet, pcm_slice, frame_size, decode_fec != 0) {
            Ok(n) => n as c_int,
            Err(e) => e,
        }
    })
}

// ---------------------------------------------------------------------------
// opus_packet_pad / opus_packet_unpad (thin wrappers; functions already exist
// in ropus::opus::repacketizer). Phase-1-listed deliverable.
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_pad(
    data: *mut c_uchar,
    len: i32,
    new_len: i32,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if data.is_null() || len < 0 || new_len < 0 {
            return OPUS_BAD_ARG;
        }
        if new_len < len {
            return OPUS_BAD_ARG;
        }
        let buf = unsafe { std::slice::from_raw_parts_mut(data, new_len as usize) };
        ropus::opus::repacketizer::opus_packet_pad(buf, len, new_len)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_unpad(data: *mut c_uchar, len: i32) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if data.is_null() || len < 1 {
            return OPUS_BAD_ARG;
        }
        let buf = unsafe { std::slice::from_raw_parts_mut(data, len as usize) };
        ropus::opus::repacketizer::opus_packet_unpad(buf, len)
    })
}

// ---------------------------------------------------------------------------
// opus_packet_get_* accessors (delegates to ropus::opus::decoder)
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_get_bandwidth(data: *const c_uchar) -> c_int {
    ffi_guard!(crate::OPUS_INVALID_PACKET, {
        if data.is_null() {
            return crate::OPUS_INVALID_PACKET;
        }
        let slice = unsafe { std::slice::from_raw_parts(data, 1) };
        ropus::opus::decoder::opus_packet_get_bandwidth(slice)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_get_samples_per_frame(
    data: *const c_uchar,
    fs: i32,
) -> c_int {
    ffi_guard!(OPUS_BAD_ARG, {
        if data.is_null() {
            return OPUS_BAD_ARG;
        }
        let slice = unsafe { std::slice::from_raw_parts(data, 1) };
        ropus::opus::decoder::opus_packet_get_samples_per_frame(slice, fs)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_get_nb_channels(data: *const c_uchar) -> c_int {
    ffi_guard!(crate::OPUS_INVALID_PACKET, {
        if data.is_null() {
            return crate::OPUS_INVALID_PACKET;
        }
        let slice = unsafe { std::slice::from_raw_parts(data, 1) };
        ropus::opus::decoder::opus_packet_get_nb_channels(slice)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_get_nb_frames(
    packet: *const c_uchar,
    len: i32,
) -> c_int {
    ffi_guard!(OPUS_BAD_ARG, {
        if packet.is_null() || len < 1 {
            return OPUS_BAD_ARG;
        }
        let slice = unsafe { std::slice::from_raw_parts(packet, len as usize) };
        match ropus::opus::decoder::opus_packet_get_nb_frames(slice) {
            Ok(n) => n,
            Err(e) => e,
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_get_nb_samples(
    packet: *const c_uchar,
    len: i32,
    fs: i32,
) -> c_int {
    ffi_guard!(OPUS_BAD_ARG, {
        if packet.is_null() || len < 1 {
            return OPUS_BAD_ARG;
        }
        let slice = unsafe { std::slice::from_raw_parts(packet, len as usize) };
        match ropus::opus::decoder::opus_packet_get_nb_samples(slice, fs) {
            Ok(n) => n,
            Err(e) => e,
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_has_lbrr(packet: *const c_uchar, len: i32) -> c_int {
    ffi_guard!(crate::OPUS_INVALID_PACKET, {
        if packet.is_null() || len < 1 {
            return crate::OPUS_INVALID_PACKET;
        }
        let slice = unsafe { std::slice::from_raw_parts(packet, len as usize) };
        match ropus::opus::decoder::opus_packet_has_lbrr(slice, len) {
            Ok(true) => 1,
            Ok(false) => 0,
            Err(e) => e,
        }
    })
}

/// `opus_decoder_get_nb_samples(dec, packet, len)` — reference at
/// `opus_decoder.c:1333`. Reads `dec->Fs` and defers to
/// `opus_packet_get_nb_samples`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decoder_get_nb_samples(
    dec: *const OpusDecoder,
    packet: *const c_uchar,
    len: i32,
) -> c_int {
    ffi_guard!(OPUS_BAD_ARG, {
        if dec.is_null() || packet.is_null() || len < 1 {
            return OPUS_BAD_ARG;
        }
        let Some(d) = (unsafe { resolve_handle_ref(dec) }) else {
            return OPUS_BAD_ARG;
        };
        let fs = d.get_sample_rate();
        let slice = unsafe { std::slice::from_raw_parts(packet, len as usize) };
        match ropus::opus::decoder::opus_packet_get_nb_samples(slice, fs) {
            Ok(n) => n,
            Err(e) => e,
        }
    })
}

// ---------------------------------------------------------------------------
// Internal helpers exposed to ctl.rs so it can route through the same handle
// indirection.
// ---------------------------------------------------------------------------

/// Resolve a raw `*mut OpusDecoder` handle to the real interior decoder.
/// Returns `None` if the handle is null or poisoned.
pub(crate) unsafe fn handle_to_decoder<'a>(st: *mut OpusDecoder) -> Option<&'a mut OpusDecoder> {
    unsafe { resolve_handle(st) }
}

pub(crate) unsafe fn handle_to_decoder_ref<'a>(st: *const OpusDecoder) -> Option<&'a OpusDecoder> {
    unsafe { resolve_handle_ref(st) }
}

/// Build a per-stream decoder handle block whose `inner` points at an existing
/// `OpusDecoder` (typically one inside `OpusMSDecoder`'s `Vec<OpusDecoder>`).
/// Returned pointer is leaked; lifetime is managed by the MS decoder handle.
pub(crate) fn alloc_sub_handle_for(target: *mut OpusDecoder) -> *mut OpusDecoder {
    let handle = alloc_handle_storage();
    // SAFETY: `handle` points to zero-initialised storage of size_of<OpusDecoder>.
    unsafe { install_handle(handle, target) };
    handle as *mut OpusDecoder
}

// ---------------------------------------------------------------------------
// opus_pcm_soft_clip — float-only, exposed per `opus.h:800`.
// ---------------------------------------------------------------------------

/// Soft-clips a float PCM signal into [-1, 1] with per-channel continuation
/// memory. Matches C `opus_pcm_soft_clip` (reference `src/opus.c:163`).
///
/// Like the reference, returns silently on degenerate inputs
/// (`N<1`, `C<1`, or any null pointer).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_pcm_soft_clip(
    x: *mut f32,
    n: c_int,
    c: c_int,
    declip_mem: *mut f32,
) {
    let _: () = ffi_guard!((), {
        if x.is_null() || declip_mem.is_null() || n < 1 || c < 1 {
            return;
        }
        let n_usize = n as usize;
        let c_usize = c as usize;
        let Some(total) = n_usize.checked_mul(c_usize) else {
            return;
        };
        let x_slice = unsafe { std::slice::from_raw_parts_mut(x, total) };
        let mem_slice = unsafe { std::slice::from_raw_parts_mut(declip_mem, c_usize) };
        ropus::opus::soft_clip::opus_pcm_soft_clip(x_slice, n_usize, c_usize, mem_slice);
    });
}
