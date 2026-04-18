//! `opus_packet_extensions_*` C ABI wrappers.
//!
//! The Rust-native extension API in [`ropus::opus::extensions`] uses typed
//! slices and lifetimes; the conformance tests call C signatures taking raw
//! pointers. This module bridges: null-checks, slice construction, and
//! translation between the raw C `opus_extension_data` struct (with
//! `*const u8 data`) and the Rust-side `OpusExtensionData` (with `&[u8]
//! data`).
//!
//! All entry points are wrapped in `ffi_guard!` so a panic in the codec
//! maps to `OPUS_INTERNAL_ERROR` instead of crossing the FFI boundary.
//!
//! Signatures mirror `reference/src/opus_private.h:246-263` exactly.

use std::os::raw::c_int;

use ropus::opus::extensions::{
    OpusExtensionData as RExt, opus_packet_extensions_count as r_count,
    opus_packet_extensions_count_ext as r_count_ext, opus_packet_extensions_generate as r_generate,
    opus_packet_extensions_parse as r_parse, opus_packet_extensions_parse_ext as r_parse_ext,
};

use crate::{OPUS_BAD_ARG, OPUS_INTERNAL_ERROR, ffi_guard};

/// C-facing extension descriptor. Layout must match the reference
/// `opus_extension_data` struct (see `reference/src/opus_private.h:68-73`).
///
/// ```c
/// typedef struct {
///    int id;
///    int frame;
///    const unsigned char *data;
///    opus_int32 len;
/// } opus_extension_data;
/// ```
#[repr(C)]
#[derive(Clone, Copy)]
pub struct OpusExtensionDataC {
    pub id: c_int,
    pub frame: c_int,
    pub data: *const u8,
    pub len: i32,
}

// -------------------- count --------------------

/// `opus_packet_extensions_count(data, len, nb_frames)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_extensions_count(
    data: *const u8,
    len: i32,
    nb_frames: c_int,
) -> i32 {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        // Mirror the sibling wrappers (count_ext, parse, parse_ext) and the
        // C iterator's `celt_assert(nb_frames <= 48)` (extensions.c:124).
        // Without this, pathological inputs bypass the ffi_guard-friendly
        // BAD_ARG path and may panic inside the iterator, which ffi_guard
        // would then surface as OPUS_INTERNAL_ERROR.
        if nb_frames < 0 || nb_frames > 48 {
            return OPUS_BAD_ARG;
        }
        if len <= 0 {
            return 0;
        }
        if data.is_null() {
            return OPUS_BAD_ARG;
        }
        let slice = unsafe { std::slice::from_raw_parts(data, len as usize) };
        r_count(slice, len, nb_frames)
    })
}

// -------------------- count_ext --------------------

/// `opus_packet_extensions_count_ext(data, len, nb_frame_exts[nb_frames], nb_frames)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_extensions_count_ext(
    data: *const u8,
    len: i32,
    nb_frame_exts: *mut i32,
    nb_frames: c_int,
) -> i32 {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if nb_frames < 0 || nb_frames > 48 {
            return OPUS_BAD_ARG;
        }
        if nb_frames > 0 && nb_frame_exts.is_null() {
            return OPUS_BAD_ARG;
        }
        // Always zero per-frame counts even on an empty packet (C does
        // OPUS_CLEAR before iterating).
        let mut counts = [0i32; 48];
        if data.is_null() || len <= 0 {
            // Fill output with zeros, return 0.
            if nb_frames > 0 && !nb_frame_exts.is_null() {
                unsafe {
                    std::ptr::write_bytes(nb_frame_exts, 0, nb_frames as usize);
                }
            }
            return 0;
        }
        let slice = unsafe { std::slice::from_raw_parts(data, len as usize) };
        let ret = r_count_ext(slice, len, &mut counts[..nb_frames as usize], nb_frames);
        if nb_frames > 0 && !nb_frame_exts.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(counts.as_ptr(), nb_frame_exts, nb_frames as usize);
            }
        }
        ret
    })
}

// -------------------- parse --------------------

/// `opus_packet_extensions_parse(data, len, extensions, *nb_extensions, nb_frames)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_extensions_parse(
    data: *const u8,
    len: i32,
    extensions: *mut OpusExtensionDataC,
    nb_extensions: *mut i32,
    nb_frames: c_int,
) -> i32 {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if nb_extensions.is_null() {
            return OPUS_BAD_ARG;
        }
        let cap = unsafe { *nb_extensions };
        if cap < 0 {
            return OPUS_BAD_ARG;
        }
        if cap > 0 && extensions.is_null() {
            return OPUS_BAD_ARG;
        }
        if nb_frames < 0 || nb_frames > 48 {
            return OPUS_BAD_ARG;
        }
        if data.is_null() || len <= 0 {
            unsafe { *nb_extensions = 0 };
            return 0;
        }
        let slice = unsafe { std::slice::from_raw_parts(data, len as usize) };
        // Per-call heap allocation: `cap` can reach ~9552 in
        // `test_random_extensions_parse`, too large for the stack.
        let mut out = vec![
            RExt {
                id: 0,
                frame: 0,
                data: &[],
                len: 0,
            };
            cap as usize
        ];
        let mut count = cap;
        let ret = r_parse(slice, len, &mut out, &mut count, nb_frames);
        // Determine how many entries were actually filled.
        // On OPUS_BUFFER_TOO_SMALL, ropus does NOT write back *count; the
        // iterator filled exactly `cap` entries before returning (matches the
        // C reference — see extensions.c:372-377, which returns without
        // touching `*nb_extensions`). On success/error, ropus wrote `count`
        // to our local.
        let filled = if ret == crate::OPUS_BUFFER_TOO_SMALL {
            cap
        } else {
            count
        };
        // Match C: write `*nb_extensions` on success/other errors, but leave
        // it untouched on OPUS_BUFFER_TOO_SMALL. The caller-visible value on
        // BUFFER_TOO_SMALL remains the input cap.
        if ret != crate::OPUS_BUFFER_TOO_SMALL {
            unsafe { *nb_extensions = count };
        }
        // Copy back filled entries regardless of success/buffer-too-small so
        // the C array has valid pointers for the j<nb_extensions loop in
        // `test_random_extensions_parse`.
        copy_extensions_to_c(&out[..filled as usize], extensions, data, len);
        ret
    })
}

/// Copy parsed Rust-side extensions into the C-facing array at `extensions`.
/// Checks that every extension's data pointer lies within `[data, data+len)`
/// in debug builds — the random-test invariant in
/// `test_opus_extensions.c:609`.
fn copy_extensions_to_c(
    out: &[RExt],
    extensions: *mut OpusExtensionDataC,
    data: *const u8,
    len: i32,
) {
    let _packet_base = data;
    let _packet_end = unsafe { data.add(len.max(0) as usize) };
    for (i, ext) in out.iter().enumerate() {
        // `ext.data` is always a sub-slice of the parse input so `as_ptr()`
        // gives a pointer into the caller's buffer even for zero-length
        // payloads (key for the `data >= payload` random-test invariant).
        let raw_ptr = ext.data.as_ptr();
        debug_assert!(
            raw_ptr >= _packet_base
                && unsafe { raw_ptr.add(ext.len.max(0) as usize) } <= _packet_end,
            "extension pointer (#{} of {}) escapes input packet: ptr={:p} len={} packet=[{:p}..{:p}] id={} frame={}",
            i,
            out.len(),
            raw_ptr,
            ext.len,
            _packet_base,
            _packet_end,
            ext.id,
            ext.frame
        );
        unsafe {
            (*extensions.add(i)).id = ext.id as c_int;
            (*extensions.add(i)).frame = ext.frame as c_int;
            (*extensions.add(i)).data = raw_ptr;
            (*extensions.add(i)).len = ext.len;
        }
    }
}

// -------------------- parse_ext --------------------

/// `opus_packet_extensions_parse_ext(data, len, extensions, *nb_extensions, nb_frame_exts, nb_frames)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_extensions_parse_ext(
    data: *const u8,
    len: i32,
    extensions: *mut OpusExtensionDataC,
    nb_extensions: *mut i32,
    nb_frame_exts: *const i32,
    nb_frames: c_int,
) -> i32 {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if nb_extensions.is_null() {
            return OPUS_BAD_ARG;
        }
        let cap = unsafe { *nb_extensions };
        if cap < 0 {
            return OPUS_BAD_ARG;
        }
        if cap > 0 && extensions.is_null() {
            return OPUS_BAD_ARG;
        }
        if nb_frames < 0 || nb_frames > 48 {
            return OPUS_BAD_ARG;
        }
        if nb_frames > 0 && nb_frame_exts.is_null() {
            return OPUS_BAD_ARG;
        }
        if data.is_null() || len <= 0 {
            unsafe { *nb_extensions = 0 };
            return 0;
        }
        let slice = unsafe { std::slice::from_raw_parts(data, len as usize) };
        let frame_counts = if nb_frames > 0 {
            unsafe { std::slice::from_raw_parts(nb_frame_exts, nb_frames as usize) }
        } else {
            &[][..]
        };
        let mut out = vec![
            RExt {
                id: 0,
                frame: 0,
                data: &[],
                len: 0,
            };
            cap as usize
        ];
        let mut count = cap;
        let ret = r_parse_ext(slice, len, &mut out, &mut count, frame_counts, nb_frames);
        // parse_ext scatters writes by per-frame index, so on
        // OPUS_BUFFER_TOO_SMALL some indices are left uninitialised — but
        // the conformance tests always size the output generously, and we
        // don't surface that path to the C caller. On success, exactly
        // `count` slots (in frame order, scattered across the prefix-sum
        // ranges) are filled; copy only those.
        //
        // Match C (reference/src/extensions.c:413-415): on
        // OPUS_BUFFER_TOO_SMALL, C returns without writing `*nb_extensions`.
        // Ropus's parse_ext also leaves `*count` untouched in that case
        // (extensions.rs around line 127), so we only mirror the write on
        // success/other errors.
        if ret != crate::OPUS_BUFFER_TOO_SMALL {
            unsafe { *nb_extensions = count };
        }
        if ret < 0 {
            return ret;
        }
        copy_extensions_to_c(&out[..count as usize], extensions, data, len);
        ret
    })
}

// -------------------- generate --------------------

/// `opus_packet_extensions_generate(data, len, extensions, nb_extensions, nb_frames, pad)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_extensions_generate(
    data: *mut u8,
    len: i32,
    extensions: *const OpusExtensionDataC,
    nb_extensions: i32,
    nb_frames: c_int,
    pad: c_int,
) -> i32 {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        if len < 0 {
            return OPUS_BAD_ARG;
        }
        if nb_extensions < 0 {
            return OPUS_BAD_ARG;
        }
        if nb_extensions > 0 && extensions.is_null() {
            return OPUS_BAD_ARG;
        }
        if nb_frames < 0 {
            return OPUS_BAD_ARG;
        }
        if len > 0 && data.is_null() {
            return OPUS_BAD_ARG;
        }
        // Snapshot C-side extensions into Rust slices. Extension payloads
        // borrowed via `*const u8` must remain live for the duration of the
        // call — the C test stacks them, so that invariant holds.
        let mut owned: Vec<RExt<'static>> = Vec::with_capacity(nb_extensions.max(0) as usize);
        for i in 0..nb_extensions as usize {
            let raw = unsafe { &*extensions.add(i) };
            // A negative length is illegal input; preserve the C semantics
            // (bad_arg). The generator checks again but we defer to it for
            // id range validation to keep a single source of truth.
            if raw.len < 0 {
                // Build an empty slice and let the generator return BAD_ARG
                // on id/len validation the same way C would.
                owned.push(RExt {
                    id: raw.id,
                    frame: raw.frame,
                    data: &[],
                    len: raw.len,
                });
                continue;
            }
            let slice: &'static [u8] = if raw.len > 0 && !raw.data.is_null() {
                unsafe {
                    std::mem::transmute::<&[u8], &'static [u8]>(std::slice::from_raw_parts(
                        raw.data,
                        raw.len as usize,
                    ))
                }
            } else {
                &[]
            };
            owned.push(RExt {
                id: raw.id,
                frame: raw.frame,
                data: slice,
                len: raw.len,
            });
        }
        let buf_slice = if len > 0 && !data.is_null() {
            Some(unsafe { std::slice::from_raw_parts_mut(data, len as usize) })
        } else {
            None
        };
        r_generate(buf_slice, len, &owned, nb_frames, pad != 0)
    })
}
