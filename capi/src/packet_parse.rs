//! `opus_packet_parse` wrapper.
//!
//! The C signature returns frame pointers (`const unsigned char *frames[48]`).
//! Our ropus `opus_packet_parse_impl` returns sizes + a payload offset; we
//! convert to frame pointers here so the conformance test parser suite
//! (`test_opus_api.c::test_parse`) can exercise it unchanged.

use std::os::raw::{c_int, c_uchar};

use ropus::opus::decoder::{MAX_FRAMES, opus_packet_parse_impl};

use crate::{OPUS_BAD_ARG, OPUS_INTERNAL_ERROR, ffi_guard};

/// `opus_packet_parse(data, len, out_toc, frames[48], size[48], payload_offset)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_parse(
    data: *const c_uchar,
    len: i32,
    out_toc: *mut c_uchar,
    frames: *mut *const c_uchar,
    size: *mut i16,
    payload_offset: *mut c_int,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        // Match reference C `opus_packet_parse_impl`: size==NULL or len<0
        // returns OPUS_BAD_ARG before any other validation. `frames` is
        // allowed to be NULL — callers opt out of receiving frame pointers
        // (see reference/src/opus.c:373-376).
        if size.is_null() || len < 0 {
            return OPUS_BAD_ARG;
        }
        if data.is_null() {
            return OPUS_BAD_ARG;
        }
        let slice = unsafe { std::slice::from_raw_parts(data, len.max(0) as usize) };
        let mut toc: u8 = 0;
        let mut sizes = [0i16; MAX_FRAMES];
        let mut pay_off: i32 = 0;
        let count = opus_packet_parse_impl(
            slice,
            len,
            false,
            &mut toc,
            &mut sizes,
            &mut pay_off,
            None,
        );
        if count <= 0 {
            return count;
        }
        // Fill outputs.
        if !out_toc.is_null() {
            unsafe { *out_toc = toc };
        }
        // Write frame pointers: frame[i] = data + pay_off + sum(sizes[0..i]).
        // `frames` is optional; only write when non-null.
        let mut cursor = pay_off as isize;
        for i in 0..count as usize {
            if !frames.is_null() {
                let frame_ptr = unsafe { data.offset(cursor) };
                unsafe { *frames.add(i) = frame_ptr };
            }
            unsafe { *size.add(i) = sizes[i] };
            cursor += sizes[i] as isize;
        }
        if !payload_offset.is_null() {
            unsafe { *payload_offset = pay_off };
        }
        count
    })
}
