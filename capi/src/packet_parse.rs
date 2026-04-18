//! `opus_packet_parse` wrapper.
//!
//! The C signature returns frame pointers (`const unsigned char *frames[48]`).
//! Our ropus `opus_packet_parse_impl` returns sizes + a payload offset; we
//! convert to frame pointers here so the conformance test parser suite
//! (`test_opus_api.c::test_parse`) can exercise it unchanged.

use std::os::raw::{c_int, c_uchar};

use ropus::opus::decoder::{
    MAX_FRAMES, PaddingInfo, opus_packet_parse_impl as ropus_opus_packet_parse_impl,
    opus_packet_parse_impl_with_padding,
};

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
        // (see reference/src/opus.c:373-376). Null `data` is only rejected
        // when `len > 0` (C would crash; we surface BAD_ARG). With `len==0`
        // and `data==NULL`, C returns OPUS_INVALID_PACKET from the length
        // check before any dereference — we flow through with an empty
        // slice so ropus returns the same.
        if size.is_null() || len < 0 {
            return OPUS_BAD_ARG;
        }
        if len > 0 && data.is_null() {
            return OPUS_BAD_ARG;
        }
        let slice = if len > 0 {
            unsafe { std::slice::from_raw_parts(data, len as usize) }
        } else {
            &[]
        };
        let mut toc: u8 = 0;
        let mut sizes = [0i16; MAX_FRAMES];
        let mut pay_off: i32 = 0;
        let count = ropus_opus_packet_parse_impl(
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

/// `opus_packet_parse_impl` — the fuller reference signature with self-delimited
/// mode and padding extraction. Backs `test_opus_extensions.c` line 691.
/// See `reference/src/opus_private.h:208-212`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_parse_impl(
    data: *const c_uchar,
    len: i32,
    self_delimited: c_int,
    out_toc: *mut c_uchar,
    frames: *mut *const c_uchar,
    size: *mut i16,
    payload_offset: *mut c_int,
    packet_offset: *mut i32,
    padding: *mut *const c_uchar,
    padding_len: *mut i32,
) -> c_int {
    ffi_guard!(OPUS_INTERNAL_ERROR, {
        // Mirror C `opus_packet_parse_impl` NULL / length preconditions.
        // C writes `*padding=NULL; *padding_len=0` unconditionally when
        // `padding!=NULL`, regardless of whether the parse ultimately
        // succeeds.
        if !padding.is_null() {
            unsafe { *padding = std::ptr::null() };
            if !padding_len.is_null() {
                unsafe { *padding_len = 0 };
            }
        }
        if size.is_null() || len < 0 {
            return OPUS_BAD_ARG;
        }
        // C `opus_packet_parse_impl` (reference/src/opus.c:246-254) rejects
        // only on size==NULL || len<0, returns OPUS_INVALID_PACKET on len==0,
        // and only dereferences `data` when len>0. Mirror that: flow
        // {data==NULL, len==0} through to the length check; reject
        // {data==NULL, len>0} with BAD_ARG (C would segfault).
        if len > 0 && data.is_null() {
            return OPUS_BAD_ARG;
        }
        let slice = if len > 0 {
            unsafe { std::slice::from_raw_parts(data, len as usize) }
        } else {
            &[]
        };
        let mut toc: u8 = 0;
        let mut sizes = [0i16; MAX_FRAMES];
        let mut pay_off: i32 = 0;
        let mut pkt_off: i32 = 0;
        let mut pad_info = PaddingInfo { offset: 0, len: 0 };
        let count = opus_packet_parse_impl_with_padding(
            slice,
            len,
            self_delimited != 0,
            &mut toc,
            &mut sizes,
            &mut pay_off,
            Some(&mut pkt_off),
            Some(&mut pad_info),
        );
        if count <= 0 {
            return count;
        }

        if !out_toc.is_null() {
            unsafe { *out_toc = toc };
        }
        // Fill frame pointers + sizes. `frames` may be NULL (caller opts out).
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
        if !packet_offset.is_null() {
            unsafe { *packet_offset = pkt_off };
        }
        if !padding.is_null() {
            // Emit padding pointer unconditionally when caller requested it
            // (matches C reference `opus.c:378-382`). Callers should inspect
            // `padding_len` to decide whether to dereference the pointer.
            unsafe {
                *padding = data.offset(pad_info.offset as isize);
                if !padding_len.is_null() {
                    *padding_len = pad_info.len;
                }
            }
        }
        count
    })
}

