//! Opus packet extensions — parsing, iteration, and generation (Opus 1.4+).
//!
//! Ported from `reference/src/extensions.c` and declared in
//! `reference/src/opus_private.h`.
//!
//! The actual implementation lives in [`crate::opus::repacketizer`]; this
//! module is the canonical public surface for extension-related types and
//! functions, separated out so callers of the higher-level extensions API
//! do not need to import from `repacketizer`. The implementation was
//! originally colocated with the repacketizer because `opus_repacketizer_out`
//! consumes and produces extension bytes; keeping the code physically
//! together avoided a cyclic module boundary during the initial port.
//!
//! Public functions mirror the C reference one-for-one:
//!
//! - [`opus_packet_extensions_count`] — count extensions (total) in padding.
//! - [`opus_packet_extensions_count_ext`] — per-frame extension counts.
//! - [`opus_packet_extensions_parse`] — extract extensions in bitstream order.
//! - [`opus_packet_extensions_parse_ext`] — extract in per-frame order.
//! - [`opus_packet_extensions_generate`] — serialize extensions into padding.
//!
//! The [`OpusExtensionIterator`] and [`OpusExtensionData`] types are also
//! re-exported here. See the repacketizer module for the implementation and
//! tests.

pub use crate::opus::repacketizer::{
    OpusExtensionData, OpusExtensionIterator, opus_packet_extensions_count,
    opus_packet_extensions_generate, opus_packet_extensions_parse,
};

use crate::opus::decoder::{OPUS_BAD_ARG, OPUS_OK};
use crate::opus::repacketizer::OpusExtensionIterator as Iter;

/// Maximum frames per Opus packet (120ms / 2.5ms = 48). Mirrors
/// `MAX_FRAMES` in the repacketizer module.
const MAX_FRAMES: usize = 48;

/// Per-frame extension count. Fills `nb_frame_exts[0..nb_frames]` with the
/// number of extensions targeting each frame and returns the total count
/// (matches C `opus_packet_extensions_count_ext`).
pub fn opus_packet_extensions_count_ext(
    data: &[u8],
    len: i32,
    nb_frame_exts: &mut [i32],
    nb_frames: i32,
) -> i32 {
    if nb_frames < 0 || nb_frames as usize > nb_frame_exts.len() {
        return OPUS_BAD_ARG;
    }
    for slot in &mut nb_frame_exts[..nb_frames as usize] {
        *slot = 0;
    }
    if len <= 0 {
        return 0;
    }
    let mut iter = Iter::new(data, len, nb_frames);
    let mut count: i32 = 0;
    loop {
        let (ret, ext) = iter.next_ext();
        if ret <= 0 {
            break;
        }
        if ext.frame < 0 || ext.frame >= nb_frames {
            // Iterator should never emit an out-of-range frame, but be
            // defensive — mirrors the celt_assert in the C parser.
            return OPUS_BAD_ARG;
        }
        nb_frame_exts[ext.frame as usize] += 1;
        count += 1;
    }
    count
}

/// Parse extensions and deliver them in per-frame order. `nb_frame_exts` must
/// contain the per-frame counts produced by [`opus_packet_extensions_count_ext`]
/// with the same `data`/`len`/`nb_frames`. Returns 0 on success or a negative
/// error code. Matches C `opus_packet_extensions_parse_ext`.
pub fn opus_packet_extensions_parse_ext<'a>(
    data: &'a [u8],
    len: i32,
    extensions: &mut [OpusExtensionData<'a>],
    nb_extensions: &mut i32,
    nb_frame_exts: &[i32],
    nb_frames: i32,
) -> i32 {
    if nb_frames < 0 || nb_frames as usize > MAX_FRAMES {
        return OPUS_BAD_ARG;
    }
    if nb_frames as usize > nb_frame_exts.len() {
        return OPUS_BAD_ARG;
    }
    // Build prefix-sum of frame counts so we can place parsed extensions at
    // the correct position in the output array. `nb_frames_cum[i]` is the
    // index where frame-`i` extensions start; `nb_frames_cum[nb_frames]` is
    // the total.
    //
    // Use `wrapping_add` to match C `int` addition semantics
    // (reference/src/extensions.c:400-405). Well-formed inputs from
    // `opus_packet_extensions_count_ext` cannot overflow, so this is
    // observationally identical for trusted inputs — the match matters only
    // for pathological differential-fuzz inputs.
    let mut nb_frames_cum = [0i32; MAX_FRAMES + 1];
    let mut prev_total: i32 = 0;
    for i in 0..nb_frames as usize {
        nb_frames_cum[i] = prev_total;
        prev_total = prev_total.wrapping_add(nb_frame_exts[i]);
    }
    nb_frames_cum[nb_frames as usize] = prev_total;

    if len <= 0 {
        *nb_extensions = 0;
        return OPUS_OK;
    }
    let max_ext = *nb_extensions;
    let mut iter = Iter::new(data, len, nb_frames);
    let mut count: i32 = 0;
    loop {
        let (ret, ext) = iter.next_ext();
        if ret < 0 {
            *nb_extensions = count;
            return ret;
        }
        if ret == 0 {
            *nb_extensions = count;
            return OPUS_OK;
        }
        let f = ext.frame;
        if f < 0 || f >= nb_frames {
            *nb_extensions = count;
            return OPUS_BAD_ARG;
        }
        let idx = nb_frames_cum[f as usize];
        if idx >= max_ext {
            return crate::opus::decoder::OPUS_BUFFER_TOO_SMALL;
        }
        extensions[idx as usize] = ext;
        nb_frames_cum[f as usize] += 1;
        count += 1;
    }
}
