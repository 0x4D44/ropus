//! Opus Repacketizer and Extensions — transport-layer packet manipulation.
//!
//! Ported from: reference/src/repacketizer.c, reference/src/extensions.c
//!
//! Provides:
//! - Merging/splitting Opus packets (OpusRepacketizer)
//! - Padding/unpadding single and multistream packets
//! - Extension parsing, iteration, and generation (Opus 1.4+)

use crate::opus::decoder::{
    OPUS_BAD_ARG, OPUS_BUFFER_TOO_SMALL, OPUS_INTERNAL_ERROR, OPUS_INVALID_PACKET, OPUS_OK,
    opus_packet_get_nb_frames, opus_packet_get_samples_per_frame,
};

/// Maximum frames per Opus packet (120ms / 2.5ms = 48).
const MAX_FRAMES: usize = 48;

// ===========================================================================
// Packet utility functions
// ===========================================================================

/// Encode a frame size into 1 or 2 bytes. Returns bytes written.
/// Matches C `encode_size` in opus.c.
pub(crate) fn encode_size(size: i32, data: &mut [u8]) -> usize {
    if size < 252 {
        data[0] = size as u8;
        1
    } else {
        data[0] = (252 + (size & 0x3)) as u8;
        data[1] = ((size - data[0] as i32) >> 2) as u8;
        2
    }
}

/// Parse a frame size from 1 or 2 bytes.
/// Returns `(bytes_consumed, frame_size)`. On error returns `(-1, -1)`.
/// Matches C `parse_size` in opus.c.
fn parse_size(data: &[u8], len: i32) -> (i32, i16) {
    if len < 1 {
        return (-1, -1);
    }
    if data[0] < 252 {
        (1, data[0] as i16)
    } else if len < 2 {
        (-1, -1)
    } else {
        (2, 4 * data[1] as i16 + data[0] as i16)
    }
}

/// Result of parsing an Opus packet for repacketizer use.
struct PacketParseResult<'a> {
    toc: u8,
    count: usize,
    frames: [&'a [u8]; MAX_FRAMES],
    sizes: [i16; MAX_FRAMES],
    padding: &'a [u8],
    padding_len: i32,
    packet_offset: i32,
}

impl<'a> PacketParseResult<'a> {
    fn new() -> Self {
        Self {
            toc: 0,
            count: 0,
            frames: [&[]; MAX_FRAMES],
            sizes: [0; MAX_FRAMES],
            padding: &[],
            padding_len: 0,
            packet_offset: 0,
        }
    }
}

/// Full packet parse returning frame slices, sizes, and padding info.
/// Matches C `opus_packet_parse_impl` in opus.c, with all output parameters.
fn parse_packet<'a>(
    data: &'a [u8],
    len: i32,
    self_delimited: bool,
) -> Result<PacketParseResult<'a>, i32> {
    if len < 0 {
        return Err(OPUS_BAD_ARG);
    }
    if len == 0 {
        return Err(OPUS_INVALID_PACKET);
    }

    let mut result = PacketParseResult::new();
    let framesize = opus_packet_get_samples_per_frame(data, 48000);

    let mut pos: usize = 0;
    let mut remaining = len;
    let mut pad: i32 = 0;

    let toc = data[0];
    pos += 1;
    remaining -= 1;
    let mut last_size: i32 = remaining;
    let mut cbr = false;
    let count: i32;

    match toc & 0x3 {
        // Code 0: one frame
        0 => {
            count = 1;
        }
        // Code 1: two CBR frames
        1 => {
            count = 2;
            cbr = true;
            if !self_delimited {
                if remaining & 0x1 != 0 {
                    return Err(OPUS_INVALID_PACKET);
                }
                last_size = remaining / 2;
                result.sizes[0] = last_size as i16;
            }
        }
        // Code 2: two VBR frames
        2 => {
            count = 2;
            let (bytes, sz) = parse_size(&data[pos..], remaining);
            if sz < 0 {
                return Err(OPUS_INVALID_PACKET);
            }
            remaining -= bytes;
            pos += bytes as usize;
            if sz as i32 > remaining {
                return Err(OPUS_INVALID_PACKET);
            }
            result.sizes[0] = sz;
            last_size = remaining - sz as i32;
        }
        // Code 3: multiple CBR/VBR frames
        _ => {
            if remaining < 1 {
                return Err(OPUS_INVALID_PACKET);
            }
            let ch = data[pos];
            pos += 1;
            remaining -= 1;
            count = (ch & 0x3F) as i32;
            if count <= 0 || framesize * count > 5760 {
                return Err(OPUS_INVALID_PACKET);
            }
            // Padding flag (bit 6)
            if ch & 0x40 != 0 {
                loop {
                    if remaining <= 0 {
                        return Err(OPUS_INVALID_PACKET);
                    }
                    let p = data[pos];
                    pos += 1;
                    remaining -= 1;
                    let tmp = if p == 255 { 254 } else { p as i32 };
                    remaining -= tmp;
                    pad += tmp;
                    if p != 255 {
                        break;
                    }
                }
            }
            if remaining < 0 {
                return Err(OPUS_INVALID_PACKET);
            }
            // VBR flag (bit 7): cbr = !(ch & 0x80)
            cbr = (ch & 0x80) == 0;
            if !cbr {
                // VBR case
                last_size = remaining;
                for i in 0..count as usize - 1 {
                    let (bytes, sz) = parse_size(&data[pos..], remaining);
                    if sz < 0 {
                        return Err(OPUS_INVALID_PACKET);
                    }
                    remaining -= bytes;
                    pos += bytes as usize;
                    if sz as i32 > remaining {
                        return Err(OPUS_INVALID_PACKET);
                    }
                    result.sizes[i] = sz;
                    last_size -= bytes + sz as i32;
                }
                if last_size < 0 {
                    return Err(OPUS_INVALID_PACKET);
                }
            } else if !self_delimited {
                // CBR case
                last_size = remaining / count;
                if last_size * count != remaining {
                    return Err(OPUS_INVALID_PACKET);
                }
                for i in 0..count as usize - 1 {
                    result.sizes[i] = last_size as i16;
                }
            }
        }
    }

    // Self-delimited framing has an extra size for the last frame
    if self_delimited {
        let (bytes, sz) = parse_size(&data[pos..], remaining);
        if sz < 0 {
            return Err(OPUS_INVALID_PACKET);
        }
        remaining -= bytes;
        pos += bytes as usize;
        if sz as i32 > remaining {
            return Err(OPUS_INVALID_PACKET);
        }
        result.sizes[count as usize - 1] = sz;
        if cbr {
            if sz as i32 * count > remaining {
                return Err(OPUS_INVALID_PACKET);
            }
            for i in 0..count as usize - 1 {
                result.sizes[i] = sz;
            }
        } else if bytes + sz as i32 > last_size {
            return Err(OPUS_INVALID_PACKET);
        }
    } else {
        // Reject frames > 1275 bytes (Opus spec maximum)
        if last_size > 1275 {
            return Err(OPUS_INVALID_PACKET);
        }
        result.sizes[count as usize - 1] = last_size as i16;
    }

    result.toc = toc;
    result.count = count as usize;

    // Fill frame slices
    let mut frame_pos = pos;
    for i in 0..count as usize {
        let sz = result.sizes[i] as usize;
        result.frames[i] = &data[frame_pos..frame_pos + sz];
        frame_pos += sz;
    }

    // Padding data follows frame data
    if pad > 0 {
        result.padding = &data[frame_pos..frame_pos + pad as usize];
    }
    result.padding_len = pad;

    // Packet offset = total consumed bytes
    result.packet_offset = pad + frame_pos as i32;

    Ok(result)
}

// ===========================================================================
// Extension types
// ===========================================================================

/// Extension data extracted from or to be embedded in Opus packet padding.
/// Matches C `opus_extension_data`.
#[derive(Clone, Copy, Debug)]
pub struct OpusExtensionData<'a> {
    pub id: i32,
    pub frame: i32,
    pub data: &'a [u8],
    pub len: i32,
}

impl<'a> OpusExtensionData<'a> {
    const EMPTY: Self = Self {
        id: 0,
        frame: 0,
        data: &[],
        len: 0,
    };
}

// ===========================================================================
// Extension iterator internals
// ===========================================================================

/// Advance past an extension payload (excluding the ID byte).
/// Returns remaining length (negative = error).
/// Matches C `skip_extension_payload` in extensions.c.
fn skip_extension_payload(
    data: &[u8],
    pos: &mut usize,
    mut remaining: i32,
    header_size: &mut i32,
    id_byte: u8,
    trailing_short_len: i32,
) -> i32 {
    *header_size = 0;
    let id = (id_byte >> 1) as i32;
    let l = (id_byte & 1) as i32;

    if (id == 0 && l == 1) || id == 2 {
        // Nothing to do: padding-end or repeat indicator
    } else if id > 0 && id < 32 {
        // Short extension: payload is L bytes (0 or 1)
        if remaining < l {
            return -1;
        }
        *pos += l as usize;
        remaining -= l;
    } else {
        // Long extension (id >= 32) or id == 0 with L == 0
        if l == 0 {
            // Payload extends to end of data minus trailing short bytes
            if remaining < trailing_short_len {
                return -1;
            }
            *pos += (remaining - trailing_short_len) as usize;
            remaining = trailing_short_len;
        } else {
            // Lacing-encoded length
            let mut bytes: i32 = 0;
            loop {
                if remaining < 1 {
                    return -1;
                }
                let lacing = data[*pos] as i32;
                *pos += 1;
                bytes += lacing;
                *header_size += 1;
                remaining -= lacing + 1;
                if lacing != 255 {
                    break;
                }
            }
            if remaining < 0 {
                return -1;
            }
            *pos += bytes as usize;
        }
    }
    remaining
}

/// Advance past a complete extension (ID byte + payload).
/// Returns remaining length (negative = error). Does not advance pos on error.
/// Matches C `skip_extension` in extensions.c.
fn skip_extension(data: &[u8], pos: &mut usize, remaining: i32, header_size: &mut i32) -> i32 {
    if remaining == 0 {
        *header_size = 0;
        return 0;
    }
    if remaining < 1 {
        return -1;
    }
    let saved_pos = *pos;
    let id_byte = data[*pos];
    *pos += 1;
    let new_remaining = skip_extension_payload(data, pos, remaining - 1, header_size, id_byte, 0);
    if new_remaining >= 0 {
        *header_size += 1; // account for the ID byte
    } else {
        *pos = saved_pos; // restore on error
    }
    new_remaining
}

// ===========================================================================
// Extension iterator
// ===========================================================================

/// Stateful iterator over Opus extension data in packet padding.
/// Handles the "Repeat These Extensions" mechanism.
/// Matches C `OpusExtensionIterator` in extensions.c.
pub struct OpusExtensionIterator<'a> {
    data: &'a [u8],
    /// Current read position (replaces C `curr_data`)
    pos: usize,
    /// Start of region being repeated (replaces C `repeat_data`)
    repeat_pos: usize,
    /// Position past the last long extension (replaces C `last_long`, None = NULL)
    last_long_pos: Option<usize>,
    /// Source scan position during repeat (replaces C `src_data`)
    src_pos: usize,
    /// Total extension data length
    len: i32,
    /// Remaining bytes from `pos`
    curr_len: i32,
    /// Length of repeated extension region
    repeat_len: i32,
    /// Remaining source length during repeat
    src_len: i32,
    /// Bytes of short extension payloads after last long extension
    trailing_short_len: i32,
    /// Total frames in the packet
    nb_frames: i32,
    /// Early termination frame limit
    frame_max: i32,
    /// Current frame index
    curr_frame: i32,
    /// Current frame in repeat iteration (0 = not repeating)
    repeat_frame: i32,
    /// L flag of the repeat indicator
    repeat_l: u8,
}

impl<'a> OpusExtensionIterator<'a> {
    /// Initialize iterator over extension payload bytes.
    /// Matches C `opus_extension_iterator_init`.
    pub fn new(data: &'a [u8], len: i32, nb_frames: i32) -> Self {
        debug_assert!(len >= 0);
        debug_assert!(nb_frames >= 0 && nb_frames <= MAX_FRAMES as i32);
        Self {
            data,
            pos: 0,
            repeat_pos: 0,
            last_long_pos: None,
            src_pos: 0,
            len,
            curr_len: len,
            repeat_len: 0,
            src_len: 0,
            trailing_short_len: 0,
            nb_frames,
            frame_max: nb_frames,
            curr_frame: 0,
            repeat_frame: 0,
            repeat_l: 0,
        }
    }

    /// Reset to beginning without reallocating.
    /// Matches C `opus_extension_iterator_reset`.
    pub fn reset(&mut self) {
        self.pos = 0;
        self.repeat_pos = 0;
        self.last_long_pos = None;
        self.curr_len = self.len;
        self.curr_frame = 0;
        self.repeat_frame = 0;
        self.trailing_short_len = 0;
    }

    /// Limit iteration to frames `[0, frame_max)`.
    /// Matches C `opus_extension_iterator_set_frame_max`.
    pub fn set_frame_max(&mut self, frame_max: i32) {
        self.frame_max = frame_max;
    }

    /// Return the next repeated extension.
    /// Returns `(status, extension)` where status is 1 if found, 0 if done,
    /// or negative on error.
    /// Matches C `opus_extension_iterator_next_repeat`.
    fn next_repeat(&mut self) -> (i32, OpusExtensionData<'a>) {
        debug_assert!(self.repeat_frame > 0);
        let mut ext = OpusExtensionData::EMPTY;

        while self.repeat_frame < self.nb_frames {
            while self.src_len > 0 {
                let mut header_size: i32 = 0;
                let mut repeat_id_byte = self.data[self.src_pos];

                // Skip past this extension in the source (original definition) list
                self.src_len =
                    skip_extension(self.data, &mut self.src_pos, self.src_len, &mut header_size);
                // We skipped this extension earlier, so it should not fail now
                debug_assert!(self.src_len >= 0);

                // Don't repeat padding or frame separators (id_byte <= 3 means id <= 1)
                if repeat_id_byte <= 3 {
                    continue;
                }

                // If repeat has L=0 and this is the last long extension in the last frame,
                // force L=0 encoding
                if self.repeat_l == 0
                    && self.repeat_frame + 1 >= self.nb_frames
                    && self.last_long_pos == Some(self.src_pos)
                {
                    repeat_id_byte &= !1;
                }

                // Advance pos through the payload data for this repeated frame
                let ext_start = self.pos;
                header_size = 0;
                self.curr_len = skip_extension_payload(
                    self.data,
                    &mut self.pos,
                    self.curr_len,
                    &mut header_size,
                    repeat_id_byte,
                    self.trailing_short_len,
                );
                if self.curr_len < 0 {
                    return (OPUS_INVALID_PACKET, ext);
                }
                debug_assert!(self.pos as i32 == self.len - self.curr_len);

                // If we were asked to stop at frame_max, skip extensions for later frames
                if self.repeat_frame >= self.frame_max {
                    continue;
                }

                ext.id = (repeat_id_byte >> 1) as i32;
                ext.frame = self.repeat_frame;
                let hdr = header_size as usize;
                ext.data = &self.data[ext_start + hdr..self.pos];
                ext.len = (self.pos - ext_start - hdr) as i32;
                return (1, ext);
            }
            // Finished repeating for this frame; reset source for next frame
            self.src_pos = self.repeat_pos;
            self.src_len = self.repeat_len;
            self.repeat_frame += 1;
        }

        // Finished all repeats
        self.repeat_pos = self.pos;
        self.last_long_pos = None;
        // If L=0, advance frame number
        if self.repeat_l == 0 {
            self.curr_frame += 1;
            if self.curr_frame >= self.nb_frames {
                self.curr_len = 0;
            }
        }
        self.repeat_frame = 0;
        (0, ext)
    }

    /// Return the next extension in bitstream order.
    /// Returns `(status, extension)` where status is 1 if found, 0 if done,
    /// or negative on error.
    /// Matches C `opus_extension_iterator_next`.
    pub fn next_ext(&mut self) -> (i32, OpusExtensionData<'a>) {
        let mut ext = OpusExtensionData::EMPTY;

        if self.curr_len < 0 {
            return (OPUS_INVALID_PACKET, ext);
        }
        // If we are in repeat mode, continue repeating
        if self.repeat_frame > 0 {
            let (ret, rext) = self.next_repeat();
            if ret != 0 {
                return (ret, rext);
            }
        }
        // Check frame_max (allows set_frame_max to be called at any point)
        if self.curr_frame >= self.frame_max {
            return (0, ext);
        }

        while self.curr_len > 0 {
            let ext_start = self.pos;
            let id = (self.data[ext_start] >> 1) as i32;
            let l = (self.data[ext_start] & 1) as i32;
            let mut header_size: i32 = 0;

            self.curr_len =
                skip_extension(self.data, &mut self.pos, self.curr_len, &mut header_size);
            if self.curr_len < 0 {
                return (OPUS_INVALID_PACKET, ext);
            }
            debug_assert!(self.pos as i32 == self.len - self.curr_len);

            if id == 1 {
                // Frame separator
                if l == 0 {
                    self.curr_frame += 1;
                } else {
                    // ID byte is at ext_start, increment value is at ext_start+1
                    let inc = self.data[ext_start + 1];
                    if inc == 0 {
                        continue; // zero increment is a no-op
                    }
                    self.curr_frame += inc as i32;
                }
                if self.curr_frame >= self.nb_frames {
                    self.curr_len = -1;
                    return (OPUS_INVALID_PACKET, ext);
                }
                if self.curr_frame >= self.frame_max {
                    self.curr_len = 0;
                }
                self.repeat_pos = self.pos;
                self.last_long_pos = None;
                self.trailing_short_len = 0;
            } else if id == 2 {
                // Repeat These Extensions
                self.repeat_l = l as u8;
                self.repeat_frame = self.curr_frame + 1;
                self.repeat_len = (ext_start - self.repeat_pos) as i32;
                self.src_pos = self.repeat_pos;
                self.src_len = self.repeat_len;
                let (ret, rext) = self.next_repeat();
                if ret != 0 {
                    return (ret, rext);
                }
            } else if id > 2 {
                // Actual extension data
                if id >= 32 {
                    self.last_long_pos = Some(self.pos);
                    self.trailing_short_len = 0;
                } else {
                    self.trailing_short_len += l;
                }
                ext.id = id;
                ext.frame = self.curr_frame;
                let hdr = header_size as usize;
                ext.data = &self.data[ext_start + hdr..self.pos];
                ext.len = (self.pos - ext_start - hdr) as i32;
                return (1, ext);
            }
        }
        (0, ext)
    }

    /// Seek forward to the next extension with the given ID.
    /// Matches C `opus_extension_iterator_find`.
    pub fn find(&mut self, id: i32) -> (i32, OpusExtensionData<'a>) {
        loop {
            let (ret, ext) = self.next_ext();
            if ret <= 0 {
                return (ret, ext);
            }
            if ext.id == id {
                return (ret, ext);
            }
        }
    }
}

// ===========================================================================
// Extension convenience functions
// ===========================================================================

/// Count extensions in padding data.
/// Matches C `opus_packet_extensions_count`.
pub(crate) fn opus_packet_extensions_count(data: &[u8], len: i32, nb_frames: i32) -> i32 {
    if len <= 0 {
        return 0;
    }
    let mut iter = OpusExtensionIterator::new(data, len, nb_frames);
    let mut count = 0;
    loop {
        let (ret, _) = iter.next_ext();
        if ret <= 0 {
            break;
        }
        count += 1;
    }
    count
}

/// Parse all extensions from padding into an array in bitstream order.
/// Matches C `opus_packet_extensions_parse`.
pub(crate) fn opus_packet_extensions_parse<'a>(
    data: &'a [u8],
    len: i32,
    extensions: &mut [OpusExtensionData<'a>],
    nb_extensions: &mut i32,
    nb_frames: i32,
) -> i32 {
    if len <= 0 {
        *nb_extensions = 0;
        return 0;
    }
    let mut iter = OpusExtensionIterator::new(data, len, nb_frames);
    let max_ext = *nb_extensions;
    let mut count: i32 = 0;
    loop {
        let (ret, ext) = iter.next_ext();
        if ret <= 0 {
            *nb_extensions = count;
            return ret;
        }
        if count == max_ext {
            return OPUS_BUFFER_TOO_SMALL;
        }
        extensions[count as usize] = ext;
        count += 1;
    }
}

// ===========================================================================
// Extension generation internals
// ===========================================================================

/// Write an extension payload (excluding ID byte) to `buf`.
/// When `dry_run` is true, no bytes are written but position is tracked.
/// Returns new position or negative error code.
/// Matches C `write_extension_payload`.
fn write_extension_payload(
    buf: &mut [u8],
    dry_run: bool,
    len: i32,
    mut pos: i32,
    ext: &OpusExtensionData,
    last: bool,
) -> i32 {
    debug_assert!(ext.id >= 3 && ext.id <= 127);
    if ext.id < 32 {
        // Short extension: payload is 0 or 1 bytes
        if ext.len < 0 || ext.len > 1 {
            return OPUS_BAD_ARG;
        }
        if ext.len > 0 {
            if len - pos < ext.len {
                return OPUS_BUFFER_TOO_SMALL;
            }
            if !dry_run {
                buf[pos as usize] = ext.data[0];
            }
            pos += 1;
        }
    } else {
        // Long extension
        if ext.len < 0 {
            return OPUS_BAD_ARG;
        }
        // If last, no length encoding needed (L=0 mode)
        let length_bytes = if last { 0 } else { 1 + ext.len / 255 };
        if len - pos < length_bytes + ext.len {
            return OPUS_BUFFER_TOO_SMALL;
        }
        if !last {
            // Lacing-encoded length
            for _ in 0..ext.len / 255 {
                if !dry_run {
                    buf[pos as usize] = 255;
                }
                pos += 1;
            }
            if !dry_run {
                buf[pos as usize] = (ext.len % 255) as u8;
            }
            pos += 1;
        }
        if !dry_run {
            let p = pos as usize;
            buf[p..p + ext.len as usize].copy_from_slice(&ext.data[..ext.len as usize]);
        }
        pos += ext.len;
    }
    pos
}

/// Write a complete extension (ID byte + payload) to `buf`.
/// Returns new position or negative error code.
/// Matches C `write_extension`.
fn write_extension(
    buf: &mut [u8],
    dry_run: bool,
    len: i32,
    mut pos: i32,
    ext: &OpusExtensionData,
    last: bool,
) -> i32 {
    if len - pos < 1 {
        return OPUS_BUFFER_TOO_SMALL;
    }
    debug_assert!(ext.id >= 3 && ext.id <= 127);
    // ID byte: (id << 1) + L, where L depends on extension type
    let l_bit = if ext.id < 32 {
        ext.len // 0 or 1
    } else if last {
        0
    } else {
        1
    };
    if !dry_run {
        buf[pos as usize] = ((ext.id << 1) + l_bit) as u8;
    }
    pos += 1;
    write_extension_payload(buf, dry_run, len, pos, ext, last)
}

/// Serialize extensions into bytes. `data` may be `None` for dry-run (size query).
/// Returns total byte count or negative error code.
/// Matches C `opus_packet_extensions_generate`.
pub(crate) fn opus_packet_extensions_generate(
    data: Option<&mut [u8]>,
    len: i32,
    extensions: &[OpusExtensionData],
    nb_frames: i32,
    pad: bool,
) -> i32 {
    debug_assert!(len >= 0);
    if nb_frames > MAX_FRAMES as i32 {
        return OPUS_BAD_ARG;
    }

    let nb_ext = extensions.len() as i32;
    let mut frame_min_idx = [nb_ext; MAX_FRAMES];
    let mut frame_max_idx = [0i32; MAX_FRAMES];
    let mut frame_repeat_idx = [0i32; MAX_FRAMES];

    // Pre-scan: find min/max extension indices per frame
    for i in 0..nb_ext {
        let f = extensions[i as usize].frame;
        if f < 0 || f >= nb_frames {
            return OPUS_BAD_ARG;
        }
        if extensions[i as usize].id < 3 || extensions[i as usize].id > 127 {
            return OPUS_BAD_ARG;
        }
        let fu = f as usize;
        if i < frame_min_idx[fu] {
            frame_min_idx[fu] = i;
        }
        if i + 1 > frame_max_idx[fu] {
            frame_max_idx[fu] = i + 1;
        }
    }
    for f in 0..nb_frames as usize {
        frame_repeat_idx[f] = frame_min_idx[f];
    }

    // Setup dry_run mode
    let dry_run = data.is_none();
    let mut dummy = [0u8; 0];
    let buf: &mut [u8] = match data {
        Some(d) => d,
        None => &mut dummy[..],
    };

    let mut curr_frame: i32 = 0;
    let mut pos: i32 = 0;
    let mut written: i32 = 0;

    for f in 0..nb_frames as usize {
        let mut last_long_idx: i32 = -1;
        let mut repeat_count: i32 = 0;

        // Determine which extensions can use the repeat mechanism
        if (f + 1) < nb_frames as usize {
            let mut i = frame_min_idx[f];
            while i < frame_max_idx[f] {
                if extensions[i as usize].frame == f as i32 {
                    // Test if we can repeat this extension in all future frames
                    let mut g = f + 1;
                    while g < nb_frames as usize {
                        if frame_repeat_idx[g] >= frame_max_idx[g] {
                            break;
                        }
                        debug_assert!(extensions[frame_repeat_idx[g] as usize].frame == g as i32);
                        if extensions[frame_repeat_idx[g] as usize].id != extensions[i as usize].id
                        {
                            break;
                        }
                        if extensions[frame_repeat_idx[g] as usize].id < 32
                            && extensions[frame_repeat_idx[g] as usize].len
                                != extensions[i as usize].len
                        {
                            break;
                        }
                        g += 1;
                    }
                    if g < nb_frames as usize {
                        break; // can't repeat
                    }
                    // We can repeat! Track last long extension index
                    if extensions[i as usize].id >= 32 {
                        last_long_idx = frame_repeat_idx[nb_frames as usize - 1];
                    }
                    // Advance repeat pointers for subsequent frames
                    for g2 in (f + 1)..nb_frames as usize {
                        let mut j = frame_repeat_idx[g2] + 1;
                        while j < frame_max_idx[g2] && extensions[j as usize].frame != g2 as i32 {
                            j += 1;
                        }
                        frame_repeat_idx[g2] = j;
                    }
                    repeat_count += 1;
                    frame_repeat_idx[f] = i;
                }
                i += 1;
            }
        }

        // Write extensions for this frame
        let mut i = frame_min_idx[f];
        while i < frame_max_idx[f] {
            if extensions[i as usize].frame == f as i32 {
                // Insert separator when frame changes
                if f as i32 != curr_frame {
                    let diff = f as i32 - curr_frame;
                    if len - pos < 2 {
                        return OPUS_BUFFER_TOO_SMALL;
                    }
                    if diff == 1 {
                        if !dry_run {
                            buf[pos as usize] = 0x02;
                        }
                        pos += 1;
                    } else {
                        if !dry_run {
                            buf[pos as usize] = 0x03;
                        }
                        pos += 1;
                        if !dry_run {
                            buf[pos as usize] = diff as u8;
                        }
                        pos += 1;
                    }
                    curr_frame = f as i32;
                }

                // Write the extension
                pos = write_extension(
                    &mut *buf,
                    dry_run,
                    len,
                    pos,
                    &extensions[i as usize],
                    written == nb_ext - 1,
                );
                if pos < 0 {
                    return pos;
                }
                written += 1;

                // Handle repeat mechanism
                if repeat_count > 0 && frame_repeat_idx[f] == i {
                    let nb_repeated = repeat_count * (nb_frames - (f as i32 + 1));
                    let last = written + nb_repeated == nb_ext
                        || (last_long_idx < 0 && i + 1 >= frame_max_idx[f]);
                    if len - pos < 1 {
                        return OPUS_BUFFER_TOO_SMALL;
                    }
                    // Repeat indicator: ID=2, L=!last
                    if !dry_run {
                        buf[pos as usize] = 0x04 + if last { 0 } else { 1 };
                    }
                    pos += 1;
                    // Write repeated payloads for subsequent frames
                    for g in (f + 1)..nb_frames as usize {
                        let mut j = frame_min_idx[g];
                        while j < frame_repeat_idx[g] {
                            if extensions[j as usize].frame == g as i32 {
                                pos = write_extension_payload(
                                    &mut *buf,
                                    dry_run,
                                    len,
                                    pos,
                                    &extensions[j as usize],
                                    last && j == last_long_idx,
                                );
                                if pos < 0 {
                                    return pos;
                                }
                                written += 1;
                            }
                            j += 1;
                        }
                        frame_min_idx[g] = j;
                    }
                    if last {
                        curr_frame += 1;
                    }
                }
            }
            i += 1;
        }
    }
    debug_assert!(written == nb_ext);

    // Pad with 0x01 bytes by prepending (shifting existing data forward)
    if pad && pos < len {
        let padding = (len - pos) as usize;
        if !dry_run {
            buf.copy_within(0..pos as usize, padding);
            for idx in 0..padding {
                buf[idx] = 0x01;
            }
        }
        pos += padding as i32;
    }
    pos
}

// ===========================================================================
// OpusRepacketizer
// ===========================================================================

/// Opus packet repacketizer: merge, split, pad, and unpad Opus packets.
/// Matches C `OpusRepacketizer`.
pub struct OpusRepacketizer<'a> {
    toc: u8,
    nb_frames: usize,
    frames: [&'a [u8]; MAX_FRAMES],
    len: [i16; MAX_FRAMES],
    framesize: i32,
    paddings: [&'a [u8]; MAX_FRAMES],
    padding_len: [i32; MAX_FRAMES],
    padding_nb_frames: [u8; MAX_FRAMES],
}

impl<'a> OpusRepacketizer<'a> {
    /// Create a new repacketizer with empty state.
    pub fn new() -> Self {
        Self {
            toc: 0,
            nb_frames: 0,
            frames: [&[]; MAX_FRAMES],
            len: [0; MAX_FRAMES],
            framesize: 0,
            paddings: [&[]; MAX_FRAMES],
            padding_len: [0; MAX_FRAMES],
            padding_nb_frames: [0; MAX_FRAMES],
        }
    }

    /// Re-initialize state (sets nb_frames = 0).
    /// Matches C `opus_repacketizer_init`.
    pub fn init(&mut self) {
        self.nb_frames = 0;
    }

    /// Internal cat with self-delimited flag.
    /// Matches C `opus_repacketizer_cat_impl`.
    pub(crate) fn cat_impl(&mut self, data: &'a [u8], len: i32, self_delimited: bool) -> i32 {
        if len < 1 {
            return OPUS_INVALID_PACKET;
        }
        // Validate/set TOC
        if self.nb_frames == 0 {
            self.toc = data[0];
            self.framesize = opus_packet_get_samples_per_frame(data, 8000);
        } else if (self.toc & 0xFC) != (data[0] & 0xFC) {
            return OPUS_INVALID_PACKET;
        }
        // Get frame count of input packet
        let curr_nb_frames = match opus_packet_get_nb_frames(data) {
            Ok(n) if n >= 1 => n,
            _ => return OPUS_INVALID_PACKET,
        };
        // Check 120ms maximum packet duration (960 samples at 8kHz)
        if (curr_nb_frames + self.nb_frames as i32) * self.framesize > 960 {
            return OPUS_INVALID_PACKET;
        }
        // Parse packet to extract frame slices and padding
        let parsed = match parse_packet(data, len, self_delimited) {
            Ok(p) if p.count >= 1 => p,
            Ok(_) => return OPUS_INVALID_PACKET,
            Err(e) => return e,
        };

        let base = self.nb_frames;
        // Store frames, sizes, and padding from parsed result
        for i in 0..parsed.count {
            self.frames[base + i] = parsed.frames[i];
            self.len[base + i] = parsed.sizes[i];
        }
        self.paddings[base] = parsed.padding;
        self.padding_len[base] = parsed.padding_len;
        self.padding_nb_frames[base] = parsed.count as u8;

        // Clear padding info for all but the first frame of this cat call
        for i in 1..parsed.count {
            self.paddings[base + i] = &[];
            self.padding_len[base + i] = 0;
            self.padding_nb_frames[base + i] = 0;
        }
        self.nb_frames = base + parsed.count;
        OPUS_OK
    }

    /// Add frames from a packet to the repacketizer.
    /// Matches C `opus_repacketizer_cat`.
    pub fn cat(&mut self, data: &'a [u8], len: i32) -> i32 {
        self.cat_impl(data, len, false)
    }

    /// Return the number of frames currently held.
    /// Matches C `opus_repacketizer_get_nb_frames`.
    pub fn get_nb_frames(&self) -> i32 {
        self.nb_frames as i32
    }

    /// Full-featured packet output: selects compact code, optional self-delimited
    /// framing, padding, and extension injection.
    /// Matches C `opus_repacketizer_out_range_impl`.
    pub(crate) fn out_range_impl(
        &self,
        begin: usize,
        end: usize,
        data: &mut [u8],
        maxlen: i32,
        self_delimited: bool,
        pad_to_max: bool,
        extensions: &[OpusExtensionData<'a>],
    ) -> i32 {
        if begin >= end || end > self.nb_frames {
            return OPUS_BAD_ARG;
        }
        let count = end - begin;
        let len = &self.len[begin..end];
        let frames = &self.frames[begin..end];

        // Initial tot_size for self-delimited overhead
        let mut tot_size: i32 = if self_delimited {
            1 + if len[count - 1] >= 252 { 1 } else { 0 }
        } else {
            0
        };

        // Collect all extensions (caller-provided + parsed from padding)
        let nb_caller_ext = extensions.len() as i32;
        let mut total_ext_count = nb_caller_ext;
        for i in begin..end {
            let n = opus_packet_extensions_count(
                self.paddings[i],
                self.padding_len[i],
                self.padding_nb_frames[i] as i32,
            );
            if n > 0 {
                total_ext_count += n;
            }
        }
        let mut all_extensions = Vec::with_capacity(total_ext_count.max(0) as usize);
        for i in 0..nb_caller_ext as usize {
            all_extensions.push(extensions[i]);
        }
        // Parse padding extensions, renumbering frame indices
        for i in begin..end {
            if self.padding_len[i] <= 0 {
                continue;
            }
            let remaining_capacity = total_ext_count - all_extensions.len() as i32;
            if remaining_capacity <= 0 {
                continue;
            }
            let mut frame_exts = vec![OpusExtensionData::EMPTY; remaining_capacity as usize];
            let mut frame_ext_count = remaining_capacity;
            let ret = opus_packet_extensions_parse(
                self.paddings[i],
                self.padding_len[i],
                &mut frame_exts,
                &mut frame_ext_count,
                self.padding_nb_frames[i] as i32,
            );
            if ret < 0 {
                return OPUS_INTERNAL_ERROR;
            }
            // Renumber frame indices relative to the output range
            for j in 0..frame_ext_count as usize {
                frame_exts[j].frame += (i - begin) as i32;
                all_extensions.push(frame_exts[j]);
            }
        }
        let ext_count = all_extensions.len() as i32;

        // Determine packet code and write header
        let mut write_pos: usize = 0;
        let mut ones_begin: usize = 0;
        let mut ones_end: usize = 0;
        let mut ext_begin: usize = 0;
        let mut ext_len: i32 = 0;

        if count == 1 {
            // Code 0
            tot_size += len[0] as i32 + 1;
            if tot_size > maxlen {
                return OPUS_BUFFER_TOO_SMALL;
            }
            data[write_pos] = self.toc & 0xFC;
            write_pos += 1;
        } else if count == 2 {
            if len[1] == len[0] {
                // Code 1 (two CBR frames)
                tot_size += 2 * len[0] as i32 + 1;
                if tot_size > maxlen {
                    return OPUS_BUFFER_TOO_SMALL;
                }
                data[write_pos] = (self.toc & 0xFC) | 0x1;
                write_pos += 1;
            } else {
                // Code 2 (two VBR frames)
                tot_size += len[0] as i32 + len[1] as i32 + 2 + if len[0] >= 252 { 1 } else { 0 };
                if tot_size > maxlen {
                    return OPUS_BUFFER_TOO_SMALL;
                }
                data[write_pos] = (self.toc & 0xFC) | 0x2;
                write_pos += 1;
                write_pos += encode_size(len[0] as i32, &mut data[write_pos..]);
            }
        }

        // Upgrade to Code 3 if needed (count > 2, or padding/extensions present)
        if count > 2 || (pad_to_max && tot_size < maxlen) || ext_count > 0 {
            // Code 3: restart from beginning
            write_pos = 0;
            tot_size = if self_delimited {
                1 + if len[count - 1] >= 252 { 1 } else { 0 }
            } else {
                0
            };

            // Determine VBR vs CBR
            let mut vbr = false;
            for i in 1..count {
                if len[i] != len[0] {
                    vbr = true;
                    break;
                }
            }

            if vbr {
                tot_size += 2; // TOC + count byte
                for i in 0..count - 1 {
                    tot_size += 1 + if len[i] >= 252 { 1 } else { 0 } + len[i] as i32;
                }
                tot_size += len[count - 1] as i32;
                if tot_size > maxlen {
                    return OPUS_BUFFER_TOO_SMALL;
                }
                data[write_pos] = (self.toc & 0xFC) | 0x3;
                write_pos += 1;
                data[write_pos] = count as u8 | 0x80;
                write_pos += 1;
            } else {
                tot_size += count as i32 * len[0] as i32 + 2;
                if tot_size > maxlen {
                    return OPUS_BUFFER_TOO_SMALL;
                }
                data[write_pos] = (self.toc & 0xFC) | 0x3;
                write_pos += 1;
                data[write_pos] = count as u8;
                write_pos += 1;
            }

            // Compute padding amount
            let mut pad_amount: i32 = if pad_to_max { maxlen - tot_size } else { 0 };

            if ext_count > 0 {
                // Dry-run to compute extension byte count
                ext_len = opus_packet_extensions_generate(
                    None,
                    maxlen - tot_size,
                    &all_extensions,
                    count as i32,
                    false,
                );
                if ext_len < 0 {
                    return ext_len;
                }
                if !pad_to_max {
                    // Need just enough padding to hold extensions + encoding overhead
                    pad_amount = ext_len
                        + if ext_len != 0 {
                            (ext_len + 253) / 254
                        } else {
                            1
                        };
                }
            }

            if pad_amount != 0 {
                let nb_255s = (pad_amount - 1) / 255;
                // Set padding flag in count byte (byte at index 1)
                data[1] |= 0x40;
                if tot_size + ext_len + nb_255s + 1 > maxlen {
                    return OPUS_BUFFER_TOO_SMALL;
                }
                ext_begin = (tot_size + pad_amount - ext_len) as usize;
                ones_begin = (tot_size + nb_255s + 1) as usize;
                ones_end = (tot_size + pad_amount - ext_len) as usize;
                // Write padding length encoding
                for _ in 0..nb_255s {
                    data[write_pos] = 255;
                    write_pos += 1;
                }
                data[write_pos] = (pad_amount - 255 * nb_255s - 1) as u8;
                write_pos += 1;
                tot_size += pad_amount;
            }

            // Write VBR frame sizes
            if vbr {
                for i in 0..count - 1 {
                    write_pos += encode_size(len[i] as i32, &mut data[write_pos..]);
                }
            }
        }

        // Self-delimited last-frame size
        if self_delimited {
            write_pos += encode_size(len[count - 1] as i32, &mut data[write_pos..]);
        }

        // Copy frame data
        for i in 0..count {
            let frame_len = len[i] as usize;
            data[write_pos..write_pos + frame_len].copy_from_slice(frames[i]);
            write_pos += frame_len;
        }

        // Generate extension bytes at ext_begin
        if ext_len > 0 {
            let ret = opus_packet_extensions_generate(
                Some(&mut data[ext_begin..ext_begin + ext_len as usize]),
                ext_len,
                &all_extensions,
                count as i32,
                false,
            );
            debug_assert!(ret == ext_len);
        }

        // Fill 0x01 padding between sequential data and extensions
        for i in ones_begin..ones_end {
            data[i] = 0x01;
        }

        // Fill trailing zeros when padding with no extensions
        if pad_to_max && ext_count == 0 {
            while write_pos < maxlen as usize {
                data[write_pos] = 0;
                write_pos += 1;
            }
        }

        tot_size
    }

    /// Emit frames `[begin, end)` as a new packet.
    /// Matches C `opus_repacketizer_out_range`.
    pub fn out_range(&self, begin: usize, end: usize, data: &mut [u8], maxlen: i32) -> i32 {
        self.out_range_impl(begin, end, data, maxlen, false, false, &[])
    }

    /// Emit all frames as a new packet.
    /// Matches C `opus_repacketizer_out`.
    pub fn out(&self, data: &mut [u8], maxlen: i32) -> i32 {
        self.out_range_impl(0, self.nb_frames, data, maxlen, false, false, &[])
    }
}

// ===========================================================================
// Pad / Unpad (single stream)
// ===========================================================================

/// Internal pad implementation with extension support.
/// Matches C `opus_packet_pad_impl`.
pub(crate) fn opus_packet_pad_impl(
    data: &mut [u8],
    len: i32,
    new_len: i32,
    pad: bool,
    extensions: &[OpusExtensionData],
) -> i32 {
    if len < 1 {
        return OPUS_BAD_ARG;
    }
    if len == new_len {
        return OPUS_OK;
    }
    if len > new_len {
        return OPUS_BAD_ARG;
    }
    // Copy original packet to temp buffer so frame pointers don't alias output
    let copy = data[..len as usize].to_vec();
    let mut rp = OpusRepacketizer::new();
    let ret = rp.cat(&copy, len);
    if ret != OPUS_OK {
        return ret;
    }
    let nb = rp.nb_frames;
    rp.out_range_impl(0, nb, data, new_len, false, pad, extensions)
}

/// Pad a packet to `new_len` bytes in-place.
/// Returns `OPUS_OK` on success or a negative error code.
/// Matches C `opus_packet_pad`.
pub fn opus_packet_pad(data: &mut [u8], len: i32, new_len: i32) -> i32 {
    let ret = opus_packet_pad_impl(data, len, new_len, true, &[]);
    if ret > 0 { OPUS_OK } else { ret }
}

/// Strip all padding from a packet in-place.
/// Returns the new (shorter) length or a negative error code.
/// Matches C `opus_packet_unpad`.
pub fn opus_packet_unpad(data: &mut [u8], len: i32) -> i32 {
    if len < 1 {
        return OPUS_BAD_ARG;
    }
    // Copy original so frame pointers don't alias the output buffer
    let copy = data[..len as usize].to_vec();
    let mut rp = OpusRepacketizer::new();
    let ret = rp.cat(&copy, len);
    if ret < 0 {
        return ret;
    }
    // Discard all padding and extensions
    for i in 0..rp.nb_frames {
        rp.padding_len[i] = 0;
        rp.paddings[i] = &[];
    }
    let nb = rp.nb_frames;
    let ret = rp.out_range_impl(0, nb, data, len, false, false, &[]);
    debug_assert!(ret > 0 && ret <= len);
    ret
}

// ===========================================================================
// Multistream pad / unpad
// ===========================================================================

/// Pad the last stream of a multistream packet.
/// Matches C `opus_multistream_packet_pad`.
pub fn opus_multistream_packet_pad(
    data: &mut [u8],
    len: i32,
    new_len: i32,
    nb_streams: i32,
) -> i32 {
    if len < 1 {
        return OPUS_BAD_ARG;
    }
    if len == new_len {
        return OPUS_OK;
    }
    if len > new_len {
        return OPUS_BAD_ARG;
    }
    let amount = new_len - len;

    // Seek to last stream by parsing self-delimited sub-packets
    let mut offset: usize = 0;
    let mut remaining = len;
    for _ in 0..nb_streams - 1 {
        if remaining <= 0 {
            return OPUS_INVALID_PACKET;
        }
        let parsed = match parse_packet(&data[offset..], remaining, true) {
            Ok(p) => p,
            Err(e) => return e,
        };
        offset += parsed.packet_offset as usize;
        remaining -= parsed.packet_offset;
    }

    // Pad only the last stream
    opus_packet_pad(&mut data[offset..], remaining, remaining + amount)
}

/// Unpad all streams of a multistream packet in-place.
/// Returns the new (shorter) total length or a negative error code.
/// Matches C `opus_multistream_packet_unpad`.
pub fn opus_multistream_packet_unpad(data: &mut [u8], len: i32, nb_streams: i32) -> i32 {
    if len < 1 {
        return OPUS_BAD_ARG;
    }
    // Copy entire buffer so frame references don't alias the output
    let copy = data[..len as usize].to_vec();
    let mut dst_pos: usize = 0;
    let mut src_pos: usize = 0;
    let mut remaining = len;

    for s in 0..nb_streams {
        let self_delimited = s != nb_streams - 1;
        if remaining <= 0 {
            return OPUS_INVALID_PACKET;
        }

        // Parse to get packet_offset
        let parsed = match parse_packet(&copy[src_pos..], remaining, self_delimited) {
            Ok(p) => p,
            Err(e) => return e,
        };
        let packet_offset = parsed.packet_offset;

        // Cat this sub-packet into a fresh repacketizer
        let mut rp = OpusRepacketizer::new();
        let ret = rp.cat_impl(
            &copy[src_pos..src_pos + packet_offset as usize],
            packet_offset,
            self_delimited,
        );
        if ret < 0 {
            return ret;
        }

        // Discard all padding and extensions
        for i in 0..rp.nb_frames {
            rp.padding_len[i] = 0;
            rp.paddings[i] = &[];
        }

        // Output to data at dst_pos
        let nb = rp.nb_frames;
        let ret = rp.out_range_impl(
            0,
            nb,
            &mut data[dst_pos..],
            remaining,
            self_delimited,
            false,
            &[],
        );
        if ret < 0 {
            return ret;
        }

        dst_pos += ret as usize;
        src_pos += packet_offset as usize;
        remaining -= packet_offset;
    }

    dst_pos as i32
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_size() {
        // Small sizes (< 252): single byte
        let mut buf = [0u8; 2];
        assert_eq!(encode_size(0, &mut buf), 1);
        assert_eq!(buf[0], 0);

        assert_eq!(encode_size(251, &mut buf), 1);
        assert_eq!(buf[0], 251);

        // Sizes >= 252: two bytes
        assert_eq!(encode_size(252, &mut buf), 2);
        let (bytes, sz) = parse_size(&buf, 2);
        assert_eq!(bytes, 2);
        assert_eq!(sz, 252);

        assert_eq!(encode_size(1275, &mut buf), 2);
        let (bytes, sz) = parse_size(&buf, 2);
        assert_eq!(bytes, 2);
        assert_eq!(sz, 1275);

        // Round-trip for all valid sizes
        for size in 0..=1275i32 {
            let n = encode_size(size, &mut buf);
            let (_, decoded) = parse_size(&buf, n as i32);
            assert_eq!(decoded as i32, size, "size={size}");
        }
    }

    #[test]
    fn test_parse_size_errors() {
        let buf = [0u8; 0];
        let (b, s) = parse_size(&buf, 0);
        assert_eq!(b, -1);
        assert_eq!(s, -1);

        let buf = [252u8]; // needs 2 bytes but only 1 available
        let (b, s) = parse_size(&buf, 1);
        assert_eq!(b, -1);
        assert_eq!(s, -1);
    }

    #[test]
    fn test_parse_size_branch_matrix() {
        let buf = [251u8];
        assert_eq!(parse_size(&buf, 1), (1, 251));

        let buf = [252u8, 1u8];
        assert_eq!(parse_size(&buf, 2), (2, 256));

        let buf = [252u8];
        assert_eq!(parse_size(&buf, 1), (-1, -1));

        let buf = [0u8];
        assert_eq!(parse_size(&buf, 0), (-1, -1));
    }

    #[test]
    fn test_parse_packet_branch_matrix() {
        let code0 = [0x08u8, 0xAA, 0xBB];
        let parsed = parse_packet(&code0, code0.len() as i32, false).unwrap();
        assert_eq!(parsed.count, 1);
        assert_eq!(parsed.sizes[0], 2);
        assert_eq!(parsed.frames[0], &[0xAA, 0xBB]);
        assert_eq!(parsed.packet_offset, 3);

        let code1 = [0x09u8, 0xCC, 0xDD];
        let parsed = parse_packet(&code1, code1.len() as i32, false).unwrap();
        assert_eq!(parsed.count, 2);
        assert_eq!(parsed.sizes[0], 1);
        assert_eq!(parsed.sizes[1], 1);
        assert_eq!(parsed.frames[0], &[0xCC]);
        assert_eq!(parsed.frames[1], &[0xDD]);

        let code2 = [0x0Au8, 0x01, 0xEE, 0xFF, 0x11];
        let parsed = parse_packet(&code2, code2.len() as i32, false).unwrap();
        assert_eq!(parsed.count, 2);
        assert_eq!(parsed.sizes[0], 1);
        assert_eq!(parsed.sizes[1], 2);
        assert_eq!(parsed.frames[0], &[0xEE]);
        assert_eq!(parsed.frames[1], &[0xFF, 0x11]);

        let code3_cbr = [0x0Bu8, 0x02, 0xAA, 0xBB, 0xCC, 0xDD];
        let parsed = parse_packet(&code3_cbr, code3_cbr.len() as i32, false).unwrap();
        assert_eq!(parsed.count, 2);
        assert_eq!(parsed.sizes[0], 2);
        assert_eq!(parsed.sizes[1], 2);
        assert_eq!(parsed.frames[0], &[0xAA, 0xBB]);
        assert_eq!(parsed.frames[1], &[0xCC, 0xDD]);

        let code3_vbr_pad = [0x0Bu8, 0xC2, 0x01, 0x01, 0xAA, 0xBB, 0xCC, 0xEE];
        let parsed = parse_packet(&code3_vbr_pad, code3_vbr_pad.len() as i32, false).unwrap();
        assert_eq!(parsed.count, 2);
        assert_eq!(parsed.sizes[0], 1);
        assert_eq!(parsed.sizes[1], 2);
        assert_eq!(parsed.frames[0], &[0xAA]);
        assert_eq!(parsed.frames[1], &[0xBB, 0xCC]);
        assert_eq!(parsed.padding_len, 1);
        assert_eq!(parsed.padding, &[0xEE]);
        assert_eq!(parsed.packet_offset, 8);

        assert!(matches!(
            parse_packet(&[0x09u8, 0xAA], 2, false),
            Err(OPUS_INVALID_PACKET)
        ));
        assert!(matches!(
            parse_packet(&[0x0Au8, 0x04, 0xAA, 0xBB], 4, false),
            Err(OPUS_INVALID_PACKET)
        ));
        assert!(matches!(
            parse_packet(&[0x0Bu8, 0x00], 2, false),
            Err(OPUS_INVALID_PACKET)
        ));
    }

    #[test]
    fn test_skip_extension_payload_branch_matrix() {
        let mut pos = 0usize;
        let mut header_size = 7;
        assert_eq!(
            skip_extension_payload(&[], &mut pos, 0, &mut header_size, 0x01, 0),
            0
        );
        assert_eq!(pos, 0);
        assert_eq!(header_size, 0);

        let short = [0xABu8];
        let mut pos = 0usize;
        let mut header_size = 0;
        assert_eq!(
            skip_extension_payload(&short, &mut pos, 1, &mut header_size, 0x0B, 0),
            0
        );
        assert_eq!(pos, 1);
        assert_eq!(header_size, 0);

        let mut pos = 0usize;
        let mut header_size = 0;
        let mut long = vec![0u8; 258];
        long[0] = 255;
        long[1] = 1;
        assert_eq!(
            skip_extension_payload(&long, &mut pos, 258, &mut header_size, 0x41, 0),
            0
        );
        assert_eq!(pos, 258);
        assert_eq!(header_size, 2);

        let mut pos = 0usize;
        let mut header_size = 0;
        assert_eq!(skip_extension(&[], &mut pos, 0, &mut header_size), 0);
        assert_eq!(pos, 0);
        assert_eq!(header_size, 0);
    }

    #[test]
    fn test_extension_generation_repeat_and_padding() {
        let ext0 = OpusExtensionData {
            id: 5,
            frame: 0,
            data: &[0x11],
            len: 1,
        };
        let ext1 = OpusExtensionData {
            id: 5,
            frame: 1,
            data: &[0x11],
            len: 1,
        };
        let ext2 = OpusExtensionData {
            id: 5,
            frame: 2,
            data: &[0x11],
            len: 1,
        };

        let size = opus_packet_extensions_generate(None, 256, &[ext0, ext1, ext2], 3, false);
        assert!(size > 0);

        let mut buf = vec![0u8; (size + 8) as usize];
        let ret =
            opus_packet_extensions_generate(Some(&mut buf), size + 8, &[ext0, ext1, ext2], 3, true);
        assert_eq!(ret, size + 8);
        assert!(buf[..8].iter().all(|&b| b == 0x01));

        let mut parsed = [OpusExtensionData::EMPTY; 4];
        let mut nb = 4;
        assert_eq!(
            opus_packet_extensions_parse(&buf[8..8 + size as usize], size, &mut parsed, &mut nb, 3),
            0
        );
        assert_eq!(nb, 3);
        assert_eq!(parsed[0].frame, 0);
        assert_eq!(parsed[1].frame, 1);
        assert_eq!(parsed[2].frame, 2);
        assert_eq!(parsed[0].data, &[0x11]);
        assert_eq!(parsed[1].data, &[0x11]);
        assert_eq!(parsed[2].data, &[0x11]);
    }

    #[test]
    fn test_repacketizer_single_frame() {
        // Code 0 SILK-only packet: TOC 0x08 = SILK narrowband 10ms, 1 frame
        let packet = [0x08u8, 0xAA, 0xBB, 0xCC];
        let mut rp = OpusRepacketizer::new();
        assert_eq!(rp.cat(&packet, 4), OPUS_OK);
        assert_eq!(rp.get_nb_frames(), 1);

        let mut out = [0u8; 256];
        let ret = rp.out(&mut out, 256);
        assert!(ret > 0);
        // Code 0: TOC byte (top 6 bits) + frame data
        assert_eq!(out[0], 0x08); // TOC with code=0
        assert_eq!(&out[1..ret as usize], &[0xAA, 0xBB, 0xCC]);
    }

    #[test]
    fn test_repacketizer_two_cbr_frames() {
        // Two identical-length packets with same TOC (top 6 bits)
        let pkt1 = [0x08u8, 0xAA, 0xBB]; // 2 bytes payload
        let pkt2 = [0x08u8, 0xCC, 0xDD]; // 2 bytes payload

        let mut rp = OpusRepacketizer::new();
        assert_eq!(rp.cat(&pkt1, 3), OPUS_OK);
        assert_eq!(rp.cat(&pkt2, 3), OPUS_OK);
        assert_eq!(rp.get_nb_frames(), 2);

        let mut out = [0u8; 256];
        let ret = rp.out(&mut out, 256);
        assert!(ret > 0);
        // Code 1 (CBR): TOC|0x01 + frame1 + frame2
        assert_eq!(out[0], 0x08 | 0x01);
        assert_eq!(&out[1..3], &[0xAA, 0xBB]);
        assert_eq!(&out[3..5], &[0xCC, 0xDD]);
        assert_eq!(ret, 5);
    }

    #[test]
    fn test_repacketizer_two_vbr_frames() {
        let pkt1 = [0x08u8, 0xAA, 0xBB]; // 2 bytes payload
        let pkt2 = [0x08u8, 0xCC, 0xDD, 0xEE]; // 3 bytes payload

        let mut rp = OpusRepacketizer::new();
        assert_eq!(rp.cat(&pkt1, 3), OPUS_OK);
        assert_eq!(rp.cat(&pkt2, 4), OPUS_OK);
        assert_eq!(rp.get_nb_frames(), 2);

        let mut out = [0u8; 256];
        let ret = rp.out(&mut out, 256);
        assert!(ret > 0);
        // Code 2 (VBR): TOC|0x02 + size(frame1) + frame1 + frame2
        assert_eq!(out[0], 0x08 | 0x02);
        assert_eq!(out[1], 2); // size of first frame
        assert_eq!(&out[2..4], &[0xAA, 0xBB]);
        assert_eq!(&out[4..7], &[0xCC, 0xDD, 0xEE]);
        assert_eq!(ret, 7);
    }

    #[test]
    fn test_repacketizer_self_delimited_cbr_roundtrip() {
        let pkt1 = [0x08u8, 0x11, 0x22];
        let pkt2 = [0x08u8, 0x33, 0x44];

        let mut rp = OpusRepacketizer::new();
        assert_eq!(rp.cat(&pkt1, 3), OPUS_OK);
        assert_eq!(rp.cat(&pkt2, 3), OPUS_OK);

        let mut out = [0u8; 32];
        let ret = rp.out_range_impl(0, 2, &mut out, 32, true, false, &[]);
        assert_eq!(ret, 6);
        assert_eq!(out[0] & 0x3, 0x1);
        assert_eq!(out[1], 2);

        let mut parsed = OpusRepacketizer::new();
        assert_eq!(parsed.cat_impl(&out, ret, true), OPUS_OK);
        assert_eq!(parsed.get_nb_frames(), 2);
        assert_eq!(parsed.frames[0], &pkt1[1..]);
        assert_eq!(parsed.frames[1], &pkt2[1..]);
    }

    #[test]
    fn test_repacketizer_self_delimited_vbr_roundtrip() {
        let pkt1 = [0x08u8, 0xAA, 0xBB];
        let pkt2 = [0x08u8, 0xCC, 0xDD, 0xEE];

        let mut rp = OpusRepacketizer::new();
        assert_eq!(rp.cat(&pkt1, 3), OPUS_OK);
        assert_eq!(rp.cat(&pkt2, 4), OPUS_OK);

        let mut out = [0u8; 32];
        let ret = rp.out_range_impl(0, 2, &mut out, 32, true, false, &[]);
        assert_eq!(ret, 8);
        assert_eq!(out[0] & 0x3, 0x2);
        assert_eq!(out[1], 2);

        let mut parsed = OpusRepacketizer::new();
        assert_eq!(parsed.cat_impl(&out, ret, true), OPUS_OK);
        assert_eq!(parsed.get_nb_frames(), 2);
        assert_eq!(parsed.frames[0], &pkt1[1..]);
        assert_eq!(parsed.frames[1], &pkt2[1..]);
    }

    #[test]
    fn test_repacketizer_self_delimited_code3_roundtrip() {
        let pkt = [0x08u8, 0x11, 0x22];
        let mut rp = OpusRepacketizer::new();
        for _ in 0..3 {
            assert_eq!(rp.cat(&pkt, pkt.len() as i32), OPUS_OK);
        }

        let mut out = [0u8; 64];
        let ret = rp.out_range_impl(0, 3, &mut out, 64, true, false, &[]);
        assert!(ret > 0);
        assert_eq!(out[0] & 0x3, 0x3);
        assert_eq!(out[1] & 0x3F, 3);
        assert_eq!(out[2], 2);
        assert_eq!(&out[3..5], &[0x11, 0x22]);
        assert_eq!(&out[5..7], &[0x11, 0x22]);
        assert_eq!(&out[7..9], &[0x11, 0x22]);

        let mut parsed = OpusRepacketizer::new();
        assert_eq!(parsed.cat_impl(&out, ret, true), OPUS_OK);
        assert_eq!(parsed.get_nb_frames(), 3);
        assert_eq!(parsed.frames[0], &pkt[1..]);
        assert_eq!(parsed.frames[1], &pkt[1..]);
        assert_eq!(parsed.frames[2], &pkt[1..]);
    }

    #[test]
    fn test_repacketizer_self_delimited_code3_vbr_roundtrip() {
        let pkt1 = [0x08u8, 0xAA, 0xBB];
        let pkt2 = [0x08u8, 0xCC, 0xDD, 0xEE];
        let pkt3 = [0x08u8, 0x11, 0x22, 0x33, 0x44];

        let mut rp = OpusRepacketizer::new();
        assert_eq!(rp.cat(&pkt1, pkt1.len() as i32), OPUS_OK);
        assert_eq!(rp.cat(&pkt2, pkt2.len() as i32), OPUS_OK);
        assert_eq!(rp.cat(&pkt3, pkt3.len() as i32), OPUS_OK);

        let mut out = [0u8; 64];
        let ret = rp.out_range_impl(0, 3, &mut out, 64, true, false, &[]);
        assert!(ret > 0);
        assert_eq!(out[0] & 0x3, 0x3);
        assert_eq!(out[1] & 0x80, 0x80);
        assert_eq!(out[2], 2);
        assert_eq!(out[3], 3);
        assert_eq!(out[4], 4);
        assert_eq!(&out[5..7], &[0xAA, 0xBB]);
        assert_eq!(&out[7..10], &[0xCC, 0xDD, 0xEE]);
        assert_eq!(&out[10..14], &[0x11, 0x22, 0x33, 0x44]);

        let mut parsed = OpusRepacketizer::new();
        assert_eq!(parsed.cat_impl(&out, ret, true), OPUS_OK);
        assert_eq!(parsed.get_nb_frames(), 3);
        assert_eq!(parsed.frames[0], &pkt1[1..]);
        assert_eq!(parsed.frames[1], &pkt2[1..]);
        assert_eq!(parsed.frames[2], &pkt3[1..]);
    }

    #[test]
    fn test_repacketizer_pad_unpad_with_extensions() {
        let original = [0x08u8, 0xAA, 0xBB, 0xCC];
        let ext = OpusExtensionData {
            id: 5,
            frame: 0,
            data: &[0x42],
            len: 1,
        };

        let mut buf = [0u8; 128];
        buf[..original.len()].copy_from_slice(&original);

        let padded_len = opus_packet_pad_impl(&mut buf, original.len() as i32, 64, true, &[ext]);
        assert_eq!(padded_len, 64);

        let mut parsed = OpusRepacketizer::new();
        assert_eq!(parsed.cat(&buf, padded_len), OPUS_OK);
        assert_eq!(parsed.get_nb_frames(), 1);
        assert!(parsed.padding_len[0] > 0);
        assert_eq!(
            opus_packet_extensions_count(parsed.paddings[0], parsed.padding_len[0], 1),
            1
        );

        let mut parsed_exts = [OpusExtensionData::EMPTY; 2];
        let mut nb_ext = 2;
        assert_eq!(
            opus_packet_extensions_parse(
                parsed.paddings[0],
                parsed.padding_len[0],
                &mut parsed_exts,
                &mut nb_ext,
                1
            ),
            0
        );
        assert_eq!(nb_ext, 1);
        assert_eq!(parsed_exts[0].id, 5);
        assert_eq!(parsed_exts[0].frame, 0);
        assert_eq!(parsed_exts[0].len, 1);
        assert_eq!(parsed_exts[0].data[0], 0x42);

        let unpadded = opus_packet_unpad(&mut buf, padded_len);
        assert_eq!(unpadded, original.len() as i32);
        assert_eq!(&buf[..original.len()], &original);
    }

    #[test]
    fn test_packet_pad_large_padding_roundtrip() {
        let original = [0x08u8, 0xAA, 0xBB, 0xCC];
        let mut buf = [0u8; 600];
        buf[..original.len()].copy_from_slice(&original);

        let ret = opus_packet_pad(&mut buf, original.len() as i32, 600);
        assert_eq!(ret, OPUS_OK);
        assert_eq!(buf[0] & 0x3, 0x3);
        assert_ne!(buf[1] & 0x40, 0);
        assert_eq!(buf[2], 255);
        assert_eq!(buf[3], 255);

        let unpadded = opus_packet_unpad(&mut buf, 600);
        assert_eq!(unpadded, original.len() as i32);
        assert_eq!(&buf[..original.len()], &original);
    }

    #[test]
    fn test_repacketizer_rejects_invalid_packet_shapes() {
        let mut rp = OpusRepacketizer::new();

        // Code 1 with odd payload length must fail.
        let odd_cbr = [0x09u8, 0xAA, 0xBB, 0xCC];
        assert_eq!(rp.cat(&odd_cbr, odd_cbr.len() as i32), OPUS_INVALID_PACKET);

        // Code 3 with a zero frame count must fail.
        let bad_count = [0x0Bu8, 0x00];
        assert_eq!(
            rp.cat(&bad_count, bad_count.len() as i32),
            OPUS_INVALID_PACKET
        );

        // Multistream pad requires self-delimited prefix streams; this one is not.
        let mut multi = [0x08u8, 0xAA, 0xBB];
        assert_eq!(
            opus_multistream_packet_pad(&mut multi, 3, 12, 2),
            OPUS_INVALID_PACKET
        );
    }

    #[test]
    fn test_repacketizer_three_frames_code3() {
        let pkt = [0x08u8, 0xAA]; // 1 byte payload each
        let mut rp = OpusRepacketizer::new();
        for _ in 0..3 {
            assert_eq!(rp.cat(&pkt, 2), OPUS_OK);
        }
        assert_eq!(rp.get_nb_frames(), 3);

        let mut out = [0u8; 256];
        let ret = rp.out(&mut out, 256);
        assert!(ret > 0);
        // Code 3 CBR: TOC|0x03, count=3, then 3 frames of 1 byte each
        assert_eq!(out[0], 0x08 | 0x03);
        assert_eq!(out[1], 3); // count (CBR, no P flag, no V flag)
        assert_eq!(&out[2..5], &[0xAA, 0xAA, 0xAA]);
        assert_eq!(ret, 5);
    }

    #[test]
    fn test_repacketizer_toc_mismatch() {
        let pkt1 = [0x08u8, 0xAA]; // SILK narrowband
        let pkt2 = [0x48u8, 0xBB]; // SILK wideband (different top 6 bits)

        let mut rp = OpusRepacketizer::new();
        assert_eq!(rp.cat(&pkt1, 2), OPUS_OK);
        assert_eq!(rp.cat(&pkt2, 2), OPUS_INVALID_PACKET);
    }

    #[test]
    fn test_repacketizer_120ms_limit() {
        // 60ms SILK frames (TOC bits 4:3 = 11): 480 samples at 8kHz
        // Two of these = 960 = limit. Third would exceed.
        let pkt = [0x18u8, 0xAA]; // 60ms SILK narrowband

        let mut rp = OpusRepacketizer::new();
        assert_eq!(rp.cat(&pkt, 2), OPUS_OK);
        assert_eq!(rp.cat(&pkt, 2), OPUS_OK);
        assert_eq!(rp.cat(&pkt, 2), OPUS_INVALID_PACKET); // exceeds 120ms
    }

    #[test]
    fn test_repacketizer_out_range() {
        let pkt1 = [0x08u8, 0xAA];
        let pkt2 = [0x08u8, 0xBB];
        let pkt3 = [0x08u8, 0xCC];

        let mut rp = OpusRepacketizer::new();
        assert_eq!(rp.cat(&pkt1, 2), OPUS_OK);
        assert_eq!(rp.cat(&pkt2, 2), OPUS_OK);
        assert_eq!(rp.cat(&pkt3, 2), OPUS_OK);

        // Extract middle frame only
        let mut out = [0u8; 256];
        let ret = rp.out_range(1, 2, &mut out, 256);
        assert!(ret > 0);
        assert_eq!(out[0], 0x08); // Code 0
        assert_eq!(out[1], 0xBB);
    }

    #[test]
    fn test_repacketizer_bad_range() {
        let pkt = [0x08u8, 0xAA];
        let mut rp = OpusRepacketizer::new();
        assert_eq!(rp.cat(&pkt, 2), OPUS_OK);

        let mut out = [0u8; 256];
        assert_eq!(rp.out_range(0, 0, &mut out, 256), OPUS_BAD_ARG);
        assert_eq!(rp.out_range(1, 0, &mut out, 256), OPUS_BAD_ARG);
        assert_eq!(rp.out_range(0, 2, &mut out, 256), OPUS_BAD_ARG);
    }

    #[test]
    fn test_repacketizer_buffer_too_small() {
        let pkt = [0x08u8, 0xAA, 0xBB, 0xCC]; // 3 byte payload
        let mut rp = OpusRepacketizer::new();
        assert_eq!(rp.cat(&pkt, 4), OPUS_OK);

        let mut out = [0u8; 2]; // too small for TOC + 3 bytes
        assert_eq!(rp.out(&mut out, 2), OPUS_BUFFER_TOO_SMALL);
    }

    #[test]
    fn test_packet_pad_unpad_roundtrip() {
        // Create a Code 0 packet: TOC + 3 bytes payload
        let original = [0x08u8, 0xAA, 0xBB, 0xCC];
        let mut buf = [0u8; 64];
        buf[..4].copy_from_slice(&original);

        // Pad to 64 bytes
        let ret = opus_packet_pad(&mut buf, 4, 64);
        assert_eq!(ret, OPUS_OK);

        // Unpad back
        let ret = opus_packet_unpad(&mut buf, 64);
        assert!(ret > 0);
        let unpadded = &buf[..ret as usize];

        // The unpadded packet should carry the same audio content.
        // It will be a Code 0 packet with the same frame data.
        // TOC top 6 bits match, bottom 2 bits = 0 (code 0)
        assert_eq!(unpadded[0] & 0xFC, 0x08);
        assert_eq!(&unpadded[1..], &[0xAA, 0xBB, 0xCC]);
    }

    #[test]
    fn test_packet_pad_edge_cases() {
        let mut buf = [0u8; 64];
        assert_eq!(opus_packet_pad(&mut buf, 0, 64), OPUS_BAD_ARG); // len < 1
        buf[0] = 0x08;
        assert_eq!(opus_packet_pad(&mut buf, 10, 10), OPUS_OK); // len == new_len
        assert_eq!(opus_packet_pad(&mut buf, 10, 5), OPUS_BAD_ARG); // len > new_len
    }

    #[test]
    fn test_repacketizer_error_branches() {
        let mut rp = OpusRepacketizer::new();

        let empty: [u8; 0] = [];
        assert_eq!(rp.cat(&empty, 0), OPUS_INVALID_PACKET);
        assert_eq!(rp.cat(&empty, -1), OPUS_INVALID_PACKET);

        let mut bad_unpad = [0x0Bu8, 0x00];
        let bad_unpad_len = bad_unpad.len() as i32;
        assert_eq!(
            opus_packet_unpad(&mut bad_unpad, bad_unpad_len),
            OPUS_INVALID_PACKET
        );

        let mut short = [0x08u8, 0xAA];
        assert_eq!(opus_packet_unpad(&mut short, 0), OPUS_BAD_ARG);
        assert_eq!(
            opus_multistream_packet_unpad(&mut short, 0, 2),
            OPUS_BAD_ARG
        );

        let mut malformed = [0x08u8, 0xAA, 0xBB, 0x08u8, 0xCC];
        let malformed_len = malformed.len() as i32;
        assert_eq!(
            opus_multistream_packet_unpad(&mut malformed, malformed_len, 2),
            OPUS_INVALID_PACKET
        );

        let mut pad_buf = [0x08u8, 0xAA, 0xBB, 0xCC];
        let bad_ext = OpusExtensionData {
            id: 5,
            frame: 0,
            data: &[0x11, 0x22],
            len: 2,
        };
        assert_eq!(
            opus_packet_pad_impl(&mut pad_buf, 4, 64, true, &[bad_ext]),
            OPUS_BAD_ARG
        );
    }

    #[test]
    fn test_packet_unpad_minimal() {
        // Already-minimal Code 0 packet (no padding)
        let mut buf = [0x08u8, 0xAA];
        let ret = opus_packet_unpad(&mut buf, 2);
        assert_eq!(ret, 2); // no change
        assert_eq!(buf, [0x08, 0xAA]);
    }

    #[test]
    fn test_repacketizer_init_reuse() {
        let pkt = [0x08u8, 0xAA];
        let mut rp = OpusRepacketizer::new();
        assert_eq!(rp.cat(&pkt, 2), OPUS_OK);
        assert_eq!(rp.get_nb_frames(), 1);

        rp.init();
        assert_eq!(rp.get_nb_frames(), 0);

        // Can reuse after init
        let pkt2 = [0x48u8, 0xBB]; // different TOC
        assert_eq!(rp.cat(&pkt2, 2), OPUS_OK);
        assert_eq!(rp.get_nb_frames(), 1);
    }

    #[test]
    fn test_extension_iterator_empty() {
        let data: &[u8] = &[];
        let mut iter = OpusExtensionIterator::new(data, 0, 1);
        let (ret, _) = iter.next_ext();
        assert_eq!(ret, 0);
    }

    #[test]
    fn test_extension_iterator_reset_and_zero_increment_separator() {
        let data = [0x03u8, 0x00, 0x0Bu8, 0xAA];
        let mut iter = OpusExtensionIterator::new(&data, data.len() as i32, 2);

        let (ret, ext) = iter.next_ext();
        assert_eq!(ret, 1);
        assert_eq!(ext.id, 5);
        assert_eq!(ext.frame, 0);
        assert_eq!(ext.len, 1);
        assert_eq!(ext.data, &[0xAA]);

        iter.reset();
        let (ret, ext) = iter.next_ext();
        assert_eq!(ret, 1);
        assert_eq!(ext.id, 5);
        assert_eq!(ext.frame, 0);
        assert_eq!(ext.len, 1);
        assert_eq!(ext.data, &[0xAA]);
    }

    #[test]
    fn test_write_extension_payload_short_zero_and_long_laced() {
        let short = OpusExtensionData {
            id: 5,
            frame: 0,
            data: &[],
            len: 0,
        };
        let mut buf = [0u8; 260];
        assert_eq!(write_extension_payload(&mut buf, false, 260, 0, &short, false), 0);

        let payload: Vec<u8> = (0..256u16).map(|v| v as u8).collect();
        let long = OpusExtensionData {
            id: 40,
            frame: 0,
            data: &payload,
            len: payload.len() as i32,
        };
        assert_eq!(write_extension_payload(&mut buf, false, 260, 0, &long, false), 258);
        assert_eq!(&buf[..2], &[255, 1]);
        assert_eq!(&buf[2..258], payload.as_slice());
    }

    #[test]
    fn test_extension_count_no_extensions() {
        let count = opus_packet_extensions_count(&[], 0, 1);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_extension_generate_and_parse_single() {
        // Generate a single short extension (id=5, frame=0, 1 byte payload)
        let ext = OpusExtensionData {
            id: 5,
            frame: 0,
            data: &[0x42],
            len: 1,
        };

        // Dry-run to get size
        let size = opus_packet_extensions_generate(None, 256, &[ext], 1, false);
        assert!(size > 0);

        // Generate
        let mut buf = vec![0u8; size as usize];
        let ret = opus_packet_extensions_generate(Some(&mut buf), size, &[ext], 1, false);
        assert_eq!(ret, size);

        // Parse back
        let mut parsed = [OpusExtensionData::EMPTY; 4];
        let mut nb = 4;
        let ret = opus_packet_extensions_parse(&buf, size, &mut parsed, &mut nb, 1);
        assert!(ret >= 0);
        assert_eq!(nb, 1);
        assert_eq!(parsed[0].id, 5);
        assert_eq!(parsed[0].frame, 0);
        assert_eq!(parsed[0].len, 1);
        assert_eq!(parsed[0].data[0], 0x42);
    }

    #[test]
    fn test_extension_generate_and_parse_long() {
        // Generate a long extension (id=32, frame=0, 10 bytes)
        let payload = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let ext = OpusExtensionData {
            id: 32,
            frame: 0,
            data: &payload,
            len: 10,
        };

        let size = opus_packet_extensions_generate(None, 256, &[ext], 1, false);
        assert!(size > 0);

        let mut buf = vec![0u8; size as usize];
        let ret = opus_packet_extensions_generate(Some(&mut buf), size, &[ext], 1, false);
        assert_eq!(ret, size);

        let mut parsed = [OpusExtensionData::EMPTY; 4];
        let mut nb = 4;
        let ret = opus_packet_extensions_parse(&buf, size, &mut parsed, &mut nb, 1);
        assert!(ret >= 0);
        assert_eq!(nb, 1);
        assert_eq!(parsed[0].id, 32);
        assert_eq!(parsed[0].len, 10);
        assert_eq!(parsed[0].data, &payload);
    }

    #[test]
    fn test_extension_multiframe() {
        // Two extensions on different frames
        let ext0 = OpusExtensionData {
            id: 5,
            frame: 0,
            data: &[0xAA],
            len: 1,
        };
        let ext1 = OpusExtensionData {
            id: 5,
            frame: 1,
            data: &[0xBB],
            len: 1,
        };

        let size = opus_packet_extensions_generate(None, 256, &[ext0, ext1], 2, false);
        assert!(size > 0);

        let mut buf = vec![0u8; size as usize];
        let ret = opus_packet_extensions_generate(Some(&mut buf), size, &[ext0, ext1], 2, false);
        assert_eq!(ret, size);

        let mut parsed = [OpusExtensionData::EMPTY; 8];
        let mut nb = 8;
        let ret = opus_packet_extensions_parse(&buf, size, &mut parsed, &mut nb, 2);
        assert!(ret >= 0);
        assert_eq!(nb, 2);
    }

    #[test]
    fn test_extension_repeat_roundtrip() {
        let ext0 = OpusExtensionData {
            id: 5,
            frame: 0,
            data: &[0x11],
            len: 1,
        };
        let ext1 = OpusExtensionData {
            id: 5,
            frame: 1,
            data: &[0x11],
            len: 1,
        };
        let ext2 = OpusExtensionData {
            id: 5,
            frame: 2,
            data: &[0x11],
            len: 1,
        };

        let size = opus_packet_extensions_generate(None, 256, &[ext0, ext1, ext2], 3, false);
        assert!(size > 0);

        let mut buf = vec![0u8; size as usize];
        let ret =
            opus_packet_extensions_generate(Some(&mut buf), size, &[ext0, ext1, ext2], 3, false);
        assert_eq!(ret, size);

        let mut iter = OpusExtensionIterator::new(&buf, size, 3);
        let mut seen = Vec::new();
        loop {
            let (ret, ext) = iter.next_ext();
            if ret <= 0 {
                break;
            }
            seen.push((ext.id, ext.frame, ext.len, ext.data[0]));
        }

        assert_eq!(
            seen,
            vec![(5, 0, 1, 0x11), (5, 1, 1, 0x11), (5, 2, 1, 0x11)]
        );
    }

    #[test]
    fn test_extension_iterator_find() {
        let ext0 = OpusExtensionData {
            id: 5,
            frame: 0,
            data: &[0xAA],
            len: 1,
        };
        let ext1 = OpusExtensionData {
            id: 32,
            frame: 0,
            data: &[0xBB, 0xCC],
            len: 2,
        };

        let size = opus_packet_extensions_generate(None, 256, &[ext0, ext1], 1, false);
        let mut buf = vec![0u8; size as usize];
        opus_packet_extensions_generate(Some(&mut buf), size, &[ext0, ext1], 1, false);

        let mut iter = OpusExtensionIterator::new(&buf, size, 1);
        let (ret, found) = iter.find(32);
        assert_eq!(ret, 1);
        assert_eq!(found.id, 32);
        assert_eq!(found.len, 2);
    }

    #[test]
    fn test_extension_iterator_reset_and_write_payload_edges() {
        let ext0 = OpusExtensionData {
            id: 5,
            frame: 0,
            data: &[0x11],
            len: 1,
        };
        let ext1 = OpusExtensionData {
            id: 40,
            frame: 2,
            data: &[1, 2, 3],
            len: 3,
        };
        let size = opus_packet_extensions_generate(None, 64, &[ext0, ext1], 3, false);
        let mut buf = vec![0u8; size as usize];
        assert_eq!(
            opus_packet_extensions_generate(Some(&mut buf), size, &[ext0, ext1], 3, false),
            size
        );

        let mut iter = OpusExtensionIterator::new(&buf, size, 3);
        let (ret, _) = iter.next_ext();
        assert!(ret > 0);
        iter.reset();
        assert_eq!(iter.pos, 0);
        assert_eq!(iter.repeat_pos, 0);
        assert_eq!(iter.curr_len, iter.len);
        assert_eq!(iter.curr_frame, 0);
        assert_eq!(iter.repeat_frame, 0);
        assert_eq!(iter.trailing_short_len, 0);

        let short_bad = OpusExtensionData {
            id: 5,
            frame: 0,
            data: &[0xAA, 0xBB],
            len: 2,
        };
        assert_eq!(
            write_extension_payload(&mut [0u8; 2], false, 2, 0, &short_bad, false),
            OPUS_BAD_ARG
        );

        let short_tight = OpusExtensionData {
            id: 5,
            frame: 0,
            data: &[0xAA],
            len: 1,
        };
        assert_eq!(
            write_extension_payload(&mut [0u8; 0], false, 0, 0, &short_tight, false),
            OPUS_BUFFER_TOO_SMALL
        );

        let long_bad = OpusExtensionData {
            id: 40,
            frame: 0,
            data: &[],
            len: -1,
        };
        assert_eq!(
            write_extension_payload(&mut [0u8; 4], false, 4, 0, &long_bad, false),
            OPUS_BAD_ARG
        );

        let long_ok = OpusExtensionData {
            id: 40,
            frame: 0,
            data: &[9, 8, 7],
            len: 3,
        };
        let mut out = [0u8; 3];
        assert_eq!(
            write_extension_payload(&mut out, false, 3, 0, &long_ok, true),
            3
        );
        assert_eq!(out, [9, 8, 7]);
    }

    #[test]
    fn test_extension_iterator_frame_max_limits_results() {
        let ext0 = OpusExtensionData {
            id: 5,
            frame: 0,
            data: &[0x11],
            len: 1,
        };
        let ext1 = OpusExtensionData {
            id: 5,
            frame: 1,
            data: &[0x22],
            len: 1,
        };
        let ext2 = OpusExtensionData {
            id: 5,
            frame: 2,
            data: &[0x33],
            len: 1,
        };

        let size = opus_packet_extensions_generate(None, 256, &[ext0, ext1, ext2], 3, false);
        assert!(size > 0);

        let mut buf = vec![0u8; size as usize];
        assert_eq!(
            opus_packet_extensions_generate(Some(&mut buf), size, &[ext0, ext1, ext2], 3, false),
            size
        );

        let mut iter = OpusExtensionIterator::new(&buf, size, 3);
        iter.set_frame_max(1);

        let (ret, ext) = iter.next_ext();
        assert_eq!(ret, 1);
        assert_eq!(ext.frame, 0);
        assert_eq!(ext.data, &[0x11]);
        assert_eq!(iter.next_ext().0, 0);
    }

    #[test]
    fn test_extension_iterator_rejects_invalid_frame_separator_increment() {
        let data = [0x03u8, 0x02];
        let mut iter = OpusExtensionIterator::new(&data, data.len() as i32, 2);

        let (ret, _) = iter.next_ext();
        assert_eq!(ret, OPUS_INVALID_PACKET);
    }

    #[test]
    fn test_packet_extensions_parse_buffer_too_small() {
        let ext0 = OpusExtensionData {
            id: 5,
            frame: 0,
            data: &[0x11],
            len: 1,
        };
        let ext1 = OpusExtensionData {
            id: 6,
            frame: 0,
            data: &[0x22],
            len: 1,
        };

        let size = opus_packet_extensions_generate(None, 256, &[ext0, ext1], 1, false);
        assert!(size > 0);

        let mut buf = vec![0u8; size as usize];
        assert_eq!(
            opus_packet_extensions_generate(Some(&mut buf), size, &[ext0, ext1], 1, false),
            size
        );

        let mut parsed = [OpusExtensionData::EMPTY; 1];
        let mut nb_ext = 1;
        assert_eq!(
            opus_packet_extensions_parse(&buf, size, &mut parsed, &mut nb_ext, 1),
            OPUS_BUFFER_TOO_SMALL
        );
    }

    #[test]
    fn test_packet_extensions_parse_zero_capacity_and_multistream_pad_edges() {
        let ext = OpusExtensionData {
            id: 5,
            frame: 0,
            data: &[0x42],
            len: 1,
        };
        let size = opus_packet_extensions_generate(None, 16, &[ext], 1, false);
        let mut buf = vec![0u8; size as usize];
        assert_eq!(
            opus_packet_extensions_generate(Some(&mut buf), size, &[ext], 1, false),
            size
        );

        let mut parsed = [];
        let mut nb = 0;
        assert_eq!(
            opus_packet_extensions_parse(&buf, size, &mut parsed, &mut nb, 1),
            OPUS_BUFFER_TOO_SMALL
        );

        let mut packet = [0x08u8, 0xAA, 0xBB];
        assert_eq!(opus_multistream_packet_pad(&mut packet, 3, 3, 1), OPUS_OK);
        assert_eq!(opus_multistream_packet_pad(&mut packet, 3, 2, 1), OPUS_BAD_ARG);
    }

    #[test]
    fn test_packet_extensions_generate_error_paths() {
        let valid = OpusExtensionData {
            id: 5,
            frame: 0,
            data: &[0x11],
            len: 1,
        };

        let bad_frame = OpusExtensionData { frame: 2, ..valid };
        assert_eq!(
            opus_packet_extensions_generate(None, 64, &[bad_frame], 1, false),
            OPUS_BAD_ARG
        );

        let bad_id = OpusExtensionData { id: 2, ..valid };
        assert_eq!(
            opus_packet_extensions_generate(None, 64, &[bad_id], 1, false),
            OPUS_BAD_ARG
        );

        let mut short = [0u8; 1];
        assert_eq!(
            opus_packet_extensions_generate(Some(&mut short), 1, &[valid], 1, false),
            OPUS_BUFFER_TOO_SMALL
        );
    }

    #[test]
    fn test_packet_extensions_and_out_range_guard_paths() {
        let ext = OpusExtensionData {
            id: 5,
            frame: 0,
            data: &[0x11],
            len: 1,
        };

        let mut parsed = [OpusExtensionData::EMPTY; 1];
        let mut nb_ext = 1;
        assert_eq!(
            opus_packet_extensions_parse(&[], 0, &mut parsed, &mut nb_ext, 1),
            0
        );
        assert_eq!(nb_ext, 0);

        assert_eq!(
            opus_packet_extensions_generate(
                None,
                64,
                &[ext],
                (MAX_FRAMES as i32) + 1,
                false
            ),
            OPUS_BAD_ARG
        );

        let mut short = [0u8; 1];
        assert_eq!(
            opus_packet_extensions_generate(Some(&mut short), 1, &[ext], 1, false),
            OPUS_BUFFER_TOO_SMALL
        );

        let pkt = [0x08u8, 0xAA];
        let mut rp = OpusRepacketizer::new();
        assert_eq!(rp.cat(&pkt, 2), OPUS_OK);

        let mut out = [0u8; 4];
        assert_eq!(rp.out_range_impl(0, 0, &mut out, 4, false, false, &[]), OPUS_BAD_ARG);
        assert_eq!(
            rp.out_range_impl(0, 1, &mut out, 1, false, false, &[]),
            OPUS_BUFFER_TOO_SMALL
        );
    }

    #[test]
    fn test_parse_code3_cbr_packet() {
        // Construct a Code 3 CBR packet: TOC|0x03, count=3, then 3 frames of 2 bytes
        let mut pkt = vec![0x08u8 | 0x03, 3]; // TOC with code 3, count=3 (CBR, no P, no V)
        pkt.extend_from_slice(&[0xAA, 0xBB]); // frame 0
        pkt.extend_from_slice(&[0xCC, 0xDD]); // frame 1
        pkt.extend_from_slice(&[0xEE, 0xFF]); // frame 2

        let mut rp = OpusRepacketizer::new();
        assert_eq!(rp.cat(&pkt, pkt.len() as i32), OPUS_OK);
        assert_eq!(rp.get_nb_frames(), 3);

        // Re-emit and verify
        let mut out = [0u8; 256];
        let ret = rp.out(&mut out, 256);
        assert!(ret > 0);
        // Should produce Code 3 CBR again
        assert_eq!(out[0] & 0x3, 0x3);
        assert_eq!(out[1] & 0x3F, 3);
    }

    #[test]
    fn test_parse_code3_vbr_packet() {
        // Code 3 VBR: TOC|0x03, count=2|0x80, size(frame0)=2, frame0, frame1(3 bytes)
        let pkt = [0x08u8 | 0x03, 2 | 0x80, 2, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE];

        let mut rp = OpusRepacketizer::new();
        assert_eq!(rp.cat(&pkt, pkt.len() as i32), OPUS_OK);
        assert_eq!(rp.get_nb_frames(), 2);
    }

    #[test]
    fn test_multistream_pad_unpad() {
        // Construct a simple 2-stream multistream packet:
        // Stream 1 (self-delimited): Code 0, TOC, self-delimited size, frame data
        // Stream 2 (normal): Code 0, TOC, frame data

        // Stream 1: TOC=0x08 (SILK NB 10ms), self-delimited size=2, frame=[0xAA, 0xBB]
        // Self-delimited Code 0: [TOC][sd_size][frame_data]
        let pkt = vec![
            0x08u8, // TOC (code 0)
            2,      // self-delimited size = 2
            0xAA, 0xBB, 0x08u8, // Stream 2 TOC
            0xCC, 0xDD,
        ];

        let original_len = pkt.len() as i32;
        let padded_len = original_len + 20;

        let mut buf = vec![0u8; padded_len as usize];
        buf[..pkt.len()].copy_from_slice(&pkt);

        // Pad
        let ret = opus_multistream_packet_pad(&mut buf, original_len, padded_len, 2);
        assert_eq!(ret, OPUS_OK);

        // Unpad
        let ret = opus_multistream_packet_unpad(&mut buf, padded_len, 2);
        assert!(ret > 0);
        assert_eq!(ret, original_len);
    }
}
