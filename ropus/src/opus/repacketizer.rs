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
pub fn opus_packet_extensions_count(data: &[u8], len: i32, nb_frames: i32) -> i32 {
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
pub fn opus_packet_extensions_parse<'a>(
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
pub fn opus_packet_extensions_generate(
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
    pub fn out_range_impl(
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
        // Parse padding extensions, renumbering frame indices.
        // Must always attempt parse when padding exists (even with 0 capacity)
        // so malformed extensions trigger OPUS_INTERNAL_ERROR, matching C.
        for i in begin..end {
            if self.padding_len[i] <= 0 {
                continue;
            }
            let remaining_capacity =
                (total_ext_count - all_extensions.len() as i32).max(0) as usize;
            let mut frame_exts = vec![OpusExtensionData::EMPTY; remaining_capacity];
            let mut frame_ext_count = remaining_capacity as i32;
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
        assert_eq!(
            write_extension_payload(&mut buf, false, 260, 0, &short, false),
            0
        );

        let payload: Vec<u8> = (0..256u16).map(|v| v as u8).collect();
        let long = OpusExtensionData {
            id: 40,
            frame: 0,
            data: &payload,
            len: payload.len() as i32,
        };
        assert_eq!(
            write_extension_payload(&mut buf, false, 260, 0, &long, false),
            258
        );
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
        assert_eq!(
            opus_multistream_packet_pad(&mut packet, 3, 2, 1),
            OPUS_BAD_ARG
        );
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
            opus_packet_extensions_generate(None, 64, &[ext], (MAX_FRAMES as i32) + 1, false),
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
        assert_eq!(
            rp.out_range_impl(0, 0, &mut out, 4, false, false, &[]),
            OPUS_BAD_ARG
        );
        assert_eq!(
            rp.out_range_impl(0, 1, &mut out, 1, false, false, &[]),
            OPUS_BUFFER_TOO_SMALL
        );
    }

    #[test]
    fn test_out_range_single_frame_extension_and_padding_paths() {
        let ext = OpusExtensionData {
            id: 5,
            frame: 0,
            data: &[0x42],
            len: 1,
        };

        let mut with_ext = [0u8; 16];
        with_ext[..3].copy_from_slice(&[0x08, 0xAA, 0xBB]);
        let ret = opus_packet_pad_impl(&mut with_ext, 3, 16, true, &[ext]);
        assert_eq!(ret, 16);
        assert_eq!(with_ext[0] & 0x3, 0x3);
        assert_ne!(with_ext[1] & 0x40, 0);

        let parsed = parse_packet(&with_ext, ret, false).unwrap();
        assert_eq!(parsed.count, 1);
        assert_eq!(parsed.frames[0], &[0xAA, 0xBB]);
        assert!(parsed.padding_len > 0);

        let mut padded = [0u8; 16];
        padded[..3].copy_from_slice(&[0x08, 0xAA, 0xBB]);
        let ret = opus_packet_pad_impl(&mut padded, 3, 16, true, &[]);
        assert_eq!(ret, 16);
        assert_eq!(padded[0] & 0x3, 0x3);
        assert_ne!(padded[1] & 0x40, 0);
        assert!(
            padded[ret as usize - 4..ret as usize]
                .iter()
                .all(|&b| b == 0)
        );
    }

    #[test]
    fn test_out_range_impl_reemits_extensions_from_stored_padding() {
        let pkt = [0x08u8, 0xAA];
        let mut rp = OpusRepacketizer::new();
        assert_eq!(rp.cat(&pkt, pkt.len() as i32), OPUS_OK);

        let ext = OpusExtensionData {
            id: 40,
            frame: 0,
            data: &[0x12, 0x34],
            len: 2,
        };
        let padding_len = opus_packet_extensions_generate(None, 32, &[ext], 1, false);
        assert!(padding_len > 0);

        let mut padding = vec![0u8; padding_len as usize];
        assert_eq!(
            opus_packet_extensions_generate(Some(&mut padding), padding_len, &[ext], 1, false),
            padding_len
        );

        rp.paddings[0] = &padding;
        rp.padding_len[0] = padding_len;
        rp.padding_nb_frames[0] = 1;

        let mut out = [0u8; 64];
        let ret = rp.out_range_impl(0, 1, &mut out, 64, false, false, &[]);
        assert!(ret > 0);
        assert_eq!(out[0] & 0x3, 0x3);

        let parsed = parse_packet(&out[..ret as usize], ret, false).unwrap();
        assert!(parsed.padding_len > 0);
        assert_eq!(
            opus_packet_extensions_count(parsed.padding, parsed.padding_len, parsed.count as i32),
            1
        );
    }

    #[test]
    fn test_out_range_impl_rejects_partially_invalid_padding_extensions() {
        let pkt = [0x08u8, 0xAA];
        let mut rp = OpusRepacketizer::new();
        assert_eq!(rp.cat(&pkt, pkt.len() as i32), OPUS_OK);

        let invalid_padding = [0x0Bu8, 0x42, 0x03, 0x02];
        rp.paddings[0] = &invalid_padding;
        rp.padding_len[0] = invalid_padding.len() as i32;
        rp.padding_nb_frames[0] = 2;

        let mut out = [0u8; 64];
        assert_eq!(
            rp.out_range_impl(0, 1, &mut out, 64, false, false, &[]),
            OPUS_INTERNAL_ERROR
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

    #[test]
    fn test_multistream_pad_and_unpad_require_additional_streams() {
        let mut pad_buf = [0x08u8, 2, 0xAA, 0xBB];
        let pad_len = pad_buf.len() as i32;
        assert_eq!(
            opus_multistream_packet_pad(&mut pad_buf, pad_len, 8, 3),
            OPUS_INVALID_PACKET
        );

        let mut unpad_buf = [0x08u8, 2, 0xAA, 0xBB, 0x08u8, 0xCC, 0xDD];
        let unpad_len = unpad_buf.len() as i32;
        assert_eq!(
            opus_multistream_packet_unpad(&mut unpad_buf, unpad_len, 3),
            OPUS_INVALID_PACKET
        );
    }

    // ===================================================================
    // Additional coverage tests
    // ===================================================================

    #[test]
    fn test_extension_long_lacing_n255_payload() {
        let payload: Vec<u8> = (0..260).map(|i| (i & 0xFF) as u8).collect();
        let ext = OpusExtensionData {
            id: 40,
            frame: 0,
            data: &payload,
            len: 260,
        };
        let size = opus_packet_extensions_generate(None, 512, &[ext], 1, false);
        assert!(size > 0);
        let mut buf = vec![0u8; size as usize];
        opus_packet_extensions_generate(Some(&mut buf), size, &[ext], 1, false);
        let mut parsed = [OpusExtensionData::EMPTY; 4];
        let mut nb = 4;
        opus_packet_extensions_parse(&buf, size, &mut parsed, &mut nb, 1);
        assert_eq!(nb, 1);
        assert_eq!(parsed[0].id, 40);
        assert_eq!(parsed[0].len, 260);
        assert_eq!(parsed[0].data, &payload[..]);
    }

    #[test]
    fn test_extension_variable_length_data_multiple_ids() {
        let ext_short0 = OpusExtensionData {
            id: 3,
            frame: 0,
            data: &[],
            len: 0,
        };
        let ext_short1 = OpusExtensionData {
            id: 5,
            frame: 0,
            data: &[0xAB],
            len: 1,
        };
        let long_payload = [0x11u8; 20];
        let ext_long = OpusExtensionData {
            id: 64,
            frame: 0,
            data: &long_payload,
            len: 20,
        };
        let exts = [ext_short0, ext_short1, ext_long];
        let size = opus_packet_extensions_generate(None, 256, &exts, 1, false);
        let mut buf = vec![0u8; size as usize];
        opus_packet_extensions_generate(Some(&mut buf), size, &exts, 1, false);
        let mut parsed = [OpusExtensionData::EMPTY; 8];
        let mut nb = 8;
        opus_packet_extensions_parse(&buf, size, &mut parsed, &mut nb, 1);
        assert_eq!(nb, 3);
        assert_eq!(parsed[0].id, 3);
        assert_eq!(parsed[1].id, 5);
        assert_eq!(parsed[2].id, 64);
        assert_eq!(parsed[2].len, 20);
    }

    #[test]
    fn test_extension_cascade_repeat_across_three_frames() {
        let payload = [0x99u8; 5];
        let exts: Vec<OpusExtensionData> = (0..3)
            .map(|f| OpusExtensionData {
                id: 40,
                frame: f,
                data: &payload,
                len: 5,
            })
            .collect();
        let size = opus_packet_extensions_generate(None, 256, &exts, 3, false);
        let mut buf = vec![0u8; size as usize];
        opus_packet_extensions_generate(Some(&mut buf), size, &exts, 3, false);
        let mut iter = OpusExtensionIterator::new(&buf, size, 3);
        let mut results = Vec::new();
        loop {
            let (ret, ext) = iter.next_ext();
            if ret <= 0 {
                break;
            }
            results.push((ext.id, ext.frame, ext.len));
        }
        assert_eq!(results.len(), 3);
        for (i, &(id, frame, len)) in results.iter().enumerate() {
            assert_eq!(id, 40);
            assert_eq!(frame, i as i32);
            assert_eq!(len, 5);
        }
    }

    #[test]
    fn test_self_delimited_code0_large_frame_size() {
        let payload = vec![0xBBu8; 300];
        let mut pkt = vec![0x08u8];
        let mut sz_buf = [0u8; 2];
        let n = encode_size(300, &mut sz_buf);
        pkt.extend_from_slice(&sz_buf[..n]);
        pkt.extend_from_slice(&payload);
        let parsed = parse_packet(&pkt, pkt.len() as i32, true).unwrap();
        assert_eq!(parsed.count, 1);
        assert_eq!(parsed.sizes[0], 300);
    }

    #[test]
    fn test_self_delimited_cbr_size_mismatch_rejected() {
        let pkt = [0x09u8, 10, 0xAA, 0xBB];
        assert!(matches!(
            parse_packet(&pkt, pkt.len() as i32, true),
            Err(OPUS_INVALID_PACKET)
        ));
    }

    #[test]
    fn test_self_delimited_vbr_code2_size_overflow() {
        let pkt = [0x0Au8, 2, 0xAA, 0xBB, 100];
        assert!(matches!(
            parse_packet(&pkt, pkt.len() as i32, true),
            Err(OPUS_INVALID_PACKET)
        ));
    }

    #[test]
    fn test_padding_with_non_zero_content() {
        let pkt = [0x08u8 | 0x03, 0x01 | 0x40, 3, 0xAA, 0xDE, 0xAD, 0xBE];
        let parsed = parse_packet(&pkt, pkt.len() as i32, false).unwrap();
        assert_eq!(parsed.count, 1);
        assert_eq!(parsed.sizes[0], 1);
        assert_eq!(parsed.padding_len, 3);
        assert_eq!(parsed.padding, &[0xDE, 0xAD, 0xBE]);
    }

    #[test]
    fn test_padding_255_encoding() {
        let mut pkt = vec![0x08u8 | 0x03, 0x01 | 0x40, 255, 10];
        pkt.extend_from_slice(&[0x42u8; 2]);
        pkt.extend_from_slice(&[0x01u8; 264]);
        let parsed = parse_packet(&pkt, pkt.len() as i32, false).unwrap();
        assert_eq!(parsed.count, 1);
        assert_eq!(parsed.sizes[0], 2);
        assert_eq!(parsed.padding_len, 264);
    }

    #[test]
    fn test_multi_frame_repacketization_four_frames() {
        let pkt = [0x08u8, 0xAA, 0xBB];
        let mut rp = OpusRepacketizer::new();
        for _ in 0..4 {
            assert_eq!(rp.cat(&pkt, pkt.len() as i32), OPUS_OK);
        }
        assert_eq!(rp.get_nb_frames(), 4);
        let mut out = [0u8; 256];
        let ret = rp.out(&mut out, 256);
        assert!(ret > 0);
        assert_eq!(out[0] & 0x3, 0x3);
        assert_eq!(out[1] & 0x3F, 4);
        let parsed = parse_packet(&out[..ret as usize], ret, false).unwrap();
        assert_eq!(parsed.count, 4);
        for i in 0..4 {
            assert_eq!(parsed.frames[i], &[0xAA, 0xBB]);
        }
    }

    #[test]
    fn test_multi_frame_vbr_different_sizes() {
        let pkt1 = [0x08u8, 0x11];
        let pkt2 = [0x08u8, 0x22, 0x33];
        let pkt3 = [0x08u8, 0x44, 0x55, 0x66];
        let mut rp = OpusRepacketizer::new();
        assert_eq!(rp.cat(&pkt1, pkt1.len() as i32), OPUS_OK);
        assert_eq!(rp.cat(&pkt2, pkt2.len() as i32), OPUS_OK);
        assert_eq!(rp.cat(&pkt3, pkt3.len() as i32), OPUS_OK);
        let mut out = [0u8; 256];
        let ret = rp.out(&mut out, 256);
        assert!(ret > 0);
        assert_eq!(out[0] & 0x3, 0x3);
        assert_ne!(out[1] & 0x80, 0);
        let parsed = parse_packet(&out[..ret as usize], ret, false).unwrap();
        assert_eq!(parsed.count, 3);
        assert_eq!(parsed.frames[0], &[0x11]);
        assert_eq!(parsed.frames[1], &[0x22, 0x33]);
        assert_eq!(parsed.frames[2], &[0x44, 0x55, 0x66]);
    }

    #[test]
    fn test_code3_vbr_manual_parse() {
        let mut pkt = vec![0x08u8 | 0x03, 3 | 0x80, 1, 3];
        pkt.push(0xAA);
        pkt.extend_from_slice(&[0xBB, 0xCC, 0xDD]);
        pkt.extend_from_slice(&[0xEE, 0xFF]);
        let parsed = parse_packet(&pkt, pkt.len() as i32, false).unwrap();
        assert_eq!(parsed.count, 3);
        assert_eq!(parsed.sizes[0], 1);
        assert_eq!(parsed.sizes[1], 3);
        assert_eq!(parsed.sizes[2], 2);
    }

    #[test]
    fn test_code3_cbr_with_padding_stripping() {
        let pkt = vec![
            0x08u8 | 0x03,
            2 | 0x40,
            5,
            0xAA,
            0xBB,
            0xCC,
            0xDD,
            0x01,
            0x01,
            0x01,
            0x01,
            0x01,
        ];
        let mut rp = OpusRepacketizer::new();
        assert_eq!(rp.cat(&pkt, pkt.len() as i32), OPUS_OK);
        let mut out = [0u8; 256];
        let ret = rp.out(&mut out, 256);
        assert!(ret > 0);
        assert_eq!(out[0] & 0x3, 0x1);
    }

    #[test]
    fn test_toc_configurations_silk_celt_hybrid() {
        assert_eq!(parse_packet(&[0x00u8, 0xAA], 2, false).unwrap().toc, 0x00);
        assert_eq!(
            parse_packet(&[0x48u8, 0xBB, 0xCC], 3, false).unwrap().toc,
            0x48
        );
        assert_eq!(parse_packet(&[0x80u8, 0xDD], 2, false).unwrap().toc, 0x80);
        assert_eq!(
            parse_packet(&[0x60u8, 0xEE, 0xFF], 3, false).unwrap().toc,
            0x60
        );
        assert_eq!(
            parse_packet(&[0x18u8, 0xAA, 0xBB], 3, false).unwrap().toc,
            0x18
        );
    }

    #[test]
    fn test_out_range_impl_various_ranges() {
        let pkts = [
            [0x08u8, 0x11],
            [0x08u8, 0x22],
            [0x08u8, 0x33],
            [0x08u8, 0x44],
        ];
        let mut rp = OpusRepacketizer::new();
        for pkt in &pkts {
            assert_eq!(rp.cat(pkt, pkt.len() as i32), OPUS_OK);
        }
        let mut out = [0u8; 256];
        let _ret = rp.out_range(0, 1, &mut out, 256);
        assert_eq!(out[0] & 0x3, 0x0);
        assert_eq!(out[1], 0x11);
        let ret = rp.out_range(2, 4, &mut out, 256);
        assert!(ret > 0);
        let _ret = rp.out_range(0, 4, &mut out, 256);
        assert_eq!(out[0] & 0x3, 0x3);
    }

    #[test]
    fn test_out_range_impl_with_pad_to_max() {
        let pkt = [0x08u8, 0xAA, 0xBB];
        let mut rp = OpusRepacketizer::new();
        assert_eq!(rp.cat(&pkt, pkt.len() as i32), OPUS_OK);
        let mut out = [0u8; 64];
        let ret = rp.out_range_impl(0, 1, &mut out, 64, false, true, &[]);
        assert_eq!(ret, 64);
        assert_eq!(out[0] & 0x3, 0x3);
        assert_ne!(out[1] & 0x40, 0);
    }

    #[test]
    fn test_code3_vbr_large_frame_size_encoding() {
        let small_payload = [0x11u8; 10];
        let large_payload = vec![0x22u8; 300];
        let mut pkt_small = vec![0x08u8];
        pkt_small.extend_from_slice(&small_payload);
        let mut pkt_large = vec![0x08u8];
        pkt_large.extend_from_slice(&large_payload);
        let mut rp = OpusRepacketizer::new();
        assert_eq!(rp.cat(&pkt_small, pkt_small.len() as i32), OPUS_OK);
        assert_eq!(rp.cat(&pkt_large, pkt_large.len() as i32), OPUS_OK);
        let mut out = [0u8; 512];
        let ret = rp.out(&mut out, 512);
        let parsed = parse_packet(&out[..ret as usize], ret, false).unwrap();
        assert_eq!(parsed.count, 2);
        assert_eq!(parsed.sizes[0] as usize, 10);
        assert_eq!(parsed.sizes[1] as usize, 300);
    }

    #[test]
    fn test_parse_negative_len_returns_bad_arg() {
        assert!(matches!(
            parse_packet(&[0x08u8, 0xAA], -1, false),
            Err(OPUS_BAD_ARG)
        ));
    }

    #[test]
    fn test_parse_code3_vbr_last_size_negative() {
        let pkt = [0x08u8 | 0x03, 2 | 0x80, 250, 0xAA, 0xBB];
        assert!(matches!(
            parse_packet(&pkt, pkt.len() as i32, false),
            Err(OPUS_INVALID_PACKET)
        ));
    }

    #[test]
    fn test_parse_code3_cbr_non_divisible_rejected() {
        let pkt = [0x08u8 | 0x03, 3, 0xAA, 0xBB, 0xCC, 0xDD];
        assert!(matches!(
            parse_packet(&pkt, pkt.len() as i32, false),
            Err(OPUS_INVALID_PACKET)
        ));
    }

    #[test]
    fn test_parse_code3_exceeds_duration_limit() {
        let pkt = [0x08u8 | 0x03, 13];
        assert!(matches!(
            parse_packet(&pkt, pkt.len() as i32, false),
            Err(OPUS_INVALID_PACKET)
        ));
    }

    #[test]
    fn test_parse_frame_size_above_1275_rejected() {
        let mut pkt = vec![0x08u8];
        pkt.extend(vec![0xAA; 1276]);
        assert!(matches!(
            parse_packet(&pkt, pkt.len() as i32, false),
            Err(OPUS_INVALID_PACKET)
        ));
    }

    #[test]
    fn test_multistream_two_streams_pad_unpad_roundtrip() {
        let pkt = vec![0x08u8, 3, 0x11, 0x22, 0x33, 0x08u8, 0x44, 0x55];
        let original_len = pkt.len() as i32;
        let padded_len = original_len + 30;
        let mut buf = vec![0u8; padded_len as usize];
        buf[..pkt.len()].copy_from_slice(&pkt);
        assert_eq!(
            opus_multistream_packet_pad(&mut buf, original_len, padded_len, 2),
            OPUS_OK
        );
        assert_eq!(
            opus_multistream_packet_unpad(&mut buf, padded_len, 2),
            original_len
        );
    }

    #[test]
    fn test_extension_frame_separator_multi_increment() {
        let ext0 = OpusExtensionData {
            id: 5,
            frame: 0,
            data: &[0x11],
            len: 1,
        };
        let ext3 = OpusExtensionData {
            id: 5,
            frame: 3,
            data: &[0x44],
            len: 1,
        };
        let size = opus_packet_extensions_generate(None, 256, &[ext0, ext3], 4, false);
        let mut buf = vec![0u8; size as usize];
        opus_packet_extensions_generate(Some(&mut buf), size, &[ext0, ext3], 4, false);
        let mut parsed = [OpusExtensionData::EMPTY; 8];
        let mut nb = 8;
        opus_packet_extensions_parse(&buf, size, &mut parsed, &mut nb, 4);
        assert_eq!(nb, 2);
        assert_eq!(parsed[0].frame, 0);
        assert_eq!(parsed[1].frame, 3);
    }

    #[test]
    fn test_skip_extension_payload_short_insufficient() {
        let mut pos = 0usize;
        let mut header_size = 0;
        assert_eq!(
            skip_extension_payload(&[], &mut pos, 0, &mut header_size, 0x0B, 0),
            -1
        );
    }

    #[test]
    fn test_skip_extension_payload_long_l0_with_trailing() {
        let data = [0x11u8, 0x22, 0x33, 0x44, 0x55];
        let mut pos = 0usize;
        let mut header_size = 0;
        let rem = skip_extension_payload(&data, &mut pos, 5, &mut header_size, 0x40, 2);
        assert_eq!(rem, 2);
        assert_eq!(pos, 3);
    }

    #[test]
    fn test_skip_extension_payload_long_l0_insufficient_trailing() {
        let mut pos = 0usize;
        let mut header_size = 0;
        assert_eq!(
            skip_extension_payload(&[], &mut pos, 1, &mut header_size, 0x40, 5),
            -1
        );
    }

    #[test]
    fn test_skip_extension_negative_remaining() {
        let mut pos = 0usize;
        let mut header_size = 0;
        assert_eq!(skip_extension(&[], &mut pos, -1, &mut header_size), -1);
        assert_eq!(pos, 0);
    }

    #[test]
    fn test_write_extension_buffer_too_small_for_id() {
        let ext = OpusExtensionData {
            id: 5,
            frame: 0,
            data: &[0x11],
            len: 1,
        };
        let mut buf = [0u8; 0];
        assert_eq!(
            write_extension(&mut buf, false, 0, 0, &ext, false),
            OPUS_BUFFER_TOO_SMALL
        );
    }

    #[test]
    fn test_write_extension_long_last_vs_not_last() {
        let payload = [0x11u8, 0x22, 0x33];
        let ext = OpusExtensionData {
            id: 40,
            frame: 0,
            data: &payload,
            len: 3,
        };
        let mut buf = [0u8; 8];
        let ret = write_extension(&mut buf, false, 8, 0, &ext, true);
        assert_eq!(ret, 4);
        assert_eq!(buf[0] & 1, 0);
        let mut buf = [0u8; 8];
        let ret = write_extension(&mut buf, false, 8, 0, &ext, false);
        assert_eq!(ret, 5);
        assert_eq!(buf[0] & 1, 1);
        assert_eq!(buf[1], 3);
    }

    // ===================================================================
    // Targeted coverage tests for remaining uncovered lines
    // ===================================================================

    /// Test out_range_impl BUFFER_TOO_SMALL paths for Code 1 CBR (line 1169)
    /// and Code 2 VBR (line 1177) with exactly 2 frames.
    #[test]
    fn test_out_range_impl_two_frame_buffer_too_small() {
        // Code 1 CBR: 2 equal frames of 3 bytes each -> needs 1+3+3=7 bytes
        let pkt = [0x08u8, 0xAA, 0xBB, 0xCC];
        let mut rp = OpusRepacketizer::new();
        assert_eq!(rp.cat(&pkt, 4), OPUS_OK);
        assert_eq!(rp.cat(&pkt, 4), OPUS_OK);
        let mut out = [0u8; 6]; // too small for 7
        assert_eq!(rp.out(&mut out, 6), OPUS_BUFFER_TOO_SMALL);

        // Code 2 VBR: frames of 2 and 3 bytes -> needs 1+1+2+3=7
        let pkt_a = [0x08u8, 0xAA, 0xBB];
        let pkt_b = [0x08u8, 0xCC, 0xDD, 0xEE];
        let mut rp2 = OpusRepacketizer::new();
        assert_eq!(rp2.cat(&pkt_a, 3), OPUS_OK);
        assert_eq!(rp2.cat(&pkt_b, 4), OPUS_OK);
        let mut out2 = [0u8; 6]; // too small for 7
        assert_eq!(rp2.out(&mut out2, 6), OPUS_BUFFER_TOO_SMALL);
    }

    /// Test out_range_impl BUFFER_TOO_SMALL for Code 3 VBR (line 1211)
    /// and Code 3 CBR (line 1220) with >2 frames.
    #[test]
    fn test_out_range_impl_code3_buffer_too_small() {
        // Code 3 CBR: 3 frames of 2 bytes each -> needs 2+2+2+2=8
        let pkt = [0x08u8, 0xAA, 0xBB];
        let mut rp = OpusRepacketizer::new();
        for _ in 0..3 {
            assert_eq!(rp.cat(&pkt, 3), OPUS_OK);
        }
        let mut out = [0u8; 7]; // too small for 8
        assert_eq!(rp.out(&mut out, 7), OPUS_BUFFER_TOO_SMALL);

        // Code 3 VBR: frames of 1, 2, 3 bytes -> needs 2+1+1+1+1+2+3=11
        let pkt1 = [0x08u8, 0x11];
        let pkt2 = [0x08u8, 0x22, 0x33];
        let pkt3 = [0x08u8, 0x44, 0x55, 0x66];
        let mut rp2 = OpusRepacketizer::new();
        assert_eq!(rp2.cat(&pkt1, 2), OPUS_OK);
        assert_eq!(rp2.cat(&pkt2, 3), OPUS_OK);
        assert_eq!(rp2.cat(&pkt3, 4), OPUS_OK);
        let mut out2 = [0u8; 5]; // too small
        assert_eq!(rp2.out(&mut out2, 5), OPUS_BUFFER_TOO_SMALL);
    }

    /// Test extension iterator calling next_ext() after curr_len becomes
    /// negative (line 534), and frame_max limiting via set_frame_max mid-iteration
    /// (line 545 / 577-579).
    #[test]
    fn test_extension_iterator_negative_curr_len_and_frame_max_mid_iter() {
        // Craft data that makes curr_len go negative: a truncated long extension
        let data = [0x41u8, 0xFF, 0xFF]; // id=32, L=1, lacing=255,255 -> wants 510 bytes
        let mut iter = OpusExtensionIterator::new(&data, 3, 1);
        let (ret, _) = iter.next_ext(); // Should fail with curr_len < 0
        assert_eq!(ret, OPUS_INVALID_PACKET);
        // Calling again should hit the curr_len < 0 early return (line 534)
        let (ret, _) = iter.next_ext();
        assert_eq!(ret, OPUS_INVALID_PACKET);

        // Test frame_max applied mid-iteration via frame separator (lines 577-579):
        // ext on frame 0, separator to frame 1, ext on frame 1
        // With frame_max=1, the separator should stop iteration
        let ext0 = OpusExtensionData {
            id: 5,
            frame: 0,
            data: &[0x11],
            len: 1,
        };
        let ext1 = OpusExtensionData {
            id: 6,
            frame: 1,
            data: &[0x22],
            len: 1,
        };
        let size = opus_packet_extensions_generate(None, 256, &[ext0, ext1], 2, false);
        let mut buf = vec![0u8; size as usize];
        opus_packet_extensions_generate(Some(&mut buf), size, &[ext0, ext1], 2, false);

        let mut iter2 = OpusExtensionIterator::new(&buf, size, 2);
        // Get first ext
        let (ret, ext) = iter2.next_ext();
        assert_eq!(ret, 1);
        assert_eq!(ext.frame, 0);
        // Now set frame_max to 1 before reading more — should stop
        iter2.set_frame_max(1);
        let (ret, _) = iter2.next_ext();
        assert_eq!(ret, 0);
    }

    /// Test extension iterator repeat mode with padding/separator skipping
    /// (line 466) and L=0 repeat ending advancing curr_frame (lines 517-521).
    #[test]
    fn test_extension_repeat_l0_long_extensions() {
        // Two frames, each with a long extension (id=40). Using repeat with L=0.
        let payload = [0x77u8; 10];
        let ext0 = OpusExtensionData {
            id: 40,
            frame: 0,
            data: &payload,
            len: 10,
        };
        let ext1 = OpusExtensionData {
            id: 40,
            frame: 1,
            data: &payload,
            len: 10,
        };
        let size = opus_packet_extensions_generate(None, 256, &[ext0, ext1], 2, false);
        assert!(size > 0);
        let mut buf = vec![0u8; size as usize];
        opus_packet_extensions_generate(Some(&mut buf), size, &[ext0, ext1], 2, false);

        let mut iter = OpusExtensionIterator::new(&buf, size, 2);
        let mut results = Vec::new();
        loop {
            let (ret, ext) = iter.next_ext();
            if ret <= 0 {
                break;
            }
            results.push((ext.id, ext.frame, ext.len));
        }
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], (40, 0, 10));
        assert_eq!(results[1], (40, 1, 10));
    }

    /// Test multistream pad/unpad with 3 streams (exercises the
    /// multistream_packet_unpad loop more thoroughly).
    #[test]
    fn test_multistream_three_streams_pad_unpad() {
        // 3 streams: first two self-delimited, last normal
        // Each stream: Code 0 packet with TOC=0x08
        let mut pkt = Vec::new();
        // Stream 1 (self-delimited): TOC + sd_size(2) + 2 bytes
        pkt.extend_from_slice(&[0x08u8, 2, 0x11, 0x22]);
        // Stream 2 (self-delimited): TOC + sd_size(1) + 1 byte
        pkt.extend_from_slice(&[0x08u8, 1, 0x33]);
        // Stream 3 (normal): TOC + 2 bytes
        pkt.extend_from_slice(&[0x08u8, 0x44, 0x55]);

        let original_len = pkt.len() as i32;
        let padded_len = original_len + 20;
        let mut buf = vec![0u8; padded_len as usize];
        buf[..pkt.len()].copy_from_slice(&pkt);

        let ret = opus_multistream_packet_pad(&mut buf, original_len, padded_len, 3);
        assert_eq!(ret, OPUS_OK);

        let ret = opus_multistream_packet_unpad(&mut buf, padded_len, 3);
        assert!(ret > 0);
        assert_eq!(ret, original_len);
    }

    /// Test parse_packet with self-delimited VBR Code 3 where the last frame
    /// size bytes plus size exceeds last_size (line 222 error).
    #[test]
    fn test_self_delimited_code3_vbr_last_size_overflow() {
        // Code 3 VBR self-delimited: TOC|0x03, count=2|0x80, size(f0)=5,
        // sd_size=200, but not enough data -> should fail
        let pkt = [0x08u8 | 0x03, 2 | 0x80, 5, 200, 0xAA];
        assert!(matches!(
            parse_packet(&pkt, pkt.len() as i32, true),
            Err(OPUS_INVALID_PACKET)
        ));
    }

    /// Test parse_packet Code 2 with self-delimited framing where the
    /// sd_size itself is invalid (line 206).
    #[test]
    fn test_self_delimited_code2_bad_sd_size() {
        // Code 2 self-delimited: TOC, size(f0)=1, then sd_size needs to be
        // parsed but we provide only 1 remaining byte that's >= 252 (needs 2)
        let pkt = [0x0Au8, 1, 0xAA, 253];
        assert!(matches!(
            parse_packet(&pkt, pkt.len() as i32, true),
            Err(OPUS_INVALID_PACKET)
        ));
    }

    /// Test extension generation with frame separator using multi-increment
    /// (diff > 1, lines 887-890) and verify dry-run vs actual write match.
    #[test]
    fn test_extension_generate_multi_frame_separator_dryrun() {
        // Extensions on frame 0 and frame 3 (skip 2 frames)
        let ext0 = OpusExtensionData {
            id: 5,
            frame: 0,
            data: &[0xAA],
            len: 1,
        };
        let ext3 = OpusExtensionData {
            id: 6,
            frame: 3,
            data: &[0xBB],
            len: 1,
        };
        let size = opus_packet_extensions_generate(None, 64, &[ext0, ext3], 4, false);
        assert!(size > 0);
        let mut buf = vec![0u8; size as usize];
        let ret = opus_packet_extensions_generate(Some(&mut buf), size, &[ext0, ext3], 4, false);
        assert_eq!(ret, size);
        // Parse back to verify correctness
        let mut parsed = [OpusExtensionData::EMPTY; 4];
        let mut nb = 4;
        let ret = opus_packet_extensions_parse(&buf, size, &mut parsed, &mut nb, 4);
        assert!(ret >= 0);
        assert_eq!(nb, 2);
        assert_eq!(parsed[0].frame, 0);
        assert_eq!(parsed[1].frame, 3);
    }

    /// Test write_extension_payload dry-run mode (line 726).
    #[test]
    fn test_write_extension_payload_dryrun() {
        let ext = OpusExtensionData {
            id: 40,
            frame: 0,
            data: &[1, 2, 3],
            len: 3,
        };
        let mut buf = [0xFFu8; 8];
        // dry_run=true: should advance position but not write
        let ret = write_extension_payload(&mut buf, true, 8, 0, &ext, false);
        assert_eq!(ret, 4); // 1 lacing byte + 3 data bytes
        assert_eq!(buf[0], 0xFF); // unchanged because dry_run

        // Short extension dry-run
        let short = OpusExtensionData {
            id: 5,
            frame: 0,
            data: &[0x42],
            len: 1,
        };
        let ret = write_extension_payload(&mut buf, true, 8, 0, &short, false);
        assert_eq!(ret, 1);
        assert_eq!(buf[0], 0xFF); // unchanged
    }

    /// Test long extension payload buffer too small (line 719).
    #[test]
    fn test_write_extension_payload_long_buffer_too_small() {
        let ext = OpusExtensionData {
            id: 40,
            frame: 0,
            data: &[1, 2, 3, 4, 5],
            len: 5,
        };
        let mut buf = [0u8; 4];
        // Not last: needs 1 lacing + 5 data = 6, only 4 available
        assert_eq!(
            write_extension_payload(&mut buf, false, 4, 0, &ext, false),
            OPUS_BUFFER_TOO_SMALL
        );
    }

    /// Test parse_packet Code 3 with padding flag, where the padding
    /// exhausts remaining bytes (line 166: remaining < 0 after padding).
    #[test]
    fn test_code3_padding_exhausts_remaining() {
        // Code 3, count=1, padding flag set, padding value=255 (needs more)
        // but only 1 byte remains after the 255 -> fails at line 152
        let pkt = [0x08u8 | 0x03, 1 | 0x40, 255];
        assert!(matches!(
            parse_packet(&pkt, pkt.len() as i32, false),
            Err(OPUS_INVALID_PACKET)
        ));

        // Padding consumes more than remaining audio data (line 166)
        let pkt2 = [0x08u8 | 0x03, 1 | 0x40, 200, 0xAA];
        assert!(matches!(
            parse_packet(&pkt2, pkt2.len() as i32, false),
            Err(OPUS_INVALID_PACKET)
        ));
    }

    /// Test opus_packet_pad_impl with extensions that cause ext_len overflow
    /// relative to available space (line 1259).
    #[test]
    fn test_pad_impl_ext_len_overflow() {
        let big_payload = [0x11u8; 50];
        let ext = OpusExtensionData {
            id: 40,
            frame: 0,
            data: &big_payload,
            len: 50,
        };
        let mut buf = [0u8; 60];
        buf[0] = 0x08;
        buf[1] = 0xAA;
        // Try to pad to 10 bytes with an extension that needs ~52 bytes
        let ret = opus_packet_pad_impl(&mut buf, 2, 10, true, &[ext]);
        assert!(ret < 0); // Should fail (buffer too small for extensions)
    }

    // =======================================================================
    // Stage 2 branch coverage additions
    // =======================================================================
    mod branch_coverage_stage2 {
        use super::*;
        use crate::opus::encoder::{OPUS_APPLICATION_AUDIO, OpusEncoder};

        fn patterned_pcm_i16(frame_size: usize, channels: usize, seed: i32) -> Vec<i16> {
            (0..frame_size * channels)
                .map(|i| {
                    let base = ((i as i32 * 7919 + seed * 911) % 28000) - 14000;
                    if channels == 2 && i % 2 == 1 {
                        (base / 2) as i16
                    } else {
                        base as i16
                    }
                })
                .collect()
        }

        /// Encode one 20ms mono frame at 16kHz. Useful for cat()/pad() inputs.
        fn mk_packet(seed: i32) -> Vec<u8> {
            let mut enc = OpusEncoder::new(16000, 1, OPUS_APPLICATION_AUDIO).unwrap();
            enc.set_bitrate(16000);
            enc.set_complexity(2);
            let pcm = patterned_pcm_i16(320, 1, seed);
            let mut buf = vec![0u8; 400];
            let n = enc.encode(&pcm, 320, &mut buf, 400).unwrap();
            buf.truncate(n as usize);
            buf
        }

        #[test]
        fn init_resets_frame_count_after_cat() {
            let pkt = mk_packet(1);
            let mut rp = OpusRepacketizer::new();
            assert_eq!(rp.cat(&pkt, pkt.len() as i32), OPUS_OK);
            assert!(rp.get_nb_frames() >= 1);
            rp.init();
            assert_eq!(rp.get_nb_frames(), 0);
        }

        #[test]
        fn cat_accumulates_multiple_packets_and_out_round_trips() {
            let a = mk_packet(2);
            let b = mk_packet(3);
            let c = mk_packet(4);
            let mut rp = OpusRepacketizer::new();
            assert_eq!(rp.cat(&a, a.len() as i32), OPUS_OK);
            assert_eq!(rp.cat(&b, b.len() as i32), OPUS_OK);
            assert_eq!(rp.cat(&c, c.len() as i32), OPUS_OK);
            assert_eq!(rp.get_nb_frames(), 3);

            let mut out = vec![0u8; 2048];
            let n = rp.out(&mut out, 2048);
            assert!(n > 0);
            // out_range with arbitrary sub-range
            let n2 = rp.out_range(1, 3, &mut out, 2048);
            assert!(n2 > 0);
        }

        #[test]
        fn out_range_bad_bounds_returns_bad_arg() {
            let a = mk_packet(5);
            let mut rp = OpusRepacketizer::new();
            rp.cat(&a, a.len() as i32);
            let mut out = vec![0u8; 256];
            // begin >= end
            assert_eq!(rp.out_range(1, 1, &mut out, 256), OPUS_BAD_ARG);
            // end > nb_frames
            assert_eq!(rp.out_range(0, 99, &mut out, 256), OPUS_BAD_ARG);
        }

        #[test]
        fn cat_rejects_invalid_length() {
            let mut rp = OpusRepacketizer::new();
            assert_eq!(rp.cat(&[], 0), OPUS_INVALID_PACKET);
        }

        #[test]
        fn cat_rejects_mismatched_toc() {
            // First pkt SILK-WB, second pkt CELT-only — TOC top bits differ.
            let mut enc1 = OpusEncoder::new(16000, 1, OPUS_APPLICATION_AUDIO).unwrap();
            enc1.set_bitrate(16000);
            let pcm = patterned_pcm_i16(320, 1, 7);
            let mut p1 = vec![0u8; 400];
            let n1 = enc1.encode(&pcm, 320, &mut p1, 400).unwrap();
            p1.truncate(n1 as usize);

            let mut enc2 = OpusEncoder::new(
                48000,
                1,
                crate::opus::encoder::OPUS_APPLICATION_RESTRICTED_LOWDELAY,
            )
            .unwrap();
            enc2.set_bitrate(64000);
            let pcm2 = patterned_pcm_i16(960, 1, 8);
            let mut p2 = vec![0u8; 1500];
            let n2 = enc2.encode(&pcm2, 960, &mut p2, 1500).unwrap();
            p2.truncate(n2 as usize);

            let mut rp = OpusRepacketizer::new();
            assert_eq!(rp.cat(&p1, p1.len() as i32), OPUS_OK);
            let r = rp.cat(&p2, p2.len() as i32);
            assert!(r < 0, "expected mismatched-TOC rejection, got {r}");
        }

        #[test]
        fn opus_packet_pad_noop_when_equal_lengths() {
            let pkt = mk_packet(10);
            let mut buf = vec![0u8; 512];
            let original_len = pkt.len() as i32;
            buf[..pkt.len()].copy_from_slice(&pkt);
            // len == new_len → OPUS_OK immediately
            assert_eq!(opus_packet_pad(&mut buf, original_len, original_len), OPUS_OK);
        }

        #[test]
        fn opus_packet_pad_grows_to_new_len() {
            let pkt = mk_packet(11);
            let original_len = pkt.len() as i32;
            let target = original_len + 40;
            let mut buf = vec![0u8; (target + 16) as usize];
            buf[..pkt.len()].copy_from_slice(&pkt);
            let r = opus_packet_pad(&mut buf, original_len, target);
            assert_eq!(r, OPUS_OK);
            // And unpad returns a length <= target
            let unp = opus_packet_unpad(&mut buf, target);
            assert!(unp > 0 && unp <= target, "unpad returned {unp}");
        }

        #[test]
        fn opus_packet_pad_rejects_shrink_and_zero_len() {
            let pkt = mk_packet(12);
            let mut buf = vec![0u8; 512];
            buf[..pkt.len()].copy_from_slice(&pkt);
            // new_len < len → OPUS_BAD_ARG
            let r = opus_packet_pad(&mut buf, pkt.len() as i32, (pkt.len() as i32) - 1);
            assert_eq!(r, OPUS_BAD_ARG);
            // len < 1 → OPUS_BAD_ARG
            assert_eq!(opus_packet_pad(&mut buf, 0, 10), OPUS_BAD_ARG);
        }

        #[test]
        fn opus_packet_pad_across_255_byte_sentinel() {
            let pkt = mk_packet(13);
            let original_len = pkt.len() as i32;
            // Pad to >255 extra bytes — hits the nb_255s loop in out_range_impl.
            let target = original_len + 260;
            let mut buf = vec![0u8; (target + 16) as usize];
            buf[..pkt.len()].copy_from_slice(&pkt);
            let r = opus_packet_pad(&mut buf, original_len, target);
            assert_eq!(r, OPUS_OK, "pad across 255-byte sentinel failed: {r}");
            let unp = opus_packet_unpad(&mut buf, target);
            assert!(unp > 0);
        }

        #[test]
        fn opus_packet_unpad_rejects_empty() {
            let mut buf = [0u8; 4];
            assert_eq!(opus_packet_unpad(&mut buf, 0), OPUS_BAD_ARG);
        }

        #[test]
        fn multistream_pad_unpad_two_stream_packet() {
            // Build a multistream packet by concatenating a self-delimited
            // sub-packet and a standard sub-packet.
            let p1 = mk_packet(20);
            let p2 = mk_packet(21);
            // Put p1 into self-delimited form via out_range_impl
            let mut rp = OpusRepacketizer::new();
            rp.cat(&p1, p1.len() as i32);
            let sd1_cap = (p1.len() + 8) as i32;
            let mut sd1 = vec![0u8; sd1_cap as usize];
            let n_sd = rp.out_range_impl(0, 1, &mut sd1, sd1_cap, true, false, &[]);
            assert!(n_sd > 0);

            let mut combined = Vec::new();
            combined.extend_from_slice(&sd1[..n_sd as usize]);
            combined.extend_from_slice(&p2);

            // Grow buffer and pad
            let new_len = combined.len() as i32 + 20;
            let mut buf = vec![0u8; (new_len + 16) as usize];
            buf[..combined.len()].copy_from_slice(&combined);
            let r = opus_multistream_packet_pad(
                &mut buf,
                combined.len() as i32,
                new_len,
                2,
            );
            assert_eq!(r, OPUS_OK);

            let unp = opus_multistream_packet_unpad(&mut buf, new_len, 2);
            assert!(unp > 0 && unp <= new_len, "got {unp}");
        }

        #[test]
        fn multistream_pad_equal_len_is_noop() {
            let pkt = mk_packet(22);
            let mut buf = vec![0u8; 256];
            buf[..pkt.len()].copy_from_slice(&pkt);
            assert_eq!(
                opus_multistream_packet_pad(&mut buf, pkt.len() as i32, pkt.len() as i32, 1),
                OPUS_OK
            );
        }

        #[test]
        fn multistream_pad_rejects_shrink_and_zero() {
            let mut buf = [0u8; 16];
            assert_eq!(opus_multistream_packet_pad(&mut buf, 0, 10, 1), OPUS_BAD_ARG);
            assert_eq!(opus_multistream_packet_pad(&mut buf, 10, 5, 1), OPUS_BAD_ARG);
        }

        #[test]
        fn multistream_unpad_rejects_empty() {
            let mut buf = [0u8; 4];
            assert_eq!(opus_multistream_packet_unpad(&mut buf, 0, 1), OPUS_BAD_ARG);
        }

        #[test]
        fn self_delimited_out_range_writes_size_prefix() {
            let pkt = mk_packet(30);
            let mut rp = OpusRepacketizer::new();
            rp.cat(&pkt, pkt.len() as i32);
            let mut out = vec![0u8; 1024];
            let n = rp.out_range_impl(0, 1, &mut out, 1024, true, false, &[]);
            assert!(n > 0);
            // self-delimited output must be strictly longer than a plain out()
            let mut out2 = vec![0u8; 1024];
            let n2 = rp.out_range_impl(0, 1, &mut out2, 1024, false, false, &[]);
            assert!(n2 > 0);
            assert!(n > n2);
        }

        #[test]
        fn out_range_small_buffer_returns_buffer_too_small() {
            let pkt = mk_packet(31);
            let mut rp = OpusRepacketizer::new();
            rp.cat(&pkt, pkt.len() as i32);
            let mut small = vec![0u8; 1];
            let r = rp.out(&mut small, 1);
            assert!(r < 0);
        }

        #[test]
        fn parse_packet_all_codes_ok_and_malformed() {
            // Code 0
            let p = [0x08u8, 0xAA, 0xBB];
            let r = parse_packet(&p, p.len() as i32, false).unwrap();
            assert_eq!(r.count, 1);
            // Code 1
            let p = [0x09u8, 0xCC, 0xDD];
            let r = parse_packet(&p, p.len() as i32, false).unwrap();
            assert_eq!(r.count, 2);
            // Code 2
            let p = [0x0Au8, 0x01, 0xEE, 0xFF, 0x11];
            let r = parse_packet(&p, p.len() as i32, false).unwrap();
            assert_eq!(r.count, 2);
            // Code 3 CBR
            let p = [0x0Bu8, 0x02, 0xAA, 0xBB, 0xCC, 0xDD];
            let r = parse_packet(&p, p.len() as i32, false).unwrap();
            assert_eq!(r.count, 2);
            // Code 3 bad count (0)
            assert!(parse_packet(&[0x0Bu8, 0x00], 2, false).is_err());
            // Code 3 with only 1 byte (no count byte)
            assert!(parse_packet(&[0x0Bu8], 1, false).is_err());
            // Len == 0 → invalid
            assert!(parse_packet(&[], 0, false).is_err());
            // len < 0 → bad arg
            assert!(matches!(parse_packet(&[0x08u8, 0xAA], -1, false), Err(OPUS_BAD_ARG)));
            // Code 1 with odd remaining (self_delimited=false)
            assert!(parse_packet(&[0x09u8, 0xAA], 2, false).is_err());
        }

        #[test]
        fn parse_packet_self_delimited_paths() {
            // Self-delimited code 0: size byte then payload
            let p = [0x08u8, 0x02, 0xAA, 0xBB];
            let r = parse_packet(&p, p.len() as i32, true).unwrap();
            assert_eq!(r.count, 1);
            // Self-delimited code 1: two CBR, size at end
            let p = [0x09u8, 0x02, 0xAA, 0xBB, 0xCC, 0xDD];
            let r = parse_packet(&p, p.len() as i32, true).unwrap();
            assert_eq!(r.count, 2);
        }

        #[test]
        fn out_range_with_extensions_triggers_code3_upgrade() {
            // Single-frame cat, then out_range_impl with an extension —
            // forces the "upgrade to code 3" branch.
            let pkt = mk_packet(40);
            let mut rp = OpusRepacketizer::new();
            rp.cat(&pkt, pkt.len() as i32);
            let payload = [0x55u8];
            let ext = OpusExtensionData {
                id: 33,
                frame: 0,
                data: &payload,
                len: 1,
            };
            let mut out = vec![0u8; 2048];
            let n = rp.out_range_impl(0, 1, &mut out, 2048, false, false, &[ext]);
            assert!(n > 0, "out_range with extension returned {n}");
        }

        #[test]
        fn out_range_pad_to_max_fills_remaining_bytes() {
            let pkt = mk_packet(41);
            let mut rp = OpusRepacketizer::new();
            rp.cat(&pkt, pkt.len() as i32);
            let target = pkt.len() as i32 + 80;
            let mut out = vec![0u8; target as usize];
            let n = rp.out_range_impl(0, 1, &mut out, target, false, true, &[]);
            assert_eq!(n, target);
        }

        // --- Targeted parse_packet error-path coverage ---

        #[test]
        fn parse_packet_code2_bad_first_size() {
            // Code 2, size byte claims 252 (needs 2 bytes) but only 1 remaining.
            let p = [0x0Au8, 252u8];
            assert!(parse_packet(&p, p.len() as i32, false).is_err());
        }

        #[test]
        fn parse_packet_code2_first_size_overflow() {
            // Code 2: first frame size > remaining
            let p = [0x0Au8, 10u8, 0xAA];
            assert!(parse_packet(&p, p.len() as i32, false).is_err());
        }

        #[test]
        fn parse_packet_code3_vbr_bad_size_and_overflow() {
            // VBR code 3, count=2 → need 1 VBR size for frame 0.
            // Byte layout: TOC=0x0B, ch=0x82 (VBR, count=2), then bad size (252 alone),
            // then frames.
            let p = [0x0Bu8, 0x82u8, 252u8];
            assert!(parse_packet(&p, p.len() as i32, false).is_err());
            // size too large
            let p = [0x0Bu8, 0x82u8, 250u8, 0xAA];
            assert!(parse_packet(&p, p.len() as i32, false).is_err());
        }

        #[test]
        fn parse_packet_code3_vbr_last_size_underflow() {
            // VBR code 3, count=3 → 2 VBR sizes; if sum of sizes > remaining,
            // last_size goes negative.
            let p = [0x0Bu8, 0x83u8, 5u8, 5u8, 0xAA];
            assert!(parse_packet(&p, p.len() as i32, false).is_err());
        }

        #[test]
        fn parse_packet_self_delimited_code3_cbr_overflow() {
            // Code 3 CBR, count=3, self-delim size*count > remaining
            let p = [0x0Bu8, 0x03u8, 100u8, 0, 0, 0];
            assert!(parse_packet(&p, p.len() as i32, true).is_err());
        }

        #[test]
        fn parse_packet_self_delimited_vbr_bytes_plus_sz_overflow() {
            // Self-delim code 3 VBR, count=2, first VBR size = 1, self-delim size huge.
            // Sets bytes + sz > last_size → branch at line 221.
            let p = [0x0Bu8, 0x82u8, 1u8, 251u8, 0xAA];
            assert!(parse_packet(&p, p.len() as i32, true).is_err());
        }

        #[test]
        fn parse_packet_self_delimited_bad_size() {
            // Self-delim code 0 but size byte claims 252 with 1 byte available.
            let p = [0x08u8, 252u8];
            assert!(parse_packet(&p, p.len() as i32, true).is_err());
        }

        #[test]
        fn parse_packet_code3_padding_runs_out() {
            // Code 3 with padding flag but no padding bytes available.
            let p = [0x0Bu8, 0x41u8];
            assert!(parse_packet(&p, p.len() as i32, false).is_err());
        }

        #[test]
        fn parse_packet_code3_framesize_times_count_exceeds_120ms() {
            // 2.5ms CELT × 49 frames > 5760 samples at 48kHz
            // TOC 0x87 = CELT-only 2.5ms + code 3
            let p = [0x87u8, 49u8];
            assert!(parse_packet(&p, p.len() as i32, false).is_err());
        }

        #[test]
        fn parse_packet_last_size_over_1275_rejected() {
            // Code 0 with remaining > 1275 → line 226 branch.
            let mut big = vec![0x08u8];
            big.extend(std::iter::repeat(0u8).take(1280));
            assert!(parse_packet(&big, big.len() as i32, false).is_err());
        }

        #[test]
        fn extension_iterator_reset_and_set_frame_max() {
            let data = [0x04u8]; // repeat-indicator with L=0, no content
            let mut iter = OpusExtensionIterator::new(&data, data.len() as i32, 2);
            iter.set_frame_max(1);
            iter.reset();
            let _ = iter.next_ext();
        }

        #[test]
        fn skip_extension_empty_and_single_byte() {
            let mut pos = 0usize;
            let mut header_size = 1;
            // Empty data with remaining=0 → returns 0 and sets header_size=0
            assert_eq!(skip_extension(&[], &mut pos, 0, &mut header_size), 0);
            assert_eq!(header_size, 0);
        }

        #[test]
        fn extensions_parse_and_count_empty() {
            let mut exts = [OpusExtensionData::EMPTY; 4];
            let mut n = 4;
            assert_eq!(
                opus_packet_extensions_parse(&[], 0, &mut exts, &mut n, 1),
                0
            );
            assert_eq!(n, 0);
            assert_eq!(opus_packet_extensions_count(&[], 0, 1), 0);
        }

        #[test]
        fn extensions_generate_bad_args() {
            // nb_frames > MAX_FRAMES → OPUS_BAD_ARG
            let ret = opus_packet_extensions_generate(None, 256, &[], (MAX_FRAMES + 1) as i32, false);
            assert_eq!(ret, OPUS_BAD_ARG);

            // extension with frame out of range
            let ext = OpusExtensionData {
                id: 33,
                frame: 99,
                data: &[],
                len: 0,
            };
            let ret = opus_packet_extensions_generate(None, 256, &[ext], 1, false);
            assert_eq!(ret, OPUS_BAD_ARG);

            // extension with invalid id
            let ext = OpusExtensionData {
                id: 1,
                frame: 0,
                data: &[],
                len: 0,
            };
            let ret = opus_packet_extensions_generate(None, 256, &[ext], 1, false);
            assert_eq!(ret, OPUS_BAD_ARG);
        }

        #[test]
        fn extensions_generate_short_with_invalid_len() {
            // Short extension (id < 32) with len=2 is invalid.
            let payload = [0x11u8];
            let ext = OpusExtensionData {
                id: 5,
                frame: 0,
                data: &payload,
                len: 2,
            };
            let ret = opus_packet_extensions_generate(None, 256, &[ext], 1, false);
            assert!(ret < 0);
        }

        #[test]
        fn out_range_with_multiple_extensions_across_frames() {
            // Cat 2 packets so nb_frames = 2, then request out_range with
            // extensions that apply to both frames — exercises the repeat
            // machinery on write side.
            let a = mk_packet(60);
            let b = mk_packet(61);
            let mut rp = OpusRepacketizer::new();
            rp.cat(&a, a.len() as i32);
            rp.cat(&b, b.len() as i32);

            let payload = [0xABu8];
            let ext0 = OpusExtensionData { id: 33, frame: 0, data: &payload, len: 1 };
            let ext1 = OpusExtensionData { id: 33, frame: 1, data: &payload, len: 1 };

            let mut out = vec![0u8; 2048];
            let n = rp.out_range_impl(0, 2, &mut out, 2048, false, false, &[ext0, ext1]);
            assert!(n > 0, "out_range_impl with 2 exts returned {n}");
        }

        #[test]
        fn pad_impl_with_caller_extension() {
            // pad_impl with an extension — exercises the extension length path.
            let pkt = mk_packet(65);
            let mut buf = vec![0u8; 512];
            buf[..pkt.len()].copy_from_slice(&pkt);
            let payload = [0x22u8];
            let ext = OpusExtensionData { id: 33, frame: 0, data: &payload, len: 1 };
            let target = pkt.len() as i32 + 30;
            let ret = opus_packet_pad_impl(&mut buf, pkt.len() as i32, target, true, &[ext]);
            assert!(ret > 0, "pad_impl with caller extension returned {ret}");
        }

        // --- More extension generation / iterator coverage ---

        #[test]
        fn extension_generate_repeat_mechanism_multi_frame() {
            // Same extension applied to all frames → exercises the repeat branch.
            let payload = [0xABu8];
            let ext0 = OpusExtensionData { id: 33, frame: 0, data: &payload, len: 1 };
            let ext1 = OpusExtensionData { id: 33, frame: 1, data: &payload, len: 1 };
            let ext2 = OpusExtensionData { id: 33, frame: 2, data: &payload, len: 1 };
            let ext3 = OpusExtensionData { id: 33, frame: 3, data: &payload, len: 1 };
            let exts = [ext0, ext1, ext2, ext3];

            let size = opus_packet_extensions_generate(None, 256, &exts, 4, false);
            assert!(size > 0);
            let mut buf = vec![0u8; size as usize];
            let s = opus_packet_extensions_generate(Some(&mut buf), size, &exts, 4, false);
            assert_eq!(s, size);

            // Parse back
            let mut parsed = [OpusExtensionData::EMPTY; 8];
            let mut n = 8;
            assert_eq!(
                opus_packet_extensions_parse(&buf, size, &mut parsed, &mut n, 4),
                0
            );
            assert_eq!(n, 4);

            // Count matches
            assert_eq!(opus_packet_extensions_count(&buf, size, 4), 4);
        }

        #[test]
        fn extension_generate_mixed_frames_with_separators() {
            // Extensions on frames 0 and 2 — forces separator emission (line 886).
            let p = [0x11u8];
            let exts = [
                OpusExtensionData { id: 5, frame: 0, data: &p, len: 1 },
                OpusExtensionData { id: 5, frame: 2, data: &p, len: 1 },
            ];
            let size = opus_packet_extensions_generate(None, 256, &exts, 3, false);
            assert!(size > 0);
            let mut buf = vec![0u8; size as usize];
            let s = opus_packet_extensions_generate(Some(&mut buf), size, &exts, 3, false);
            assert_eq!(s, size);

            // Parse back
            let mut parsed = [OpusExtensionData::EMPTY; 4];
            let mut n = 4;
            assert_eq!(
                opus_packet_extensions_parse(&buf, size, &mut parsed, &mut n, 3),
                0
            );
        }

        #[test]
        fn extension_generate_adjacent_separator_diff_1() {
            // diff == 1 branch (line 886)
            let p = [0x11u8];
            let exts = [
                OpusExtensionData { id: 5, frame: 0, data: &p, len: 1 },
                OpusExtensionData { id: 5, frame: 1, data: &p, len: 1 },
            ];
            let size = opus_packet_extensions_generate(None, 256, &exts, 2, false);
            let mut buf = vec![0u8; size as usize];
            let _ = opus_packet_extensions_generate(Some(&mut buf), size, &exts, 2, false);
        }

        #[test]
        fn extension_iterator_find_and_next_ext_empty() {
            let data: [u8; 0] = [];
            let mut iter = OpusExtensionIterator::new(&data, 0, 1);
            let (ret, _) = iter.next_ext();
            assert_eq!(ret, 0);
            let (ret2, _) = iter.find(33);
            assert_eq!(ret2, 0);
        }

        #[test]
        fn extension_iterator_finds_matching_id() {
            // Build a buffer with one extension id=33, payload 0xAA
            let buf = [66u8, 1u8, 0xAAu8]; // id=33 L=0, length=1, payload
            let mut iter = OpusExtensionIterator::new(&buf, buf.len() as i32, 1);
            let (ret, ext) = iter.find(33);
            assert_eq!(ret, 1);
            assert_eq!(ext.id, 33);
        }

        #[test]
        fn unpad_on_padded_real_packet() {
            // Generate encoded pkt, pad it, then unpad — round-trips the
            // unpad path fully.
            let pkt = mk_packet(70);
            let new_len = pkt.len() as i32 + 60;
            let mut buf = vec![0u8; (new_len + 16) as usize];
            buf[..pkt.len()].copy_from_slice(&pkt);
            let _ = opus_packet_pad(&mut buf, pkt.len() as i32, new_len);
            let r = opus_packet_unpad(&mut buf, new_len);
            assert!(r > 0);
            assert!(r <= new_len);
        }

        #[test]
        fn multistream_unpad_multiple_streams_round_trip() {
            // 3-stream packet: sd(p1) + sd(p2) + p3
            let p1 = mk_packet(80);
            let p2 = mk_packet(81);
            let p3 = mk_packet(82);

            let mut combined = Vec::new();
            for (i, p) in [&p1, &p2, &p3].into_iter().enumerate() {
                let mut rp = OpusRepacketizer::new();
                rp.cat(p, p.len() as i32);
                let cap = (p.len() + 8) as i32;
                let mut sd = vec![0u8; cap as usize];
                let is_last = i == 2;
                let k = rp.out_range_impl(0, 1, &mut sd, cap, !is_last, false, &[]);
                assert!(k > 0);
                combined.extend_from_slice(&sd[..k as usize]);
            }

            let new_len = combined.len() as i32 + 30;
            let mut buf = vec![0u8; (new_len + 16) as usize];
            buf[..combined.len()].copy_from_slice(&combined);
            let r = opus_multistream_packet_pad(&mut buf, combined.len() as i32, new_len, 3);
            // May fail depending on padding heuristic, but branch is taken.
            let _ = r;

            // Unpad
            let r = opus_multistream_packet_unpad(&mut buf, combined.len() as i32, 3);
            let _ = r;
        }
    }
}
