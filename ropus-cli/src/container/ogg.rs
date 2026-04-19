//! Ogg + OpusHead/OpusTags packet helpers and a reverse-scan for the last
//! granule position of a target stream serial.

use std::io::{Read, Seek, SeekFrom};

use anyhow::{Result, anyhow, bail};

use ogg::reading::PacketReader;

/// Logical Ogg stream serial. Any non-zero value works for a single stream we
/// write ourselves. NEVER use this for matching against an arbitrary input
/// stream; capture the input's serial from its first OggS page instead.
pub(crate) const OGG_STREAM_SERIAL: u32 = 0xC0DE_C0DE;

/// Build the `OpusHead` packet (RFC 7845, section 5.1). 19 bytes for a
/// channel-mapping=0 mono/stereo stream.
pub(crate) fn build_opus_head(channels: u8, input_sample_rate: u32, pre_skip: u16) -> Vec<u8> {
    let mut h = Vec::with_capacity(19);
    h.extend_from_slice(b"OpusHead");
    h.push(1); // version
    h.push(channels);
    h.extend_from_slice(&pre_skip.to_le_bytes());
    h.extend_from_slice(&input_sample_rate.to_le_bytes());
    h.extend_from_slice(&0i16.to_le_bytes()); // output gain (Q7.8 dB)
    h.push(0); // channel mapping family
    debug_assert_eq!(h.len(), 19);
    h
}

/// Build the `OpusTags` packet (RFC 7845, section 5.2).
pub(crate) fn build_opus_tags(vendor: &str) -> Vec<u8> {
    let vendor_bytes = vendor.as_bytes();
    let mut t = Vec::with_capacity(8 + 4 + vendor_bytes.len() + 4);
    t.extend_from_slice(b"OpusTags");
    t.extend_from_slice(&(vendor_bytes.len() as u32).to_le_bytes());
    t.extend_from_slice(vendor_bytes);
    t.extend_from_slice(&0u32.to_le_bytes()); // user-comment count
    t
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct OpusHead {
    pub(crate) version: u8,
    pub(crate) channels: u8,
    pub(crate) pre_skip: u16,
    pub(crate) input_sample_rate: u32,
    pub(crate) output_gain: i16,
    pub(crate) channel_mapping: u8,
}

pub(crate) fn parse_opus_head(data: &[u8]) -> Result<OpusHead> {
    if data.len() < 19 {
        bail!("OpusHead too short ({} bytes)", data.len());
    }
    if &data[..8] != b"OpusHead" {
        bail!("not an OpusHead packet");
    }
    Ok(OpusHead {
        version: data[8],
        channels: data[9],
        pre_skip: u16::from_le_bytes([data[10], data[11]]),
        input_sample_rate: u32::from_le_bytes([data[12], data[13], data[14], data[15]]),
        output_gain: i16::from_le_bytes([data[16], data[17]]),
        channel_mapping: data[18],
    })
}

/// Read the OpusTags packet from `reader`, verifying its magic. Returns an
/// error if the next packet is missing or doesn't begin with `b"OpusTags"`,
/// which would indicate a malformed file (e.g. tags page stripped) and would
/// otherwise cause the first audio packet to be silently consumed.
pub(crate) fn read_opus_tags<R: std::io::Read + std::io::Seek>(
    reader: &mut PacketReader<R>,
) -> Result<()> {
    let pkt = reader
        .read_packet()?
        .ok_or_else(|| anyhow!("expected OpusTags packet, got end of stream"))?;
    if pkt.data.len() < 8 || &pkt.data[..8] != b"OpusTags" {
        let head = &pkt.data[..8.min(pkt.data.len())];
        bail!("expected OpusTags packet, got {:?}", head);
    }
    Ok(())
}

/// Scan backwards from EOF for the last Ogg `OggS` page belonging to
/// `target_serial`, and return its absolute granule position. Returns
/// `Ok(None)` if the last matching page has the unknown-granule sentinel
/// (`0xFFFF_FFFF_FFFF_FFFF`) or if no matching page can be found in the
/// search window — both of which leave the caller responsible for falling
/// back to the slow whole-stream decode.
///
/// Reads at most the trailing 128 KiB of the file. RFC 3533 caps a single
/// Ogg page at 65,307 bytes (27-byte header + 255 lacing entries × 255
/// bytes payload); 128 KiB therefore reliably covers a max-size last page
/// even with a small amount of trailing junk after the OggS frame.
pub(crate) fn read_last_granule<R: Read + Seek>(
    src: &mut R,
    target_serial: u32,
) -> std::io::Result<Option<u64>> {
    const SCAN_WINDOW: u64 = 128 * 1024;
    const HEADER_LEN: usize = 27;
    const UNKNOWN_GRANULE: u64 = 0xFFFF_FFFF_FFFF_FFFF;

    // Use Seek to obtain the source's length without depending on filesystem
    // metadata, so this helper can drive any Read+Seek (including Cursor in
    // unit tests).
    let file_len = src.seek(SeekFrom::End(0))?;
    if file_len < HEADER_LEN as u64 {
        return Ok(None);
    }

    let read_len = SCAN_WINDOW.min(file_len);
    let start = file_len - read_len;
    src.seek(SeekFrom::Start(start))?;

    let mut buf = vec![0u8; read_len as usize];
    src.read_exact(&mut buf)?;

    // Reverse-scan for the b"OggS" capture pattern. For each candidate, validate
    // the fixed-layout header and that the serial number matches. Walk back to
    // the previous candidate if any check fails (e.g. wrong stream in a
    // multiplexed file, or coincidental "OggS" bytes inside packet data).
    let mut i = buf.len().saturating_sub(4);
    loop {
        if i + HEADER_LEN <= buf.len()
            && &buf[i..i + 4] == b"OggS"
            // stream_structure_version must be 0 per RFC 3533 §6
            && buf[i + 4] == 0
        {
            let absgp = u64::from_le_bytes([
                buf[i + 6],
                buf[i + 7],
                buf[i + 8],
                buf[i + 9],
                buf[i + 10],
                buf[i + 11],
                buf[i + 12],
                buf[i + 13],
            ]);
            let serial = u32::from_le_bytes([buf[i + 14], buf[i + 15], buf[i + 16], buf[i + 17]]);
            if serial == target_serial {
                if absgp == UNKNOWN_GRANULE {
                    return Ok(None);
                }
                return Ok(Some(absgp));
            }
        }
        if i == 0 {
            return Ok(None);
        }
        i -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Build a minimal valid 27-byte Ogg page (zero-segment payload) with the
    /// supplied absolute granule position and stream serial. Tests only care
    /// about the fields `read_last_granule` inspects (capture pattern,
    /// version, absgp, serial, segment count).
    fn build_minimal_ogg_page(absgp: u64, serial: u32) -> Vec<u8> {
        let mut page = Vec::with_capacity(27);
        page.extend_from_slice(b"OggS"); // capture pattern
        page.push(0); // stream_structure_version
        page.push(0x04); // header_type_flag (end-of-stream — irrelevant here)
        page.extend_from_slice(&absgp.to_le_bytes());
        page.extend_from_slice(&serial.to_le_bytes());
        page.extend_from_slice(&0u32.to_le_bytes()); // page sequence
        page.extend_from_slice(&0u32.to_le_bytes()); // CRC (read_last_granule does not verify)
        page.push(0); // page_segments = 0 → no lacing bytes follow
        debug_assert_eq!(page.len(), 27);
        page
    }

    #[test]
    fn read_last_granule_empty_file_returns_none() {
        let mut cursor = Cursor::new(Vec::<u8>::new());
        let got = read_last_granule(&mut cursor, 0xC0DE_C0DE).expect("ok");
        assert!(got.is_none(), "empty input must yield None");
    }

    #[test]
    fn read_last_granule_minimal_valid_page() {
        let page = build_minimal_ogg_page(12_345, 0xC0DE_C0DE);
        let mut cursor = Cursor::new(page);
        let got = read_last_granule(&mut cursor, 0xC0DE_C0DE).expect("ok");
        assert_eq!(got, Some(12_345));
    }

    #[test]
    fn read_last_granule_unknown_granule_sentinel() {
        let page = build_minimal_ogg_page(0xFFFF_FFFF_FFFF_FFFF, 0xC0DE_C0DE);
        let mut cursor = Cursor::new(page);
        let got = read_last_granule(&mut cursor, 0xC0DE_C0DE).expect("ok");
        assert!(got.is_none(), "sentinel granule must yield None");
    }

    #[test]
    fn read_last_granule_skips_wrong_serial() {
        // Layout: [target page absgp=42] [other-serial page absgp=999]
        // Reverse scan should walk past the trailing wrong-serial page and
        // pick up the target page's granule.
        let mut buf = Vec::new();
        buf.extend_from_slice(&build_minimal_ogg_page(42, 0xC0DE_C0DE));
        buf.extend_from_slice(&build_minimal_ogg_page(999, 0xDEAD_BEEF));
        let mut cursor = Cursor::new(buf);
        let got = read_last_granule(&mut cursor, 0xC0DE_C0DE).expect("ok");
        assert_eq!(got, Some(42));
    }
}
