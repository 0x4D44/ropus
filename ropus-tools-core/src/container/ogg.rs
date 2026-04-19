//! Ogg + OpusHead/OpusTags packet helpers and a reverse-scan for the last
//! granule position of a target stream serial.

use std::io::{Read, Seek, SeekFrom};

use anyhow::{Result, anyhow, bail};

/// Logical Ogg stream serial. Any non-zero value works for a single stream we
/// write ourselves. NEVER use this for matching against an arbitrary input
/// stream; capture the input's serial from its first OggS page instead.
pub const OGG_STREAM_SERIAL: u32 = 0xC0DE_C0DE;

/// Build the `OpusHead` packet (RFC 7845, section 5.1). 19 bytes for a
/// channel-mapping=0 mono/stereo stream.
pub fn build_opus_head(channels: u8, input_sample_rate: u32, pre_skip: u16) -> Vec<u8> {
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

/// Parsed or about-to-be-written `OpusTags` comment header (RFC 7845 §5.2).
///
/// Each entry in `comments` is one raw `"KEY=value"` string per the Vorbis
/// comment spec. We store the on-disk form rather than a split `(key, value)`
/// because it preserves original ordering and byte-identity on round-trip,
/// which matters for differential testing and for real files that contain
/// duplicate keys, non-standard tag names, or `=` characters inside values.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct OpusTags {
    pub vendor: String,
    pub comments: Vec<String>,
}

impl OpusTags {
    /// Parse an `OpusTags` packet payload into vendor + comments.
    ///
    /// Rejects: missing magic, length fields that overrun the buffer, non-UTF-8
    /// vendor bytes, non-UTF-8 comment bytes. Lax on duplicates, non-standard
    /// tag names, and empty values — those are preserved verbatim.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            bail!(
                "OpusTags packet too short ({} bytes; need at least 8 for magic)",
                data.len()
            );
        }
        if &data[..8] != b"OpusTags" {
            bail!(
                "OpusTags magic missing (got {:?})",
                &data[..8.min(data.len())]
            );
        }

        let mut pos = 8usize;
        let vendor_len = read_u32_le(data, &mut pos)
            .ok_or_else(|| anyhow!("OpusTags truncated before vendor_length"))?
            as usize;
        // Overflow-safe: on 32-bit targets `pos + vendor_len` could wrap; use
        // saturating_sub on the slice length instead.
        if vendor_len > data.len().saturating_sub(pos) {
            bail!(
                "OpusTags vendor_length {} overruns packet ({} bytes remaining)",
                vendor_len,
                data.len() - pos
            );
        }
        let vendor = std::str::from_utf8(&data[pos..pos + vendor_len])
            .map_err(|e| anyhow!("OpusTags vendor is not valid UTF-8: {e}"))?
            .to_string();
        pos += vendor_len;

        let comment_count = read_u32_le(data, &mut pos)
            .ok_or_else(|| anyhow!("OpusTags truncated before user_comment_count"))?
            as usize;

        // DoS guard: each comment has a 4-byte length prefix as its minimum
        // serialized size, so a valid comment_count cannot exceed
        // remaining_bytes / 4. Without this check, a crafted u32::MAX value
        // would request ~96 GB from Vec::with_capacity and abort.
        if comment_count > data.len().saturating_sub(pos) / 4 {
            bail!(
                "OpusTags user_comment_count {} exceeds remaining buffer ({} bytes)",
                comment_count,
                data.len().saturating_sub(pos)
            );
        }

        let mut comments = Vec::with_capacity(comment_count);
        for i in 0..comment_count {
            let len = read_u32_le(data, &mut pos).ok_or_else(|| {
                anyhow!("OpusTags truncated before length of comment #{i}")
            })? as usize;
            if len > data.len().saturating_sub(pos) {
                bail!(
                    "OpusTags comment #{i} length {} overruns packet ({} bytes remaining)",
                    len,
                    data.len() - pos
                );
            }
            let s = std::str::from_utf8(&data[pos..pos + len])
                .map_err(|e| anyhow!("OpusTags comment #{i} is not valid UTF-8: {e}"))?
                .to_string();
            comments.push(s);
            pos += len;
        }

        Ok(OpusTags { vendor, comments })
    }

    /// Serialize to the on-disk `OpusTags` packet payload. Little-endian,
    /// length-prefixed, no framing bit, no padding — RFC 7845 §5.2 exactly.
    pub fn encode(&self) -> Vec<u8> {
        let vendor_bytes = self.vendor.as_bytes();
        // No pre-computed capacity: summing lengths could overflow for
        // pathological inputs, and extend_from_slice reallocates cheaply
        // enough that it's not worth the risk for tags packets.
        let mut out = Vec::new();
        out.extend_from_slice(b"OpusTags");
        out.extend_from_slice(&(vendor_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(vendor_bytes);
        out.extend_from_slice(&(self.comments.len() as u32).to_le_bytes());
        for c in &self.comments {
            out.extend_from_slice(&(c.len() as u32).to_le_bytes());
            out.extend_from_slice(c.as_bytes());
        }
        out
    }

    /// Case-insensitive key lookup. Splits each comment on the first `=` only
    /// (so `KEY=a=b=c` returns `a=b=c`). Entries without an `=` are skipped
    /// rather than aborting the search — Vorbis comments are "mostly" but not
    /// strictly `KEY=value`, and one malformed line shouldn't hide the rest.
    pub fn get(&self, key: &str) -> Option<&str> {
        for c in &self.comments {
            if let Some((k, v)) = c.split_once('=') {
                if k.eq_ignore_ascii_case(key) {
                    return Some(v);
                }
            }
        }
        None
    }
}

/// Read 4 LE bytes at `*pos` and advance. Returns `None` if fewer than 4 bytes
/// remain — callers turn that into a descriptive error.
fn read_u32_le(data: &[u8], pos: &mut usize) -> Option<u32> {
    // Overflow-safe bounds check: on 32-bit targets `*pos + 4` could wrap when
    // `*pos` is close to usize::MAX.
    if 4 > data.len().saturating_sub(*pos) {
        return None;
    }
    let v = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    Some(v)
}

#[derive(Debug, Clone, Copy)]
pub struct OpusHead {
    pub version: u8,
    pub channels: u8,
    pub pre_skip: u16,
    pub input_sample_rate: u32,
    pub output_gain: i16,
    pub channel_mapping: u8,
}

pub fn parse_opus_head(data: &[u8]) -> Result<OpusHead> {
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
pub fn read_last_granule<R: Read + Seek>(
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

    // -- OpusTags ----------------------------------------------------------

    #[test]
    fn opus_tags_round_trip_encode_parse() {
        let tags = OpusTags {
            vendor: "ropusenc 0.1.0".to_string(),
            comments: vec![
                "ARTIST=Example".to_string(),
                "TITLE=Test".to_string(),
                "ENCODER=ropusenc".to_string(),
            ],
        };
        let bytes = tags.encode();
        let parsed = OpusTags::parse(&bytes).expect("parse round-tripped bytes");
        assert_eq!(parsed, tags);
    }

    #[test]
    fn opus_tags_round_trip_zero_comments() {
        // Production encode path (commands::encode) writes an OpusTags packet
        // with an empty comments vector. Exercise that explicitly.
        let tags = OpusTags {
            vendor: "ropusenc 0.1.0".to_string(),
            comments: vec![],
        };
        let bytes = tags.encode();
        let parsed = OpusTags::parse(&bytes).expect("parse");
        assert_eq!(parsed.vendor, tags.vendor);
        assert!(parsed.comments.is_empty());
    }

    #[test]
    fn opus_tags_parse_known_fixture() {
        // Hand-built payload:
        //   "OpusTags" (8)                            = 8
        //   vendor_length = 11 LE                     + 4 = 12
        //   "libopus 1.3" (11)                        + 11 = 23 (header+vendor)
        //   user_comment_count = 1 LE                 + 4 = 27
        //   comment_length = 15 LE                    + 4 = 31
        //   "ENCODER=opusenc" (15)                    + 15 = 46
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"OpusTags");
        bytes.extend_from_slice(&11u32.to_le_bytes());
        bytes.extend_from_slice(b"libopus 1.3");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&15u32.to_le_bytes());
        bytes.extend_from_slice(b"ENCODER=opusenc");
        assert_eq!(bytes.len(), 46, "fixture byte count must match hand math");

        let parsed = OpusTags::parse(&bytes).expect("parse fixture");
        assert_eq!(parsed.vendor, "libopus 1.3");
        assert_eq!(parsed.comments, vec!["ENCODER=opusenc".to_string()]);
    }

    #[test]
    fn opus_tags_parse_rejects_short_buffer() {
        assert!(OpusTags::parse(&[0u8; 4]).is_err());
    }

    #[test]
    fn opus_tags_parse_rejects_wrong_magic() {
        // 8 bytes, but not the "OpusTags" magic. Must be rejected before we
        // ever look at a length field.
        let buf = b"NotOpus!";
        assert!(OpusTags::parse(buf).is_err());
    }

    #[test]
    fn opus_tags_parse_rejects_overlong_vendor_length() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"OpusTags");
        bytes.extend_from_slice(&100u32.to_le_bytes()); // claims 100 bytes of vendor
        bytes.extend_from_slice(&[b'A'; 10]); // but only 10 provided
        let err = OpusTags::parse(&bytes).expect_err("must reject overrun");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("overruns"),
            "expected overrun error, got: {msg}"
        );
    }

    #[test]
    fn opus_tags_parse_rejects_non_utf8_vendor() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"OpusTags");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&[0xFF, 0xFE]); // invalid UTF-8
        bytes.extend_from_slice(&0u32.to_le_bytes()); // zero comments
        let err = OpusTags::parse(&bytes).expect_err("must reject non-UTF-8 vendor");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("UTF-8"),
            "expected UTF-8 error, got: {msg}"
        );
    }

    #[test]
    fn opus_tags_get_is_case_insensitive_on_key() {
        let tags = OpusTags {
            vendor: "v".to_string(),
            comments: vec!["ARTIST=Foo".to_string()],
        };
        assert_eq!(tags.get("artist"), Some("Foo"));
        assert_eq!(tags.get("ARTIST"), Some("Foo"));
        assert_eq!(tags.get("ArTiSt"), Some("Foo"));
        assert_eq!(tags.get("missing"), None);
    }

    #[test]
    fn opus_tags_get_splits_on_first_equals_only() {
        // Values may legitimately contain '=', notably METADATA_BLOCK_PICTURE
        // base64 payloads. We split the key off on the first '=' and return
        // everything after it verbatim.
        let tags = OpusTags {
            vendor: "v".to_string(),
            comments: vec!["KEY=a=b=c".to_string()],
        };
        assert_eq!(tags.get("key"), Some("a=b=c"));
    }
}
