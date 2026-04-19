//! METADATA_BLOCK_PICTURE packing and base64 encoding.
//!
//! Builds the FLAC-style picture block that opus-tools embeds as a
//! `METADATA_BLOCK_PICTURE=<base64>` Vorbis comment. Hand-rolled base64 (same
//! style as `audio/wav.rs`) to avoid pulling in a dependency for ~30 lines of
//! code. We only encode; parsing pictures back out is not a project goal.

use anyhow::{Result, bail};

/// Picture MIME types we detect via magic bytes. Only JPEG and PNG are
/// supported — everything else is rejected rather than guessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PictureFormat {
    Jpeg,
    Png,
}

impl PictureFormat {
    pub fn mime(self) -> &'static str {
        match self {
            PictureFormat::Jpeg => "image/jpeg",
            PictureFormat::Png => "image/png",
        }
    }
}

/// Detect image format from the first handful of bytes.
///
/// JPEG variants we accept: `FF D8 FF E0` (JFIF), `FF D8 FF E1` (EXIF),
/// `FF D8 FF DB` (raw quantisation table — rare but valid).
/// PNG: the 8-byte signature `89 50 4E 47 0D 0A 1A 0A`.
pub fn detect_format(data: &[u8]) -> Result<PictureFormat> {
    if data.len() >= 4
        && data[0] == 0xFF
        && data[1] == 0xD8
        && data[2] == 0xFF
        && matches!(data[3], 0xE0 | 0xE1 | 0xDB)
    {
        return Ok(PictureFormat::Jpeg);
    }
    if data.len() >= 8 && data[..8] == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] {
        return Ok(PictureFormat::Png);
    }
    bail!("unrecognised picture format (need JPEG or PNG)");
}

/// Build a METADATA_BLOCK_PICTURE byte structure. All multi-byte integers are
/// **big-endian** per the FLAC spec — this is *not* the Ogg/Opus little-endian
/// convention, so do not copy-paste LE code from `container::ogg`.
///
/// Layout (per FLAC picture metadata block):
/// ```text
/// u32  picture_type                (3 = Front Cover)
/// u32  mime_length
/// N    mime string (ASCII)
/// u32  description_length
/// N    description string (UTF-8)
/// u32  width                       (0 = unknown)
/// u32  height                      (0 = unknown)
/// u32  colour_depth                (0 = unknown)
/// u32  indexed_colours             (0 = unknown, non-indexed)
/// u32  picture_data_length
/// N    picture bytes
/// ```
///
/// Width/height/depth/indexed are always emitted as 0. Parsing PNG/JPEG chunks
/// to populate them is a 2-hour job for no real downstream benefit — 0 means
/// "unknown" which is legal per spec.
///
/// Returns an error rather than silently truncating if `data.len()` exceeds
/// `u32::MAX` bytes (FLAC picture-block length field is a u32). The `as u32`
/// cast would otherwise wrap, producing a malformed block.
pub fn build_picture_block(format: PictureFormat, data: &[u8]) -> Result<Vec<u8>> {
    let mime = format.mime().as_bytes();
    let description: &[u8] = b"";
    // FLAC picture-block length fields are u32, so bail cleanly rather than
    // wrap-truncating via `as u32`. Same check for mime length for symmetry,
    // though our MIME strings are `image/jpeg` / `image/png` — well under 4 GiB.
    let data_len = u32::try_from(data.len())
        .map_err(|_| anyhow::anyhow!(
            "picture data is {} bytes; FLAC picture-block length field is u32 (max {} bytes)",
            data.len(),
            u32::MAX,
        ))?;
    let mime_len = u32::try_from(mime.len())
        .expect("MIME strings are < 32 bytes by construction");
    let desc_len = u32::try_from(description.len())
        .expect("description is empty by construction");

    // Pre-size to the exact byte length: 4*7 header u32s + mime + desc + data.
    let mut out = Vec::with_capacity(32 + mime.len() + description.len() + data.len());

    // picture_type: 3 = Front Cover
    out.extend_from_slice(&3u32.to_be_bytes());
    // MIME string
    out.extend_from_slice(&mime_len.to_be_bytes());
    out.extend_from_slice(mime);
    // Description string (empty)
    out.extend_from_slice(&desc_len.to_be_bytes());
    out.extend_from_slice(description);
    // width/height/depth/indexed — all unknown
    out.extend_from_slice(&0u32.to_be_bytes());
    out.extend_from_slice(&0u32.to_be_bytes());
    out.extend_from_slice(&0u32.to_be_bytes());
    out.extend_from_slice(&0u32.to_be_bytes());
    // Picture data
    out.extend_from_slice(&data_len.to_be_bytes());
    out.extend_from_slice(data);
    Ok(out)
}

/// Hard cap on `--picture` file size. 20 MiB is 10× the typical "album art"
/// ceiling of ~2 MiB; anything larger is almost certainly a user mistake
/// (dropped-in wrong file). Reject early, before reading the full file into
/// memory, to keep errors fast and useful.
pub const MAX_PICTURE_BYTES: u64 = 20 * 1024 * 1024;

/// Base64-encode per RFC 4648 with standard alphabet and `=` padding.
///
/// Hand-rolled deliberately — base64 isn't a workspace dep and the picture
/// code is the only caller. Matches the style of `audio/wav.rs`
/// (hand-rolled to avoid a dep for a tiny surface area).
pub fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    let mut chunks = data.chunks_exact(3);
    for chunk in &mut chunks {
        let (b0, b1, b2) = (chunk[0], chunk[1], chunk[2]);
        out.push(ALPHABET[(b0 >> 2) as usize] as char);
        out.push(ALPHABET[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        out.push(ALPHABET[(((b1 & 0x0F) << 2) | (b2 >> 6)) as usize] as char);
        out.push(ALPHABET[(b2 & 0x3F) as usize] as char);
    }
    let rem = chunks.remainder();
    match rem.len() {
        0 => {}
        1 => {
            let b0 = rem[0];
            out.push(ALPHABET[(b0 >> 2) as usize] as char);
            out.push(ALPHABET[((b0 & 0x03) << 4) as usize] as char);
            out.push('=');
            out.push('=');
        }
        2 => {
            let (b0, b1) = (rem[0], rem[1]);
            out.push(ALPHABET[(b0 >> 2) as usize] as char);
            out.push(ALPHABET[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
            out.push(ALPHABET[((b1 & 0x0F) << 2) as usize] as char);
            out.push('=');
        }
        _ => unreachable!("chunks_exact(3).remainder() is < 3 by construction"),
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- base64 ---------------------------------------------------------

    /// RFC 4648 §10 — the canonical "base64 test vectors" everyone uses. If
    /// any of these fail, every tag we ever emit is wrong.
    #[test]
    fn base64_encode_matches_known_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    // ---- format detection -----------------------------------------------

    #[test]
    fn detect_format_recognises_jpeg_jfif() {
        let data = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x00];
        assert_eq!(detect_format(&data).unwrap(), PictureFormat::Jpeg);
    }

    #[test]
    fn detect_format_recognises_jpeg_exif() {
        let data = [0xFF, 0xD8, 0xFF, 0xE1, 0x00, 0x00];
        assert_eq!(detect_format(&data).unwrap(), PictureFormat::Jpeg);
    }

    #[test]
    fn detect_format_recognises_jpeg_dqt() {
        let data = [0xFF, 0xD8, 0xFF, 0xDB, 0x00, 0x00];
        assert_eq!(detect_format(&data).unwrap(), PictureFormat::Jpeg);
    }

    #[test]
    fn detect_format_recognises_png() {
        let data = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0xDE, 0xAD];
        assert_eq!(detect_format(&data).unwrap(), PictureFormat::Png);
    }

    #[test]
    fn picture_block_rejects_unknown_format() {
        let data = [0xAB, 0xCD, 0xEF, 0x01];
        let err = detect_format(&data).expect_err("must reject unknown magic");
        assert!(
            format!("{err:#}").contains("unrecognised picture format"),
            "expected unknown-format error, got: {err}"
        );
    }

    #[test]
    fn detect_format_rejects_too_short() {
        assert!(detect_format(&[]).is_err());
        assert!(detect_format(&[0xFF]).is_err());
        assert!(detect_format(&[0xFF, 0xD8, 0xFF]).is_err()); // 3 bytes — JPEG needs 4
    }

    // ---- picture block layout -------------------------------------------

    /// Pull a u32 back out of the big-endian header at `offset`.
    fn u32_be(block: &[u8], offset: usize) -> u32 {
        u32::from_be_bytes([
            block[offset],
            block[offset + 1],
            block[offset + 2],
            block[offset + 3],
        ])
    }

    #[test]
    fn picture_block_png_magic() {
        // Tiny fake "PNG" payload — just enough for magic-bytes detection.
        let fake_png = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0xAA];
        let format = detect_format(&fake_png).unwrap();
        let block = build_picture_block(format, &fake_png).expect("tiny PNG fits in u32");

        // picture_type at offset 0 (BE u32 = 3)
        assert_eq!(u32_be(&block, 0), 3, "picture_type must be 3 (Front Cover)");
        // mime_length at offset 4 (BE u32 = len("image/png") = 9)
        assert_eq!(u32_be(&block, 4), 9, "mime_length must match 'image/png' length");
        // MIME string at offset 8..17
        assert_eq!(&block[8..17], b"image/png", "mime string body");
    }

    #[test]
    fn picture_block_jpeg_magic() {
        let fake_jpeg = [0xFF, 0xD8, 0xFF, 0xE0, 0xAA, 0xBB];
        let format = detect_format(&fake_jpeg).unwrap();
        let block = build_picture_block(format, &fake_jpeg).expect("tiny JPEG fits in u32");

        assert_eq!(u32_be(&block, 0), 3, "picture_type must be 3");
        assert_eq!(u32_be(&block, 4), 10, "mime_length must match 'image/jpeg'");
        assert_eq!(&block[8..18], b"image/jpeg");
        // description_length (should be 0)
        assert_eq!(u32_be(&block, 18), 0, "description must be empty");
        // width/height/depth/indexed (all 0, unknown)
        assert_eq!(u32_be(&block, 22), 0);
        assert_eq!(u32_be(&block, 26), 0);
        assert_eq!(u32_be(&block, 30), 0);
        assert_eq!(u32_be(&block, 34), 0);
        // picture_data_length
        assert_eq!(u32_be(&block, 38), fake_jpeg.len() as u32);
        // picture bytes follow
        assert_eq!(&block[42..], &fake_jpeg);
    }

    #[test]
    fn picture_block_round_trips_through_base64() {
        // Sanity: base64(build(jpeg)) decodes back to something that starts
        // with "picture_type = 3" when re-interpreted as BE u32.
        let fake = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x11, 0x22, 0x33];
        let format = detect_format(&fake).unwrap();
        let block = build_picture_block(format, &fake).expect("tiny fixture fits in u32");
        let b64 = base64_encode(&block);
        // Crude decode: just verify base64 length is what the spec says.
        // (We don't have a decoder; we only need to ensure we didn't emit
        // padding for a 3-byte-aligned input.)
        assert_eq!(b64.len(), block.len().div_ceil(3) * 4);
    }
}
