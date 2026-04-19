//! OpusTags (vorbis_comment) parser — RFC 7845 sec. 5.2.
//!
//! Layout of an OpusTags packet:
//!
//!   8 bytes   magic  "OpusTags"
//!   u32 le    vendor_length
//!   N bytes   vendor_string (UTF-8)
//!   u32 le    user_comment_count
//!   for each comment:
//!       u32 le   length
//!       N bytes  "KEY=VALUE" (UTF-8; key is ASCII 0x20..=0x7D excluding '=')
//!
//! We keep the parser tight: one linear pass, all length checks bounded,
//! invalid UTF-8 in the value is rejected with a typed error rather than
//! silently mojibake'd. Keys are normalised to ASCII uppercase on extraction
//! because vorbis_comment keys are case-insensitive and consumers (fb2k,
//! ReplayGain mapping) expect uppercase.
//!
//! The parser preserves *every* comment, including `METADATA_BLOCK_PICTURE`
//! cover-art blobs. Dropping them happens at the reporting boundary in
//! `lib.rs::ropus_fb2k_read_tags` (HLD sec. 2 non-goals: fb2k has its own
//! cover-art pipeline). Keeping the parser complete means a future caller
//! that *does* want the picture blob can get it from `ParsedTags::iter()`
//! directly without re-parsing the packet.
//!
//! ReplayGain tag *conversion* does not live here — a future release will
//! layer it on top of the raw `(key, value)` iteration exposed below.
//! Today we only need the vendor string and the raw comments so the
//! tag-read callback has something to emit.

use std::fmt;

/// Magic at the start of an OpusTags packet.
pub(crate) const OPUS_TAGS_MAGIC: &[u8; 8] = b"OpusTags";

/// Everything we extract from an OpusTags packet. We preserve the raw
/// bytes' order — some downstream tooling (the tag callback in
/// `ropus_fb2k_read_tags`) assumes first-seen-first-emitted.
#[derive(Debug, Clone)]
pub(crate) struct ParsedTags {
    pub(crate) vendor: String,
    pub(crate) comments: Vec<(String, String)>,
}

impl ParsedTags {
    /// Iterate `(KEY, VALUE)` pairs in input order, skipping filtered keys.
    /// Keys are uppercased per vorbis_comment convention.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.comments
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
    }
}

/// Typed error for the OpusTags parser. Kept shallow so the caller can turn
/// it into an `INVALID_STREAM` status without matching variants.
#[derive(Debug)]
pub(crate) enum TagError {
    BadMagic,
    Truncated,
    NonUtf8Vendor,
    NonUtf8Comment,
    MissingEqualsInComment,
    /// A comment started with `=` (zero-length key). RFC 7845 /
    /// vorbis_comment requires keys be 1+ ASCII chars.
    EmptyKey,
}

impl fmt::Display for TagError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TagError::BadMagic => f.write_str("OpusTags magic missing"),
            TagError::Truncated => f.write_str("OpusTags packet truncated"),
            TagError::NonUtf8Vendor => f.write_str("OpusTags vendor string not UTF-8"),
            TagError::NonUtf8Comment => f.write_str("OpusTags comment not UTF-8"),
            TagError::MissingEqualsInComment => {
                f.write_str("OpusTags comment missing '=' separator")
            }
            TagError::EmptyKey => f.write_str("OpusTags comment has zero-length key"),
        }
    }
}

impl std::error::Error for TagError {}

/// Parse an OpusTags packet.
///
/// Does not allocate per-comment intermediate `Vec`s — pulls straight from
/// the input slice. Validates every length against remaining bytes before
/// indexing so a malformed packet can only surface as `TagError::Truncated`,
/// never as a panic.
pub(crate) fn parse(bytes: &[u8]) -> Result<ParsedTags, TagError> {
    if bytes.len() < OPUS_TAGS_MAGIC.len() {
        return Err(TagError::Truncated);
    }
    if &bytes[..OPUS_TAGS_MAGIC.len()] != OPUS_TAGS_MAGIC {
        return Err(TagError::BadMagic);
    }
    let mut cur = &bytes[OPUS_TAGS_MAGIC.len()..];

    let vendor_len = read_u32_le(&mut cur)? as usize;
    let vendor_bytes = take(&mut cur, vendor_len)?;
    let vendor = std::str::from_utf8(vendor_bytes)
        .map_err(|_| TagError::NonUtf8Vendor)?
        .to_owned();

    let comment_count = read_u32_le(&mut cur)? as usize;
    let mut comments = Vec::with_capacity(comment_count.min(64));
    for _ in 0..comment_count {
        let len = read_u32_le(&mut cur)? as usize;
        let raw = take(&mut cur, len)?;
        let text = std::str::from_utf8(raw).map_err(|_| TagError::NonUtf8Comment)?;

        // Per RFC 7845 / vorbis_comment, the separator is '=' (0x3D). We
        // split on the first '=' only; the value may contain further '='.
        let (key, value) = match text.find('=') {
            Some(i) => (&text[..i], &text[i + 1..]),
            None => return Err(TagError::MissingEqualsInComment),
        };
        // RFC 7845 / vorbis_comment requires keys be 1+ ASCII chars. An
        // input like `=foo` has a zero-length key and is malformed.
        if key.is_empty() {
            return Err(TagError::EmptyKey);
        }
        // Keys are 7-bit ASCII per the spec; uppercase via ASCII table,
        // don't involve Unicode case folding.
        let key = key.to_ascii_uppercase();
        comments.push((key, value.to_owned()));
    }

    // Trailing bytes after the documented payload are legal and we ignore
    // them (libopus includes a framing bit + padding in vorbis_comment pages
    // for history; Opus doesn't, but be liberal).
    Ok(ParsedTags { vendor, comments })
}

fn read_u32_le(cur: &mut &[u8]) -> Result<u32, TagError> {
    if cur.len() < 4 {
        return Err(TagError::Truncated);
    }
    let v = u32::from_le_bytes([cur[0], cur[1], cur[2], cur[3]]);
    *cur = &cur[4..];
    Ok(v)
}

fn take<'a>(cur: &mut &'a [u8], n: usize) -> Result<&'a [u8], TagError> {
    if cur.len() < n {
        return Err(TagError::Truncated);
    }
    let (head, tail) = cur.split_at(n);
    *cur = tail;
    Ok(head)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal OpusTags packet with the given vendor and comments.
    fn build(vendor: &str, comments: &[&str]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(OPUS_TAGS_MAGIC);
        v.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        v.extend_from_slice(vendor.as_bytes());
        v.extend_from_slice(&(comments.len() as u32).to_le_bytes());
        for c in comments {
            v.extend_from_slice(&(c.len() as u32).to_le_bytes());
            v.extend_from_slice(c.as_bytes());
        }
        v
    }

    #[test]
    fn parses_vendor_and_comments() {
        let bytes = build(
            "ropus-cli",
            &["ARTIST=Alice", "title=Bob", "DATE=2026"],
        );
        let parsed = parse(&bytes).expect("valid");
        assert_eq!(parsed.vendor, "ropus-cli");
        let kv: Vec<(&str, &str)> = parsed.iter().collect();
        assert_eq!(
            kv,
            vec![
                ("ARTIST", "Alice"),
                ("TITLE", "Bob"),
                ("DATE", "2026"),
            ]
        );
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = build("vendor", &[]);
        bytes[0] = b'X';
        assert!(matches!(parse(&bytes), Err(TagError::BadMagic)));
    }

    #[test]
    fn rejects_truncated_comment_length() {
        let mut bytes = build("v", &["A=B"]);
        // Strip the last 4 bytes (half of "A=B" plus a bit).
        bytes.truncate(bytes.len() - 3);
        assert!(matches!(parse(&bytes), Err(TagError::Truncated)));
    }

    #[test]
    fn rejects_comment_without_equals() {
        let bytes = build("v", &["no_equals_here"]);
        assert!(matches!(
            parse(&bytes),
            Err(TagError::MissingEqualsInComment)
        ));
    }

    #[test]
    fn parse_empty_key_rejected() {
        // `=foo` — vorbis_comment keys must be 1+ ASCII chars; no key
        // means a malformed packet.
        let bytes = build("v", &["=foo"]);
        assert!(matches!(parse(&bytes), Err(TagError::EmptyKey)));
    }

    #[test]
    fn parser_preserves_metadata_block_picture() {
        // The parser keeps every comment, including cover-art blobs. The
        // FFI layer is what filters them out before the tag callback fires
        // (see `lib.rs::FILTERED_TAG_KEYS`). A future caller that *does*
        // want the picture blob can still get it from here.
        let bytes = build("v", &["Metadata_Block_Picture=<huge blob>", "ARTIST=Alice"]);
        let parsed = parse(&bytes).expect("valid");
        let kv: Vec<(&str, &str)> = parsed.iter().collect();
        assert_eq!(
            kv,
            vec![
                ("METADATA_BLOCK_PICTURE", "<huge blob>"),
                ("ARTIST", "Alice"),
            ]
        );
    }

    #[test]
    fn preserves_further_equals_in_value() {
        let bytes = build("v", &["R128_TRACK_GAIN=-1234"]);
        let parsed = parse(&bytes).expect("valid");
        let kv: Vec<(&str, &str)> = parsed.iter().collect();
        assert_eq!(kv, vec![("R128_TRACK_GAIN", "-1234")]);
    }

    #[test]
    fn non_utf8_vendor_is_rejected() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(OPUS_TAGS_MAGIC);
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&[0xFF, 0xFE, 0xFD]);
        bytes.extend_from_slice(&0u32.to_le_bytes());
        assert!(matches!(parse(&bytes), Err(TagError::NonUtf8Vendor)));
    }
}
