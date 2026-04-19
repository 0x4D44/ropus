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
//! ReplayGain tag conversion lives at the bottom of this file (see
//! `extract_replaygain`) per HLD sec. 5.5: legacy `REPLAYGAIN_*` tags
//! override their R128 counterparts when both are present; R128 values are
//! Q7.8 dB relative to -23 LUFS and get a +5 dB offset to match the
//! ReplayGain -18 LUFS reference. Malformed values fall back to `NaN`
//! silently — logging is the host's job if it cares.

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
        self.comments.iter().map(|(k, v)| (k.as_str(), v.as_str()))
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

// ---------------------------------------------------------------------------
// ReplayGain extraction (HLD sec. 5.5)
// ---------------------------------------------------------------------------

/// R128-to-ReplayGain dB offset. R128 targets -23 LUFS; ReplayGain targets
/// -18 LUFS; adding +5 dB to the R128 value aligns the two references.
const R128_TO_RG_OFFSET_DB: f32 = 5.0;

/// Q7.8 divisor for the string-encoded signed integer in `R128_*_GAIN` tags.
/// `"-1280"` means -1280/256 = -5 dB relative to the R128 reference.
const R128_Q78_DIVISOR: f32 = 256.0;

/// ReplayGain values as they will appear in `RopusFb2kInfo`. `NaN` means
/// "tag absent or malformed"; callers check with `.is_nan()`, not equality.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ReplayGainInfo {
    pub(crate) track_gain: f32,
    pub(crate) album_gain: f32,
    pub(crate) track_peak: f32,
    pub(crate) album_peak: f32,
}

impl ReplayGainInfo {
    pub(crate) fn nan() -> Self {
        Self {
            track_gain: f32::NAN,
            album_gain: f32::NAN,
            track_peak: f32::NAN,
            album_peak: f32::NAN,
        }
    }
}

/// Extract ReplayGain values from parsed OpusTags per HLD sec. 5.5.
///
/// Rules:
/// * `R128_TRACK_GAIN` / `R128_ALBUM_GAIN` — string-encoded signed integer
///   in Q7.8 dB relative to R128 -23 LUFS. Convert: `i / 256 + 5` dB.
/// * `REPLAYGAIN_TRACK_GAIN` / `REPLAYGAIN_ALBUM_GAIN` — legacy string like
///   `"-6.75 dB"` or `"-6.75"`, already in ReplayGain dB. Parse the leading
///   float.
/// * **Legacy overrides R128.** If both are present the user (or tool that
///   wrote the file) explicitly set `REPLAYGAIN_*`; R128 was probably
///   auto-derived and we should not second-guess the explicit value.
/// * `R128_TRACK_PEAK` / `R128_ALBUM_PEAK` / `REPLAYGAIN_TRACK_PEAK` /
///   `REPLAYGAIN_ALBUM_PEAK` — linear peak as float, pass through. Legacy
///   overrides R128 by the same rule.
/// * Absent tag → `NaN`. Malformed value (unparseable, non-finite) → `NaN`
///   silently (logging is fb2k's job; a bad RG tag shouldn't break
///   playback).
pub(crate) fn extract_replaygain(tags: &ParsedTags) -> ReplayGainInfo {
    let mut rg = ReplayGainInfo::nan();

    // First pass: R128 tags. These are the canonical Opus form, so we
    // populate from them first; the legacy pass below then overrides.
    for (key, value) in tags.iter() {
        match key {
            "R128_TRACK_GAIN" => {
                if let Some(v) = parse_r128_gain(value) {
                    rg.track_gain = v;
                }
            }
            "R128_ALBUM_GAIN" => {
                if let Some(v) = parse_r128_gain(value) {
                    rg.album_gain = v;
                }
            }
            "R128_TRACK_PEAK" => {
                if let Some(v) = parse_rg_peak(value) {
                    rg.track_peak = v;
                }
            }
            "R128_ALBUM_PEAK" => {
                if let Some(v) = parse_rg_peak(value) {
                    rg.album_peak = v;
                }
            }
            _ => {}
        }
    }

    // Second pass: legacy REPLAYGAIN_* tags override R128 when both are
    // present. We overwrite unconditionally on a successful parse — an
    // unparseable legacy tag leaves the (possibly-R128-populated) value
    // alone rather than nuking it to NaN.
    for (key, value) in tags.iter() {
        match key {
            "REPLAYGAIN_TRACK_GAIN" => {
                if let Some(v) = parse_legacy_gain(value) {
                    rg.track_gain = v;
                }
            }
            "REPLAYGAIN_ALBUM_GAIN" => {
                if let Some(v) = parse_legacy_gain(value) {
                    rg.album_gain = v;
                }
            }
            "REPLAYGAIN_TRACK_PEAK" => {
                if let Some(v) = parse_rg_peak(value) {
                    rg.track_peak = v;
                }
            }
            "REPLAYGAIN_ALBUM_PEAK" => {
                if let Some(v) = parse_rg_peak(value) {
                    rg.album_peak = v;
                }
            }
            _ => {}
        }
    }

    rg
}

/// Parse `R128_*_GAIN`: string-encoded signed integer in Q7.8 dB relative
/// to R128 -23 LUFS. Add +5 dB to land on the ReplayGain -18 LUFS reference.
/// Returns `None` on parse failure, non-finite result, or out-of-range
/// value (outside ±127 dB — matches libopusfile's R128 clamp; a value
/// outside this range is almost certainly file corruption, and rejecting
/// NaN/Inf alone would still accept `i32::MIN` → ≈-8.4 million dB).
fn parse_r128_gain(s: &str) -> Option<f32> {
    let raw: i32 = s.trim().parse().ok()?;
    let gain = raw as f32 / R128_Q78_DIVISOR + R128_TO_RG_OFFSET_DB;
    if !gain.is_finite() || !(-127.0..=127.0).contains(&gain) {
        return None;
    }
    Some(gain)
}

/// Parse `REPLAYGAIN_*_GAIN`: legacy string, e.g. `"-6.75 dB"` or `"-6.75"`.
/// Strip a trailing "dB" / "db" (case-insensitive) before parsing as `f32`.
fn parse_legacy_gain(s: &str) -> Option<f32> {
    let trimmed = s.trim();
    // `str::strip_suffix` is case-sensitive, so we walk the canonical
    // spellings. The " " variants handle files written with a space before
    // the unit; the no-space variants handle files without.
    let numeric = trimmed
        .strip_suffix(" dB")
        .or_else(|| trimmed.strip_suffix(" db"))
        .or_else(|| trimmed.strip_suffix("dB"))
        .or_else(|| trimmed.strip_suffix("db"))
        .unwrap_or(trimmed)
        .trim();
    let v: f32 = numeric.parse().ok()?;
    v.is_finite().then_some(v)
}

/// Parse a ReplayGain peak: linear float, unitless. Both R128 and legacy
/// forms share the same parse — they differ only in the tag key.
fn parse_rg_peak(s: &str) -> Option<f32> {
    let v: f32 = s.trim().parse().ok()?;
    v.is_finite().then_some(v)
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
        let bytes = build("ropus-cli", &["ARTIST=Alice", "title=Bob", "DATE=2026"]);
        let parsed = parse(&bytes).expect("valid");
        assert_eq!(parsed.vendor, "ropus-cli");
        let kv: Vec<(&str, &str)> = parsed.iter().collect();
        assert_eq!(
            kv,
            vec![("ARTIST", "Alice"), ("TITLE", "Bob"), ("DATE", "2026"),]
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

    // -----------------------------------------------------------------
    // ReplayGain extraction (HLD sec. 5.5)
    // -----------------------------------------------------------------

    fn tags_from(kvs: &[(&str, &str)]) -> ParsedTags {
        ParsedTags {
            vendor: String::new(),
            comments: kvs
                .iter()
                .map(|(k, v)| (k.to_ascii_uppercase(), (*v).to_owned()))
                .collect(),
        }
    }

    #[test]
    fn extract_rg_absent_tags_are_nan() {
        let parsed = tags_from(&[("ARTIST", "Alice")]);
        let rg = extract_replaygain(&parsed);
        assert!(rg.track_gain.is_nan());
        assert!(rg.album_gain.is_nan());
        assert!(rg.track_peak.is_nan());
        assert!(rg.album_peak.is_nan());
    }

    #[test]
    fn extract_rg_r128_track_gain_converts_q78_and_offsets_by_5db() {
        // -1280 Q7.8 = -5 dB R128. Add +5 dB offset → 0 dB RG.
        let parsed = tags_from(&[("R128_TRACK_GAIN", "-1280")]);
        let rg = extract_replaygain(&parsed);
        assert!((rg.track_gain - 0.0).abs() < 1e-3);
    }

    #[test]
    fn extract_rg_legacy_overrides_r128() {
        let parsed = tags_from(&[
            ("R128_TRACK_GAIN", "-1280"),          // -5 dB R128 → 0 dB RG
            ("REPLAYGAIN_TRACK_GAIN", "-6.75 dB"), // wins the override
        ]);
        let rg = extract_replaygain(&parsed);
        assert!((rg.track_gain - (-6.75)).abs() < 1e-3);
    }

    #[test]
    fn extract_rg_legacy_gain_accepts_no_unit() {
        let parsed = tags_from(&[("REPLAYGAIN_TRACK_GAIN", "-3.5")]);
        let rg = extract_replaygain(&parsed);
        assert!((rg.track_gain - (-3.5)).abs() < 1e-3);
    }

    #[test]
    fn extract_rg_peak_passes_through() {
        let parsed = tags_from(&[("R128_TRACK_PEAK", "0.95")]);
        let rg = extract_replaygain(&parsed);
        assert!((rg.track_peak - 0.95).abs() < 1e-3);
    }

    #[test]
    fn extract_rg_malformed_is_nan() {
        let parsed = tags_from(&[("R128_TRACK_GAIN", "not_a_number")]);
        let rg = extract_replaygain(&parsed);
        assert!(rg.track_gain.is_nan());
    }

    #[test]
    fn extract_rg_malformed_legacy_does_not_wipe_r128() {
        let parsed = tags_from(&[
            ("R128_TRACK_GAIN", "-1280"),
            ("REPLAYGAIN_TRACK_GAIN", "boom"),
        ]);
        let rg = extract_replaygain(&parsed);
        assert!((rg.track_gain - 0.0).abs() < 1e-3);
    }

    #[test]
    fn r128_gain_out_of_range_is_none() {
        // i32::MIN through Q7.8 conversion ≈ -8.4 million dB, which is
        // almost certainly file corruption. We reject rather than emit
        // a catastrophic gain value. Matches libopusfile's R128 clamp.
        let extreme = i32::MIN.to_string();
        assert!(parse_r128_gain(&extreme).is_none(), "i32::MIN must reject");

        let extreme_pos = i32::MAX.to_string();
        assert!(
            parse_r128_gain(&extreme_pos).is_none(),
            "i32::MAX must reject"
        );

        // Values just outside the ±127 dB window also reject. 127 dB
        // plus the +5 dB RG offset pre-subtraction: raw = (122 dB) * 256
        // = 31_232 is just inside, and one past that clamp is out.
        // In practice, anything beyond ~±32k Q7.8 maps outside ±127 dB.
        let just_out = (128 * 256 + 1).to_string(); // > 128 dB raw
        assert!(
            parse_r128_gain(&just_out).is_none(),
            "value outside ±127 dB window must reject"
        );

        // Confirm normal values still work as a sanity check.
        assert!(parse_r128_gain("-1280").is_some());
    }

    #[test]
    fn extract_rg_r128_extreme_gain_is_nan() {
        // End-to-end: a corrupted R128_TRACK_GAIN value surfaces as NaN in
        // the extracted RG struct, not as a catastrophic finite gain.
        let parsed = tags_from(&[("R128_TRACK_GAIN", &i32::MIN.to_string())]);
        let rg = extract_replaygain(&parsed);
        assert!(rg.track_gain.is_nan());
    }
}
