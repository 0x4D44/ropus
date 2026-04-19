//! Opus Table of Contents (TOC) byte decoder, per RFC 6716 §3.1.
//!
//! Every Opus packet opens with a single TOC byte that encodes three things:
//!   - `config` (5 bits) → mode + bandwidth + frame size (Table 2 of the RFC)
//!   - `stereo` (1 bit)  → 0 = mono, 1 = stereo
//!   - frame-count code `c` (2 bits):
//!     0 → 1 frame
//!     1 → 2 equal-size (CBR) frames
//!     2 → 2 variable-size (VBR) frames
//!     3 → arbitrary frame count, encoded in byte 1 bits 0-5
//!
//! This module is pure bit-twiddling on one or two bytes; no codec state, no
//! allocations. `ropusinfo --extended` uses it to print a per-packet summary.
//!
//! Scope: we decode as much as the RFC byte layout exposes. We do not validate
//! per-config consistency (e.g. whether `stereo=1` actually makes sense for
//! the encoder mode) — that is the decoder's job, not ours.
//!
//! References:
//!   - RFC 6716 §3.1, Figure 1 and Table 2

/// Opus high-level codec mode, derived from the TOC `config` field (bits 7-3).
///
/// Three variants, matching the three encoder paths libopus exposes:
///   - `SilkOnly`: SILK narrowband/mediumband/wideband
///   - `Hybrid`:   SILK low band + CELT high band for SWB/FB
///   - `CeltOnly`: CELT at any bandwidth
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusMode {
    SilkOnly,
    Hybrid,
    CeltOnly,
}

impl OpusMode {
    /// Human-readable mode tag used by `ropusinfo --extended`. Matches the
    /// conventional labels opus-tools uses so `ropusinfo | grep SILK-NB`
    /// works for users coming from `opusinfo -e`.
    pub fn label(self, bw: OpusBandwidth) -> &'static str {
        match (self, bw) {
            (OpusMode::SilkOnly, OpusBandwidth::Nb) => "SILK-NB",
            (OpusMode::SilkOnly, OpusBandwidth::Mb) => "SILK-MB",
            (OpusMode::SilkOnly, OpusBandwidth::Wb) => "SILK-WB",
            (OpusMode::Hybrid, OpusBandwidth::Swb) => "Hybrid-SWB",
            (OpusMode::Hybrid, OpusBandwidth::Fb) => "Hybrid-FB",
            (OpusMode::CeltOnly, OpusBandwidth::Nb) => "CELT-NB",
            (OpusMode::CeltOnly, OpusBandwidth::Wb) => "CELT-WB",
            (OpusMode::CeltOnly, OpusBandwidth::Swb) => "CELT-SWB",
            (OpusMode::CeltOnly, OpusBandwidth::Fb) => "CELT-FB",
            // Table 2 never produces these combinations (e.g. Hybrid-NB, SILK-FB);
            // a caller that hands us one has a bug upstream. Fall back to a
            // label that makes the mismatch visible rather than silently lying.
            _ => "?",
        }
    }
}

/// Opus audio bandwidth, derived from the TOC `config` field alongside mode
/// and frame size. Matches the RFC §2 abbreviations exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusBandwidth {
    /// Narrowband, 4 kHz
    Nb,
    /// Mediumband, 6 kHz
    Mb,
    /// Wideband, 8 kHz
    Wb,
    /// Super-wideband, 12 kHz
    Swb,
    /// Fullband, 20 kHz
    Fb,
}

impl OpusBandwidth {
    pub fn label(self) -> &'static str {
        match self {
            OpusBandwidth::Nb => "NB",
            OpusBandwidth::Mb => "MB",
            OpusBandwidth::Wb => "WB",
            OpusBandwidth::Swb => "SWB",
            OpusBandwidth::Fb => "FB",
        }
    }
}

/// Decoded TOC byte + optional second-byte frame count.
///
/// `frames` is `Some(n)` when the frame count is known from the packet alone
/// (codes 0/1/2 always; code 3 when byte 1 is available). `None` means we have
/// a code-3 packet with no byte 1 to consult — unusual but possible with a
/// truncated packet read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpusToc {
    pub config: u8,
    pub mode: OpusMode,
    pub bandwidth: OpusBandwidth,
    /// Frame duration in hundredths of a ms (so 2.5 ms → 250, 20 ms → 2000).
    /// Integer representation keeps equality comparisons simple and avoids
    /// f32 precision surprises in tests; callers format via `frame_size_ms()`.
    pub frame_size_cms: u16,
    pub stereo: bool,
    /// Frame-count code `c` (0..=3), straight from bits 1-0 of byte 0.
    pub code: u8,
    /// Number of frames in the packet. See the struct doc for the `None` case.
    pub frames: Option<u8>,
}

impl OpusToc {
    /// Pretty-print the frame size as the conventional "2.5", "5", "10", "20",
    /// "40", "60" ms the RFC tables use. Integer division when possible so
    /// "20" prints without a trailing ".0".
    pub fn frame_size_ms(&self) -> String {
        let cms = self.frame_size_cms;
        if cms.is_multiple_of(100) {
            format!("{}", cms / 100)
        } else {
            // Only the 2.5 ms case (cms=250) produces a fractional label under
            // RFC 6716 Table 2. Hardcoding the single decimal keeps the output
            // deterministic without pulling in a general-purpose formatter.
            format!("{}.{}", cms / 100, (cms % 100) / 10)
        }
    }
}

/// Decode the TOC byte + optional byte 1 of an Opus packet.
///
/// `packet` must be at least 1 byte long (the TOC byte itself). For
/// frame-count code 3 we additionally read byte 1 when it exists, to recover
/// the 6-bit frame count. A code-3 packet with only 1 byte is legal to parse
/// (the RFC says the encoder is buggy, but info tools shouldn't refuse to
/// display it) — we surface that as `frames: None`.
///
/// Returns `None` if the packet is empty.
pub fn decode_toc(packet: &[u8]) -> Option<OpusToc> {
    let byte0 = *packet.first()?;
    let config = (byte0 >> 3) & 0x1F;
    let stereo = (byte0 & 0x04) != 0;
    let code = byte0 & 0x03;

    let (mode, bandwidth, frame_size_cms) = config_table(config);

    // Frame-count decode. Codes 0/1/2 have deterministic counts; code 3 pulls
    // the count from byte 1 bits 0-5 (bits 6 and 7 are padding and VBR flags
    // respectively, which we don't surface here — `ropusinfo --extended`
    // doesn't print them).
    let frames = match code {
        0 => Some(1),
        1 | 2 => Some(2),
        3 => packet.get(1).map(|b| b & 0x3F),
        _ => unreachable!("code is masked to 2 bits"),
    };

    Some(OpusToc {
        config,
        mode,
        bandwidth,
        frame_size_cms,
        stereo,
        code,
        frames,
    })
}

/// RFC 6716 Table 2 as a function: `config` → (mode, bandwidth, frame size).
///
/// Frame size is returned in hundredths of a ms so "2.5 ms" stays integer.
/// The 32 configs tile four tiers × four bandwidths × four frame sizes, but
/// not uniformly — the Hybrid tier only has two frame sizes, so a single
/// formulaic lookup would be less readable than the flat match. Inline comments
/// tag each arm with the RFC row for auditability.
fn config_table(config: u8) -> (OpusMode, OpusBandwidth, u16) {
    use OpusBandwidth::*;
    use OpusMode::*;
    match config {
        // SILK-only, {10, 20, 40, 60} ms
        0 => (SilkOnly, Nb, 1000),
        1 => (SilkOnly, Nb, 2000),
        2 => (SilkOnly, Nb, 4000),
        3 => (SilkOnly, Nb, 6000),
        4 => (SilkOnly, Mb, 1000),
        5 => (SilkOnly, Mb, 2000),
        6 => (SilkOnly, Mb, 4000),
        7 => (SilkOnly, Mb, 6000),
        8 => (SilkOnly, Wb, 1000),
        9 => (SilkOnly, Wb, 2000),
        10 => (SilkOnly, Wb, 4000),
        11 => (SilkOnly, Wb, 6000),
        // Hybrid, {10, 20} ms
        12 => (Hybrid, Swb, 1000),
        13 => (Hybrid, Swb, 2000),
        14 => (Hybrid, Fb, 1000),
        15 => (Hybrid, Fb, 2000),
        // CELT-only, {2.5, 5, 10, 20} ms
        16 => (CeltOnly, Nb, 250),
        17 => (CeltOnly, Nb, 500),
        18 => (CeltOnly, Nb, 1000),
        19 => (CeltOnly, Nb, 2000),
        20 => (CeltOnly, Wb, 250),
        21 => (CeltOnly, Wb, 500),
        22 => (CeltOnly, Wb, 1000),
        23 => (CeltOnly, Wb, 2000),
        24 => (CeltOnly, Swb, 250),
        25 => (CeltOnly, Swb, 500),
        26 => (CeltOnly, Swb, 1000),
        27 => (CeltOnly, Swb, 2000),
        28 => (CeltOnly, Fb, 250),
        29 => (CeltOnly, Fb, 500),
        30 => (CeltOnly, Fb, 1000),
        31 => (CeltOnly, Fb, 2000),
        // `config` is produced by a 5-bit mask (`>> 3 & 0x1F`) so it is by
        // construction in 0..=31; unreachable covers the exhaustiveness
        // checker without masking real bugs.
        _ => unreachable!("config is 5 bits, already masked to 0..=31"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a TOC byte from its constituent fields. Useful for the property-
    /// style tests below that need many (config, stereo, code) combinations.
    fn toc_byte(config: u8, stereo: bool, code: u8) -> u8 {
        assert!(config < 32, "config is 5 bits");
        assert!(code < 4, "code is 2 bits");
        (config << 3) | ((stereo as u8) << 2) | code
    }

    #[test]
    fn toc_config_0_is_silk_nb_10ms() {
        let toc = decode_toc(&[toc_byte(0, false, 0)]).expect("non-empty");
        assert_eq!(toc.mode, OpusMode::SilkOnly);
        assert_eq!(toc.bandwidth, OpusBandwidth::Nb);
        assert_eq!(toc.frame_size_cms, 1000);
        assert_eq!(toc.frame_size_ms(), "10");
        assert!(!toc.stereo);
        assert_eq!(toc.frames, Some(1));
    }

    #[test]
    fn toc_config_15_is_hybrid_fb_20ms() {
        let toc = decode_toc(&[toc_byte(15, false, 0)]).expect("non-empty");
        assert_eq!(toc.mode, OpusMode::Hybrid);
        assert_eq!(toc.bandwidth, OpusBandwidth::Fb);
        assert_eq!(toc.frame_size_cms, 2000);
        assert_eq!(toc.frame_size_ms(), "20");
    }

    #[test]
    fn toc_config_19_is_celt_nb_20ms() {
        let toc = decode_toc(&[toc_byte(19, false, 0)]).expect("non-empty");
        assert_eq!(toc.mode, OpusMode::CeltOnly);
        assert_eq!(toc.bandwidth, OpusBandwidth::Nb);
        assert_eq!(toc.frame_size_cms, 2000);
    }

    #[test]
    fn toc_fixture_0x9c_is_celt_nb_20ms_stereo_code0() {
        // Hand-rolled fixture: byte 0 = 0b10011_100 = 0x9C
        //   config = 0b10011 = 19  (CELT-NB, 20 ms)
        //   stereo = 1
        //   code   = 0 → 1 frame
        // Guards against a reordered config-table row (e.g. a CELT-NB vs
        // CELT-WB swap would decode the same byte to the wrong bandwidth).
        let toc = decode_toc(&[0x9C]).expect("non-empty");
        assert_eq!(toc.config, 19);
        assert_eq!(toc.mode, OpusMode::CeltOnly);
        assert_eq!(toc.bandwidth, OpusBandwidth::Nb);
        assert_eq!(toc.frame_size_cms, 2000);
        assert_eq!(toc.frame_size_ms(), "20");
        assert!(toc.stereo);
        assert_eq!(toc.code, 0);
        assert_eq!(toc.frames, Some(1));
        assert_eq!(toc.mode.label(toc.bandwidth), "CELT-NB");
    }

    #[test]
    fn toc_config_31_is_celt_fb_20ms() {
        let toc = decode_toc(&[toc_byte(31, false, 0)]).expect("non-empty");
        assert_eq!(toc.mode, OpusMode::CeltOnly);
        assert_eq!(toc.bandwidth, OpusBandwidth::Fb);
        assert_eq!(toc.frame_size_cms, 2000);
    }

    #[test]
    fn toc_config_16_is_celt_nb_2_5ms() {
        // 2.5 ms is the only config that needs the fractional-ms formatter
        // branch; exercise it so future edits don't regress the "2.5" label.
        let toc = decode_toc(&[toc_byte(16, false, 0)]).expect("non-empty");
        assert_eq!(toc.frame_size_cms, 250);
        assert_eq!(toc.frame_size_ms(), "2.5");
    }

    #[test]
    fn toc_stereo_flag_extracted() {
        let mono = decode_toc(&[toc_byte(0, false, 0)]).expect("non-empty");
        let stereo = decode_toc(&[toc_byte(0, true, 0)]).expect("non-empty");
        assert!(!mono.stereo);
        assert!(stereo.stereo);
    }

    #[test]
    fn toc_frame_count_code_0_is_1_frame() {
        let toc = decode_toc(&[toc_byte(0, false, 0)]).expect("non-empty");
        assert_eq!(toc.code, 0);
        assert_eq!(toc.frames, Some(1));
    }

    #[test]
    fn toc_frame_count_code_1_and_2_are_2_frames() {
        let code1 = decode_toc(&[toc_byte(0, false, 1)]).expect("non-empty");
        let code2 = decode_toc(&[toc_byte(0, false, 2)]).expect("non-empty");
        assert_eq!(code1.frames, Some(2));
        assert_eq!(code2.frames, Some(2));
    }

    #[test]
    fn toc_frame_count_code_3_reads_byte1_count() {
        // Code 3 + byte 1 = 0x82 (binary 1000_0010):
        //   bits 0-5 = count   = 0b00_0010 = 2
        //   bit 6    = padding = 0
        //   bit 7    = VBR     = 1
        // The RFC bit layout is: "v p M M M M M M" reading MSB→LSB, i.e.
        //   byte1 = (vbr << 7) | (padding << 6) | frame_count
        // We only extract the frame count (low 6 bits), so 0x82 & 0x3F = 2.
        let toc = decode_toc(&[toc_byte(0, false, 3), 0x82]).expect("non-empty");
        assert_eq!(toc.code, 3);
        assert_eq!(toc.frames, Some(2));
    }

    #[test]
    fn toc_frame_count_code_3_without_byte1_is_unknown() {
        // Truncated packet (just the TOC byte) with code 3: we can't know the
        // frame count; surface that as None rather than fabricating a 1.
        let toc = decode_toc(&[toc_byte(0, false, 3)]).expect("non-empty");
        assert_eq!(toc.code, 3);
        assert!(toc.frames.is_none());
    }

    #[test]
    fn toc_empty_packet_returns_none() {
        assert!(decode_toc(&[]).is_none());
    }

    #[test]
    fn toc_labels_match_rfc_abbreviations() {
        // Cross-check the user-facing label strings against the RFC §2
        // abbreviations; `ropusinfo --extended` prints these so a typo here
        // breaks scripted grep pipelines.
        let t = decode_toc(&[toc_byte(0, false, 0)]).expect("t0");
        assert_eq!(t.mode.label(t.bandwidth), "SILK-NB");
        let t = decode_toc(&[toc_byte(8, false, 0)]).expect("t8");
        assert_eq!(t.mode.label(t.bandwidth), "SILK-WB");
        let t = decode_toc(&[toc_byte(15, false, 0)]).expect("t15");
        assert_eq!(t.mode.label(t.bandwidth), "Hybrid-FB");
        let t = decode_toc(&[toc_byte(28, false, 0)]).expect("t28");
        assert_eq!(t.mode.label(t.bandwidth), "CELT-FB");
    }
}
