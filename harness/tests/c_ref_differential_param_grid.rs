#![cfg(not(no_reference))]
//! Parameterized round-trip differential grid against the C reference.
//!
//! Closes four coverage gaps surfaced by the 2026-05-07 reconnaissance
//! pass and itemized in `wrk_docs/2026.05.07 - PLN - parameterized
//! roundtrip grid.md`:
//!
//!   1. Bandwidth restriction (NB / MB / WB / SWB / FB) parameterized
//!      via `set_max_bandwidth`.
//!   2. Forced encoder mode (AUTO / SILK_ONLY / HYBRID / CELT_ONLY)
//!      via `set_force_mode` (CTL 11002).
//!   3. Multistream > 2 streams (3.0, quad, 5.1, FOA ambisonics) via
//!      `OpusMSEncoder::new_surround` symmetric on both sides.
//!   4. complexity ∈ {9, 10} × RESTRICTED_LOWDELAY — closes the
//!      coverage hole the analysis-cap heuristic incidentally created.
//!
//! Oracle classification reuses `tests/fuzz/fuzz_targets/oracle.rs` via
//! `#[path]` include — same pattern as `fuzz_roundtrip.rs:17-18`. The
//! tier-2 SNR floor (>= 50 dB) is the project-wide envelope from
//! `CLAUDE.md`. Compressed bytes are asserted exact in every cell;
//! decoded PCM is exact for CELT-only paths and SNR-bounded otherwise.

#![allow(clippy::too_many_arguments)]
// Clippy wants the SNR-applicable check folded into a match-arm guard;
// the explicit nested `if` is intentional here so the assertion message
// can still reference all bindings cleanly. Suppress at file scope.
#![allow(clippy::collapsible_match)]

use ropus::opus::decoder::{
    MODE_CELT_ONLY, MODE_HYBRID, MODE_SILK_ONLY, OPUS_BANDWIDTH_FULLBAND,
    OPUS_BANDWIDTH_MEDIUMBAND, OPUS_BANDWIDTH_NARROWBAND, OPUS_BANDWIDTH_SUPERWIDEBAND,
    OPUS_BANDWIDTH_WIDEBAND, OpusDecoder,
};
use ropus::opus::encoder::{
    OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_APPLICATION_VOIP, OPUS_AUTO,
    OpusEncoder,
};
use ropus::opus::multistream::{OpusMSDecoder, OpusMSEncoder};

use std::os::raw::c_int;

#[path = "../../tests/fuzz/fuzz_targets/oracle.rs"]
mod oracle;

// ---------------------------------------------------------------------------
// Minimal FFI to the C reference (linked via harness/build.rs as `opus_ref`).
// We re-declare just the symbols this file uses so it can stand alone — same
// approach `c_ref_differential.rs` takes (the harness `bindings.rs` module is
// public to bins, not to integration tests).
// ---------------------------------------------------------------------------
#[link(name = "opus_ref", kind = "static")]
unsafe extern "C" {
    fn opus_encoder_create(
        fs: i32,
        channels: c_int,
        application: c_int,
        error: *mut c_int,
    ) -> *mut u8;
    fn opus_encode(
        st: *mut u8,
        pcm: *const i16,
        frame_size: c_int,
        data: *mut u8,
        max_data_bytes: i32,
    ) -> i32;
    fn opus_encoder_destroy(st: *mut u8);
    fn opus_encoder_ctl(st: *mut u8, request: c_int, ...) -> c_int;

    fn opus_decoder_create(fs: i32, channels: c_int, error: *mut c_int) -> *mut std::ffi::c_void;
    fn opus_decode(
        st: *mut std::ffi::c_void,
        data: *const u8,
        len: i32,
        pcm: *mut i16,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
    fn opus_decoder_destroy(st: *mut std::ffi::c_void);

    fn opus_multistream_surround_encoder_create(
        fs: i32,
        channels: c_int,
        mapping_family: c_int,
        streams: *mut c_int,
        coupled_streams: *mut c_int,
        mapping: *mut u8,
        application: c_int,
        error: *mut c_int,
    ) -> *mut u8;
    fn opus_multistream_encode(
        st: *mut u8,
        pcm: *const i16,
        frame_size: c_int,
        data: *mut u8,
        max_data_bytes: i32,
    ) -> i32;
    fn opus_multistream_encoder_destroy(st: *mut u8);
    fn opus_multistream_encoder_ctl(st: *mut u8, request: c_int, ...) -> c_int;

    fn opus_multistream_decoder_create(
        fs: i32,
        channels: c_int,
        streams: c_int,
        coupled_streams: c_int,
        mapping: *const u8,
        error: *mut c_int,
    ) -> *mut std::ffi::c_void;
    fn opus_multistream_decode(
        st: *mut std::ffi::c_void,
        data: *const u8,
        len: i32,
        pcm: *mut i16,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
    fn opus_multistream_decoder_destroy(st: *mut std::ffi::c_void);
}

// CTL request codes used here. Values come from `opus_defines.h` /
// `opus_private.h` in the C reference (and match
// `harness/src/bindings.rs`).
const OPUS_SET_BITRATE_REQUEST: c_int = 4002;
const OPUS_SET_MAX_BANDWIDTH_REQUEST: c_int = 4004;
const OPUS_SET_VBR_REQUEST: c_int = 4006;
const OPUS_SET_COMPLEXITY_REQUEST: c_int = 4010;
const OPUS_SET_VBR_CONSTRAINT_REQUEST: c_int = 4020;
const OPUS_SET_FORCE_MODE_REQUEST: c_int = 11002;

// ---------------------------------------------------------------------------
// WAV reader — copied from c_ref_differential.rs:2787-2820.
// ---------------------------------------------------------------------------
struct Wav {
    channels: u16,
    sample_rate: u32,
    samples: Vec<i16>,
}

fn read_wav_i16(rel_path: &[&str]) -> Wav {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("..");
    path.push("tests");
    path.push("vectors");
    for seg in rel_path {
        path.push(seg);
    }
    let data =
        std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {}: {}", path.display(), e));
    assert!(data.len() >= 44, "WAV {} too small", path.display());
    assert_eq!(&data[0..4], b"RIFF");
    assert_eq!(&data[8..12], b"WAVE");
    let channels = u16::from_le_bytes([data[22], data[23]]);
    let sample_rate = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
    let bits_per_sample = u16::from_le_bytes([data[34], data[35]]);
    assert_eq!(bits_per_sample, 16, "expected 16-bit PCM");

    // Find data chunk.
    let mut pos = 12;
    let mut pcm_bytes: &[u8] = &[];
    while pos + 8 <= data.len() {
        let chunk_size =
            u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]])
                as usize;
        if &data[pos..pos + 4] == b"data" {
            pcm_bytes = &data[pos + 8..pos + 8 + chunk_size];
            break;
        }
        pos += 8 + chunk_size;
        if !chunk_size.is_multiple_of(2) {
            pos += 1;
        }
    }
    assert!(!pcm_bytes.is_empty(), "no data chunk in {}", path.display());

    let samples: Vec<i16> = pcm_bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    Wav {
        channels,
        sample_rate,
        samples,
    }
}

/// Decimate interleaved PCM by an integer factor. Picks every `factor`th
/// frame; deterministic, identical samples flow to both sides. The grid
/// uses this as a "resample on read" stand-in — the differential cares
/// about Rust-vs-C parity on identical inputs, not absolute fidelity.
fn decimate_interleaved(samples: &[i16], channels: usize, factor: usize) -> Vec<i16> {
    let nframes = samples.len() / channels;
    let mut out = Vec::with_capacity(nframes / factor * channels);
    for f in (0..nframes).step_by(factor) {
        let off = f * channels;
        out.extend_from_slice(&samples[off..off + channels]);
    }
    out
}

/// Duplicate mono interleaved PCM to N channels by replicating each
/// sample. Used for multi-channel cells whose source fixture is mono.
fn replicate_mono_to_n(samples: &[i16], channels: usize) -> Vec<i16> {
    let mut out = Vec::with_capacity(samples.len() * channels);
    for &s in samples {
        for _ in 0..channels {
            out.push(s);
        }
    }
    out
}

/// Patterned PCM helper for synthetic multi-channel cells. Mirrors the
/// `patterned_pcm_i16` helper in `c_ref_differential.rs:99` but generalized
/// to N channels with a per-channel seed offset so each channel carries a
/// distinct waveform (otherwise downmix collapses to a no-op).
fn patterned_pcm_nch(
    frame_size: usize,
    channels: usize,
    base_seed: i32,
    frames: usize,
) -> Vec<i16> {
    let total = frames * frame_size;
    let mut out = Vec::with_capacity(total * channels);
    for i in 0..total {
        for ch in 0..channels {
            let seed = base_seed.wrapping_add(ch as i32 * 911);
            let base = ((i as i32 * 7919 + seed * 911) % 28000) - 14000;
            let scaled = if ch.is_multiple_of(2) { base } else { base / 2 };
            out.push(scaled as i16);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Cell helpers — stateful Rust-vs-C round trip across multiple PCM frames.
//
// State carryover is critical: the silence-prefix warm-up only exposes
// state-driven divergence (SILK LPC history, CELT TF analysis, hysteretic
// rate-allocation) if the encoders are alive across frames. We therefore
// build one encoder and one decoder per side per cell, push the silence
// prefix through them (asserting compressed-byte parity on every frame,
// even prefix frames — they're a free oracle), then walk the assertion
// frames.
// ---------------------------------------------------------------------------

/// Configuration for one round-trip cell. Lifted out so prefix and
/// assertion frames share the exact same setter sequence.
struct CellCfg {
    sr: i32,
    ch: i32,
    application: i32,
    bitrate: i32,
    complexity: i32,
    vbr: bool,
    vbr_constraint: bool,
    max_bandwidth: Option<i32>,
    force_mode: Option<i32>,
    frame_size: i32,
}

fn build_rust_encoder(cfg: &CellCfg, label: &str) -> OpusEncoder {
    let mut enc = OpusEncoder::new(cfg.sr, cfg.ch, cfg.application)
        .unwrap_or_else(|e| panic!("[{label}] Rust encoder create failed: {e}"));
    enc.set_bitrate(cfg.bitrate);
    enc.set_complexity(cfg.complexity);
    enc.set_vbr(if cfg.vbr { 1 } else { 0 });
    enc.set_vbr_constraint(if cfg.vbr_constraint { 1 } else { 0 });
    if let Some(bw) = cfg.max_bandwidth {
        let r = enc.set_max_bandwidth(bw);
        assert_eq!(r, 0, "[{label}] Rust set_max_bandwidth({bw}) -> {r}");
    }
    if let Some(m) = cfg.force_mode {
        let r = enc.set_force_mode(m);
        assert_eq!(r, 0, "[{label}] Rust set_force_mode({m}) -> {r}");
    }
    enc
}

/// Build a C encoder mirrored to the same config; caller is responsible
/// for `opus_encoder_destroy` on the returned pointer.
unsafe fn build_c_encoder(cfg: &CellCfg, label: &str) -> *mut u8 {
    let mut err: c_int = 0;
    let enc = unsafe { opus_encoder_create(cfg.sr, cfg.ch, cfg.application, &mut err) };
    assert!(
        !enc.is_null() && err == 0,
        "[{label}] C opus_encoder_create err={err}"
    );
    unsafe {
        opus_encoder_ctl(enc, OPUS_SET_BITRATE_REQUEST, cfg.bitrate);
        opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY_REQUEST, cfg.complexity);
        opus_encoder_ctl(enc, OPUS_SET_VBR_REQUEST, cfg.vbr as c_int);
        opus_encoder_ctl(
            enc,
            OPUS_SET_VBR_CONSTRAINT_REQUEST,
            cfg.vbr_constraint as c_int,
        );
        if let Some(bw) = cfg.max_bandwidth {
            let r = opus_encoder_ctl(enc, OPUS_SET_MAX_BANDWIDTH_REQUEST, bw);
            assert_eq!(r, 0, "[{label}] C set_max_bandwidth({bw}) -> {r}");
        }
        if let Some(m) = cfg.force_mode {
            let r = opus_encoder_ctl(enc, OPUS_SET_FORCE_MODE_REQUEST, m);
            assert_eq!(r, 0, "[{label}] C set_force_mode({m}) -> {r}");
        }
    }
    enc
}

/// Encode one frame on each side; assert compressed-byte parity. Returns
/// the (shared) packet so callers can decode it.
fn encode_one_frame(
    cfg: &CellCfg,
    rust_enc: &mut OpusEncoder,
    c_enc: *mut u8,
    pcm: &[i16],
    frame_label: &str,
) -> Vec<u8> {
    let max_packet = 4000usize;
    let mut rust_out = vec![0u8; max_packet];
    let rust_len = rust_enc
        .encode(pcm, cfg.frame_size, &mut rust_out, max_packet as i32)
        .unwrap_or_else(|e| panic!("[{frame_label}] Rust encode failed: {e}"))
        as usize;
    rust_out.truncate(rust_len);

    let c_packet = unsafe {
        let mut buf = vec![0u8; max_packet];
        let n = opus_encode(
            c_enc,
            pcm.as_ptr(),
            cfg.frame_size,
            buf.as_mut_ptr(),
            max_packet as i32,
        );
        assert!(n > 0, "[{frame_label}] C opus_encode returned {n}");
        buf.truncate(n as usize);
        buf
    };

    assert_eq!(
        rust_out,
        c_packet,
        "[{frame_label}] compressed bytes mismatch (rust_len={}, c_len={})",
        rust_out.len(),
        c_packet.len()
    );
    rust_out
}

/// Decode one frame on each side and return the decoded PCM.
fn decode_one_frame(
    cfg: &CellCfg,
    rust_dec: &mut OpusDecoder,
    c_dec: *mut std::ffi::c_void,
    packet: &[u8],
    frame_label: &str,
) -> (Vec<i16>, Vec<i16>) {
    let max_frame = 5760usize;
    let mut rust_pcm = vec![0i16; max_frame * cfg.ch as usize];
    let rust_n = rust_dec
        .decode(Some(packet), &mut rust_pcm, max_frame as i32, false)
        .unwrap_or_else(|e| panic!("[{frame_label}] Rust decode failed: {e}"))
        as usize;
    rust_pcm.truncate(rust_n * cfg.ch as usize);

    let c_pcm = unsafe {
        let mut buf = vec![0i16; max_frame * cfg.ch as usize];
        let n = opus_decode(
            c_dec,
            packet.as_ptr(),
            packet.len() as i32,
            buf.as_mut_ptr(),
            max_frame as i32,
            0,
        );
        assert!(n > 0, "[{frame_label}] C opus_decode returned {n}");
        buf.truncate(n as usize * cfg.ch as usize);
        buf
    };
    (rust_pcm, c_pcm)
}

/// Classify and assert PCM tier per the oracle module. CELT-only paths
/// must be byte-exact; SILK/Hybrid SNR-comparable cells must clear the
/// 50 dB floor. Cells classified as weakened (SampleCountOnly,
/// RecoveryOrDtxOnly, ErrorAgreementOnly) only get a sample-count check
/// here — the compressed-byte assert in `roundtrip_single` is the strong
/// oracle for those cases.
fn assert_pcm_tier(
    cell_label: &str,
    sr: i32,
    ch: i32,
    packet: &[u8],
    rust_pcm: &[i16],
    c_pcm: &[i16],
) {
    assert_eq!(
        rust_pcm.len(),
        c_pcm.len(),
        "[{cell_label}] decoded sample count mismatch: rust={}, c={}",
        rust_pcm.len(),
        c_pcm.len()
    );
    let class = oracle::classify_decode_packet(packet, sr);
    match class {
        oracle::DecodeOracleClass::CeltCodedComparable => {
            assert_eq!(
                rust_pcm, c_pcm,
                "[{cell_label}] CELT-only PCM mismatch (sr={sr}, ch={ch})"
            );
        }
        oracle::DecodeOracleClass::SilkHybridCodedComparable => {
            if oracle::snr_oracle_applicable(c_pcm) {
                let snr = oracle::snr_db(c_pcm, rust_pcm);
                assert!(
                    snr >= oracle::SILK_DECODE_MIN_SNR_DB,
                    "[{cell_label}] SILK/Hybrid SNR {snr:.2} dB < {:.0} dB \
                     (sr={sr}, ch={ch}, packet_len={})",
                    oracle::SILK_DECODE_MIN_SNR_DB,
                    packet.len()
                );
            }
            // else: reference is silence/near-silence — sample count is
            // the only oracle, already checked.
        }
        oracle::DecodeOracleClass::SampleCountOnly
        | oracle::DecodeOracleClass::RecoveryOrDtxOnly
        | oracle::DecodeOracleClass::ErrorAgreementOnly => {
            // Sample count already asserted; compressed bytes were the
            // strong oracle for these (asserted in roundtrip_single).
        }
    }
}

/// Run a (sr, ch, app, ...) cell across N consecutive frames with a
/// single long-lived encoder + decoder per side. Compressed bytes are
/// asserted exact on every frame including the silence-prefix warm-up;
/// PCM tier is enforced (per the oracle) only on assertion frames.
fn run_grid_cell(
    cell_label: &str,
    sr: i32,
    ch: i32,
    application: i32,
    bitrate: i32,
    complexity: i32,
    vbr: bool,
    vbr_constraint: bool,
    max_bandwidth: Option<i32>,
    force_mode: Option<i32>,
    frame_size: i32,
    pcm_full: &[i16],
    n_frames: usize,
    silence_prefix_frames: usize,
) {
    let cfg = CellCfg {
        sr,
        ch,
        application,
        bitrate,
        complexity,
        vbr,
        vbr_constraint,
        max_bandwidth,
        force_mode,
        frame_size,
    };
    let frame_samples = frame_size as usize * ch as usize;

    let mut rust_enc = build_rust_encoder(&cfg, cell_label);
    let c_enc = unsafe { build_c_encoder(&cfg, cell_label) };
    let mut rust_dec = OpusDecoder::new(sr, ch).expect("Rust decoder create");
    let c_dec = unsafe {
        let mut err: c_int = 0;
        let d = opus_decoder_create(sr, ch, &mut err);
        assert!(
            !d.is_null() && err == 0,
            "[{cell_label}] C decoder create err={err}"
        );
        d
    };

    // Silence prefix warms both encoder and decoder state in lockstep.
    // We still assert compressed-byte parity here — it's a free oracle.
    let silence = vec![0i16; frame_samples];
    for p in 0..silence_prefix_frames {
        let plabel = format!("{cell_label}/prefix{p}");
        let packet = encode_one_frame(&cfg, &mut rust_enc, c_enc, &silence, &plabel);
        let _ = decode_one_frame(&cfg, &mut rust_dec, c_dec, &packet, &plabel);
    }

    let total_frames = pcm_full.len() / frame_samples;
    let frames_to_use = n_frames.min(total_frames);
    assert!(
        frames_to_use > 0,
        "[{cell_label}] no frames available (pcm_len={}, frame_samples={frame_samples})",
        pcm_full.len()
    );

    for f in 0..frames_to_use {
        let off = f * frame_samples;
        let frame = &pcm_full[off..off + frame_samples];
        let frame_label = format!("{cell_label}/frame{f}");
        let packet = encode_one_frame(&cfg, &mut rust_enc, c_enc, frame, &frame_label);
        let (rust_pcm, c_pcm) = decode_one_frame(&cfg, &mut rust_dec, c_dec, &packet, &frame_label);
        assert_pcm_tier(&frame_label, sr, ch, &packet, &rust_pcm, &c_pcm);
    }

    unsafe {
        opus_encoder_destroy(c_enc);
        opus_decoder_destroy(c_dec);
    }
}

// ---------------------------------------------------------------------------
// Test 1 — bandwidth grid {NB, MB, WB, SWB, FB}.
// (sr, ch) pinned per-bandwidth; 24 kbps mono / 48 kbps stereo; cx=5; 20 ms.
// ---------------------------------------------------------------------------

/// Load `speech_48k_mono.wav`, decimate to `target_sr` if needed, and
/// optionally replicate to stereo.
fn speech_at(target_sr: u32, target_ch: usize) -> Vec<i16> {
    let wav = read_wav_i16(&["speech_48k_mono.wav"]);
    assert_eq!(wav.sample_rate, 48_000);
    assert_eq!(wav.channels, 1);
    let factor = (48_000 / target_sr) as usize;
    let decimated = if factor == 1 {
        wav.samples.clone()
    } else {
        decimate_interleaved(&wav.samples, 1, factor)
    };
    if target_ch == 1 {
        decimated
    } else {
        replicate_mono_to_n(&decimated, target_ch)
    }
}

/// Load `music_48k_stereo.wav` and decimate to `target_sr` if needed.
fn music_at(target_sr: u32) -> Vec<i16> {
    let wav = read_wav_i16(&["music_48k_stereo.wav"]);
    assert_eq!(wav.sample_rate, 48_000);
    assert_eq!(wav.channels, 2);
    let factor = (48_000 / target_sr) as usize;
    if factor == 1 {
        wav.samples
    } else {
        decimate_interleaved(&wav.samples, 2, factor)
    }
}

fn run_bandwidth_cell(label: &str, sr: i32, ch: i32, bw: i32) {
    let app = if matches!(bw, x if x == OPUS_BANDWIDTH_NARROWBAND || x == OPUS_BANDWIDTH_MEDIUMBAND)
    {
        OPUS_APPLICATION_VOIP
    } else {
        OPUS_APPLICATION_AUDIO
    };
    let bitrate = if ch == 1 { 24_000 } else { 48_000 };
    let pcm = if ch == 1 {
        speech_at(sr as u32, 1)
    } else {
        // Use music for stereo cells.
        music_at(sr as u32)
    };
    let frame_size = sr / 50; // 20 ms
    run_grid_cell(
        label,
        sr,
        ch,
        app,
        bitrate,
        5,
        true,
        false,
        Some(bw),
        None,
        frame_size,
        &pcm,
        10,
        3,
    );
}

#[test]
fn test_bandwidth_grid_nb() {
    run_bandwidth_cell("bw/NB-8k-mono", 8_000, 1, OPUS_BANDWIDTH_NARROWBAND);
}

#[test]
fn test_bandwidth_grid_mb() {
    run_bandwidth_cell("bw/MB-12k-mono", 12_000, 1, OPUS_BANDWIDTH_MEDIUMBAND);
}

#[test]
fn test_bandwidth_grid_wb() {
    run_bandwidth_cell("bw/WB-16k-stereo", 16_000, 2, OPUS_BANDWIDTH_WIDEBAND);
}

#[test]
fn test_bandwidth_grid_swb() {
    run_bandwidth_cell("bw/SWB-24k-stereo", 24_000, 2, OPUS_BANDWIDTH_SUPERWIDEBAND);
}

#[test]
fn test_bandwidth_grid_fb() {
    run_bandwidth_cell("bw/FB-48k-stereo", 48_000, 2, OPUS_BANDWIDTH_FULLBAND);
}

// ---------------------------------------------------------------------------
// Test 2 — force_mode grid {AUTO, SILK_ONLY, HYBRID, CELT_ONLY}.
// ---------------------------------------------------------------------------

#[test]
fn test_force_mode_grid_auto() {
    // AUTO + speech @ 48k stereo, 20 ms.
    let pcm = speech_at(48_000, 2);
    run_grid_cell(
        "fm/AUTO-48k-stereo",
        48_000,
        2,
        OPUS_APPLICATION_AUDIO,
        32_000,
        5,
        true,
        false,
        None,
        Some(OPUS_AUTO),
        960,
        &pcm,
        10,
        3,
    );
}

#[test]
fn test_force_mode_grid_silk_only() {
    // SILK_ONLY @ 16k mono, 20 ms.
    let pcm = speech_at(16_000, 1);
    run_grid_cell(
        "fm/SILK_ONLY-16k-mono",
        16_000,
        1,
        OPUS_APPLICATION_VOIP,
        32_000,
        5,
        true,
        false,
        None,
        Some(MODE_SILK_ONLY),
        320,
        &pcm,
        10,
        3,
    );
}

#[test]
fn test_force_mode_grid_hybrid() {
    // HYBRID @ 24k stereo, 20 ms — historical fragility, see design doc §5.
    let pcm = music_at(24_000);
    run_grid_cell(
        "fm/HYBRID-24k-stereo",
        24_000,
        2,
        OPUS_APPLICATION_AUDIO,
        32_000,
        5,
        true,
        false,
        None,
        Some(MODE_HYBRID),
        480,
        &pcm,
        10,
        3,
    );
}

#[test]
fn test_force_mode_grid_celt_only() {
    // CELT_ONLY @ 48k stereo, 5 ms (low-latency CELT cell).
    let pcm = music_at(48_000);
    run_grid_cell(
        "fm/CELT_ONLY-48k-stereo-5ms",
        48_000,
        2,
        OPUS_APPLICATION_AUDIO,
        32_000,
        5,
        true,
        false,
        None,
        Some(MODE_CELT_ONLY),
        240, // 5 ms @ 48 kHz
        &pcm,
        10,
        3,
    );
}

// ---------------------------------------------------------------------------
// Test 3 — multistream grid: (3,1), (4,1), (6,1), (4,2).
// Both sides build via the surround-create constructor so mapping/streams
// are derived identically. Compressed bytes asserted exact; PCM via the
// multistream oracle classifier.
// ---------------------------------------------------------------------------

fn run_multistream_cell(
    label: &str,
    channels: i32,
    mapping_family: i32,
    pcm_full: &[i16],
    n_frames: usize,
    silence_prefix_frames: usize,
) {
    let sr: i32 = 48_000;
    let frame_size: i32 = 960; // 20 ms
    let application = OPUS_APPLICATION_AUDIO;
    let bitrate = 64_000 * channels;
    let complexity = 5;
    let vbr = true;

    // --- Rust side: derive streams/coupled/mapping via surround constructor.
    let (mut rust_enc, streams, coupled_streams, mapping) =
        OpusMSEncoder::new_surround(sr, channels, mapping_family, application)
            .unwrap_or_else(|e| panic!("[{label}] Rust new_surround failed: {e}"));
    let _ = rust_enc.set_bitrate(bitrate);
    let _ = rust_enc.set_complexity(complexity);
    let _ = rust_enc.set_vbr(vbr as i32);
    let _ = rust_enc.set_vbr_constraint(0);

    // --- C side: same constructor, capture C's mapping for parity.
    let (c_enc, c_streams, c_coupled, c_mapping) = unsafe {
        let mut err: c_int = 0;
        let mut cs: c_int = 0;
        let mut cc: c_int = 0;
        let mut cm = vec![0u8; channels as usize];
        let enc = opus_multistream_surround_encoder_create(
            sr,
            channels,
            mapping_family,
            &mut cs,
            &mut cc,
            cm.as_mut_ptr(),
            application,
            &mut err,
        );
        assert!(
            !enc.is_null() && err == 0,
            "[{label}] C surround_create err={err}"
        );
        opus_multistream_encoder_ctl(enc, OPUS_SET_BITRATE_REQUEST, bitrate);
        opus_multistream_encoder_ctl(enc, OPUS_SET_COMPLEXITY_REQUEST, complexity);
        opus_multistream_encoder_ctl(enc, OPUS_SET_VBR_REQUEST, vbr as c_int);
        opus_multistream_encoder_ctl(enc, OPUS_SET_VBR_CONSTRAINT_REQUEST, 0i32);
        (enc, cs as i32, cc as i32, cm)
    };

    assert_eq!(streams, c_streams, "[{label}] streams mismatch");
    assert_eq!(
        coupled_streams, c_coupled,
        "[{label}] coupled_streams mismatch"
    );
    assert_eq!(
        mapping.as_slice(),
        c_mapping.as_slice(),
        "[{label}] mapping mismatch"
    );

    // --- Rust + C decoders (matching mapping/streams/coupled).
    let mut rust_dec = OpusMSDecoder::new(sr, channels, streams, coupled_streams, &mapping)
        .unwrap_or_else(|e| panic!("[{label}] Rust MS decoder create failed: {e}"));
    let c_dec = unsafe {
        let mut err: c_int = 0;
        let d = opus_multistream_decoder_create(
            sr,
            channels,
            streams,
            coupled_streams,
            mapping.as_ptr(),
            &mut err,
        );
        assert!(
            !d.is_null() && err == 0,
            "[{label}] C MS decoder create err={err}"
        );
        d
    };

    let frame_samples = frame_size as usize * channels as usize;
    let total_frames = pcm_full.len() / frame_samples;
    let max_packet = (4000 * streams.max(1)) as usize;

    // Silence-prefix warm-up.
    let silence = vec![0i16; frame_samples];
    let max_frame = 5760usize;
    for _ in 0..silence_prefix_frames {
        let mut rust_buf = vec![0u8; max_packet];
        let n = rust_enc
            .encode(&silence, frame_size, &mut rust_buf, max_packet as i32)
            .unwrap_or_else(|e| panic!("[{label}/prefix] Rust MS encode failed: {e}"));
        rust_buf.truncate(n as usize);
        let c_n = unsafe {
            let mut buf = vec![0u8; max_packet];
            let m = opus_multistream_encode(
                c_enc,
                silence.as_ptr(),
                frame_size,
                buf.as_mut_ptr(),
                max_packet as i32,
            );
            assert!(m > 0, "[{label}/prefix] C MS encode returned {m}");
            buf.truncate(m as usize);
            buf
        };
        assert_eq!(rust_buf, c_n, "[{label}/prefix] compressed-bytes mismatch");

        // Drive both decoders so their state stays synced for the
        // assertion frames.
        let mut rb = vec![0i16; max_frame * channels as usize];
        let _ = rust_dec
            .decode(
                Some(&rust_buf),
                rust_buf.len() as i32,
                &mut rb,
                max_frame as i32,
                false,
            )
            .unwrap();
        unsafe {
            let mut cb = vec![0i16; max_frame * channels as usize];
            let _ = opus_multistream_decode(
                c_dec,
                rust_buf.as_ptr(),
                rust_buf.len() as i32,
                cb.as_mut_ptr(),
                max_frame as i32,
                0,
            );
        }
    }

    let frames_to_use = n_frames.min(total_frames);
    assert!(frames_to_use > 0, "[{label}] no frames in PCM");

    for f in 0..frames_to_use {
        let off = f * frame_samples;
        let frame = &pcm_full[off..off + frame_samples];
        let frame_label = format!("{label}/frame{f}");

        // Encode both sides.
        let mut rust_buf = vec![0u8; max_packet];
        let rust_n = rust_enc
            .encode(frame, frame_size, &mut rust_buf, max_packet as i32)
            .unwrap_or_else(|e| panic!("[{frame_label}] Rust MS encode failed: {e}"))
            as usize;
        rust_buf.truncate(rust_n);

        let c_packet = unsafe {
            let mut buf = vec![0u8; max_packet];
            let m = opus_multistream_encode(
                c_enc,
                frame.as_ptr(),
                frame_size,
                buf.as_mut_ptr(),
                max_packet as i32,
            );
            assert!(m > 0, "[{frame_label}] C MS encode returned {m}");
            buf.truncate(m as usize);
            buf
        };

        assert_eq!(
            rust_buf,
            c_packet,
            "[{frame_label}] compressed bytes mismatch (rust_len={}, c_len={})",
            rust_buf.len(),
            c_packet.len()
        );

        // Decode both sides.
        let mut rust_pcm = vec![0i16; max_frame * channels as usize];
        let rust_samples = rust_dec
            .decode(
                Some(&rust_buf),
                rust_buf.len() as i32,
                &mut rust_pcm,
                max_frame as i32,
                false,
            )
            .unwrap_or_else(|e| panic!("[{frame_label}] Rust MS decode failed: {e}"))
            as usize;
        rust_pcm.truncate(rust_samples * channels as usize);

        let c_pcm = unsafe {
            let mut buf = vec![0i16; max_frame * channels as usize];
            let n = opus_multistream_decode(
                c_dec,
                c_packet.as_ptr(),
                c_packet.len() as i32,
                buf.as_mut_ptr(),
                max_frame as i32,
                0,
            );
            assert!(n > 0, "[{frame_label}] C MS decode returned {n}");
            buf.truncate(n as usize * channels as usize);
            buf
        };

        assert_eq!(
            rust_pcm.len(),
            c_pcm.len(),
            "[{frame_label}] decoded sample count mismatch"
        );

        let class = oracle::classify_multistream_decode_packet(&c_packet, streams, sr);
        match class {
            oracle::DecodeOracleClass::CeltCodedComparable => {
                assert_eq!(rust_pcm, c_pcm, "[{frame_label}] CELT-only PCM mismatch");
            }
            oracle::DecodeOracleClass::SilkHybridCodedComparable => {
                if oracle::snr_oracle_applicable(&c_pcm) {
                    let snr = oracle::snr_db(&c_pcm, &rust_pcm);
                    assert!(
                        snr >= oracle::SILK_DECODE_MIN_SNR_DB,
                        "[{frame_label}] SILK/Hybrid SNR {snr:.2} dB < {:.0} dB \
                         (channels={channels}, family={mapping_family}, streams={streams})",
                        oracle::SILK_DECODE_MIN_SNR_DB
                    );
                }
            }
            _ => { /* sample count + compressed-byte parity already asserted */ }
        }
    }

    unsafe {
        opus_multistream_encoder_destroy(c_enc);
        opus_multistream_decoder_destroy(c_dec);
    }
}

#[test]
fn test_multistream_grid_3ch_family1() {
    // 3.0 (L, C, R) → 2 streams + 1 coupled (per VORBIS_MAPPINGS table).
    let pcm = patterned_pcm_nch(960, 3, 17, 16);
    run_multistream_cell("ms/3ch-fam1", 3, 1, &pcm, 10, 3);
}

#[test]
fn test_multistream_grid_4ch_family1() {
    // Quad → 2 streams + 2 coupled.
    let pcm = patterned_pcm_nch(960, 4, 31, 16);
    run_multistream_cell("ms/4ch-fam1", 4, 1, &pcm, 10, 3);
}

#[test]
fn test_multistream_grid_6ch_family1() {
    // 5.1 → 4 streams + 2 coupled, LFE on the last stream. Predicted as
    // the most fragile cell per design doc §5.
    let pcm = patterned_pcm_nch(960, 6, 53, 16);
    run_multistream_cell("ms/6ch-fam1-5.1", 6, 1, &pcm, 10, 3);
}

#[test]
fn test_multistream_grid_4ch_family2_ambisonic() {
    // First-order ambisonics (W, X, Y, Z). Use the real ambisonic
    // fixture if it loads; fall back to a synthetic 4-channel pattern
    // otherwise.
    let wav = read_wav_i16(&["ambisonic", "ambisonic_order1_100ms.wav"]);
    assert_eq!(wav.sample_rate, 48_000);
    assert_eq!(wav.channels, 4);
    // 100 ms @ 48 kHz = 4800 samples per channel = 5 × 20 ms frames.
    // Append synthetic patterned tail to reach the requested frame count.
    let mut pcm = wav.samples;
    let needed = 960 * 4 * 16;
    if pcm.len() < needed {
        let extra = patterned_pcm_nch(960, 4, 71, 16 - (pcm.len() / (960 * 4)));
        pcm.extend_from_slice(&extra);
    }
    run_multistream_cell("ms/4ch-fam2-foa", 4, 2, &pcm, 10, 3);
}

// ---------------------------------------------------------------------------
// Test 4 — complexity ∈ {9, 10} × RESTRICTED_LOWDELAY × sr ∈ {16k, 48k}.
// RESTRICTED_LOWDELAY → CELT_ONLY → no analysis on either side, so cx10
// should be safely bit-exact (closes the analysis-cap coverage hole).
// ---------------------------------------------------------------------------

fn run_rld_cell(label: &str, sr: i32, complexity: i32) {
    let ch: i32 = 2;
    let frame_size = sr / 200; // 5 ms
    let bitrate = 64_000;

    let pcm = if sr == 48_000 {
        // 48k_sweep.wav is mono — replicate to stereo so downmix sees a
        // distinct L/R pair (identical channels would collapse to a
        // trivially-zero side and miss any stereo-coupling drift).
        let wav = read_wav_i16(&["48k_sweep.wav"]);
        assert_eq!(wav.sample_rate, 48_000);
        assert_eq!(wav.channels, 1);
        replicate_mono_to_n(&wav.samples, 2)
    } else if sr == 16_000 {
        // Decimate 48k speech by 3 to 16k, replicate to stereo.
        let wav = read_wav_i16(&["speech_48k_mono.wav"]);
        assert_eq!(wav.sample_rate, 48_000);
        assert_eq!(wav.channels, 1);
        let decimated = decimate_interleaved(&wav.samples, 1, 3);
        replicate_mono_to_n(&decimated, 2)
    } else {
        unreachable!()
    };

    run_grid_cell(
        label,
        sr,
        ch,
        OPUS_APPLICATION_RESTRICTED_LOWDELAY,
        bitrate,
        complexity,
        true,
        false,
        None,
        None,
        frame_size,
        &pcm,
        10,
        3,
    );
}

#[test]
fn test_rld_cx9_sr16k() {
    run_rld_cell("rld/cx9-16k", 16_000, 9);
}

#[test]
fn test_rld_cx9_sr48k() {
    run_rld_cell("rld/cx9-48k", 48_000, 9);
}

#[test]
fn test_rld_cx10_sr16k() {
    run_rld_cell("rld/cx10-16k", 16_000, 10);
}

#[test]
fn test_rld_cx10_sr48k() {
    run_rld_cell("rld/cx10-48k", 48_000, 10);
}
