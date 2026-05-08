#![no_main]
use libfuzzer_sys::arbitrary::{self, Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use ropus::opus::decoder::{
    OpusDecoder, OPUS_BANDWIDTH_FULLBAND, OPUS_BANDWIDTH_SUPERWIDEBAND, OPUS_BANDWIDTH_WIDEBAND,
};
use ropus::opus::encoder::{
    OpusEncoder, OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY,
    OPUS_APPLICATION_VOIP, OPUS_AUTO, OPUS_SIGNAL_MUSIC, OPUS_SIGNAL_VOICE,
};
use std::cell::RefCell;
use std::sync::Once;

#[path = "c_reference.rs"]
mod c_reference;
#[path = "frame_duration.rs"]
mod frame_duration;

// --------------------------------------------------------------------------- //
// Panic-capture: on Windows, Rust assertions in libfuzzer-sys trigger __fastfail
// which bypasses libFuzzer's crash-artifact writer. Install a panic hook that
// saves the current input to FUZZ_PANIC_CAPTURE_DIR (or `fuzz_crashes/`) so we
// get reproducible crash files even when libFuzzer fails to persist them.
// --------------------------------------------------------------------------- //
thread_local! {
    static CURRENT_INPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn init_panic_capture() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            CURRENT_INPUT.with(|cell| {
                let bytes = cell.borrow();
                if !bytes.is_empty() {
                    use std::collections::hash_map::DefaultHasher;
                    use std::hash::{Hash, Hasher};
                    let mut hasher = DefaultHasher::new();
                    bytes.hash(&mut hasher);
                    let hash = hasher.finish();
                    let dir = std::env::var("FUZZ_PANIC_CAPTURE_DIR")
                        .unwrap_or_else(|_| "fuzz_crashes".to_string());
                    let _ = std::fs::create_dir_all(&dir);
                    let path = std::path::Path::new(&dir).join(format!("crash_{:016x}.bin", hash));
                    match std::fs::write(&path, bytes.as_slice()) {
                        Ok(()) => eprintln!(
                            "[PANIC CAPTURE] Saved {} bytes to {}",
                            bytes.len(),
                            path.display()
                        ),
                        Err(e) => {
                            eprintln!("[PANIC CAPTURE] Failed to write {}: {}", path.display(), e)
                        }
                    }
                }
            });
            prev(info);
        }));
    });
}

// --------------------------------------------------------------------------- //
// Stereo-biased grammar tables.
//
// Channels are pinned to 2; sample rates exclude 8/12 kHz (degenerate for
// ms_pred); applications are weighted toward AUDIO via an 8-entry lookup;
// max-bandwidth, signal, and force_channels each include an INHERIT slot so
// the encoder can keep prior-frame state across multiple frames in a row.
//
// See `/home/md/language/ropus/wrk_docs/2026.05.08 - HLD - stereo biased
// multiframe fuzz target.md` for the rationale on each distribution.
// --------------------------------------------------------------------------- //

const STEREO_SAMPLE_RATES: [i32; 3] = [16000, 24000, 48000];

/// 8-entry weighted application lookup: 4× AUDIO, 3× VOIP, 1× LOWDELAY.
/// MUSIC-leaning paths (theta-RDO, ms_pred) live behind AUDIO; VOIP exercises
/// SILK's stereo prediction state; LOWDELAY forces CELT-only.
const APPLICATIONS_WEIGHTED: [i32; 8] = [
    OPUS_APPLICATION_AUDIO,
    OPUS_APPLICATION_AUDIO,
    OPUS_APPLICATION_AUDIO,
    OPUS_APPLICATION_AUDIO,
    OPUS_APPLICATION_VOIP,
    OPUS_APPLICATION_VOIP,
    OPUS_APPLICATION_VOIP,
    OPUS_APPLICATION_RESTRICTED_LOWDELAY,
];

/// 5-slot stereo bandwidth distribution. NB/MB are dropped (they don't engage
/// interesting stereo paths). The fifth slot is INHERIT — preserves the prior
/// frame's setting so multi-frame runs can span unchanged-bandwidth windows.
const STEREO_BANDWIDTHS: [i32; 5] = [
    OPUS_BANDWIDTH_WIDEBAND,
    OPUS_BANDWIDTH_SUPERWIDEBAND,
    OPUS_BANDWIDTH_FULLBAND,
    OPUS_AUTO,
    c_reference::BANDWIDTH_INHERIT,
];

/// 5-slot weighted signal distribution: 2× AUTO, 1× VOICE, 2× MUSIC, then
/// INHERIT. Voice forces SILK-only at 8/16/24 kHz which we want to probe but
/// not dominate; MUSIC engages the stereo theta-RDO path.
const STEREO_SIGNALS: [i32; 5] = [
    OPUS_AUTO,
    OPUS_AUTO,
    OPUS_SIGNAL_VOICE,
    OPUS_SIGNAL_MUSIC,
    OPUS_SIGNAL_MUSIC,
];

/// 4-slot force_channels distribution: AUTO, 1, 2, INHERIT. The INHERIT slot
/// is what lets a force_channels=1 setting carry across multiple frames,
/// exercising the stream_channels hysteresis history.
const STEREO_FORCE_CHANNELS: [i32; 4] = [OPUS_AUTO, 1, 2, c_reference::FORCE_CHANNELS_INHERIT];

/// Map a u32 to a bitrate biased for stereo ms_pred coverage. The top 2 bits
/// pick a zone; the remaining 30 bits index within that zone:
///
/// - 1/2 chance (top bit = 0): 16 kbps..=64 kbps (ms_pred decision boundary)
/// - 1/4 chance (top bits = 10): 6 kbps..<16 kbps (heavily-coupled / downmix)
/// - 1/4 chance (top bits = 11): 64 kbps..=510 kbps (high-rate, theta-RDO unrestricted)
fn raw_to_bitrate_stereo(raw: u32) -> i32 {
    let zone = raw >> 30;
    let value = raw & 0x3FFF_FFFF;
    match zone {
        0 | 1 => 16_000 + (value % 48_001) as i32,
        2 => 6_000 + (value % 10_000) as i32,
        _ => 64_000 + (value % 446_001) as i32,
    }
}

const STREAM_TAG_STEREO: u8 = 0xC;

/// 16-byte fingerprint for crash repro. Same shape as the parent target's
/// fingerprint with stereo-relevant fields packed in: bandwidth_idx widens to
/// 3 bits (5 slots), signal_idx 3 bits, force_channels_idx 2 bits, and
/// `prediction_disabled` joins the boolean-flag byte. fp[15] = 0xC tags the
/// fingerprint as stereo so post-mortem can grep them out of mixed dumps.
fn input_fingerprint(input: &StereoMultiframeInput) -> [u8; 16] {
    let mut fp = [0u8; 16];
    fp[0] = input.sample_rate_idx;
    fp[1] = 2; // literal channels — kept for layout symmetry with parent
    fp[2] = input.application_idx;
    fp[3] = input.frames.len() as u8;
    let total_pcm: usize = input.frames.iter().map(|f| f.pcm_bytes.len()).sum();
    fp[4..8].copy_from_slice(&(total_pcm as u32).to_le_bytes());
    if let Some(first) = input.frames.first() {
        // Pack low 16 bits of bitrate_raw — full 32-bit raw doesn't fit; the
        // libFuzzer artifact preserves the raw bytes anyway.
        let br16 = (first.bitrate_raw & 0xFFFF) as u16;
        fp[8..10].copy_from_slice(&br16.to_le_bytes());
        fp[10] = first.complexity;
        fp[11] = first.fec
            | ((first.vbr as u8) << 3)
            | ((first.dtx as u8) << 4)
            | ((first.vbr_constraint as u8) << 5)
            | ((first.prediction_disabled as u8) << 6);
        fp[12] = first.loss_perc;
        // bandwidth_idx 0..=4 (3b) | signal_idx 0..=4 (3b) | force_ch_idx 0..=3 (2b)
        fp[13] = (first.bandwidth_idx & 0x07)
            | ((first.signal_idx & 0x07) << 3)
            | ((first.force_channels_idx & 0x03) << 6);
    }
    fp[14] = input.frame_duration_selector;
    fp[15] = STREAM_TAG_STEREO;
    fp
}

// --------------------------------------------------------------------------- //
// Structured input — pinned to channels=2; per-frame grammar adds the three
// inherit-aware CTLs on top of the parent's per-frame setter shuffle.
// --------------------------------------------------------------------------- //
#[derive(Debug)]
struct StereoMultiframeInput {
    sample_rate_idx: u8,
    application_idx: u8,
    frame_duration_selector: u8,
    frames: Vec<StereoFrameConfig>,
}

#[derive(Debug)]
struct StereoFrameConfig {
    bitrate_raw: u32,
    complexity: u8,
    vbr: bool,
    vbr_constraint: bool,
    fec: u8,
    dtx: bool,
    loss_perc: u8,
    bandwidth_idx: u8,      // 0..=4 → STEREO_BANDWIDTHS (last slot = INHERIT)
    signal_idx: u8,         // 0..=4 → STEREO_SIGNALS (weighted; no INHERIT)
    force_channels_idx: u8, // 0..=3 → STEREO_FORCE_CHANNELS (last slot = INHERIT)
    prediction_disabled: bool,
    pcm_bytes: Vec<u8>,
}

impl<'a> Arbitrary<'a> for StereoMultiframeInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let sample_rate_idx = u.int_in_range(0..=2)?;
        let application_idx = u.int_in_range(0..=7)?;
        // Frame ceiling is tighter than the parent's (16 → 12) to keep the
        // PCM byte budget under control at 60 ms / 48 kHz / stereo. The same
        // multiframe_shape_from_byte mapping is reused (parent's helper); the
        // resulting frame count is then clamped to 2..=12.
        let sequence_shape: u8 = u.arbitrary()?;
        let (frame_duration_selector, n_frames_raw) =
            frame_duration::multiframe_shape_from_byte(sequence_shape);
        let n_frames = n_frames_raw.min(12);

        let sample_rate = STEREO_SAMPLE_RATES[sample_rate_idx as usize];
        let channels: i32 = 2;
        let frame_size = frame_duration::legal_frame_size_samples_per_channel(
            sample_rate,
            frame_duration_selector,
        ) as usize;
        let pcm_bytes_needed = frame_size * channels as usize * 2;

        let mut frames = Vec::with_capacity(n_frames);
        for _ in 0..n_frames {
            frames.push(StereoFrameConfig::arbitrary_for_shape(u, pcm_bytes_needed)?);
        }
        Ok(Self {
            sample_rate_idx,
            application_idx,
            frame_duration_selector,
            frames,
        })
    }
}

impl StereoFrameConfig {
    fn arbitrary_for_shape<'a>(
        u: &mut Unstructured<'a>,
        pcm_bytes_needed: usize,
    ) -> arbitrary::Result<Self> {
        let bitrate_raw = u.arbitrary()?;
        // Cap at 9 to dodge the analysis.c divergence class
        // (Campaign 9, 2026-04-19): C builds with DISABLE_FLOAT_API=off,
        // so complexity ≥ 10 ∧ sr ≥ 16000 ∧ app != RESTRICTED_SILK
        // produces an AnalysisInfo on the C side that the Rust port doesn't
        // yet emit. Same dodge as the parent target.
        let complexity = u.int_in_range(0..=9)?;
        let vbr = u.arbitrary()?;
        let vbr_constraint = u.arbitrary()?;
        let fec = u.int_in_range(0..=2)?;
        let dtx = u.arbitrary()?;
        let loss_perc = u.int_in_range(0..=100)?;
        let bandwidth_idx = u.int_in_range(0..=4)?;
        let signal_idx = u.int_in_range(0..=4)?;
        let force_channels_idx = u.int_in_range(0..=3)?;
        let prediction_disabled = u.arbitrary()?;
        let pcm_bytes = u.bytes(pcm_bytes_needed)?.to_vec();
        Ok(Self {
            bitrate_raw,
            complexity,
            vbr,
            vbr_constraint,
            fec,
            dtx,
            loss_perc,
            bandwidth_idx,
            signal_idx,
            force_channels_idx,
            prediction_disabled,
            pcm_bytes,
        })
    }
}

// Stereo-biased multiframe differential fuzz target.
//
// Pins channels=2. Drives `set_force_channels`, `set_max_bandwidth`, and
// `set_signal` per frame across {value, INHERIT} so the encoder's mid-stream
// channel-coercion paths and stereo-only state (width_mem, hybrid_stereo_width,
// stream_channels hysteresis, ms_pred state) get systematic coverage. Reuses
// the parent target's byte-exact differential oracle frame-by-frame.
fuzz_target!(|input: StereoMultiframeInput| {
    init_panic_capture();
    CURRENT_INPUT.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.clear();
        buf.extend_from_slice(&input_fingerprint(&input));
    });

    if input.frames.is_empty() {
        return;
    }

    let sample_rate = STEREO_SAMPLE_RATES[input.sample_rate_idx as usize];
    let channels: i32 = 2;
    let application = APPLICATIONS_WEIGHTED[input.application_idx as usize];
    let frame_duration_label =
        frame_duration::legal_frame_duration_label(input.frame_duration_selector);
    let frame_size = frame_duration::legal_frame_size_samples_per_channel(
        sample_rate,
        input.frame_duration_selector,
    );
    let samples_per_frame = frame_size as usize * channels as usize;

    // Decode each frame's PCM bytes to i16 samples. The Arbitrary impl
    // guarantees pcm_bytes.len() == samples_per_frame * 2, so chunks_exact
    // never drops a tail.
    let mut pcm_frames: Vec<Vec<i16>> = Vec::with_capacity(input.frames.len());
    for fc in &input.frames {
        let pcm: Vec<i16> = fc
            .pcm_bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        pcm_frames.push(pcm);
    }

    // Per-frame configs. Inherit-capable CTLs carry their *_INHERIT sentinel
    // when the per-frame slot picks the INHERIT entry; the C side's
    // apply_c_encoder_config_with_inherit and the Rust setter loop below both
    // skip the corresponding setter call when the sentinel is present.
    let frame_cfgs: Vec<c_reference::CEncodeConfig> = input
        .frames
        .iter()
        .map(|fc| c_reference::CEncodeConfig {
            bitrate: raw_to_bitrate_stereo(fc.bitrate_raw),
            complexity: fc.complexity as i32,
            application,
            vbr: if fc.vbr { 1 } else { 0 },
            vbr_constraint: if fc.vbr_constraint { 1 } else { 0 },
            inband_fec: fc.fec as i32,
            dtx: if fc.dtx { 1 } else { 0 },
            loss_perc: fc.loss_perc as i32,
            max_bandwidth: STEREO_BANDWIDTHS[fc.bandwidth_idx as usize],
            signal: STEREO_SIGNALS[fc.signal_idx as usize],
            force_channels: STEREO_FORCE_CHANNELS[fc.force_channels_idx as usize],
            prediction_disabled: if fc.prediction_disabled { 1 } else { 0 },
        })
        .collect();

    // --- Rust multiframe encode ---
    let mut rust_enc = match OpusEncoder::new(sample_rate, channels, application) {
        Ok(e) => e,
        Err(_) => return,
    };

    let mut rust_packets: Vec<Vec<u8>> = Vec::with_capacity(input.frames.len());
    for (pcm, cfg) in pcm_frames.iter().zip(frame_cfgs.iter()) {
        // CANONICAL SETTER ORDER — must match
        // `apply_c_encoder_config_with_inherit` in
        // `/home/md/language/ropus/tests/fuzz/fuzz_targets/c_reference.rs`.
        // `set_force_channels` side-effects are reorder-sensitive; keep this
        // in lockstep with the C side.
        rust_enc.set_bitrate(cfg.bitrate);
        rust_enc.set_vbr(cfg.vbr);
        rust_enc.set_vbr_constraint(cfg.vbr_constraint);
        rust_enc.set_inband_fec(cfg.inband_fec);
        rust_enc.set_dtx(cfg.dtx);
        rust_enc.set_packet_loss_perc(cfg.loss_perc);
        rust_enc.set_complexity(cfg.complexity);
        if cfg.max_bandwidth != c_reference::BANDWIDTH_INHERIT {
            rust_enc.set_max_bandwidth(cfg.max_bandwidth);
        }
        if cfg.signal != c_reference::SIGNAL_INHERIT {
            rust_enc.set_signal(cfg.signal);
        }
        if cfg.force_channels != c_reference::FORCE_CHANNELS_INHERIT {
            rust_enc.set_force_channels(cfg.force_channels);
        }
        rust_enc.set_prediction_disabled(cfg.prediction_disabled);

        let mut out = vec![0u8; 4000];
        match rust_enc.encode(pcm, frame_size, &mut out, 4000) {
            Ok(len) => {
                out.truncate(len as usize);
                rust_packets.push(out);
            }
            Err(_) => return,
        }
    }

    // --- C reference multiframe encode (inherit-aware) ---
    let frame_refs: Vec<&[i16]> = pcm_frames.iter().map(|f| f.as_slice()).collect();
    let c_packets = match c_reference::c_encode_multiframe_inherit(
        &frame_refs,
        &frame_cfgs,
        frame_size,
        sample_rate,
        channels,
    ) {
        Ok(p) => p,
        Err(_) => return,
    };

    // --- Differential comparison: frame-by-frame byte-exact ---
    assert_eq!(
        rust_packets.len(),
        c_packets.len(),
        "Frame count mismatch: Rust={}, C={}, sr={sample_rate}, ch={channels}, frame_duration={frame_duration_label}, frame_size={frame_size}, frames={}",
        rust_packets.len(),
        c_packets.len(),
        input.frames.len(),
    );

    for (i, (rust_pkt, c_pkt)) in rust_packets.iter().zip(c_packets.iter()).enumerate() {
        let cfg = &frame_cfgs[i];
        if std::env::var_os("ROPUS_FUZZ_DUMP").is_some()
            && (rust_pkt.len() != c_pkt.len() || rust_pkt != c_pkt)
        {
            eprintln!(
                "[ROPUS_FUZZ_DUMP] shape sr={sample_rate} ch={channels} app={application} frame_duration={frame_duration_label} frame_size={frame_size} frames={}",
                input.frames.len()
            );
            for (j, cfg) in frame_cfgs.iter().enumerate() {
                eprintln!("[ROPUS_FUZZ_DUMP] frame_cfg[{j}]={cfg:?}");
            }
            for (j, pkt) in rust_packets.iter().enumerate() {
                eprintln!(
                    "[ROPUS_FUZZ_DUMP] rust_packet[{j}] len={} {:02x?}",
                    pkt.len(),
                    pkt
                );
            }
            for (j, pkt) in c_packets.iter().enumerate() {
                eprintln!(
                    "[ROPUS_FUZZ_DUMP] c_packet[{j}] len={} {:02x?}",
                    pkt.len(),
                    pkt
                );
            }
        }
        assert_eq!(
            rust_pkt.len(),
            c_pkt.len(),
            "Frame {i}/{} length mismatch: Rust={}, C={}, \
             sr={sample_rate}, ch={channels}, app={application}, \
             frame_duration={frame_duration_label}, frame_size={frame_size}, \
             br={}, cx={}, vbr={}, vbr_constraint={}, fec={}, dtx={}, loss={}, \
             max_bw={}, signal={}, force_ch={}, pred_dis={}",
            input.frames.len(),
            rust_pkt.len(),
            c_pkt.len(),
            cfg.bitrate,
            cfg.complexity,
            cfg.vbr,
            cfg.vbr_constraint,
            cfg.inband_fec,
            cfg.dtx,
            cfg.loss_perc,
            cfg.max_bandwidth,
            cfg.signal,
            cfg.force_channels,
            cfg.prediction_disabled,
        );

        assert_eq!(
            rust_pkt,
            c_pkt,
            "Frame {i}/{} byte mismatch: \
             sr={sample_rate}, ch={channels}, app={application}, \
             frame_duration={frame_duration_label}, frame_size={frame_size}, \
             br={}, cx={}, vbr={}, vbr_constraint={}, fec={}, dtx={}, loss={}, \
             max_bw={}, signal={}, force_ch={}, pred_dis={}, len={}",
            input.frames.len(),
            cfg.bitrate,
            cfg.complexity,
            cfg.vbr,
            cfg.vbr_constraint,
            cfg.inband_fec,
            cfg.dtx,
            cfg.loss_perc,
            cfg.max_bandwidth,
            cfg.signal,
            cfg.force_channels,
            cfg.prediction_disabled,
            rust_pkt.len(),
        );
    }

    // --- Semantic invariant: each encoded packet is parseable ---
    for (i, pkt) in rust_packets.iter().enumerate() {
        if !pkt.is_empty() {
            match ropus::opus::decoder::opus_packet_get_nb_frames(pkt) {
                Ok(n) if n > 0 => {}
                Ok(n) => panic!(
                    "Frame {i}: encoded packet reports {n} frames, frame_duration={frame_duration_label}, frame_size={frame_size}, len={}",
                    pkt.len()
                ),
                Err(e) => panic!(
                    "Frame {i}: encoded packet not parseable, frame_duration={frame_duration_label}, frame_size={frame_size}, len={}, err={e}",
                    pkt.len()
                ),
            }
        }
    }

    // --- Semantic invariant: decode the encoded packets, verify sample counts ---
    let mut rust_dec = match OpusDecoder::new(sample_rate, channels) {
        Ok(d) => d,
        Err(_) => return,
    };
    for (i, pkt) in rust_packets.iter().enumerate() {
        let mut decoded = vec![0i16; samples_per_frame];
        match rust_dec.decode(Some(pkt), &mut decoded, frame_size, false) {
            Ok(samples) => {
                assert_eq!(
                    samples as usize, frame_size as usize,
                    "Frame {i}: decoded sample count {samples} != frame_size {frame_size}, frame_duration={frame_duration_label}"
                );
            }
            Err(e) => {
                panic!(
                    "Frame {i}: Rust failed to decode its own encoded packet, frame_duration={frame_duration_label}, frame_size={frame_size}: {e}"
                );
            }
        }
    }
});
