#![no_main]
use libfuzzer_sys::arbitrary::{self, Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use ropus::opus::decoder::{
    OpusDecoder, MODE_CELT_ONLY, MODE_HYBRID, MODE_SILK_ONLY, OPUS_BANDWIDTH_FULLBAND,
    OPUS_BANDWIDTH_NARROWBAND, OPUS_BANDWIDTH_SUPERWIDEBAND, OPUS_BANDWIDTH_WIDEBAND,
};
use ropus::opus::encoder::{
    OpusEncoder, OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_VOIP, OPUS_AUTO, OPUS_SIGNAL_MUSIC,
    OPUS_SIGNAL_VOICE,
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
// Mode-transition grammar tables.
//
// Sample rates that can reach HYBRID/CELT; 8 kHz is included so the
// SILK_ONLY-NB ↔ CELT_ONLY-NB transition is reachable, but 12 kHz and up are
// needed for SWB/FB and therefore HYBRID. RESTRICTED_LOWDELAY is omitted —
// it pins to CELT_ONLY (encoder.rs:1864-1865) and would skip the transitions
// this target is built to find. `set_application` rejects mid-stream changes
// (encoder.rs:3849-3851), so application is fixed for the whole sequence.
// --------------------------------------------------------------------------- //
const SAMPLE_RATES: [i32; 4] = [8000, 16000, 24000, 48000];
const APPLICATIONS: [i32; 2] = [OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_VOIP];

// MEDIUMBAND is intentionally absent — encoder.rs:2020-2022 skips it on the
// auto path, so it isn't user-reachable through the mode-decision logic this
// target exercises.
const BANDWIDTHS: [i32; 4] = [
    OPUS_BANDWIDTH_NARROWBAND,
    OPUS_BANDWIDTH_WIDEBAND,
    OPUS_BANDWIDTH_SUPERWIDEBAND,
    OPUS_BANDWIDTH_FULLBAND,
];

const SIGNALS: [i32; 3] = [OPUS_AUTO, OPUS_SIGNAL_VOICE, OPUS_SIGNAL_MUSIC];

const FORCED_MODES: [i32; 4] = [OPUS_AUTO, MODE_SILK_ONLY, MODE_HYBRID, MODE_CELT_ONLY];

/// Map Arbitrary bytes to a bitrate biased toward the auto-mode decision
/// boundary. The boundary depends on `voice_est`, which the analyser computes
/// from PCM content and isn't observable at fuzz-input-construction time, so
/// the centre values target the mid-voice (voice_est ≈ 64) boundary where the
/// threshold sits inside a 12 kbps window. Distribution:
///
/// - 70% within ±6 kbps of the per-config centre
/// - 20% in the WB↔SWB band (9..15 kbps), driving the SILK↔HYBRID promotion
///   gate (encoder.rs:2110-2114)
/// - 10% uniform [6 kbps, 510 kbps] safety net
///
/// Forced-mode frames bypass the auto-decision logic, so the boundary is moot
/// — they get a uniform [6 kbps, 96 kbps] range covering CELT rate-allocation
/// transitions and the hybrid SWB/FB rate split.
fn boundary_biased_bitrate(
    raw: u16,
    channels: i32,
    application: i32,
    signal: i32,
    forced_mode: i32,
) -> i32 {
    if forced_mode != OPUS_AUTO {
        return 6_000 + (raw as i32 % 90_001);
    }

    let centre_bps: i32 = match (channels, signal) {
        (1, OPUS_SIGNAL_VOICE) => 60_000, // mono voice peak (~71.6k - 8k VOIP - 4k hyst)
        (1, OPUS_SIGNAL_MUSIC) => 14_000, // mono music peak (~14.7k)
        (1, _) => 30_000,                 // mono auto (mid-voice)
        (2, OPUS_SIGNAL_VOICE) => 44_000, // stereo voice peak (~50.5k)
        (2, OPUS_SIGNAL_MUSIC) => 14_000, // stereo music peak
        (2, _) => 25_000,                 // stereo auto
        _ => 30_000,
    };
    let centre = if application == OPUS_APPLICATION_VOIP {
        centre_bps + 8_000 // VOIP threshold boost (encoder.rs:1878-1880)
    } else {
        centre_bps
    };

    let bucket = raw % 10;
    if bucket < 7 {
        let span = 12_000_i32; // ±6 kbps
        let offset = (raw as i32 % span) - span / 2;
        (centre + offset).clamp(6_000, 510_000)
    } else if bucket < 9 {
        9_000 + (raw as i32 % 6_001) // WB↔SWB band: 9..15 kbps
    } else {
        6_000 + (raw as i32 % 504_001) // uniform fallback
    }
}

const STREAM_TAG_MODE: u8 = 0xD;

/// 16-byte fingerprint for crash repro. libFuzzer keeps the raw byte input in
/// the artifact directory; this is the supplementary shape hint the panic hook
/// writes alongside. fp[15] = 0xD tags the fingerprint as mode-transition so
/// post-mortem tools can grep these out of mixed dumps.
fn input_fingerprint(input: &ModeTransitionInput) -> [u8; 16] {
    let mut fp = [0u8; 16];
    fp[0] = input.sample_rate_idx;
    fp[1] = input.channels_minus_one;
    fp[2] = input.application_idx;
    fp[3] = input.frames.len() as u8;
    let total_pcm: usize = input.frames.iter().map(|f| f.pcm_bytes.len()).sum();
    fp[4..8].copy_from_slice(&(total_pcm as u32).to_le_bytes());
    if let Some(first) = input.frames.first() {
        fp[8..10].copy_from_slice(&first.bitrate_selector.to_le_bytes());
        fp[10] = first.complexity;
        fp[11] = first.fec
            | ((first.vbr as u8) << 3)
            | ((first.dtx as u8) << 4)
            | ((first.vbr_constraint as u8) << 5);
        fp[12] = first.loss_perc;
        // max_bw_idx 0..=3 (2b) | signal_idx 0..=2 (2b) | forced_mode_idx 0..=3 (2b)
        fp[13] = (first.max_bandwidth_idx & 0x03)
            | ((first.signal_idx & 0x03) << 2)
            | ((first.forced_mode_idx & 0x03) << 4);
    }
    fp[14] = input.frame_duration_selector;
    fp[15] = STREAM_TAG_MODE;
    fp
}

// --------------------------------------------------------------------------- //
// Structured input — boundary-biased grammar for mode transitions. Application
// is fixed at construction (mid-stream `set_application` returns OPUS_BAD_ARG,
// encoder.rs:3849-3851). Frame durations are restricted to 10/20/40/60 ms;
// shorter durations force CELT_ONLY (encoder.rs:1915-1916) and would make the
// target degenerate.
// --------------------------------------------------------------------------- //
#[derive(Debug)]
struct ModeTransitionInput {
    sample_rate_idx: u8,
    channels_minus_one: u8,
    application_idx: u8,
    /// Already restricted to 2..=5 (10/20/40/60 ms) by `Arbitrary::arbitrary`.
    frame_duration_selector: u8,
    frames: Vec<FrameKnobs>,
}

#[derive(Debug)]
struct FrameKnobs {
    /// Boundary-biased bitrate selector, decoded by `boundary_biased_bitrate`.
    bitrate_selector: u16,
    max_bandwidth_idx: u8,
    signal_idx: u8,
    forced_mode_idx: u8,
    complexity: u8,
    vbr: bool,
    vbr_constraint: bool,
    fec: u8,
    dtx: bool,
    loss_perc: u8,
    pcm_bytes: Vec<u8>,
}

impl<'a> Arbitrary<'a> for ModeTransitionInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let sample_rate_idx = u.int_in_range(0..=(SAMPLE_RATES.len() as u8 - 1))?;
        let channels_minus_one = u.int_in_range(0..=1)?;
        let application_idx = u.int_in_range(0..=(APPLICATIONS.len() as u8 - 1))?;
        // Frame durations < 10 ms force CELT_ONLY (encoder.rs:1915-1916); pin
        // the selector to 2..=5 (10/20/40/60 ms). `legal_frame_duration_index`
        // takes selector % 6, so 2..=5 maps directly to those slots.
        let frame_duration_selector = u.int_in_range(2u8..=5u8)?;

        let sample_rate = SAMPLE_RATES[sample_rate_idx as usize];
        let channels = channels_minus_one as i32 + 1;
        let frame_size = frame_duration::legal_frame_size_samples_per_channel(
            sample_rate,
            frame_duration_selector,
        ) as usize;
        let pcm_bytes_needed = frame_size * channels as usize * 2;

        // 2..=12 frames — slightly tighter than fuzz_encode_multiframe's 2..=16
        // since the per-frame Arbitrary cost is higher (extra knobs) and 12
        // frames is enough cross-frame state to expose state-carry bugs.
        let n_frames = u.int_in_range(2usize..=12usize)?;

        let mut frames = Vec::with_capacity(n_frames);
        for _ in 0..n_frames {
            frames.push(FrameKnobs::arbitrary_for_shape(u, pcm_bytes_needed)?);
        }
        Ok(Self {
            sample_rate_idx,
            channels_minus_one,
            application_idx,
            frame_duration_selector,
            frames,
        })
    }
}

impl FrameKnobs {
    fn arbitrary_for_shape<'a>(
        u: &mut Unstructured<'a>,
        pcm_bytes_needed: usize,
    ) -> arbitrary::Result<Self> {
        let bitrate_selector = u.arbitrary()?;
        let max_bandwidth_idx = u.int_in_range(0..=(BANDWIDTHS.len() as u8 - 1))?;
        let signal_idx = u.int_in_range(0..=(SIGNALS.len() as u8 - 1))?;
        // 60% AUTO, 40% forced (split evenly across SILK/HYBRID/CELT). Keeps
        // the auto-decision arithmetic well-covered while still exercising the
        // transition handlers directly through `set_force_mode`.
        let forced_mode_idx = {
            let r: u8 = u.arbitrary()?;
            if r % 5 < 3 {
                0
            } else {
                1 + (r % 3)
            }
        };
        // Cap at 9 to dodge the analysis.c divergence class (Campaign 9,
        // 2026-04-19): C builds with DISABLE_FLOAT_API=off, so complexity ≥ 10
        // ∧ sr ≥ 16000 produces an AnalysisInfo on the C side that the Rust
        // port doesn't yet emit. Same dodge as fuzz_encode_multiframe.
        let complexity = u.int_in_range(0..=9)?;
        let vbr = u.arbitrary()?;
        let vbr_constraint = u.arbitrary()?;
        let fec = u.int_in_range(0..=2)?;
        let dtx = u.arbitrary()?;
        let loss_perc = u.int_in_range(0..=100)?;
        let pcm_bytes = u.bytes(pcm_bytes_needed)?.to_vec();
        Ok(Self {
            bitrate_selector,
            max_bandwidth_idx,
            signal_idx,
            forced_mode_idx,
            complexity,
            vbr,
            vbr_constraint,
            fec,
            dtx,
            loss_perc,
            pcm_bytes,
        })
    }
}

// Mode-transition differential fuzz target.
//
// Per-frame draws a bitrate biased toward the SILK/HYBRID ↔ CELT_ONLY
// auto-decision boundary, plus per-frame `max_bandwidth`, `signal`, and
// (with 40% probability) `force_mode` to drive transitions across consecutive
// frames. Compares Rust vs C reference frame-by-frame for byte-exact match.
//
// This catches state-carry bugs on the mode-transition surface — the same
// defect class as the 2026-05-07 wins (PLC short-frame `delay_buffer`,
// SILK LBRR `ec_prev_*`), but in the part of the codebase where it's most
// concentrated (mode-decision + redundancy + SILK re-init + bandwidth
// coercion).
fuzz_target!(|input: ModeTransitionInput| {
    init_panic_capture();
    CURRENT_INPUT.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.clear();
        buf.extend_from_slice(&input_fingerprint(&input));
    });

    if input.frames.is_empty() {
        return;
    }

    let sample_rate = SAMPLE_RATES[input.sample_rate_idx as usize];
    let channels = input.channels_minus_one as i32 + 1;
    let application = APPLICATIONS[input.application_idx as usize];
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

    // Per-frame configs. The `forced_mode` field carries OPUS_AUTO when the
    // frame is in auto-decision mode; both Rust and C apply it identically
    // through their respective force-mode setters in this case.
    let frame_cfgs: Vec<c_reference::CModeTransitionConfig> = input
        .frames
        .iter()
        .map(|fc| {
            let signal = SIGNALS[fc.signal_idx as usize];
            let forced_mode = FORCED_MODES[fc.forced_mode_idx as usize];
            let bitrate = boundary_biased_bitrate(
                fc.bitrate_selector,
                channels,
                application,
                signal,
                forced_mode,
            );
            c_reference::CModeTransitionConfig {
                bitrate,
                complexity: fc.complexity as i32,
                application,
                vbr: if fc.vbr { 1 } else { 0 },
                vbr_constraint: if fc.vbr_constraint { 1 } else { 0 },
                inband_fec: fc.fec as i32,
                dtx: if fc.dtx { 1 } else { 0 },
                loss_perc: fc.loss_perc as i32,
                max_bandwidth: BANDWIDTHS[fc.max_bandwidth_idx as usize],
                signal,
                forced_mode,
            }
        })
        .collect();

    // --- Rust multiframe encode ---
    let mut rust_enc = match OpusEncoder::new(sample_rate, channels, application) {
        Ok(e) => e,
        Err(_) => return,
    };

    let log_modes = std::env::var_os("ROPUS_FUZZ_LOG_MODES").is_some();
    let mut rust_packets: Vec<Vec<u8>> = Vec::with_capacity(input.frames.len());
    for (i, (pcm, cfg)) in pcm_frames.iter().zip(frame_cfgs.iter()).enumerate() {
        // CANONICAL SETTER ORDER — must match `apply_c_mode_transition_config`
        // in `/home/md/language/ropus/tests/fuzz/fuzz_targets/c_reference.rs`.
        // The mode-flip logic in `set_force_mode` (writes `user_forced_mode`)
        // is reorder-sensitive against the bandwidth/signal CTLs that feed
        // the auto-decision threshold; keep this in lockstep with the C side.
        rust_enc.set_bitrate(cfg.bitrate);
        rust_enc.set_vbr(cfg.vbr);
        rust_enc.set_vbr_constraint(cfg.vbr_constraint);
        rust_enc.set_inband_fec(cfg.inband_fec);
        rust_enc.set_dtx(cfg.dtx);
        rust_enc.set_packet_loss_perc(cfg.loss_perc);
        rust_enc.set_complexity(cfg.complexity);
        rust_enc.set_max_bandwidth(cfg.max_bandwidth);
        rust_enc.set_signal(cfg.signal);
        rust_enc.set_force_mode(cfg.forced_mode);

        let mut out = vec![0u8; 4000];
        match rust_enc.encode(pcm, frame_size, &mut out, 4000) {
            Ok(len) => {
                out.truncate(len as usize);
                rust_packets.push(out);
                if log_modes {
                    let m = rust_enc.get_mode();
                    eprintln!(
                        "[MODES] sr={sample_rate} ch={channels} app={application} \
                         frame {i}: br={} max_bw={} signal={} forced={} -> mode={}",
                        cfg.bitrate, cfg.max_bandwidth, cfg.signal, cfg.forced_mode, m
                    );
                }
            }
            Err(_) => return,
        }
    }

    // --- C reference multiframe encode ---
    let frame_refs: Vec<&[i16]> = pcm_frames.iter().map(|f| f.as_slice()).collect();
    let c_packets = match c_reference::c_encode_mode_transitions(
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
             max_bw={}, signal={}, forced_mode={}",
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
            cfg.forced_mode,
        );

        assert_eq!(
            rust_pkt,
            c_pkt,
            "Frame {i}/{} byte mismatch: \
             sr={sample_rate}, ch={channels}, app={application}, \
             frame_duration={frame_duration_label}, frame_size={frame_size}, \
             br={}, cx={}, vbr={}, vbr_constraint={}, fec={}, dtx={}, loss={}, \
             max_bw={}, signal={}, forced_mode={}, len={}",
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
            cfg.forced_mode,
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
