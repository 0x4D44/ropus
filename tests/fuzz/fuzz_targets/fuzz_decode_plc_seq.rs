#![cfg_attr(not(test), no_main)]
//! Multi-packet differential decode with random packet drops (PLC).
//!
//! Coverage gap this target attacks: every existing decode target operates
//! on a single packet through a fresh decoder. The 24h fuzz campaign #1
//! (closed 2026-05-02) surfaced a SILK recovery-divergence class
//! (`silk_decode_recovery_divergence_loud`) that the SNR oracle had to be
//! disabled for — it fires when malformed-but-parseable packets push both
//! decoders into recovery state, and the per-packet oracle can't
//! distinguish "legal-DTX-style 0-byte sub-frame" from "torn-bitstream
//! recovery". A multi-packet stateful sequence with PLC interleaved is the
//! natural shape to surface this differential at scale.
//!
//! Approach:
//!   1. Encode N (2..=12) frames of arbitrary PCM through ONE Rust encoder.
//!      No C-side encode — sidesteps the encoder state-divergence classes.
//!   2. Apply a per-frame drop schedule (one bit per frame).
//!   3. Replay through ONE Rust decoder + ONE C decoder; PLC on dropped
//!      frames via opus_decode(NULL, 0, ...).
//!   4. Per-frame: assert sample-count parity. CELT-only kept frames also
//!      assert PCM bit-equality. SILK/Hybrid PCM is uncomparable (same
//!      shelf as fuzz_decode); the SNR oracle is intentionally NOT applied
//!      here — this target's job is to surface decoder state-divergence,
//!      not relitigate the recovery-divergence class.

#[cfg(not(test))]
use libfuzzer_sys::arbitrary::{self, Arbitrary, Unstructured};
#[cfg(not(test))]
use libfuzzer_sys::fuzz_target;
#[cfg(not(test))]
use ropus::opus::decoder::OpusDecoder;
#[cfg(not(test))]
use ropus::opus::encoder::{
    OpusEncoder, OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY,
    OPUS_APPLICATION_VOIP,
};
#[cfg(not(test))]
use std::cell::RefCell;
#[cfg(not(test))]
use std::sync::Once;

#[cfg(not(test))]
#[path = "c_reference.rs"]
mod c_reference;
#[allow(dead_code)]
#[path = "frame_duration.rs"]
mod frame_duration;

// --------------------------------------------------------------------------- //
// Panic-capture (fingerprint-only — same pattern as fuzz_encode_multiframe).
// --------------------------------------------------------------------------- //
thread_local! {
    #[cfg(not(test))]
    static CURRENT_INPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

#[cfg(not(test))]
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
                    let _ = std::fs::write(&path, bytes.as_slice());
                }
            });
            prev(info);
        }));
    });
}

#[cfg(not(test))]
const SAMPLE_RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];
#[cfg(not(test))]
const APPLICATIONS: [i32; 3] = [
    OPUS_APPLICATION_VOIP,
    OPUS_APPLICATION_AUDIO,
    OPUS_APPLICATION_RESTRICTED_LOWDELAY,
];

pub(crate) fn plc_decode_frame_size_samples_per_channel(sample_rate: i32, drop_mask: u32) -> i32 {
    let frame_duration_selector = frame_duration::plc_seq_frame_duration_selector(drop_mask);
    frame_duration::legal_frame_size_samples_per_channel(sample_rate, frame_duration_selector)
}

pub(crate) fn plc_decode_capacity_samples(
    sample_rate: i32,
    channels: i32,
    drop_mask: u32,
) -> usize {
    plc_decode_frame_size_samples_per_channel(sample_rate, drop_mask) as usize * channels as usize
}

#[cfg(not(test))]
fn raw_to_bitrate(raw: u16) -> i32 {
    6000 + (raw as i32 % 504_001)
}

#[cfg(not(test))]
#[derive(Debug)]
struct PlcSeqInput {
    sample_rate_idx: u8,
    channels_minus_one: u8,
    application_idx: u8,
    bitrate_raw: u16,
    complexity: u8,
    vbr: bool,
    inband_fec: u8,
    loss_perc: u8,
    /// One bit per frame — drop schedule. Bits are consumed lo→hi,
    /// frame[i] dropped when bit i is set.
    drop_mask: u32,
    /// 2..=12 frames; per-frame PCM length is derived from sr/ch/duration.
    frames: Vec<Vec<u8>>,
}

#[cfg(not(test))]
impl<'a> Arbitrary<'a> for PlcSeqInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let sample_rate_idx = u.int_in_range(0..=4)?;
        let channels_minus_one = u.int_in_range(0..=1)?;
        let application_idx = u.int_in_range(0..=2)?;
        let bitrate_raw = u.arbitrary()?;
        let complexity = u.int_in_range(0..=9)?;
        let vbr = u.arbitrary()?;
        let inband_fec = u.int_in_range(0..=2)?;
        let loss_perc = u.int_in_range(0..=100)?;
        let drop_mask = u.arbitrary()?;

        let sample_rate = SAMPLE_RATES[sample_rate_idx as usize];
        let channels = channels_minus_one as i32 + 1;
        let frame_size = plc_decode_frame_size_samples_per_channel(sample_rate, drop_mask) as usize;
        let pcm_bytes_per_frame = frame_size * channels as usize * 2;

        // Budget the frame count to whatever fits in remaining input. PCM
        // bytes are the dominant cost; 60 ms at sr=48000 stereo eats
        // 11520 bytes/frame and the launcher caps -max_len at 32 KB. Without
        // this the arbitrary parse would NotEnoughData on every iteration and
        // the body would never run.
        let avail = u.len();
        let max_fit = (avail / pcm_bytes_per_frame.max(1)).min(12);
        if max_fit < 2 {
            return Err(arbitrary::Error::NotEnoughData);
        }
        let n_frames = u.int_in_range(2..=max_fit)?;

        let mut frames = Vec::with_capacity(n_frames);
        for _ in 0..n_frames {
            frames.push(u.bytes(pcm_bytes_per_frame)?.to_vec());
        }

        Ok(Self {
            sample_rate_idx,
            channels_minus_one,
            application_idx,
            bitrate_raw,
            complexity,
            vbr,
            inband_fec,
            loss_perc,
            drop_mask,
            frames,
        })
    }
}

#[cfg(not(test))]
fuzz_target!(|input: PlcSeqInput| {
    init_panic_capture();

    // Save a small fingerprint for crash repro alongside the libFuzzer artifact.
    CURRENT_INPUT.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.clear();
        buf.push(input.sample_rate_idx);
        buf.push(input.channels_minus_one);
        buf.push(input.application_idx);
        buf.extend_from_slice(&input.bitrate_raw.to_le_bytes());
        buf.push(input.complexity);
        buf.push(input.vbr as u8);
        buf.push(input.inband_fec);
        buf.push(input.loss_perc);
        buf.extend_from_slice(&input.drop_mask.to_le_bytes());
        buf.push(frame_duration::plc_seq_frame_duration_selector(
            input.drop_mask,
        ));
        buf.push(input.frames.len() as u8);
    });

    let sample_rate = SAMPLE_RATES[input.sample_rate_idx as usize];
    let channels = input.channels_minus_one as i32 + 1;
    let application = APPLICATIONS[input.application_idx as usize];
    let frame_duration_selector = frame_duration::plc_seq_frame_duration_selector(input.drop_mask);
    let frame_duration_label = frame_duration::legal_frame_duration_label(frame_duration_selector);
    let frame_size = plc_decode_frame_size_samples_per_channel(sample_rate, input.drop_mask);
    let samples_per_frame = plc_decode_capacity_samples(sample_rate, channels, input.drop_mask);

    // --- Encode N packets through ONE Rust encoder ---
    let mut enc = match OpusEncoder::new(sample_rate, channels, application) {
        Ok(e) => e,
        Err(_) => return,
    };
    enc.set_bitrate(raw_to_bitrate(input.bitrate_raw));
    enc.set_vbr(if input.vbr { 1 } else { 0 });
    enc.set_inband_fec(input.inband_fec as i32);
    enc.set_packet_loss_perc(input.loss_perc as i32);
    enc.set_complexity(input.complexity as i32);

    let mut packets: Vec<Vec<u8>> = Vec::with_capacity(input.frames.len());
    for fc in &input.frames {
        let pcm: Vec<i16> = fc
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        if pcm.len() != samples_per_frame {
            return;
        }
        let mut out = vec![0u8; 4000];
        match enc.encode(&pcm, frame_size, &mut out, 4000) {
            Ok(len) => {
                out.truncate(len as usize);
                packets.push(out);
            }
            Err(_) => return,
        }
    }
    if packets.is_empty() {
        return;
    }

    // --- Build drop schedule from the bitmask ---
    // Special case: never drop the first frame. PLC requires the decoder to
    // have already consumed at least one packet to seed its prediction
    // history; both Rust and C return error on cold-start PLC, but the
    // exact error code differs (and is genuinely "implementation-defined"
    // per RFC 6716). Sidestep the noise.
    let dropped: Vec<bool> = (0..packets.len())
        .map(|i| i > 0 && (input.drop_mask >> (i % 32)) & 1 == 1)
        .collect();

    // --- Rust decode of the sequence (one stateful decoder) ---
    let mut rust_dec = match OpusDecoder::new(sample_rate, channels) {
        Ok(d) => d,
        Err(_) => return,
    };
    let mut rust_per_frame: Vec<Result<Vec<i16>, String>> = Vec::with_capacity(packets.len());
    let max_pcm = samples_per_frame;
    for (pkt, &drop) in packets.iter().zip(dropped.iter()) {
        let mut pcm = vec![0i16; max_pcm];
        let arg = if drop { None } else { Some(pkt.as_slice()) };
        match rust_dec.decode(arg, &mut pcm, frame_size, false) {
            Ok(samples) => {
                pcm.truncate(samples as usize * channels as usize);
                rust_per_frame.push(Ok(pcm));
            }
            Err(e) => rust_per_frame.push(Err(format!("{e}"))),
        }
    }

    // --- C decode of the same sequence ---
    let pkt_refs: Vec<&[u8]> = packets.iter().map(|p| p.as_slice()).collect();
    let c_results = match c_reference::c_decode_seq_with_drops(
        &pkt_refs,
        &dropped,
        sample_rate,
        channels,
        frame_size,
    ) {
        Ok(r) => r,
        Err(_) => return,
    };

    // --- Per-frame parity ---
    assert_eq!(
        rust_per_frame.len(),
        c_results.len(),
        "Frame count mismatch in PLC seq: Rust={} C={}, frame_duration={frame_duration_label}, frame_size={frame_size}",
        rust_per_frame.len(),
        c_results.len()
    );

    for (i, ((r, c), &drop)) in rust_per_frame
        .iter()
        .zip(c_results.iter())
        .zip(dropped.iter())
        .enumerate()
    {
        match (r, c) {
            (Ok(r_pcm), Ok(c_pcm)) => {
                assert_eq!(
                    r_pcm.len(),
                    c_pcm.len(),
                    "Frame {i} PCM length mismatch: Rust={} C={}, dropped={drop}, \
                     sr={sample_rate}, ch={channels}, app={application}, \
                     frame_duration={frame_duration_label}, frame_size={frame_size}",
                    r_pcm.len(),
                    c_pcm.len()
                );
                // CELT-only frames must be bit-exact. SILK/Hybrid + PLC are
                // uncomparable here — same shelf as fuzz_decode. The
                // sample-count parity above is the only oracle for those.
                let pkt = &packets[i];
                let celt_only_pkt = !drop && !pkt.is_empty() && (pkt[0] & 0x80) != 0;
                if celt_only_pkt {
                    assert_eq!(
                        r_pcm, c_pcm,
                        "Frame {i} CELT PCM mismatch in PLC seq: \
                         sr={sample_rate}, ch={channels}, app={application}, \
                         frame_duration={frame_duration_label}, frame_size={frame_size}, \
                         drop_mask=0x{:08x}",
                        input.drop_mask
                    );
                }
            }
            (Err(_), Err(_)) => {
                // Both errored on this frame — fine. PLC + recovery state
                // can produce different error codes; that's not a finding.
            }
            (Ok(r_pcm), Err(c_err)) => {
                panic!(
                    "Frame {i} PLC seq: Rust ok ({} samples) but C errored ({c_err}), \
                     dropped={drop}, sr={sample_rate}, ch={channels}, app={application}, \
                     frame_duration={frame_duration_label}, frame_size={frame_size}, \
                     drop_mask=0x{:08x}",
                    r_pcm.len() / channels as usize,
                    input.drop_mask
                );
            }
            (Err(r_err), Ok(c_pcm)) => {
                panic!(
                    "Frame {i} PLC seq: C ok ({} samples) but Rust errored ({r_err}), \
                     dropped={drop}, sr={sample_rate}, ch={channels}, app={application}, \
                     frame_duration={frame_duration_label}, frame_size={frame_size}, \
                     drop_mask=0x{:08x}",
                    c_pcm.len() / channels as usize,
                    input.drop_mask
                );
            }
        }
    }
});
