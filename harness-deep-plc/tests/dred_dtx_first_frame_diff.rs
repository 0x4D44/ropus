#![cfg(not(no_reference))]
//! Stage 2 (TDD) — DRED first-frame DTX-shift differential.
//!
//! Locks F48 from the campaign HLD: in a multi-frame packet, when sub-frame
//! 0 (and possibly 1..k-1) is DTX-dropped, DRED must attach to sub-frame
//! `dtx_count` (the first non-DTX sub-frame), not to sub-frame 0. The C
//! reference does this at `opus_encoder.c:1777` via
//! `first_frame = (i==0) || (i==dtx_count)`. ropus currently uses `i==0`
//! only, so DRED rides on the doomed buffer when sub-frame 0 DTXes.
//!
//! ## Differential scope deviation (HLD §5 #13 → Stage 2 instruction §12)
//!
//! The supervisor brief asks for **both** Rust and C output packets to be
//! parsed and the DRED extension's sub-frame index compared. The current
//! `harness-deep-plc/dred_encode_shim.c::ropus_test_c_encoder_new` does
//! NOT call `OPUS_SET_DTX`, and Stage 2's editable scope is restricted to
//! `ropus/src/opus/encoder.rs` and `harness-deep-plc/tests/` — the shim is
//! out of bounds. We run the comparison Rust-side only: the assertion is that
//! the Rust encoder attaches DRED to the independently observed first non-DTX
//! sub-frame. The "matches C" comparison reduces to "matches the C contract
//! that Stage 3 is implementing" — explicit and documented. The test is
//! ignored by default because it needs optional reference/DNN assets and a
//! platform-independent DTX fixture.
//!
//! Stage 3 (or a subsequent stage) is free to add an `OPUS_SET_DTX` knob
//! to the shim and tighten this test to a true Rust-vs-C diff. The
//! current form still locks F48 against regression once Stage 3 lands.
//!
//! ## Failure semantics
//!
//! Stage 2 stubs `compute_dred_bitrate`/`estimate_dred_bitrate` with
//! `unimplemented!()`. Encoding a frame through the Rust path therefore
//! panics — the desired Stage 2 TDD signal. Once Stage 3 lands and F48
//! is wired through `encode_multiframe`, the assertion below must pass.

use ropus::dnn::embedded_weights::WEIGHTS_BLOB;
use ropus::opus::decoder::{
    PaddingInfo, opus_packet_get_samples_per_frame, opus_packet_parse_impl_with_padding,
};
use ropus::opus::dred::OpusDREDDecoder;
use ropus::opus::encoder::{OPUS_APPLICATION_VOIP as ROPUS_APP_VOIP, OpusEncoder};
use ropus::opus::repacketizer::OpusExtensionIterator;

const TARGET_FS: i32 = 48000;
const SUB_FRAME_SIZE: i32 = 960; // 20 ms @ 48 kHz.
const NB_SUB_FRAMES: i32 = 3; // 60 ms multi-frame packet.
const PACKET_FRAME_SIZE: i32 = SUB_FRAME_SIZE * NB_SUB_FRAMES;
const MAX_PACKET: usize = 2000;
const DRED_DURATION_2_5MS: i32 = 100;
const NUM_FRAMES_TO_TRY: usize = 25; // Scan multiple packets — DRED is gated.

fn require_weights() {
    assert!(
        !WEIGHTS_BLOB.is_empty(),
        "DRED DTX differential requires embedded weights; run `cargo run -p fetch-assets -- all`"
    );
}

/// Build a 60 ms PCM buffer with sub-frame 0 silent and sub-frames 1..N
/// carrying a 440 Hz tone at 18000 amplitude. Used to coax SILK into
/// emitting a DTX frame at index 0 followed by signal-bearing frames.
fn synth_multi_frame_pcm(pre_offset_samples: usize) -> Vec<i16> {
    let total = (PACKET_FRAME_SIZE as usize) * NUM_FRAMES_TO_TRY;
    let mut pcm = vec![0i16; total];
    for k in 0..NUM_FRAMES_TO_TRY {
        let base = k * PACKET_FRAME_SIZE as usize;
        // Sub-frame 0 stays silent; sub-frames 1..NB_SUB_FRAMES carry tone.
        for sf in 1..NB_SUB_FRAMES as usize {
            let sf_start = base + sf * SUB_FRAME_SIZE as usize;
            for i in 0..SUB_FRAME_SIZE as usize {
                let n = (pre_offset_samples + sf_start + i) as f64;
                let phase = 2.0 * std::f64::consts::PI * 440.0 * n / 48000.0;
                pcm[sf_start + i] = (phase.sin() * 18000.0) as i16;
            }
        }
    }
    pcm
}

/// Locate the DRED extension's sub-frame index inside `packet`. Returns
/// `Some((nb_frames, ext_frame, dtx_count))` when found, else `None`. Mirrors the
/// `dred_find_payload` walk in `ropus/src/opus/dred.rs:302` but exposes
/// the per-extension `frame` field instead of the parsed payload.
fn locate_dred_subframe(packet: &[u8]) -> Option<(i32, i32, i32)> {
    use ropus::dnn::dred::DRED_EXTENSION_ID;

    let mut toc: u8 = 0;
    let mut sizes = [0i16; 48];
    let mut payload_offset: i32 = 0;
    let mut padding = PaddingInfo { offset: 0, len: 0 };
    let nb_frames = opus_packet_parse_impl_with_padding(
        packet,
        packet.len() as i32,
        false,
        &mut toc,
        &mut sizes,
        &mut payload_offset,
        None,
        Some(&mut padding),
    );
    if nb_frames < 0 || padding.len == 0 {
        return None;
    }
    let pad_start = padding.offset;
    let pad_end = pad_start + padding.len as usize;
    if pad_end > packet.len() {
        return None;
    }
    let mut iter = OpusExtensionIterator::new(&packet[pad_start..pad_end], padding.len, nb_frames);
    loop {
        let (r, ext) = iter.find(DRED_EXTENSION_ID as i32);
        if r <= 0 {
            return None;
        }
        // Match the experimental version prefix to filter out stale extensions.
        if ext.len as usize >= 2 && ext.data[0] == b'D' {
            // The decoder treats one-byte/zero-byte frame payloads as DTX
            // (`decode_multiframe`, frame_arg=None). Count the leading DTX
            // frames independently instead of assuming frame zero was dropped.
            let dtx_count = (0..nb_frames as usize)
                .take_while(|&i| sizes[i] <= 1)
                .count() as i32;
            return Some((nb_frames, ext.frame, dtx_count));
        }
    }
}

#[test]
#[ignore = "requires DNN weights and a DTX-capable differential fixture"]
fn rust_dred_attaches_to_first_non_dtx_subframe() {
    require_weights();

    let mut enc =
        OpusEncoder::new(TARGET_FS, 1, ROPUS_APP_VOIP).expect("OpusEncoder::new(48k, mono, VOIP)");
    // Configuration meant to coax DTX on the silent sub-frame and DRED on
    // the surviving sub-frames.
    assert_eq!(enc.set_bitrate(32_000), 0);
    assert_eq!(enc.set_complexity(5), 0);
    assert_eq!(enc.set_inband_fec(1), 0);
    assert_eq!(enc.set_packet_loss_perc(20), 0);
    assert_eq!(enc.set_dtx(1), 0);
    assert_eq!(enc.set_dred_duration(DRED_DURATION_2_5MS), 0);

    let pcm = synth_multi_frame_pcm(0);

    // DTX needs a sustained inactive interval before it drops a sub-frame.
    // Pre-warm this encoder so the active packets below can prove a leading
    // DTX run rather than merely observing an ordinary DRED extension.
    let silence = vec![0i16; PACKET_FRAME_SIZE as usize];
    let mut warmup_packet = vec![0u8; MAX_PACKET];
    for _ in 0..10 {
        enc.encode(
            &silence,
            PACKET_FRAME_SIZE,
            &mut warmup_packet,
            MAX_PACKET as i32,
        )
        .expect("DTX warm-up encode");
    }

    // Rust packet stream — scan up to NUM_FRAMES_TO_TRY 60 ms packets and
    // find one that (a) contains a DRED extension and (b) was emitted from
    // a multi-frame packet. Stage 2: this loop panics inside `encode`
    // because `compute_dred_bitrate`/`estimate_dred_bitrate` are stubs.
    let mut found = None;
    for k in 0..NUM_FRAMES_TO_TRY {
        let start = k * PACKET_FRAME_SIZE as usize;
        let end = start + PACKET_FRAME_SIZE as usize;
        let mut packet = vec![0u8; MAX_PACKET];
        let nb = enc
            .encode(
                &pcm[start..end],
                PACKET_FRAME_SIZE,
                &mut packet,
                MAX_PACKET as i32,
            )
            .expect("encode");
        packet.truncate(nb as usize);

        let samples_per_frame = opus_packet_get_samples_per_frame(&packet, TARGET_FS);
        let inferred_nb_frames = if samples_per_frame > 0 {
            PACKET_FRAME_SIZE / samples_per_frame
        } else {
            0
        };
        if inferred_nb_frames < 2 {
            // Single-frame packet — F48 isn't exercised here; keep scanning.
            continue;
        }
        if let Some((nb_frames, ext_frame, dtx_count)) = locate_dred_subframe(&packet) {
            if dtx_count == 0 {
                continue;
            }
            // Cross-check: parse with the high-level decoder too.
            let decoder = OpusDREDDecoder::new();
            assert!(decoder.loaded(), "DRED decoder weights not loaded");
            let dred = decoder
                .parse(&packet, 2 * TARGET_FS, TARGET_FS)
                .expect("dred parse should not error");
            eprintln!(
                "[rust DTX-first-frame] packet {k}: nb_frames={nb_frames} \
                 ext_frame={ext_frame} nb_latents={} dred_offset={}",
                dred.nb_latents, dred.dred_offset
            );
            found = Some((nb_frames, ext_frame, dtx_count));
            break;
        }
    }

    let (nb_frames, ext_frame, dtx_count) = found.expect(
        "DTX differential fixture was inconclusive: no multi-frame DRED packet with a leading DTX frame was observed",
    );

    // F48 contract: DRED must attach to the first non-DTX sub-frame.
    assert!(
        nb_frames >= 2,
        "test must observe a multi-frame packet (got nb_frames={nb_frames})"
    );
    assert!(
        dtx_count > 0 && dtx_count < nb_frames,
        "fixture must prove a leading DTX run (dtx_count={dtx_count}, nb_frames={nb_frames})"
    );
    assert_eq!(
        ext_frame, dtx_count,
        "DRED must attach to first non-DTX frame, not frame zero"
    );
}
