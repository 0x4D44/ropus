#![no_main]
use libfuzzer_sys::fuzz_target;
use mdopus::opus::decoder::{
    opus_packet_get_bandwidth, opus_packet_get_nb_frames, opus_packet_get_samples_per_frame,
    OpusDecoder, OPUS_BANDWIDTH_FULLBAND, OPUS_BANDWIDTH_SUPERWIDEBAND,
};

#[path = "c_reference.rs"]
mod c_reference;

// Rust-only safety fuzz target for the decoder — extended with differential
// comparison for CELT-only packets (where SILK numerical divergence is not an issue).
//
// Tests: panics, OOB, infinite loops, and correctness for CELT-only paths.

const SAMPLE_RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];
const MAX_FRAME: i32 = 5760;

// CELT-only packets have bandwidth >= superwideband (TOC config >= 16).
// In these modes there's no SILK component, so numerical output must match exactly.
fn is_celt_only_packet(packet: &[u8]) -> bool {
    if packet.is_empty() {
        return false;
    }
    let bw = opus_packet_get_bandwidth(packet);
    bw == OPUS_BANDWIDTH_SUPERWIDEBAND || bw == OPUS_BANDWIDTH_FULLBAND
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }

    let sr_idx = (data[0] as usize) % SAMPLE_RATES.len();
    let sample_rate = SAMPLE_RATES[sr_idx];
    let channels = if data[1] & 1 == 0 { 1 } else { 2 };
    let packet = &data[2..];

    if packet.is_empty() {
        return;
    }

    let mut dec = match OpusDecoder::new(sample_rate, channels) {
        Ok(d) => d,
        Err(_) => return,
    };
    let max_pcm = MAX_FRAME as usize * channels as usize;
    let mut pcm = vec![0i16; max_pcm];

    // Decode the packet — must not panic
    let rust_ret = dec.decode(Some(packet), &mut pcm, MAX_FRAME, false);

    // --- Semantic invariant: decoded sample count matches packet structure ---
    if let Ok(rust_samples) = rust_ret {
        let expected_spf = opus_packet_get_samples_per_frame(packet, sample_rate);
        let expected_frames = opus_packet_get_nb_frames(packet);
        if let Ok(nf) = expected_frames {
            if nf > 0 && expected_spf > 0 {
                let expected_total = expected_spf * nf;
                assert_eq!(
                    rust_samples, expected_total,
                    "Decoded samples ({rust_samples}) != expected ({expected_total} = {expected_spf} * {nf}), \
                     sr={sample_rate}, ch={channels}"
                );
            }
        }
    }

    // --- Differential comparison for CELT-only packets ---
    if is_celt_only_packet(packet) {
        let c_ret = c_reference::c_decode(packet, sample_rate, channels);

        match (&rust_ret, &c_ret) {
            (Ok(rust_samples), Ok(c_pcm)) => {
                let rust_samples = *rust_samples as usize;
                let c_samples = c_pcm.len() / channels as usize;

                assert_eq!(
                    rust_samples, c_samples,
                    "CELT-only sample count mismatch: Rust={rust_samples}, C={c_samples}, \
                     sr={sample_rate}, ch={channels}, pkt_len={}",
                    packet.len()
                );

                let rust_slice = &pcm[..rust_samples * channels as usize];
                assert_eq!(
                    rust_slice, &c_pcm[..],
                    "CELT-only PCM mismatch: sr={sample_rate}, ch={channels}, pkt_len={}, \
                     samples={rust_samples}",
                    packet.len()
                );
            }
            (Err(_), Err(_)) => {}
            (Ok(rust_samples), Err(c_err)) => {
                panic!(
                    "CELT-only: Rust decoded ({rust_samples} samples) but C errored ({c_err}), \
                     sr={sample_rate}, ch={channels}, pkt_len={}",
                    packet.len()
                );
            }
            (Err(rust_err), Ok(c_pcm)) => {
                panic!(
                    "CELT-only: C decoded ({} samples) but Rust errored ({rust_err}), \
                     sr={sample_rate}, ch={channels}, pkt_len={}",
                    c_pcm.len() / channels as usize,
                    packet.len()
                );
            }
        }
    }

    // --- Semantic invariant: decode determinism ---
    // Decoding the same packet with a fresh decoder must produce identical output
    if rust_ret.is_ok() {
        let mut dec2 = match OpusDecoder::new(sample_rate, channels) {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut pcm2 = vec![0i16; max_pcm];
        let ret2 = dec2.decode(Some(packet), &mut pcm2, MAX_FRAME, false);
        if let (Ok(n1), Ok(n2)) = (rust_ret, ret2) {
            assert_eq!(n1, n2, "Decode determinism: sample count differs");
            assert_eq!(
                &pcm[..n1 as usize * channels as usize],
                &pcm2[..n2 as usize * channels as usize],
                "Decode determinism: PCM output differs for identical packet"
            );
        }
    }

    // Also test PLC after decode
    let plc_frame = sample_rate / 50;
    let mut plc_pcm = vec![0i16; plc_frame as usize * channels as usize];
    let _ = dec.decode(None, &mut plc_pcm, plc_frame, false);

    // Test FEC decode
    let _ = dec.decode(Some(packet), &mut pcm, MAX_FRAME, true);
});
