// Property-based tests for Opus packet format invariants.
// Uses proptest to verify TOC byte parsing, frame counts, bandwidth, and channel consistency.

use crate::opus::decoder::{
    opus_packet_get_bandwidth, opus_packet_get_nb_channels, opus_packet_get_nb_frames,
    opus_packet_get_nb_samples, opus_packet_get_samples_per_frame, OPUS_BANDWIDTH_FULLBAND,
    OPUS_BANDWIDTH_NARROWBAND, OPUS_INVALID_PACKET,
};
use proptest::prelude::*;

fn valid_sample_rates() -> impl Strategy<Value = i32> {
    prop::sample::select(&[8000_i32, 12000, 16000, 24000, 48000][..])
}

proptest! {
    // -----------------------------------------------------------------------
    // 1. For TOC codes 0, 1, 2: nb_frames * samples_per_frame == nb_samples
    // -----------------------------------------------------------------------
    #[test]
    fn prop_toc_self_consistency_codes_012(
        toc in (0u8..=255).prop_filter("not code 3", |t| t & 3 != 3),
        fs in valid_sample_rates(),
    ) {
        let code = toc & 3;

        let pkt: Vec<u8> = match code {
            0 => vec![toc],       // 1 frame, no payload needed
            1 => vec![toc],       // 2 CBR frames, no payload needed
            2 => vec![toc, 0],    // 2 VBR frames, first frame size = 0
            _ => unreachable!(),
        };

        let nb_frames = opus_packet_get_nb_frames(&pkt).unwrap();
        let spf = opus_packet_get_samples_per_frame(&pkt, fs);
        let nb_samples = opus_packet_get_nb_samples(&pkt, fs);

        // nb_samples may fail if total exceeds 120ms; if so, skip
        if let Ok(ns) = nb_samples {
            prop_assert_eq!(nb_frames * spf, ns);
        }
    }

    // -----------------------------------------------------------------------
    // 2. Code 3 packets: if get_nb_frames succeeds then the identity holds;
    //    if it fails then get_nb_samples also fails.
    // -----------------------------------------------------------------------
    #[test]
    fn prop_toc_self_consistency_code3(
        toc_base in 0u8..=255,
        second_byte in 0u8..=255,
        fs in valid_sample_rates(),
    ) {
        let toc = toc_base | 0x03; // force code 3
        let pkt = vec![toc, second_byte];

        match opus_packet_get_nb_frames(&pkt) {
            Ok(nb_frames) => {
                let spf = opus_packet_get_samples_per_frame(&pkt, fs);
                match opus_packet_get_nb_samples(&pkt, fs) {
                    Ok(ns) => prop_assert_eq!(nb_frames * spf, ns),
                    // nb_samples can fail if total > 120ms — that's fine
                    Err(_) => {}
                }
            }
            Err(_) => {
                // If get_nb_frames fails, get_nb_samples must also fail
                prop_assert!(opus_packet_get_nb_samples(&pkt, fs).is_err());
            }
        }
    }

    // -----------------------------------------------------------------------
    // 3. Bandwidth is always in the valid range [NB..FB]
    // -----------------------------------------------------------------------
    #[test]
    fn prop_bandwidth_valid_range(toc in 0u8..=255) {
        let bw = opus_packet_get_bandwidth(&[toc]);
        prop_assert!(
            bw >= OPUS_BANDWIDTH_NARROWBAND && bw <= OPUS_BANDWIDTH_FULLBAND,
            "bandwidth {} out of range for toc {:#04x}", bw, toc
        );
    }

    // -----------------------------------------------------------------------
    // 4. Bandwidth is monotonic (non-decreasing) within each mode as the
    //    bandwidth bits increase.
    // -----------------------------------------------------------------------
    #[test]
    fn prop_bandwidth_monotonic_within_mode(base_toc in 0u8..=255) {
        // Determine the mode from the base TOC byte
        let is_celt = base_toc & 0x80 != 0;
        let is_hybrid = !is_celt && (base_toc & 0x60) == 0x60;

        if is_celt {
            // CELT: bits 6:5 encode bandwidth (4 values)
            let mut prev_bw = 0i32;
            for bw_bits in 0u8..4 {
                let toc = (base_toc & 0x9F) | (bw_bits << 5);
                let bw = opus_packet_get_bandwidth(&[toc]);
                prop_assert!(bw >= prev_bw, "CELT bandwidth not monotonic at bw_bits={}", bw_bits);
                prev_bw = bw;
            }
        } else if is_hybrid {
            // Hybrid: bit 4 encodes bandwidth (2 values: SWB, FB)
            let toc_swb = base_toc & !0x10;
            let toc_fb = base_toc | 0x10;
            let bw_swb = opus_packet_get_bandwidth(&[toc_swb]);
            let bw_fb = opus_packet_get_bandwidth(&[toc_fb]);
            prop_assert!(bw_fb >= bw_swb, "Hybrid bandwidth not monotonic");
        } else {
            // SILK: bits 6:5 encode bandwidth (4 values: NB, MB, WB, (unused but valid))
            let mut prev_bw = 0i32;
            for bw_bits in 0u8..4 {
                let toc = (base_toc & 0x9F) | (bw_bits << 5);
                // Skip if this makes it hybrid (bits 6:5 == 11 with bit 7 == 0)
                if (toc & 0x60) == 0x60 {
                    continue;
                }
                let bw = opus_packet_get_bandwidth(&[toc]);
                prop_assert!(bw >= prev_bw, "SILK bandwidth not monotonic at bw_bits={}", bw_bits);
                prev_bw = bw;
            }
        }
    }

    // -----------------------------------------------------------------------
    // 5. Channel count is always 1 or 2, determined by bit 2.
    // -----------------------------------------------------------------------
    #[test]
    fn prop_channel_count_binary(toc in 0u8..=255) {
        let ch = opus_packet_get_nb_channels(&[toc]);
        prop_assert!(ch == 1 || ch == 2, "channel count {} not 1 or 2", ch);
        if toc & 0x04 != 0 {
            prop_assert_eq!(ch, 2);
        } else {
            prop_assert_eq!(ch, 1);
        }
    }

    // -----------------------------------------------------------------------
    // 6. Frame count bounds: code 0 -> 1, code 1/2 -> 2, code 3 -> 0..63.
    //    Error only possible for code 3 with packet.len() == 1.
    // -----------------------------------------------------------------------
    #[test]
    fn prop_frame_count_bounds(
        first_byte in 0u8..=255,
        extra_bytes in prop::collection::vec(any::<u8>(), 0..1275),
    ) {
        let mut pkt = vec![first_byte];
        pkt.extend_from_slice(&extra_bytes);
        // packet length is 1..=1275

        let code = first_byte & 0x3;
        match opus_packet_get_nb_frames(&pkt) {
            Ok(count) => {
                match code {
                    0 => prop_assert_eq!(count, 1),
                    1 | 2 => prop_assert_eq!(count, 2),
                    3 => {
                        prop_assert!(count >= 0 && count <= 63,
                            "code 3 frame count {} out of range", count);
                    }
                    _ => unreachable!(),
                }
            }
            Err(e) => {
                // Error should only happen for code 3 with a 1-byte packet
                prop_assert_eq!(code, 3, "unexpected error for code {}", code);
                prop_assert_eq!(pkt.len(), 1, "unexpected error for pkt len {}", pkt.len());
                prop_assert_eq!(e, OPUS_INVALID_PACKET);
            }
        }
    }

    // -----------------------------------------------------------------------
    // 7. samples_per_frame returns one of the valid Opus frame durations.
    // -----------------------------------------------------------------------
    #[test]
    fn prop_samples_per_frame_valid_durations(toc in 0u8..=255, fs in valid_sample_rates()) {
        let spf = opus_packet_get_samples_per_frame(&[toc], fs);
        let valid = [
            fs / 400,          // 2.5ms
            fs / 200,          // 5ms
            fs / 100,          // 10ms
            fs / 50,           // 20ms
            fs * 40 / 1000,    // 40ms
            fs * 60 / 1000,    // 60ms
        ];
        prop_assert!(
            valid.contains(&spf),
            "samples_per_frame {} not in valid set {:?} for toc={:#04x} fs={}",
            spf, valid, toc, fs
        );
    }
}
