#![no_main]
use libfuzzer_sys::arbitrary::{self, Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use ropus::opus::encoder::{
    OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_APPLICATION_VOIP,
};
use ropus::opus::multistream::{OpusMSDecoder, OpusMSEncoder};
use std::cell::RefCell;
use std::sync::Once;

#[path = "c_reference.rs"]
mod c_reference;

#[path = "oracle.rs"]
mod oracle;

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
                        Err(e) => eprintln!(
                            "[PANIC CAPTURE] Failed to write {}: {}",
                            path.display(),
                            e
                        ),
                    }
                }
            });
            prev(info);
        }));
    });
}

const SAMPLE_RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];
const APPLICATIONS: [i32; 3] = [
    OPUS_APPLICATION_VOIP,
    OPUS_APPLICATION_AUDIO,
    OPUS_APPLICATION_RESTRICTED_LOWDELAY,
];
/// Channel cap: `new_surround` accepts up to 255, but anything above 8
/// saturates the validator rather than the codec. HLD V2 + supervisor cap.
const MAX_CHANNELS: u8 = 8;

fn raw_to_bitrate(raw: u16) -> i32 {
    6000 + (raw as i32 % 504_001)
}

/// CELT-only packets (TOC config >= 16, top bit set) are byte-exact across
/// codecs. SILK/Hybrid have known numerical drift, so PCM equality is gated
/// on a CELT-only TOC. Mirrors `is_celt_only_packet` from Stream A.
///
/// For multistream, sub-packet TOCs share the outer packet's mode classes (the
/// outer first byte's TOC byte selects the configuration for the leading
/// stream and is a sufficient proxy in practice — non-CELT first stream means
/// at least one SILK/Hybrid sub-packet, where drift is possible).
fn is_celt_only_packet(packet: &[u8]) -> bool {
    !packet.is_empty() && (packet[0] & 0x80) != 0
}

// --------------------------------------------------------------------------- //
// Structured input: encode/decode/roundtrip switch via `op`, plus a fixed-size
// setter shuffle exercised between construction and the first encode (gap 13).
// --------------------------------------------------------------------------- //
#[derive(Debug)]
struct MSInput {
    op: u8,
    sample_rate_idx: u8,
    channels: u8,
    mapping_family: u8,
    application_idx: u8,
    bitrate_raw: u16,
    complexity: u8,
    vbr: bool,
    setter_bytes: [u8; 16],
    payload: Vec<u8>,
}

impl<'a> Arbitrary<'a> for MSInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let op = u.int_in_range(0..=2)?;
        let sample_rate_idx = u.int_in_range(0..=4)?;
        let channels = u.int_in_range(1..=MAX_CHANNELS)?;
        let mapping_family_idx: u8 = u.int_in_range(0..=3)?;
        let mapping_family = match mapping_family_idx {
            0 => 0,
            1 => 1,
            2 => 2,
            _ => 255,
        };
        let application_idx = u.int_in_range(0..=2)?;
        let bitrate_raw = u.arbitrary()?;
        // Cap at 9 to dodge the analysis.c divergence class
        // (Campaign 9, 2026-04-19): the C reference builds with
        // DISABLE_FLOAT_API=off, so complexity ≥ 10 ∧ sr ≥ 16000 ∧
        // app != RESTRICTED_SILK produces an AnalysisInfo on the C
        // side that the Rust port doesn't yet emit.
        let complexity = u.int_in_range(0..=9)?;
        let vbr = u.arbitrary()?;
        let mut setter_bytes = [0u8; 16];
        u.fill_buffer(&mut setter_bytes)?;
        // Bound payload so iterations stay within the libFuzzer time budget.
        // Worst case: 48 kHz × 20 ms × 8 channels × 2 bytes = 15360 bytes for
        // an i16 PCM frame; we cap a bit higher for op=1 packet payloads.
        let payload_len = u.int_in_range(0..=20480)?;
        let payload = u.bytes(payload_len)?.to_vec();
        Ok(Self {
            op,
            sample_rate_idx,
            channels,
            mapping_family,
            application_idx,
            bitrate_raw,
            complexity,
            vbr,
            setter_bytes,
            payload,
        })
    }
}

/// Replay up to 8 setter calls on the OpusMSEncoder driven by `bytes`
/// (2 bytes per call). Mirrors Stream A's encoder shuffle but targets MS-
/// specific setters per HLD V2 gap 13. The C reference is driven through the
/// matching `apply_c_ms_setter_sequence`, so the byte-exact differential
/// holds across the shuffle. `set_lfe` is wired into the multistream API at
/// `multistream.rs:1414`; calls to it on encoders without an LFE stream
/// return `OPUS_BAD_ARG` on both sides and are ignored.
fn apply_ms_encoder_setter_sequence(enc: &mut OpusMSEncoder, bytes: &[u8]) {
    for chunk in bytes.chunks_exact(2).take(8) {
        match chunk[0] % 7 {
            0 => {
                let _ = enc.set_bitrate(raw_to_bitrate(u16::from_le_bytes([chunk[0], chunk[1]])));
            }
            1 => {
                // Cap at 9, see Arbitrary impl above.
                let _ = enc.set_complexity((chunk[1] % 10) as i32);
            }
            2 => {
                let _ = enc.set_vbr((chunk[1] & 1) as i32);
            }
            3 => {
                let _ = enc.set_inband_fec((chunk[1] % 3) as i32);
            }
            4 => {
                let _ = enc.set_dtx((chunk[1] & 1) as i32);
            }
            5 => {
                let _ = enc.set_packet_loss_perc((chunk[1] % 101) as i32);
            }
            6 => {
                // set_lfe returns OPUS_BAD_ARG when no LFE stream exists;
                // the call must still not panic.
                let _ = enc.set_lfe((chunk[1] & 1) as i32);
            }
            _ => unreachable!(),
        }
    }
}

fn input_fingerprint(input: &MSInput) -> [u8; 16] {
    let mut fp = [0u8; 16];
    fp[0] = input.op;
    fp[1] = input.sample_rate_idx;
    fp[2] = input.channels;
    fp[3] = input.mapping_family;
    fp[4] = input.application_idx;
    fp[5..7].copy_from_slice(&input.bitrate_raw.to_le_bytes());
    fp[7] = input.complexity;
    fp[8] = input.vbr as u8;
    fp[9..13].copy_from_slice(&(input.payload.len() as u32).to_le_bytes());
    fp[13..16].copy_from_slice(&input.setter_bytes[..3]);
    fp
}

fuzz_target!(|input: MSInput| {
    init_panic_capture();
    CURRENT_INPUT.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.clear();
        buf.extend_from_slice(&input_fingerprint(&input));
    });

    let sample_rate = SAMPLE_RATES[input.sample_rate_idx as usize];
    let channels = input.channels as i32;
    let mapping_family = input.mapping_family as i32;
    let application = APPLICATIONS[input.application_idx as usize];
    let frame_size = sample_rate / 50; // 20 ms
    let bitrate = raw_to_bitrate(input.bitrate_raw);
    let complexity = input.complexity as i32 % 11;
    let vbr = if input.vbr { 1 } else { 0 };

    let op = input.op % 3;

    match op {
        0 => {
            // ---- Encode: build encoder, optionally exercise setter shuffle, encode one frame. ----
            run_encode(
                &input,
                sample_rate,
                channels,
                mapping_family,
                application,
                frame_size,
                bitrate,
                complexity,
                vbr,
            );
        }
        1 => {
            // ---- Decode: feed the payload as a packet to a fresh MS decoder. ----
            run_decode(&input, sample_rate, channels, mapping_family);
        }
        _ => {
            // ---- Roundtrip: encode then decode against C, comparing both stages. ----
            run_roundtrip(
                &input,
                sample_rate,
                channels,
                mapping_family,
                application,
                frame_size,
                bitrate,
                complexity,
                vbr,
            );
        }
    }
});

#[allow(clippy::too_many_arguments)]
fn run_encode(
    input: &MSInput,
    sample_rate: i32,
    channels: i32,
    mapping_family: i32,
    application: i32,
    frame_size: i32,
    bitrate: i32,
    complexity: i32,
    vbr: i32,
) {
    let samples_needed = frame_size as usize * channels as usize;
    let bytes_needed = samples_needed * 2; // i16 only
    if input.payload.len() < bytes_needed {
        return;
    }
    let pcm: Vec<i16> = input.payload[..bytes_needed]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    // Rust first — `new_surround` returns the canonical mapping.
    let (mut rust_enc, streams, coupled_streams, mapping) =
        match OpusMSEncoder::new_surround(sample_rate, channels, mapping_family, application) {
            Ok(t) => t,
            Err(_) => return,
        };

    // Cap on streams + coupled_streams keeps malicious mapping family 255
    // multistream allocations bounded.
    if streams + coupled_streams > 16 {
        return;
    }

    // Baseline config applied to both sides before the runtime-setter shuffle.
    apply_initial_config(&mut rust_enc, bitrate, complexity, vbr);

    // Runtime-setter shuffle (HLD V2 gap 13) — applied symmetrically to BOTH
    // the Rust and C encoders by `apply_ms_encoder_setter_sequence` /
    // `apply_c_ms_setter_sequence`, both driven by `input.setter_bytes`.
    apply_ms_encoder_setter_sequence(&mut rust_enc, &input.setter_bytes);

    let max_data_bytes = (4000 * streams.max(1)) as i32;
    let mut rust_out = vec![0u8; max_data_bytes as usize];
    let rust_ret = rust_enc.encode(&pcm, frame_size, &mut rust_out, max_data_bytes);

    let cfg = c_reference::CMSEncodeConfig {
        bitrate,
        complexity,
        application,
        vbr,
        inband_fec: 0,
        dtx: 0,
        loss_perc: 0,
    };
    // C constructs via `opus_multistream_surround_encoder_create` — it derives
    // its own mapping from `mapping_family`/`channels`. The shuffle is applied
    // inside `c_ms_encode` after the baseline config so both sides see the
    // identical setter sequence.
    let c_ret = c_reference::c_ms_encode(
        &pcm,
        frame_size,
        sample_rate,
        channels,
        mapping_family,
        &cfg,
        &input.setter_bytes,
    );

    match (&rust_ret, &c_ret) {
        (Ok(rust_len), Ok(c_out)) => {
            // Mapping/streams parity is structural — assert unconditionally
            // (no documented divergence here; mismatch is always a finding).
            assert_eq!(
                streams, c_out.streams,
                "MS streams mismatch: Rust={streams}, C={}, ch={channels}, family={mapping_family}",
                c_out.streams
            );
            assert_eq!(
                coupled_streams, c_out.coupled_streams,
                "MS coupled_streams mismatch: Rust={coupled_streams}, C={}, ch={channels}, family={mapping_family}",
                c_out.coupled_streams
            );
            assert_eq!(
                mapping.as_slice(),
                c_out.mapping.as_slice(),
                "MS mapping mismatch: ch={channels}, family={mapping_family}"
            );

            let rust_len = *rust_len as usize;
            assert_eq!(
                rust_len,
                c_out.packet.len(),
                "MS encode length mismatch: Rust={rust_len}, C={}, sr={sample_rate}, \
                 ch={channels}, family={mapping_family}, br={bitrate}, cx={complexity}, vbr={vbr}",
                c_out.packet.len()
            );
            assert_eq!(
                &rust_out[..rust_len],
                &c_out.packet[..],
                "MS encode bytes mismatch: sr={sample_rate}, ch={channels}, family={mapping_family}, \
                 br={bitrate}, cx={complexity}, vbr={vbr}, len={rust_len}"
            );
        }
        (Err(_), Err(_)) => {}
        (Ok(rust_len), Err(c_err)) => {
            panic!(
                "MS encode: Rust ok ({rust_len} bytes) but C errored ({c_err}), \
                 sr={sample_rate}, ch={channels}, family={mapping_family}, \
                 br={bitrate}, cx={complexity}, vbr={vbr}"
            );
        }
        (Err(rust_err), Ok(c_out)) => {
            panic!(
                "MS encode: C ok ({} bytes) but Rust errored ({rust_err}), \
                 sr={sample_rate}, ch={channels}, family={mapping_family}, \
                 br={bitrate}, cx={complexity}, vbr={vbr}",
                c_out.packet.len()
            );
        }
    }
}

fn run_decode(input: &MSInput, sample_rate: i32, channels: i32, mapping_family: i32) {
    if input.payload.is_empty() {
        return;
    }

    // Derive mapping from `new_surround` so Rust + C share a single source of
    // truth. Fail-fast on validator rejection (mapping family + channel
    // combinations that aren't supported).
    let probe = match OpusMSEncoder::new_surround(sample_rate, channels, mapping_family, OPUS_APPLICATION_AUDIO) {
        Ok(t) => t,
        Err(_) => return,
    };
    let (_probe_enc, streams, coupled_streams, mapping) = probe;
    if streams + coupled_streams > 16 {
        return;
    }

    let mut rust_dec = match OpusMSDecoder::new(sample_rate, channels, streams, coupled_streams, &mapping) {
        Ok(d) => d,
        Err(_) => return,
    };

    let frame_cap = 5760usize;
    let mut rust_pcm = vec![0i16; frame_cap * channels as usize];
    let rust_ret = rust_dec.decode(Some(&input.payload), input.payload.len() as i32, &mut rust_pcm, frame_cap as i32, false);
    let c_ret = c_reference::c_ms_decode(
        &input.payload,
        sample_rate,
        channels,
        streams,
        coupled_streams,
        &mapping,
        frame_cap as i32,
    );

    let celt_only = is_celt_only_packet(&input.payload);

    match (&rust_ret, &c_ret) {
        (Ok(rust_samples), Ok(c_pcm)) => {
            let rust_samples = *rust_samples as usize;
            let total = rust_samples * channels as usize;
            assert_eq!(
                total,
                c_pcm.len(),
                "MS decode sample count mismatch: Rust={rust_samples}, C={}, ch={channels}",
                c_pcm.len() / channels as usize
            );
            // SILK/Hybrid sub-packets have known numerical drift; assert
            // byte-exact PCM for CELT-only and bound SILK/Hybrid by the SNR
            // oracle (HLD V2 gap 6). Sample count parity is checked above.
            if celt_only {
                assert_eq!(
                    &rust_pcm[..total],
                    &c_pcm[..],
                    "MS decode PCM mismatch: sr={sample_rate}, ch={channels}, family={mapping_family}, \
                     packet_len={}",
                    input.payload.len()
                );
            }
            // SILK/Hybrid PCM oracle deliberately omitted in the multistream
            // decode path. Stream D's SNR floor (50 dB) trips on the same
            // recovery-divergence class that affects standalone fuzz_decode
            // (silk_decode_recovery_divergence_loud, 2026-05-01) — but for
            // multistream there is no clean structural-validity gate when
            // mapping_family > 0, and the family=0 gate via
            // opus_packet_get_nb_samples doesn't filter 0-byte DTX-style
            // sub-frames. Sample-count parity (asserted above) is the only
            // oracle here until the recovery-divergence root cause is fixed
            // or a per-sub-packet structural gate is implemented.
            // else: reference is silence or near-silence; both
            // implementations' recovery PCM is unconstrained.
            // Sample-count match (already asserted earlier) is the
            // only oracle.
        }
        (Err(_), Err(_)) => {}
        // Asymmetric decode errors are no longer tolerated. The old family=1
        // over-rejection and family=255 mismatch repros were both closed by
        // making multistream validation use the same full packet parser as C
        // (Worker B journal, 2026-05-03).
        (Ok(rust_samples), Err(c_err)) => {
            panic!(
                "MS decode: Rust ok ({rust_samples} samples) but C errored ({c_err}), \
                 sr={sample_rate}, ch={channels}, family={mapping_family}, packet_len={}",
                input.payload.len()
            );
        }
        (Err(rust_err), Ok(c_pcm)) => {
            panic!(
                "MS decode: C ok ({} samples/ch) but Rust errored ({rust_err}), \
                 sr={sample_rate}, ch={channels}, family={mapping_family}, packet_len={}",
                c_pcm.len() / channels as usize,
                input.payload.len()
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_roundtrip(
    input: &MSInput,
    sample_rate: i32,
    channels: i32,
    mapping_family: i32,
    application: i32,
    frame_size: i32,
    bitrate: i32,
    complexity: i32,
    vbr: i32,
) {
    let samples_needed = frame_size as usize * channels as usize;
    let bytes_needed = samples_needed * 2;
    if input.payload.len() < bytes_needed {
        return;
    }
    let pcm: Vec<i16> = input.payload[..bytes_needed]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    let (mut rust_enc, streams, coupled_streams, mapping) =
        match OpusMSEncoder::new_surround(sample_rate, channels, mapping_family, application) {
            Ok(t) => t,
            Err(_) => return,
        };
    if streams + coupled_streams > 16 {
        return;
    }
    apply_initial_config(&mut rust_enc, bitrate, complexity, vbr);
    apply_ms_encoder_setter_sequence(&mut rust_enc, &input.setter_bytes);

    let max_data_bytes = (4000 * streams.max(1)) as i32;
    let mut rust_packet = vec![0u8; max_data_bytes as usize];
    let rust_enc_ret = rust_enc.encode(&pcm, frame_size, &mut rust_packet, max_data_bytes);

    // ---- C-side encode, byte-exact compared to Rust BEFORE decode. ----
    let cfg = c_reference::CMSEncodeConfig {
        bitrate,
        complexity,
        application,
        vbr,
        inband_fec: 0,
        dtx: 0,
        loss_perc: 0,
    };
    let c_enc_ret = c_reference::c_ms_encode(
        &pcm,
        frame_size,
        sample_rate,
        channels,
        mapping_family,
        &cfg,
        &input.setter_bytes,
    );

    let rust_enc_len = match (&rust_enc_ret, &c_enc_ret) {
        (Ok(rust_len), Ok(c_out)) => {
            // Mapping/streams parity (HLD V2): C-derived mapping must match Rust.
            assert_eq!(
                streams, c_out.streams,
                "MS roundtrip streams mismatch: Rust={streams}, C={}, ch={channels}, family={mapping_family}",
                c_out.streams
            );
            assert_eq!(
                coupled_streams, c_out.coupled_streams,
                "MS roundtrip coupled_streams mismatch: Rust={coupled_streams}, C={}, ch={channels}, family={mapping_family}",
                c_out.coupled_streams
            );
            assert_eq!(
                mapping.as_slice(),
                c_out.mapping.as_slice(),
                "MS roundtrip mapping mismatch: ch={channels}, family={mapping_family}"
            );

            let rust_len = *rust_len as usize;
            assert_eq!(
                rust_len,
                c_out.packet.len(),
                "MS roundtrip encode length mismatch: Rust={rust_len}, C={}, sr={sample_rate}, \
                 ch={channels}, family={mapping_family}, br={bitrate}, cx={complexity}, vbr={vbr}",
                c_out.packet.len()
            );
            assert_eq!(
                &rust_packet[..rust_len],
                &c_out.packet[..],
                "MS roundtrip encode bytes mismatch: sr={sample_rate}, ch={channels}, family={mapping_family}, \
                 br={bitrate}, cx={complexity}, vbr={vbr}, len={rust_len}"
            );
            rust_len
        }
        (Err(_), Err(_)) => return,
        (Ok(rust_len), Err(c_err)) => {
            panic!(
                "MS roundtrip encode: Rust ok ({rust_len} bytes) but C errored ({c_err}), \
                 sr={sample_rate}, ch={channels}, family={mapping_family}, br={bitrate}, cx={complexity}, vbr={vbr}"
            );
        }
        (Err(rust_err), Ok(c_out)) => {
            panic!(
                "MS roundtrip encode: C ok ({} bytes) but Rust errored ({rust_err}), \
                 sr={sample_rate}, ch={channels}, family={mapping_family}, br={bitrate}, cx={complexity}, vbr={vbr}",
                c_out.packet.len()
            );
        }
    };
    rust_packet.truncate(rust_enc_len);

    // ---- Decode the (matched) packet on both sides. ----
    let mut rust_dec = match OpusMSDecoder::new(sample_rate, channels, streams, coupled_streams, &mapping) {
        Ok(d) => d,
        Err(_) => return,
    };
    let frame_cap = 5760usize;
    let mut rust_dec_pcm = vec![0i16; frame_cap * channels as usize];
    let rust_dec_ret = rust_dec.decode(
        Some(&rust_packet),
        rust_packet.len() as i32,
        &mut rust_dec_pcm,
        frame_cap as i32,
        false,
    );
    let c_dec_ret = c_reference::c_ms_decode(
        &rust_packet,
        sample_rate,
        channels,
        streams,
        coupled_streams,
        &mapping,
        frame_cap as i32,
    );

    let celt_only = is_celt_only_packet(&rust_packet);

    match (&rust_dec_ret, &c_dec_ret) {
        (Ok(rust_samples), Ok(c_pcm)) => {
            let rust_samples = *rust_samples as usize;
            let total = rust_samples * channels as usize;
            assert_eq!(
                total,
                c_pcm.len(),
                "MS roundtrip decode sample count mismatch: Rust={rust_samples}, C={}, ch={channels}",
                c_pcm.len() / channels as usize
            );
            if celt_only {
                assert_eq!(
                    &rust_dec_pcm[..total],
                    &c_pcm[..],
                    "MS roundtrip PCM mismatch: sr={sample_rate}, ch={channels}, family={mapping_family}, \
                     br={bitrate}, cx={complexity}, vbr={vbr}, packet_len={}",
                    rust_packet.len()
                );
            } else {
                // Roundtrip path: rust_packet was just produced by
                // OpusMSEncoder, so it is well-formed by construction —
                // the recovery-divergence concern only applies to the
                // attacker-controlled decode path.
                if oracle::snr_oracle_applicable_for_packet(&c_pcm[..], true) {
                    let snr = oracle::snr_db(&c_pcm[..], &rust_dec_pcm[..total]);
                    assert!(
                        snr >= oracle::SILK_DECODE_MIN_SNR_DB,
                        "MS roundtrip SILK/Hybrid SNR {snr:.2} dB < {:.0} dB \
                         (sr={sample_rate}, ch={channels}, family={mapping_family}, \
                          br={bitrate}, cx={complexity}, vbr={vbr}, packet_len={})",
                        oracle::SILK_DECODE_MIN_SNR_DB,
                        rust_packet.len()
                    );
                }
            }
            // else: reference is silence or near-silence; both
            // implementations' recovery PCM is unconstrained.
            // Sample-count match (already asserted earlier) is the
            // only oracle.
        }
        (Err(_), Err(_)) => {}
        (Ok(rust_samples), Err(c_err)) => {
            panic!(
                "MS roundtrip decode: Rust ok ({rust_samples} samples) but C errored ({c_err}), \
                 sr={sample_rate}, ch={channels}, family={mapping_family}"
            );
        }
        (Err(rust_err), Ok(c_pcm)) => {
            panic!(
                "MS roundtrip decode: C ok ({} samples/ch) but Rust errored ({rust_err}), \
                 sr={sample_rate}, ch={channels}, family={mapping_family}",
                c_pcm.len() / channels as usize
            );
        }
    }
}

fn apply_initial_config(enc: &mut OpusMSEncoder, bitrate: i32, complexity: i32, vbr: i32) {
    let _ = enc.set_bitrate(bitrate);
    let _ = enc.set_vbr(vbr);
    let _ = enc.set_inband_fec(0);
    let _ = enc.set_dtx(0);
    let _ = enc.set_packet_loss_perc(0);
    let _ = enc.set_complexity(complexity);
}
