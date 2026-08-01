//! projection_roundtrip: Compare ropus's projection encoder/decoder
//! against the C reference byte-for-byte (encode) and sample-for-sample (decode).
//!
//! Usage:
//!   projection_roundtrip                          # run all 5 orders (default)
//!   projection_roundtrip <path.wav>               # run a single fixture
//!   projection_roundtrip --bitrate N              # run all 5 at a given bitrate
//!   projection_roundtrip <path.wav> --bitrate N   # single fixture + bitrate
//!
//! With no path argument the binary iterates over the five committed fixtures
//! in `tests/vectors/ambisonic/` (orders 1..=5 → 4, 9, 16, 25, 36 channels).
//! With a path argument it runs that single fixture.
//!
//! The input must be a 16-bit PCM WAV at 48 kHz with a channel count that
//! matches an ambisonics layout (4, 9, 16, 25, or 36 ACN channels with an
//! optional pair of non-diegetic channels).
//!
//! For each fixture the test encodes 20 ms frames in lock-step through both
//! codecs and checks:
//!   1. C and Rust encoded bytes are identical.
//!   2. Decoded PCM samples are identical from both decoders consuming the
//!      (identical) encoded bitstream.

#![allow(clippy::needless_range_loop)]

use ropus_harness::bindings;
use ropus_harness::wav::{Pcm16Wav, read_pcm16_wav};

use std::os::raw::{c_int, c_uchar};
use std::path::{Path, PathBuf};
use std::process;

use ropus::opus::encoder::OPUS_APPLICATION_AUDIO;
use ropus::opus::multistream::{OpusProjectionDecoder, OpusProjectionEncoder};

// ---------------------------------------------------------------------------
// WAV reader (16-bit PCM only, any channel count)
// ---------------------------------------------------------------------------

fn read_wav(path: &Path) -> Pcm16Wav {
    read_pcm16_wav(path).unwrap_or_else(|e| {
        eprintln!("ERROR: {e}");
        process::exit(1);
    })
}

// ---------------------------------------------------------------------------
// C reference projection wrapper
// ---------------------------------------------------------------------------

struct CEncoder {
    st: *mut bindings::OpusProjectionEncoder,
    streams: c_int,
    coupled_streams: c_int,
    demixing_matrix: Vec<u8>,
    demixing_matrix_gain: c_int,
}

impl CEncoder {
    fn new(fs: i32, channels: i32, bitrate: i32) -> Self {
        unsafe {
            let mut streams: c_int = 0;
            let mut coupled: c_int = 0;
            let mut err: c_int = 0;
            let st = bindings::opus_projection_ambisonics_encoder_create(
                fs,
                channels,
                3, // ambisonics mapping family
                &mut streams,
                &mut coupled,
                OPUS_APPLICATION_AUDIO,
                &mut err,
            );
            if st.is_null() || err != 0 {
                panic!("C: opus_projection_ambisonics_encoder_create failed: err={err}");
            }

            // Bitrate
            let _ = bindings::opus_projection_encoder_ctl(
                st,
                4002, // OPUS_SET_BITRATE_REQUEST
                bitrate as c_int,
            );

            // Query the demixing matrix size and data.
            let mut matrix_size: c_int = 0;
            let _ = bindings::opus_projection_encoder_ctl(
                st,
                6003, // OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST
                &mut matrix_size as *mut c_int,
            );
            let mut matrix = vec![0u8; matrix_size as usize];
            let _ = bindings::opus_projection_encoder_ctl(
                st,
                6005, // OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST
                matrix.as_mut_ptr(),
                matrix_size as c_int,
            );

            let mut gain: c_int = 0;
            let _ = bindings::opus_projection_encoder_ctl(
                st,
                6001, // OPUS_PROJECTION_GET_DEMIXING_MATRIX_GAIN_REQUEST
                &mut gain as *mut c_int,
            );

            CEncoder {
                st,
                streams,
                coupled_streams: coupled,
                demixing_matrix: matrix,
                demixing_matrix_gain: gain,
            }
        }
    }

    fn encode(&mut self, pcm: &[i16], frame_size: i32, buf: &mut [u8]) -> i32 {
        unsafe {
            bindings::opus_projection_encode(
                self.st,
                pcm.as_ptr(),
                frame_size,
                buf.as_mut_ptr(),
                buf.len() as i32,
            )
        }
    }
}

impl Drop for CEncoder {
    fn drop(&mut self) {
        unsafe { bindings::opus_projection_encoder_destroy(self.st) }
    }
}

struct CDecoder {
    st: *mut bindings::OpusProjectionDecoder,
}

impl CDecoder {
    fn new(
        fs: i32,
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        demixing_matrix: &[u8],
    ) -> Self {
        unsafe {
            let mut err: c_int = 0;
            let st = bindings::opus_projection_decoder_create(
                fs,
                channels,
                streams,
                coupled_streams,
                demixing_matrix.as_ptr(),
                demixing_matrix.len() as i32,
                &mut err,
            );
            if st.is_null() || err != 0 {
                panic!("C: opus_projection_decoder_create failed: err={err}");
            }
            CDecoder { st }
        }
    }

    fn decode(&mut self, packet: &[u8], pcm: &mut [i16], frame_size: i32) -> i32 {
        unsafe {
            bindings::opus_projection_decode(
                self.st,
                packet.as_ptr() as *const c_uchar,
                packet.len() as i32,
                pcm.as_mut_ptr(),
                frame_size,
                0,
            )
        }
    }
}

impl Drop for CDecoder {
    fn drop(&mut self) {
        unsafe { bindings::opus_projection_decoder_destroy(self.st) }
    }
}

// ---------------------------------------------------------------------------
// Per-fixture comparison driver
// ---------------------------------------------------------------------------

/// Result of running one fixture end-to-end.
struct FixtureResult {
    total_frames: usize,
    processed_frames: usize,
    encode_mismatches: usize,
    decode_mismatches: usize,
    encode_first_mismatch: Option<(usize, usize)>,
    decode_first_mismatch: Option<(usize, usize)>,
}

impl FixtureResult {
    fn passed(&self) -> bool {
        self.processed_frames > 0
            && self.processed_frames == self.total_frames
            && self.encode_mismatches == 0
            && self.decode_mismatches == 0
    }
}

fn complete_frame_count(sample_count: usize, samples_per_frame: usize) -> Option<usize> {
    let frames = sample_count.checked_div(samples_per_frame)?;
    (frames > 0).then_some(frames)
}

fn run_fixture(path: &Path, bitrate: i32) -> FixtureResult {
    let wav = read_wav(path);
    println!(
        "Loaded {}: {} Hz, {} channels, {} samples",
        path.display(),
        wav.sample_rate,
        wav.channels,
        wav.samples.len() / wav.channels as usize
    );

    let fs: i32 = wav.sample_rate as i32;
    let channels: i32 = wav.channels as i32;
    let frame_size: i32 = fs / 50; // 20 ms
    let samples_per_frame = (frame_size as usize) * channels as usize;
    let Some(total_frames) = complete_frame_count(wav.samples.len(), samples_per_frame) else {
        eprintln!(
            "ERROR: {} contains no complete 20 ms frame (need {} interleaved samples, found {})",
            path.display(),
            samples_per_frame,
            wav.samples.len()
        );
        return FixtureResult {
            total_frames: 0,
            processed_frames: 0,
            encode_mismatches: 0,
            decode_mismatches: 0,
            encode_first_mismatch: None,
            decode_first_mismatch: None,
        };
    };

    // Build both encoders
    let mut c_enc = CEncoder::new(fs, channels, bitrate);
    let (mut r_enc, r_streams, r_coupled) =
        OpusProjectionEncoder::new(fs, channels, 3, OPUS_APPLICATION_AUDIO)
            .expect("ropus: projection encoder create");
    let _ = r_enc.set_bitrate(bitrate);

    // Stream-count sanity: both codecs must agree on layout.
    assert_eq!(
        c_enc.streams, r_streams as c_int,
        "stream count mismatch: C={}, ropus={}",
        c_enc.streams, r_streams
    );
    assert_eq!(
        c_enc.coupled_streams, r_coupled as c_int,
        "coupled stream count mismatch"
    );

    // Demixing matrix consistency: ropus computes the same matrix data and gain
    // via its public getters.
    let r_matrix = r_enc.get_demixing_matrix();
    let r_gain = r_enc.get_demixing_matrix_gain();
    assert_eq!(
        r_matrix.len(),
        c_enc.demixing_matrix.len(),
        "demixing matrix size mismatch: ropus={}, C={}",
        r_matrix.len(),
        c_enc.demixing_matrix.len()
    );
    if r_matrix != c_enc.demixing_matrix {
        let first_diff = r_matrix
            .iter()
            .zip(&c_enc.demixing_matrix)
            .position(|(a, b)| a != b)
            .unwrap();
        panic!(
            "demixing matrix differs at byte {}: ropus={} C={}",
            first_diff, r_matrix[first_diff], c_enc.demixing_matrix[first_diff]
        );
    }
    assert_eq!(
        r_gain, c_enc.demixing_matrix_gain,
        "demixing matrix gain mismatch: ropus={}, C={}",
        r_gain, c_enc.demixing_matrix_gain
    );

    // Build the decoders once we know the demixing matrix.
    let mut c_dec = CDecoder::new(
        fs,
        channels,
        c_enc.streams,
        c_enc.coupled_streams,
        &c_enc.demixing_matrix,
    );
    let mut r_dec = OpusProjectionDecoder::new(
        fs,
        channels,
        r_streams,
        r_coupled,
        &r_matrix,
        r_matrix.len() as i32,
    )
    .expect("ropus: projection decoder create");

    // Encode + decode in 20 ms frames.
    let max_packet = 1500 * c_enc.streams as usize;
    let mut c_buf = vec![0u8; max_packet];
    let mut r_buf = vec![0u8; max_packet];
    let mut c_pcm = vec![0i16; samples_per_frame];
    let mut r_pcm = vec![0i16; samples_per_frame];

    let mut encode_mismatches = 0usize;
    let mut decode_mismatches = 0usize;
    let mut processed_frames = 0usize;
    let mut encode_first_mismatch: Option<(usize, usize)> = None;
    let mut decode_first_mismatch: Option<(usize, usize)> = None;

    for frame_idx in 0..total_frames {
        let off = frame_idx * samples_per_frame;
        let input = &wav.samples[off..off + samples_per_frame];

        let c_len = c_enc.encode(input, frame_size, &mut c_buf);
        let r_len = r_enc
            .encode(input, frame_size, &mut r_buf, max_packet as i32)
            .expect("ropus: encode");

        assert!(
            c_len > 0,
            "C encode failed on frame {} (len={})",
            frame_idx,
            c_len
        );
        assert!(r_len > 0, "ropus encode failed on frame {}", frame_idx);
        assert_eq!(
            c_len, r_len,
            "frame {} size differs: C={}, ropus={}",
            frame_idx, c_len, r_len
        );

        let c_slice = &c_buf[..c_len as usize];
        let r_slice = &r_buf[..r_len as usize];
        if c_slice != r_slice {
            encode_mismatches += 1;
            if encode_first_mismatch.is_none() {
                let off = c_slice
                    .iter()
                    .zip(r_slice)
                    .position(|(a, b)| a != b)
                    .unwrap_or(0);
                encode_first_mismatch = Some((frame_idx, off));
            }
        }

        // Decode both outputs (using the ropus bytes so inputs are identical;
        // encode match above guarantees they are the same buffer).
        c_pcm.fill(0);
        r_pcm.fill(0);
        let c_ret = c_dec.decode(r_slice, &mut c_pcm, frame_size);
        let r_ret = r_dec
            .decode(
                Some(r_slice),
                r_slice.len() as i32,
                &mut r_pcm,
                frame_size,
                false,
            )
            .expect("ropus: decode");
        assert_eq!(
            c_ret, r_ret as c_int,
            "frame {} decoded sample count differs",
            frame_idx
        );
        if c_pcm != r_pcm {
            decode_mismatches += 1;
            if decode_first_mismatch.is_none() {
                let off = c_pcm
                    .iter()
                    .zip(&r_pcm)
                    .position(|(a, b)| a != b)
                    .unwrap_or(0);
                decode_first_mismatch = Some((frame_idx, off));
            }
        }
        processed_frames += 1;
    }

    println!(
        "  Processed {} frames ({} samples/frame x {} channels)",
        processed_frames, frame_size, channels
    );
    println!(
        "  Encode: {}/{} frames byte-identical",
        processed_frames - encode_mismatches,
        processed_frames
    );
    println!(
        "  Decode: {}/{} frames sample-identical",
        processed_frames - decode_mismatches,
        processed_frames
    );

    if let Some((frame, off)) = encode_first_mismatch {
        println!("  First encode divergence: frame {frame}, byte {off}");
    }
    if let Some((frame, off)) = decode_first_mismatch {
        println!("  First decode divergence: frame {frame}, sample {off}");
    }

    FixtureResult {
        total_frames,
        processed_frames,
        encode_mismatches,
        decode_mismatches,
        encode_first_mismatch,
        decode_first_mismatch,
    }
}

// ---------------------------------------------------------------------------
// Fixture-set resolution
// ---------------------------------------------------------------------------

/// Return the list of (order, path) pairs that make up the default sweep.
/// Resolved against CARGO_MANIFEST_DIR so the binary works no matter which
/// directory cargo invoked it from.
fn default_fixtures() -> Vec<(u32, PathBuf)> {
    // CARGO_MANIFEST_DIR is baked in at compile time; from the harness crate
    // root we step up once to the workspace root, then into tests/vectors.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let root = Path::new(manifest_dir).parent().unwrap_or(Path::new("."));
    let base = root.join("tests").join("vectors").join("ambisonic");
    (1u32..=5u32)
        .map(|o| (o, base.join(format!("ambisonic_order{o}_100ms.wav"))))
        .collect()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

pub fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut bitrate: i32 = 128_000;
    let mut single_path: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--bitrate" => {
                bitrate = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(128_000);
                i += 2;
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage:\n  \
                     {} [--bitrate N]              # run all 5 ambisonic orders\n  \
                     {} <path.wav> [--bitrate N]   # run one fixture",
                    args[0], args[0]
                );
                process::exit(0);
            }
            other if other.starts_with("--") => {
                eprintln!("Unknown arg: {other}");
                process::exit(1);
            }
            _ => {
                single_path = Some(PathBuf::from(&args[i]));
                i += 1;
            }
        }
    }

    // Build the work list.
    let fixtures: Vec<(String, PathBuf)> = match single_path {
        Some(p) => vec![(format!("user fixture {}", p.display()), p)],
        None => default_fixtures()
            .into_iter()
            .map(|(order, path)| (format!("order {order}"), path))
            .collect(),
    };

    let mut any_missing = false;
    for (_, path) in &fixtures {
        if !path.exists() {
            eprintln!("ERROR: fixture not found: {}", path.display());
            any_missing = true;
        }
    }
    if any_missing {
        eprintln!(
            "Hint: generate with `rustc tools/make_ambisonic_fixture.rs -o mk && \
             ./mk <order> tests/vectors/ambisonic/ambisonic_order<order>_100ms.wav`"
        );
        process::exit(1);
    }

    let mut any_fail = false;
    let mut summary: Vec<(String, FixtureResult)> = Vec::new();

    for (label, path) in &fixtures {
        println!("--- {} ---", label);
        let res = run_fixture(path, bitrate);
        if !res.passed() {
            any_fail = true;
        }
        summary.push((label.clone(), res));
    }

    println!();
    println!("=== SUMMARY ===");
    for (label, res) in &summary {
        let status = if res.passed() { "PASS" } else { "FAIL" };
        println!(
            "  {status} {label}: {} frames, encode {}/{}, decode {}/{}",
            res.total_frames,
            res.total_frames - res.encode_mismatches,
            res.total_frames,
            res.total_frames - res.decode_mismatches,
            res.total_frames,
        );
        if !res.passed() {
            if let Some((f, off)) = res.encode_first_mismatch {
                println!("    first encode divergence: frame {f}, byte {off}");
            }
            if let Some((f, off)) = res.decode_first_mismatch {
                println!("    first decode divergence: frame {f}, sample {off}");
            }
        }
    }

    if any_fail {
        println!("FAIL");
        process::exit(1);
    } else {
        println!("PASS: all fixtures byte-exact encode AND sample-exact decode");
        process::exit(0);
    }
}

#[cfg(test)]
mod tests {
    use super::{FixtureResult, complete_frame_count};

    fn clean_result_for_samples(sample_count: usize, samples_per_frame: usize) -> FixtureResult {
        FixtureResult {
            total_frames: sample_count / samples_per_frame,
            processed_frames: sample_count / samples_per_frame,
            encode_mismatches: 0,
            decode_mismatches: 0,
            encode_first_mismatch: None,
            decode_first_mismatch: None,
        }
    }

    #[test]
    fn empty_fixture_cannot_pass_without_processing_a_frame() {
        assert_eq!(complete_frame_count(0, 3_840), None);
    }

    #[test]
    fn partial_only_fixture_cannot_pass_without_processing_a_frame() {
        assert_eq!(complete_frame_count(3_839, 3_840), None);
    }

    #[test]
    fn fixture_cannot_pass_until_every_planned_frame_is_processed() {
        let mut result = clean_result_for_samples(3_840, 3_840);
        result.processed_frames = 0;
        assert!(!result.passed());
    }

    #[test]
    fn clean_fixture_with_one_processed_frame_passes() {
        assert_eq!(complete_frame_count(3_840, 3_840), Some(1));
        assert!(clean_result_for_samples(3_840, 3_840).passed());
    }
}
