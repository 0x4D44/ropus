//! corpus_diff: Walk a directory of real-world Opus files and diff ropus vs
//! the C reference, sample-for-sample.
//!
//! Usage:
//!   corpus_diff <dir>
//!
//! Scans `<dir>` non-recursively for files ending in `.opus`, `.ogg`, or
//! `.webm`. For each file it parses the Ogg container (no libopusfile
//! dependency — just raw OggS + `OpusHead`), then decodes every audio packet
//! through both ropus and the C reference `libopus`, asserting the PCM output
//! matches sample-for-sample.
//!
//! Why this exists. The conformance suite compares ropus output against the C
//! reference on synthetic vectors we build ourselves. That catches ropus-
//! vs-reference drift but cannot catch encoder-specific bitstream quirks
//! produced in the wild (FFmpeg's native `opus` encoder, WebRTC, Android
//! AOSP, streaming services). This binary provides the diff *mechanism* —
//! real-world coverage is gated on whoever populates `<dir>` with genuinely
//! non-reference-encoded samples. The bundled fetch script is a baseline of
//! reference-encoded files only; see `tests/vectors/real_world/README.md`.
//!
//! Scope.
//!   - Non-recursive directory walk (flat corpus — fetch script pulls into
//!     one directory).
//!   - Ogg container only (`.opus` / `.ogg`). `.webm` files are picked up but
//!     not parsed — logged and skipped.
//!   - Channel-mapping family 0 (mono/stereo) only. Surround / ambisonic
//!     files are logged and skipped; those have their own targeted harness
//!     (`projection_roundtrip`).
//!   - Per-file panics are caught and logged; we continue to the next file
//!     rather than aborting the whole run.
//!
//! Exit codes (stable contract — CI should gate on these).
//!   0 — at least one supported file decoded nonzero audio and every decoded
//!       file matched sample-for-sample.
//!   1 — one or more files mismatched or panicked, the directory argument was
//!       missing / unreadable, OR candidate files existed but none decoded
//!       nonzero audio for comparison.
//!   2 — directory exists but contains no candidate files. Distinct from 0
//!       so a CI pipeline cannot silently pass against an unpopulated
//!       corpus; gate the step on `fetch_corpus.sh` or a manual populate
//!       having completed first.
//!   3 — every candidate is a deferred container (e.g. WebM). Distinct from 0
//!       so a release-preflight cannot satisfy a corpus claim with only
//!       deferred-container entries.

#![allow(clippy::needless_range_loop, clippy::collapsible_if)]

use ropus_harness::bindings;

use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::BufReader;
use std::os::raw::{c_int, c_uchar};
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::process;

use ogg::reading::PacketReader;
use ropus::OpusDecoder;

// Output sample rate. Opus always decodes to 48 kHz regardless of the
// `input_sample_rate` field in OpusHead (RFC 7845 §5.1).
const OPUS_SAMPLE_RATE_HZ: i32 = 48_000;

// Max per-channel samples a single Opus frame can produce (120 ms @ 48 kHz).
const MAX_FRAME_SAMPLES_PER_CH: usize = 5760;

// ---------------------------------------------------------------------------
// File classification
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CorpusKind {
    Opus, // .opus / .ogg — Ogg container we can parse
    Webm, // .webm — Matroska container, not supported here (logged and skipped)
}

fn classify(path: &Path) -> Option<CorpusKind> {
    match path.extension().and_then(OsStr::to_str) {
        Some(ext) => match ext.to_ascii_lowercase().as_str() {
            "opus" | "ogg" => Some(CorpusKind::Opus),
            "webm" => Some(CorpusKind::Webm),
            _ => None,
        },
        None => None,
    }
}

// ---------------------------------------------------------------------------
// Ogg + OpusHead parsing
// ---------------------------------------------------------------------------

/// Parsed `OpusHead` fields we need for decoder setup. Deliberately duplicates
/// the parsing in `ropus-fb2k/src/reader.rs` and `ropus-cli/src/container/ogg.rs`
/// — those modules are crate-private and pulling them in would require
/// promoting helpers we don't otherwise want on the public surface. The
/// parser is ~20 lines and RFC 7845 is stable; duplication is cheaper than
/// refactoring two other crates.
#[derive(Debug, Clone, Copy)]
struct OpusHead {
    channels: u8,
    channel_mapping: u8,
}

fn parse_opus_head(data: &[u8]) -> Result<OpusHead, String> {
    if data.len() < 19 {
        return Err(format!(
            "OpusHead too short ({} bytes, need 19)",
            data.len()
        ));
    }
    if &data[..8] != b"OpusHead" {
        return Err("OpusHead magic missing".into());
    }
    Ok(OpusHead {
        channels: data[9],
        channel_mapping: data[18],
    })
}

// ---------------------------------------------------------------------------
// File-level decode + diff
// ---------------------------------------------------------------------------

/// Outcome of comparing one file's output from both decoders.
enum FileOutcome {
    /// Every decoded sample matched.
    Match {
        packets: usize,
        samples_per_ch: usize,
    },
    /// Container or header parse failed — file is malformed. Not a ropus bug.
    Skipped(String),
    /// Decoders disagreed. Includes first-diff context.
    Mismatch(Mismatch),
    /// Either decoder returned an error on the same packet. We treat matching
    /// errors as a pass (both implementations agreed the file is bad at the
    /// same point) and mismatched errors as a real mismatch.
    DecoderError(Mismatch),
}

struct Mismatch {
    packet_index: usize,
    sample_index: usize, // per-channel sample index into this packet
    channel: usize,
    ropus_value: i32,
    cref_value: i32,
    note: String,
}

fn diff_file(path: &Path) -> FileOutcome {
    // Open and wrap in a buffered reader. `ogg::PacketReader` needs `Read +
    // Seek`; `BufReader<File>` satisfies both.
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => return FileOutcome::Skipped(format!("open failed: {e}")),
    };
    let mut reader = PacketReader::new(BufReader::new(file));

    // Page 1: OpusHead.
    let head_pkt = match reader.read_packet() {
        Ok(Some(pkt)) => pkt,
        Ok(None) => return FileOutcome::Skipped("empty stream (no pages)".into()),
        Err(e) => return FileOutcome::Skipped(format!("read OpusHead page: {e}")),
    };
    let head = match parse_opus_head(&head_pkt.data) {
        Ok(h) => h,
        Err(e) => return FileOutcome::Skipped(e),
    };
    if head.channel_mapping != 0 {
        return FileOutcome::Skipped(format!(
            "channel_mapping_family={} (only family 0 tested here)",
            head.channel_mapping
        ));
    }
    if head.channels == 0 || head.channels > 2 {
        return FileOutcome::Skipped(format!(
            "channels={} (only mono/stereo tested here)",
            head.channels
        ));
    }
    let channels = head.channels as i32;

    // Page 2: OpusTags. Read and discard; we only need to consume the page so
    // the next `read_packet()` call lands on the first audio packet.
    match reader.read_packet() {
        Ok(Some(_)) => {}
        Ok(None) => return FileOutcome::Skipped("stream ended before OpusTags".into()),
        Err(e) => return FileOutcome::Skipped(format!("read OpusTags page: {e}")),
    };

    // Spin up both decoders.
    let mut ropus_dec = match OpusDecoder::new(OPUS_SAMPLE_RATE_HZ, channels) {
        Ok(d) => d,
        Err(code) => {
            return FileOutcome::Skipped(format!("ropus OpusDecoder::new failed (code {code})"));
        }
    };

    let cref_dec = CrefDecoder::new(OPUS_SAMPLE_RATE_HZ, channels as c_int);
    let cref_dec = match cref_dec {
        Ok(d) => d,
        Err(e) => return FileOutcome::Skipped(format!("C reference decoder alloc failed: {e}")),
    };

    let mut packets_decoded = 0usize;
    let mut total_samples_per_ch = 0usize;

    // Per-packet decode scratch. Sized for a full 120 ms frame.
    let mut ropus_pcm = vec![0i16; MAX_FRAME_SAMPLES_PER_CH * channels as usize];
    let mut cref_pcm = vec![0i16; MAX_FRAME_SAMPLES_PER_CH * channels as usize];

    loop {
        let pkt = match reader.read_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break, // clean EOF
            Err(e) => {
                return FileOutcome::Skipped(format!(
                    "Ogg read error after {packets_decoded} packets: {e}"
                ));
            }
        };

        let ropus_result = ropus_dec.decode(
            Some(&pkt.data),
            &mut ropus_pcm,
            MAX_FRAME_SAMPLES_PER_CH as i32,
            false,
        );
        let cref_result =
            cref_dec.decode(&pkt.data, &mut cref_pcm, MAX_FRAME_SAMPLES_PER_CH as i32);

        match (ropus_result, cref_result) {
            (Ok(r_n), Ok(c_n)) => {
                if r_n != c_n {
                    return FileOutcome::Mismatch(Mismatch {
                        packet_index: packets_decoded,
                        sample_index: 0,
                        channel: 0,
                        ropus_value: r_n,
                        cref_value: c_n,
                        note: format!("decoded sample-count differs: ropus={r_n}, cref={c_n}"),
                    });
                }
                let n = r_n as usize;
                for sample in 0..n {
                    for ch in 0..channels as usize {
                        let idx = sample * channels as usize + ch;
                        if ropus_pcm[idx] != cref_pcm[idx] {
                            return FileOutcome::Mismatch(Mismatch {
                                packet_index: packets_decoded,
                                sample_index: sample,
                                channel: ch,
                                ropus_value: ropus_pcm[idx] as i32,
                                cref_value: cref_pcm[idx] as i32,
                                note: "PCM sample mismatch".into(),
                            });
                        }
                    }
                }
                packets_decoded += 1;
                total_samples_per_ch += n;
            }
            (Err(r_code), Err(c_code)) => {
                // Both decoders rejected the same packet. Treat this as a
                // "matched error" — the file is likely truncated or the
                // packet is malformed; both implementations agree, which is
                // what we care about. Bail cleanly; the corpus run continues.
                if r_code != c_code {
                    return FileOutcome::DecoderError(Mismatch {
                        packet_index: packets_decoded,
                        sample_index: 0,
                        channel: 0,
                        ropus_value: r_code,
                        cref_value: c_code,
                        note: "both decoders errored but with different codes".into(),
                    });
                }
                break;
            }
            (Ok(r_n), Err(c_code)) => {
                return FileOutcome::DecoderError(Mismatch {
                    packet_index: packets_decoded,
                    sample_index: 0,
                    channel: 0,
                    ropus_value: r_n,
                    cref_value: c_code,
                    note: "ropus decoded ok; cref returned error".into(),
                });
            }
            (Err(r_code), Ok(c_n)) => {
                return FileOutcome::DecoderError(Mismatch {
                    packet_index: packets_decoded,
                    sample_index: 0,
                    channel: 0,
                    ropus_value: r_code,
                    cref_value: c_n,
                    note: "cref decoded ok; ropus returned error".into(),
                });
            }
        }
    }

    FileOutcome::Match {
        packets: packets_decoded,
        samples_per_ch: total_samples_per_ch,
    }
}

// ---------------------------------------------------------------------------
// C reference decoder wrapper
// ---------------------------------------------------------------------------

/// Thin RAII wrapper around `opus_decoder_create` / `opus_decoder_destroy`.
/// Exists so Drop handles cleanup even if the decode loop returns early.
struct CrefDecoder {
    st: *mut bindings::OpusDecoder,
}

impl CrefDecoder {
    fn new(fs: i32, channels: c_int) -> Result<Self, String> {
        let mut err: c_int = 0;
        let st = unsafe { bindings::opus_decoder_create(fs, channels, &mut err) };
        if st.is_null() || err != bindings::OPUS_OK {
            return Err(format!(
                "opus_decoder_create returned err={err} ({})",
                bindings::error_string(err)
            ));
        }
        Ok(Self { st })
    }

    fn decode(&self, data: &[u8], pcm: &mut [i16], frame_size: i32) -> Result<i32, i32> {
        let ret = unsafe {
            bindings::opus_decode(
                self.st,
                data.as_ptr() as *const c_uchar,
                data.len() as i32,
                pcm.as_mut_ptr(),
                frame_size,
                0, // decode_fec = false
            )
        };
        if ret < 0 { Err(ret) } else { Ok(ret) }
    }
}

impl Drop for CrefDecoder {
    fn drop(&mut self) {
        if !self.st.is_null() {
            unsafe { bindings::opus_decoder_destroy(self.st) };
        }
    }
}

// ---------------------------------------------------------------------------
// Directory walk + driver
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct RunStats {
    candidates: usize,
    decoded_and_compared: usize,
    zero_audio: usize,
    skipped: usize,
    deferred: usize,
    mismatched: usize,
    panicked: usize,
}

impl RunStats {
    fn exit_code(&self) -> i32 {
        if self.candidates == 0 {
            return 2;
        }
        if self.deferred == self.candidates {
            return 3;
        }
        if self.mismatched > 0 || self.panicked > 0 || self.decoded_and_compared == 0 {
            return 1;
        }
        0
    }

    fn record_match(&mut self, samples_per_ch: usize) {
        if samples_per_ch == 0 {
            self.zero_audio += 1;
        } else {
            self.decoded_and_compared += 1;
        }
    }

    fn summary_line(&self) -> String {
        format!(
            "CORPUS_DIFF_SUMMARY candidates={} decoded_and_compared={} zero_audio={} skipped={} deferred={} mismatched={} panicked={}",
            self.candidates,
            self.decoded_and_compared,
            self.zero_audio,
            self.skipped,
            self.deferred,
            self.mismatched,
            self.panicked
        )
    }
}

/// Per-file dispatch decision: extracted so the routing rule for known-
/// deferred containers vs decodable Ogg streams is unit-testable without
/// writing real container files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileDispatch {
    Decode,
    Defer { reason: &'static str },
}

fn dispatch_action(kind: CorpusKind) -> FileDispatch {
    match kind {
        CorpusKind::Opus => FileDispatch::Decode,
        CorpusKind::Webm => FileDispatch::Defer {
            reason: "webm-matroska-container-deferred",
        },
    }
}

fn gather_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let entries = fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut files = Vec::new();
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                eprintln!("WARN: skipping unreadable entry: {e}");
                continue;
            }
        };
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if classify(&path).is_some() {
            files.push(path);
        }
    }
    files.sort(); // deterministic iteration order
    Ok(files)
}

pub fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: corpus_diff <directory>");
        process::exit(1);
    }
    let dir = PathBuf::from(&args[1]);

    if !dir.is_dir() {
        eprintln!("ERROR: not a directory: {}", dir.display());
        process::exit(1);
    }

    let files = match gather_files(&dir) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("ERROR: cannot list {}: {}", dir.display(), e);
            process::exit(1);
        }
    };

    if files.is_empty() {
        eprintln!(
            "corpus_diff: no candidate files found in {} — populate via \
             tools/fetch_corpus.sh or drop .opus/.ogg/.webm files in manually.",
            dir.display()
        );
        eprintln!(
            "exit code 2 is deliberate (distinct from 0 = 'all matched'); \
             CI should gate this step on the corpus being populated."
        );
        process::exit(2);
    }

    println!(
        "corpus_diff: scanning {} file(s) in {}",
        files.len(),
        dir.display()
    );

    let mut stats = RunStats {
        candidates: files.len(),
        ..RunStats::default()
    };

    for path in &files {
        let display = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());

        let kind = classify(path).expect("classifier admitted only known kinds");
        if let FileDispatch::Defer { reason } = dispatch_action(kind) {
            let relpath = path
                .strip_prefix(&dir)
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|_| path.display().to_string());
            println!("  DEFER {relpath} reason={reason}");
            stats.deferred += 1;
            continue;
        }

        // Catch panics from either decoder so one bad file can't tank the
        // whole run. The decoders *should* return Err on malformed input
        // rather than panic, but the corpus is specifically files we haven't
        // vetted — belt-and-braces.
        let outcome = panic::catch_unwind(AssertUnwindSafe(|| diff_file(path)));

        match outcome {
            Ok(FileOutcome::Match {
                packets,
                samples_per_ch,
            }) => {
                if samples_per_ch == 0 {
                    println!("  ZERO {display} ({packets} packets, 0 samples/ch)");
                } else {
                    println!("  OK   {display} ({packets} packets, {samples_per_ch} samples/ch)");
                }
                stats.record_match(samples_per_ch);
            }
            Ok(FileOutcome::Skipped(reason)) => {
                println!("  SKIP {display} ({reason})");
                stats.skipped += 1;
            }
            Ok(FileOutcome::Mismatch(m)) => {
                println!(
                    "  FAIL {display} — packet {} sample {} ch {}: ropus={} cref={} ({})",
                    m.packet_index, m.sample_index, m.channel, m.ropus_value, m.cref_value, m.note
                );
                stats.mismatched += 1;
            }
            Ok(FileOutcome::DecoderError(m)) => {
                println!(
                    "  FAIL {display} — packet {}: ropus={} cref={} ({})",
                    m.packet_index, m.ropus_value, m.cref_value, m.note
                );
                stats.mismatched += 1;
            }
            Err(payload) => {
                let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "<non-string panic>".to_string()
                };
                println!("  PANIC {display} — {msg}");
                stats.panicked += 1;
            }
        }
    }

    println!(
        "---\n{} decoded-and-compared, {} zero-audio, {} skipped, {} deferred, {} mismatched, {} panicked (of {} total)",
        stats.decoded_and_compared,
        stats.zero_audio,
        stats.skipped,
        stats.deferred,
        stats.mismatched,
        stats.panicked,
        stats.candidates
    );
    println!("{}", stats.summary_line());

    process::exit(stats.exit_code());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_all_deferred_set_returns_exit_three() {
        let stats = RunStats {
            candidates: 2,
            deferred: 2,
            ..RunStats::default()
        };

        assert_eq!(stats.exit_code(), 3);
    }

    #[test]
    fn corpus_one_match_with_deferred_is_still_green() {
        let mut stats = RunStats {
            candidates: 2,
            deferred: 1,
            ..RunStats::default()
        };
        stats.record_match(960);

        assert_eq!(stats.decoded_and_compared, 1);
        assert_eq!(stats.exit_code(), 0);
    }

    #[test]
    fn corpus_diff_summary_line_includes_deferred_count() {
        let stats = RunStats {
            candidates: 5,
            deferred: 3,
            skipped: 1,
            decoded_and_compared: 1,
            ..RunStats::default()
        };
        let line = stats.summary_line();
        assert!(
            line.contains("deferred=3"),
            "missing deferred= field: {line}"
        );
        let skipped_pos = line.find("skipped=").expect("skipped= present");
        let deferred_pos = line.find("deferred=").expect("deferred= present");
        let mismatched_pos = line.find("mismatched=").expect("mismatched= present");
        assert!(
            skipped_pos < deferred_pos && deferred_pos < mismatched_pos,
            "expected order skipped<deferred<mismatched: {line}"
        );
    }

    #[test]
    fn webm_dispatch_increments_deferred_not_skipped() {
        let mut stats = RunStats {
            candidates: 1,
            ..RunStats::default()
        };
        let action = dispatch_action(CorpusKind::Webm);
        assert_eq!(
            action,
            FileDispatch::Defer {
                reason: "webm-matroska-container-deferred"
            }
        );
        match action {
            FileDispatch::Defer { .. } => stats.deferred += 1,
            FileDispatch::Decode => stats.decoded_and_compared += 1,
        }
        assert_eq!(stats.deferred, 1);
        assert_eq!(stats.skipped, 0);
        assert_eq!(stats.exit_code(), 3);
    }

    #[test]
    fn corpus_skipped_only_candidate_set_is_non_green() {
        let stats = RunStats {
            candidates: 2,
            skipped: 2,
            ..RunStats::default()
        };

        assert_eq!(stats.exit_code(), 1);
        assert!(stats.summary_line().contains("decoded_and_compared=0"));
    }

    #[test]
    fn corpus_zero_audio_only_candidate_set_is_non_green() {
        let mut stats = RunStats {
            candidates: 1,
            ..RunStats::default()
        };
        stats.record_match(0);

        assert_eq!(stats.zero_audio, 1);
        assert_eq!(stats.decoded_and_compared, 0);
        assert_eq!(stats.exit_code(), 1);
    }

    #[test]
    fn corpus_supported_nonzero_match_is_green_even_with_exploratory_skips() {
        let mut stats = RunStats {
            candidates: 2,
            skipped: 1,
            ..RunStats::default()
        };
        stats.record_match(960);

        assert_eq!(stats.decoded_and_compared, 1);
        assert_eq!(stats.exit_code(), 0);
    }

    #[test]
    fn corpus_mismatch_or_panic_is_non_green() {
        let mut mismatch = RunStats {
            candidates: 1,
            mismatched: 1,
            ..RunStats::default()
        };
        mismatch.record_match(960);
        assert_eq!(mismatch.exit_code(), 1);

        let mut panic = RunStats {
            candidates: 1,
            panicked: 1,
            ..RunStats::default()
        };
        panic.record_match(960);
        assert_eq!(panic.exit_code(), 1);
    }

    #[test]
    fn corpus_empty_candidate_set_preserves_exit_two() {
        assert_eq!(RunStats::default().exit_code(), 2);
    }
}
