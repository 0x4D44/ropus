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
//!   1 — one or more files were malformed, mismatched, or panicked, the
//!       directory argument was missing / unreadable, OR candidate files
//!       existed but none decoded nonzero audio for comparison.
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
use std::io::{BufReader, Read};
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

fn validate_opus_tags(data: &[u8]) -> Result<(), String> {
    if data.len() < 8 {
        return Err(format!("OpusTags too short ({} bytes, need 8)", data.len()));
    }
    if &data[..8] != b"OpusTags" {
        return Err("OpusTags magic missing".into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// File-level decode + diff
// ---------------------------------------------------------------------------

/// Validate Ogg page framing before handing the stream to `PacketReader`.
///
/// `PacketReader` deliberately treats incomplete trailing bytes as clean EOF
/// while searching for the next page. That is useful for seeking, but a
/// corpus comparator must reject a file that ends in a partial page instead
/// of counting the already-decoded prefix as a complete match.
fn validate_ogg_container(path: &Path) -> Result<(), String> {
    let mut file = File::open(path).map_err(|e| format!("open failed: {e}"))?;
    let mut page_index = 0usize;
    let mut header = [0u8; 27];

    loop {
        let first_len = file
            .read(&mut header[..1])
            .map_err(|e| format!("read Ogg page {page_index}: {e}"))?;
        if first_len == 0 {
            if page_index == 0 {
                return Err("empty stream (no pages)".into());
            }
            return Ok(());
        }
        file.read_exact(&mut header[1..])
            .map_err(|e| format!("truncated Ogg page header at page {page_index}: {e}"))?;
        if &header[..4] != b"OggS" {
            return Err(format!("missing OggS capture pattern at page {page_index}"));
        }
        if header[4] != 0 {
            return Err(format!(
                "unsupported Ogg stream structure version {} at page {page_index}",
                header[4]
            ));
        }

        let segment_count = header[26] as usize;
        let mut segments = vec![0u8; segment_count];
        file.read_exact(&mut segments)
            .map_err(|e| format!("truncated Ogg segment table at page {page_index}: {e}"))?;
        let payload_len: usize = segments.iter().map(|&segment| segment as usize).sum();
        let mut payload = vec![0u8; payload_len];
        file.read_exact(&mut payload)
            .map_err(|e| format!("truncated Ogg payload at page {page_index}: {e}"))?;
        page_index += 1;
    }
}

/// Outcome of comparing one file's output from both decoders.
enum FileOutcome {
    /// Every decoded sample matched.
    Match {
        packets: usize,
        samples_per_ch: usize,
    },
    /// The container or stream is malformed. This is always non-green, but is
    /// distinct from a decoder mismatch so corpus reports can identify bad
    /// input rather than blaming ropus.
    Malformed(String),
    /// A valid but unsupported stream family or channel layout. This is an
    /// exploratory skip and does not fail a run that has a supported match.
    Skipped(String),
    /// Decoders disagreed. Includes first-diff context.
    Mismatch(Mismatch),
    /// Either decoder returned an error on the same packet. Matching errors
    /// are classified as malformed input; mismatched errors remain a real
    /// decoder mismatch.
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
    if let Err(e) = validate_ogg_container(path) {
        return FileOutcome::Malformed(e);
    }

    // Open and wrap in a buffered reader. `ogg::PacketReader` needs `Read +
    // Seek`; `BufReader<File>` satisfies both.
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => return FileOutcome::Malformed(format!("open failed: {e}")),
    };
    let mut reader = PacketReader::new(BufReader::new(file));

    // Page 1: OpusHead.
    let head_pkt = match reader.read_packet() {
        Ok(Some(pkt)) => pkt,
        Ok(None) => return FileOutcome::Malformed("empty stream (no pages)".into()),
        Err(e) => return FileOutcome::Malformed(format!("read OpusHead page: {e}")),
    };
    let head = match parse_opus_head(&head_pkt.data) {
        Ok(h) => h,
        Err(e) => return FileOutcome::Malformed(e),
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

    // Page 2: OpusTags. Consume it so the next `read_packet()` call lands on
    // the first audio packet, but validate the packet before doing so. Without
    // this check an audio packet in the tags slot is silently discarded.
    let tags_pkt = match reader.read_packet() {
        Ok(Some(pkt)) => pkt,
        Ok(None) => return FileOutcome::Malformed("stream ended before OpusTags".into()),
        Err(e) => return FileOutcome::Malformed(format!("read OpusTags page: {e}")),
    };
    if let Err(e) = validate_opus_tags(&tags_pkt.data) {
        return FileOutcome::Malformed(e);
    }

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
                return FileOutcome::Malformed(format!(
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
                // Both decoders rejected the same packet. A matching decoder
                // error still means the file was not completely decoded, so
                // it must not be reported as a successful prefix match.
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
                return FileOutcome::Malformed(format!(
                    "both decoders rejected packet {packets_decoded}: ropus={r_code}, cref={c_code}"
                ));
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
    malformed: usize,
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
        if self.malformed > 0
            || self.mismatched > 0
            || self.panicked > 0
            || self.decoded_and_compared == 0
        {
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
            "CORPUS_DIFF_SUMMARY candidates={} decoded_and_compared={} zero_audio={} skipped={} deferred={} malformed={} mismatched={} panicked={}",
            self.candidates,
            self.decoded_and_compared,
            self.zero_audio,
            self.skipped,
            self.deferred,
            self.malformed,
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
            Ok(FileOutcome::Malformed(reason)) => {
                println!("  FAIL {display} — malformed: {reason}");
                stats.malformed += 1;
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
        "---\n{} decoded-and-compared, {} zero-audio, {} skipped, {} deferred, {} malformed, {} mismatched, {} panicked (of {} total)",
        stats.decoded_and_compared,
        stats.zero_audio,
        stats.skipped,
        stats.deferred,
        stats.malformed,
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
    use std::io::Cursor;

    use ogg::writing::{PacketWriteEndInfo, PacketWriter};
    use ropus::opus::encoder::{OPUS_APPLICATION_AUDIO, OpusEncoder};
    use tempfile::NamedTempFile;

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
            malformed: 3,
            decoded_and_compared: 1,
            ..RunStats::default()
        };
        let line = stats.summary_line();
        assert!(
            line.contains("deferred=3"),
            "missing deferred= field: {line}"
        );
        assert!(
            line.contains("malformed=3"),
            "missing malformed= field: {line}"
        );
        let skipped_pos = line.find("skipped=").expect("skipped= present");
        let deferred_pos = line.find("deferred=").expect("deferred= present");
        let malformed_pos = line.find("malformed=").expect("malformed= present");
        let mismatched_pos = line.find("mismatched=").expect("mismatched= present");
        assert!(
            skipped_pos < deferred_pos
                && deferred_pos < malformed_pos
                && malformed_pos < mismatched_pos,
            "expected order skipped<deferred<malformed<mismatched: {line}"
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
    fn corpus_malformed_is_non_green_even_with_a_valid_prefix() {
        let mut stats = RunStats {
            candidates: 2,
            malformed: 1,
            ..RunStats::default()
        };
        stats.record_match(960);

        assert_eq!(stats.decoded_and_compared, 1);
        assert_eq!(stats.exit_code(), 1);
        assert!(stats.summary_line().contains("malformed=1"));
    }

    #[test]
    fn opus_tags_magic_is_required() {
        assert!(validate_opus_tags(b"OpusTags").is_ok());
        assert!(
            validate_opus_tags(b"NotTags!")
                .expect_err("wrong magic must be rejected")
                .contains("magic")
        );
        assert!(
            validate_opus_tags(b"short")
                .expect_err("short tags must be rejected")
                .contains("too short")
        );
    }

    #[test]
    fn missing_opus_tags_is_malformed() {
        let file = write_test_ogg(b"NotTags!", &test_audio_packet());

        match diff_file(file.path()) {
            FileOutcome::Malformed(reason) => assert!(
                reason.contains("OpusTags"),
                "unexpected malformed reason: {reason}"
            ),
            _ => panic!("missing OpusTags must not be treated as a match or skip"),
        }
    }

    #[test]
    fn valid_prefix_with_malformed_tail_is_not_a_match() {
        let file = write_test_ogg(b"OpusTags", &test_audio_packet());
        let mut bytes = std::fs::read(file.path()).expect("read generated Ogg");
        bytes.extend_from_slice(b"OggS\x00");
        std::fs::write(file.path(), bytes).expect("append malformed Ogg tail");

        match diff_file(file.path()) {
            FileOutcome::Malformed(reason) => assert!(
                reason.contains("truncated Ogg page header"),
                "expected a trailing-page framing error, got: {reason}"
            ),
            FileOutcome::Match {
                packets,
                samples_per_ch,
            } => panic!("unexpected match: packets={packets}, samples_per_ch={samples_per_ch}"),
            FileOutcome::Skipped(reason) => panic!("unexpected skip: {reason}"),
            FileOutcome::Mismatch(_) => panic!("unexpected PCM mismatch"),
            FileOutcome::DecoderError(_) => panic!("unexpected decoder error"),
        }
    }

    #[test]
    fn matching_decoder_error_after_valid_prefix_is_malformed() {
        let valid_audio = test_audio_packet();
        let invalid_audio = [0xff];
        let file = write_test_ogg_packets(b"OpusTags", &[&valid_audio, &invalid_audio]);

        match diff_file(file.path()) {
            FileOutcome::Malformed(reason) => assert!(
                reason.contains("both decoders rejected packet 1"),
                "expected a matching decoder error, got: {reason}"
            ),
            FileOutcome::Match {
                packets,
                samples_per_ch,
            } => panic!("unexpected match: packets={packets}, samples_per_ch={samples_per_ch}"),
            FileOutcome::Skipped(reason) => panic!("unexpected skip: {reason}"),
            FileOutcome::Mismatch(_) => panic!("unexpected PCM mismatch"),
            FileOutcome::DecoderError(_) => panic!("unexpected decoder error mismatch"),
        }
    }

    fn test_audio_packet() -> Vec<u8> {
        let mut encoder = OpusEncoder::new(OPUS_SAMPLE_RATE_HZ, 1, OPUS_APPLICATION_AUDIO)
            .expect("create test encoder");
        let pcm = vec![0i16; 960];
        let mut packet = vec![0u8; 4000];
        let max_packet_bytes = packet.len() as i32;
        let length = encoder
            .encode(&pcm, 960, &mut packet, max_packet_bytes)
            .expect("encode test packet");
        packet.truncate(length as usize);
        packet
    }

    fn write_test_ogg(tags: &[u8], audio: &[u8]) -> NamedTempFile {
        write_test_ogg_packets(tags, &[audio])
    }

    fn write_test_ogg_packets(tags: &[u8], audio_packets: &[&[u8]]) -> NamedTempFile {
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = PacketWriter::new(&mut cursor);
            writer
                .write_packet(
                    b"OpusHead\x01\x01\x00\x00\x80\xbb\x00\x00\x00\x00\x00".to_vec(),
                    0x524f5055,
                    PacketWriteEndInfo::EndPage,
                    0,
                )
                .expect("write OpusHead");
            writer
                .write_packet(tags.to_vec(), 0x524f5055, PacketWriteEndInfo::EndPage, 0)
                .expect("write OpusTags");
            for (index, audio) in audio_packets.iter().enumerate() {
                writer
                    .write_packet(
                        audio.to_vec(),
                        0x524f5055,
                        PacketWriteEndInfo::EndPage,
                        960 * (index as u64 + 1),
                    )
                    .expect("write audio");
            }
        }

        let file = NamedTempFile::new().expect("create Ogg temp file");
        std::fs::write(file.path(), cursor.into_inner()).expect("write Ogg temp file");
        file
    }

    #[test]
    fn corpus_empty_candidate_set_preserves_exit_two() {
        assert_eq!(RunStats::default().exit_code(), 2);
    }
}
