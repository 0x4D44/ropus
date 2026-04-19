//! Plain option structs consumed by `commands::*`.
//!
//! Deliberately clap-agnostic: each binary (`ropusenc`, `ropusdec`, …) owns
//! its own `#[derive(Parser)]` definition and maps it to one of these structs.
//! Keeping clap out of the library surface means tests, GUIs, and the fb2k
//! plugin can call the commands without depending on clap.

use std::path::PathBuf;

use ropus::{Application, FrameDuration, Signal};

#[derive(Debug)]
pub struct EncodeOptions {
    pub input: PathBuf,
    pub output: Option<PathBuf>,
    pub bitrate: Option<u32>,
    pub complexity: Option<u8>,
    pub application: Application,
    pub vbr: bool,
    /// When true, enables constrained VBR (`--cvbr`). Only meaningful if
    /// `vbr` is also true — libopus treats constrained VBR on top of CBR as
    /// a no-op. Mutually exclusive with `--hard-cbr` at the CLI layer.
    pub vbr_constraint: bool,
    /// Signal-content hint. Defaults to `Signal::Auto`; `--music` / `--speech`
    /// pin it to `Signal::Music` / `Signal::Voice`.
    pub signal: Signal,
    /// Per-packet frame duration. Default `Ms20` reproduces the current
    /// hardcoded behaviour; `--framesize` picks one of the supported values.
    pub frame_duration: FrameDuration,
    /// Packet-loss percentage hint (0..=100) for the encoder's FEC planning.
    /// 0 disables the hint; higher values trade bitrate for robustness.
    pub expect_loss: u8,
    /// When true and input is stereo, mix L+R to mono before resample. No-op
    /// for mono input; error for >2-channel input.
    pub downmix_to_mono: bool,
    /// Override the hardcoded Ogg stream serial. `None` uses the library's
    /// default constant (`container::ogg::OGG_STREAM_SERIAL`).
    pub serial: Option<u32>,
    /// Optional path to a JPEG/PNG cover image. If set, the encoder wraps the
    /// file bytes in a METADATA_BLOCK_PICTURE structure and appends it to
    /// `comments` as a base64-encoded `METADATA_BLOCK_PICTURE=…` entry.
    pub picture_path: Option<PathBuf>,
    /// Written verbatim into the OpusTags `vendor` field. Each binary passes
    /// its own `CARGO_PKG_NAME` so downstream `opusinfo` output identifies
    /// which tool produced the file.
    pub vendor: String,
    /// Each entry is a raw `"KEY=value"` Vorbis comment to emit in OpusTags.
    /// Empty vector means "no user comments" — valid per RFC 7845 and the
    /// current default behaviour. Step 3 of the opus-tools-parity HLD wires
    /// CLI flags (`--artist`, `--title`, `--comment`, …) into this field.
    pub comments: Vec<String>,
}

#[derive(Debug)]
pub struct DecodeOptions {
    pub input: PathBuf,
    pub output: Option<PathBuf>,
    /// Emit 32-bit IEEE float samples instead of 16-bit PCM. Flips both the
    /// decoder path (`decode_float`) and the WAV writer (format code 3 + fact
    /// chunk). Dither is silently a no-op in this mode.
    pub float: bool,
    /// Skip the WAV header and write raw interleaved samples (LE) to the
    /// output file. Combines with `float`: `raw && float` = raw f32 LE,
    /// `raw && !float` = raw i16 LE.
    pub raw: bool,
    /// Target sample rate. `None` keeps the codec's native 48 kHz. When set,
    /// post-decode resample runs *after* the pre-skip trim so the resampler
    /// can't smear leading silence into real output.
    pub rate: Option<u32>,
    /// User gain in dB, summed with `OpusHead.output_gain` and pushed through
    /// `Decoder::set_gain` before any samples emerge. Sum is range-checked by
    /// libopus (`±128 dB`); out-of-range surfaces as a clean error, not a
    /// panic.
    pub gain_db: f32,
    /// Apply TPDF dither to i16 output. Default `true` matches `opusdec`.
    /// Ignored for the float path — nothing to dither when bit depth is 32.
    pub dither: bool,
    /// Simulated random packet loss percentage (0..=100) for PLC exercising.
    /// 0 means "never drop" and short-circuits the PRNG entirely so the output
    /// is bit-identical to leaving the flag unset.
    pub packet_loss_pct: u8,
}

#[derive(Debug)]
pub struct InfoOptions {
    pub input: PathBuf,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LoopMode {
    #[default]
    Off,
    All,
    Single,
}

#[derive(Debug)]
pub struct PlayOptions {
    pub input: PathBuf,
    pub volume: Option<f32>,
    /// Repeat behaviour for directory playback.
    pub loop_mode: LoopMode,
}
