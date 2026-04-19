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
