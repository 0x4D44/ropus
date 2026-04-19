//! Plain option structs consumed by `commands::*`.
//!
//! Deliberately clap-agnostic: each binary (`ropusenc`, `ropusdec`, …) owns
//! its own `#[derive(Parser)]` definition and maps it to one of these structs.
//! Keeping clap out of the library surface means tests, GUIs, and the fb2k
//! plugin can call the commands without depending on clap.

use std::path::PathBuf;

use ropus::Application;

#[derive(Debug)]
pub struct EncodeOptions {
    pub input: PathBuf,
    pub output: Option<PathBuf>,
    pub bitrate: Option<u32>,
    pub complexity: Option<u8>,
    pub application: Application,
    pub vbr: bool,
    /// Written verbatim into the OpusTags `vendor` field. Each binary passes
    /// its own `CARGO_PKG_NAME` so downstream `opusinfo` output identifies
    /// which tool produced the file.
    pub vendor: String,
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

#[derive(Debug)]
pub struct PlayOptions {
    pub input: PathBuf,
    pub volume: Option<f32>,
}
