//! Shared library backing the ropus command-line tools
//! (`ropusenc`, `ropusdec`, `ropusinfo`, `ropusplay`).
//!
//! Each binary owns its own clap surface and build-info plumbing; everything
//! else — audio decode/resample, Ogg container R/W, the four command
//! implementations, and the banner/error-chain helpers — lives here so each
//! `main.rs` stays a thin dispatch layer.
//!
//! Module layout:
//! ```text
//! prelude  ──► (binary main: argv sniff + anyhow-chain exit code)
//! commands ──► audio::{decode, resample, wav}
//!         └──► container::ogg
//!         └──► ui, util, consts, options
//! ```
//!
//! `OGG_STREAM_SERIAL` deliberately lives inside `container::ogg` because it
//! belongs to the encode-side writer — reverse-scanning a file for a last
//! granule uses the input's own serial, never this fabricated one.

pub mod audio;
pub mod commands;
pub mod consts;
pub mod container;
pub mod options;
pub mod prelude;
pub mod ui;
pub mod util;
