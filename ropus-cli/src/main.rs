//! ropus-cli — command-line front-end for the ropus Opus codec.
//!
//! Subcommands:
//!   encode      Decode any audio input (WAV/MP3/FLAC/OGG/AAC/...) and re-encode
//!               to a `.opus` file in the standard Ogg container.
//!   decode      Decode a `.opus` file to a 16-bit PCM `.wav` file.
//!   transcode   Alias for `encode`; familiar verb for ffmpeg users.
//!   play        Play any audio file via the system default audio device.
//!   info        Print stream info for an Opus file.
//!
//! Banner is printed as the first line on every invocation unless `--quiet`
//! / `-q` is supplied (so output remains pipe-friendly).
//!
//! The codec is `ropus`; this binary is a thin glue layer around it: it
//! handles container parsing/writing, sample-rate conversion and audio I/O.
//!
//! Module layout:
//! ```text
//! main ──► commands ──► audio::{decode, resample, wav}
//!                  └──► container::ogg
//!                  └──► cli, ui, util, consts
//! ```
//!
//! `consts.rs` is the single home for shared numeric constants
//! (sample rate, frame sizing). `OGG_STREAM_SERIAL` deliberately
//! lives in `container::ogg` because only the encoder writes it;
//! readers use the input stream's actual serial.

mod audio;
mod cli;
mod commands;
mod consts;
mod container;
mod ui;
mod util;

use std::env;
use std::process::ExitCode;

use clap::Parser;
use colored::*;

use crate::cli::{Cli, Command};
use crate::ui::print_banner;

fn main() -> ExitCode {
    // Detect --quiet / -q and --no-color from raw argv before clap runs, so the
    // banner can be suppressed even when the user passes --help/--version
    // (clap exits before our main body runs after a parse).
    let raw: Vec<String> = env::args().collect();
    let quiet_early = raw.iter().any(|a| a == "-q" || a == "--quiet");
    let no_color_early = raw.iter().any(|a| a == "--no-color");

    if no_color_early {
        colored::control::set_override(false);
    }

    if !quiet_early {
        print_banner();
    }

    let cli = Cli::parse();

    let result = match cli.cmd {
        Command::Encode(a) | Command::Transcode(a) => commands::encode(a),
        Command::Decode(a) => commands::decode(a),
        Command::Play(a) => commands::play(a),
        Command::Info(a) => commands::info(a),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{} {}", "error:".red().bold(), e);
            for cause in e.chain().skip(1) {
                eprintln!("  {} {}", "caused by:".red(), cause);
            }
            ExitCode::FAILURE
        }
    }
}
