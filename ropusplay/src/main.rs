//! ropusplay — play any audio file via the default output device.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{ArgAction, Parser};
use ropus_tools_core::options::PlayOptions;
use ropus_tools_core::prelude::{self, PreludeFlags};
use ropus_tools_core::{commands, ui};

#[derive(Parser, Debug)]
#[command(
    name = "ropusplay",
    version,
    about = "Play an audio file via the default output device using the ropus codec",
    color = clap::ColorChoice::Auto,
)]
struct Args {
    /// Input audio file.
    input: PathBuf,

    /// Playback volume in [0.0, 1.0]. Defaults to 1.0.
    #[arg(long)]
    volume: Option<f32>,

    #[arg(short, long, action = ArgAction::SetTrue)]
    quiet: bool,

    #[arg(long, action = ArgAction::SetTrue)]
    no_color: bool,
}

fn main() -> ExitCode {
    let PreludeFlags { quiet, no_color: _ } = prelude::run_prelude();
    if !quiet {
        ui::print_banner(
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION"),
            env!("BUILD_TIMESTAMP"),
            env!("BUILD_GIT_SHA"),
        );
    }
    let args = Args::parse();
    let opts = PlayOptions {
        input: args.input,
        volume: args.volume,
    };
    prelude::run(commands::play(opts))
}
