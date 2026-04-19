//! ropusdec — decode an Ogg Opus file to a 16-bit PCM WAV.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{ArgAction, Parser};
use ropus_tools_core::options::DecodeOptions;
use ropus_tools_core::prelude::{self, PreludeFlags};
use ropus_tools_core::{commands, ui};

#[derive(Parser, Debug)]
#[command(
    name = "ropusdec",
    version,
    about = "Decode an Ogg Opus file to 16-bit PCM WAV using the ropus codec",
    color = clap::ColorChoice::Auto,
)]
struct Args {
    /// Input .opus file.
    input: PathBuf,

    /// Output .wav file. Defaults to <input>.wav next to the input.
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,

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
    let opts = DecodeOptions {
        input: args.input,
        output: args.output,
    };
    prelude::run(commands::decode(opts))
}
