//! ropusplay — play any audio file via the default output device.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{ArgAction, Parser};
use ropus_tools_core::options::{LoopMode, PlayOptions};
use ropus_tools_core::prelude::{self, PreludeFlags};
use ropus_tools_core::{commands, ui};

/// Clap-facing mirror of `LoopMode`. Lives here so `ropus-tools-core` does not
/// need a clap dep; the `From` impl below keeps the library type the source of
/// truth for playback semantics.
#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum LoopArg {
    Off,
    All,
    Single,
}

impl From<LoopArg> for LoopMode {
    fn from(a: LoopArg) -> Self {
        match a {
            LoopArg::Off => LoopMode::Off,
            LoopArg::All => LoopMode::All,
            LoopArg::Single => LoopMode::Single,
        }
    }
}

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

    /// Repeat mode when input is a directory.
    #[arg(long = "loop", value_enum, ignore_case = true, default_value_t = LoopArg::Off)]
    loop_mode: LoopArg,

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
        loop_mode: args.loop_mode.into(),
        quiet,
    };
    prelude::run(commands::play(opts))
}
