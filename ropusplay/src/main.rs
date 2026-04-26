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
    /// Input audio file. Ignored when `--list-devices` is set.
    #[arg(required_unless_present = "list_devices")]
    input: Option<PathBuf>,

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

    /// Enumerate available cpal output devices (one per line) and exit.
    #[arg(long, action = ArgAction::SetTrue)]
    list_devices: bool,

    /// Exact (case-sensitive) name of the cpal output device to open.
    /// Defaults to the host's default output device.
    #[arg(long, value_name = "NAME")]
    device: Option<String>,

    /// Linear-dB gain applied to f32 PCM samples before playback.
    /// Range `[-128.0, 128.0]` (matches libopus `OPUS_SET_GAIN`);
    /// 0.0 is a no-op. NaN / ±∞ are rejected.
    #[arg(
        long,
        value_name = "DB",
        default_value_t = 0.0,
        allow_hyphen_values = true
    )]
    gain: f32,
}

fn main() -> ExitCode {
    let PreludeFlags {
        quiet, no_color: _, ..
    } = prelude::run_prelude();
    if !quiet {
        ui::print_banner(
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION"),
            env!("BUILD_TIMESTAMP"),
            env!("BUILD_GIT_SHA"),
        );
    }
    let args = Args::parse();
    // `input` is only optional at the clap layer — `required_unless_present`
    // guarantees it is `Some` on every non-`--list-devices` invocation, and
    // the `--list-devices` branch in `commands::play` returns before ever
    // looking at `input`. A `PathBuf::new()` fallback keeps the type shape
    // simple without introducing an Option into the library surface.
    let opts = PlayOptions {
        input: args.input.unwrap_or_default(),
        volume: args.volume,
        loop_mode: args.loop_mode.into(),
        quiet,
        device: args.device,
        list_devices: args.list_devices,
        gain_db: args.gain,
    };
    prelude::run(commands::play(opts))
}
