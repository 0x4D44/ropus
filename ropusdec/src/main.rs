//! ropusdec — decode an Ogg Opus file to a WAV (i16 PCM or f32 IEEE) or raw
//! interleaved PCM.

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
    about = "Decode an Ogg Opus file to WAV or raw PCM using the ropus codec",
    color = clap::ColorChoice::Auto,
)]
struct Args {
    /// Input .opus file.
    input: PathBuf,

    /// Output path. Defaults to `<input>.wav` (or `<input>.pcm` with `--raw`).
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,

    /// Emit 32-bit IEEE float samples (WAV format code 3 + fact chunk, or
    /// raw f32 LE when combined with `--raw`). Disables dither silently.
    #[arg(long, action = ArgAction::SetTrue)]
    float: bool,

    /// Target sample rate in Hz. When set, decoded 48 kHz PCM is resampled
    /// *after* the pre-skip trim and before dither/write. Accepted range:
    /// 8000..=192000. Default keeps the codec's native 48 kHz.
    #[arg(long, value_name = "HZ")]
    rate: Option<u32>,

    /// User gain in dB. Added on top of the header `output_gain` and applied
    /// through the decoder's `set_gain` (fixed-point, pre-clamp). Total
    /// range is ±128 dB; out of range surfaces as a clean error.
    #[arg(long, value_name = "DB", default_value_t = 0.0, allow_negative_numbers = true)]
    gain: f32,

    /// Skip TPDF dither on the i16 output. No-op for `--float`.
    #[arg(long = "no-dither", action = ArgAction::SetTrue)]
    no_dither: bool,

    /// Write raw interleaved samples (LE) with no WAV header. Combines with
    /// `--float` for raw f32 LE output.
    #[arg(long, action = ArgAction::SetTrue)]
    raw: bool,

    /// Simulate random packet loss (0..=100 %) to exercise PLC. Deterministic
    /// seed — the same value reproduces the same dropped-packet pattern.
    #[arg(long = "packet-loss", value_name = "PCT", default_value_t = 0)]
    packet_loss: u8,

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
    if args.packet_loss > 100 {
        eprintln!(
            "error: --packet-loss {} is out of range (accepted: 0..=100)",
            args.packet_loss
        );
        return ExitCode::from(1);
    }
    let opts = DecodeOptions {
        input: args.input,
        output: args.output,
        float: args.float,
        raw: args.raw,
        rate: args.rate,
        gain_db: args.gain,
        dither: !args.no_dither,
        packet_loss_pct: args.packet_loss,
    };
    prelude::run(commands::decode(opts))
}
