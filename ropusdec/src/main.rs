//! ropusdec — decode an Ogg Opus file to a WAV (i16 PCM or f32 IEEE) or raw
//! interleaved PCM.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{ArgAction, Parser};
use ropus_tools_core::options::DecodeOptions;
use ropus_tools_core::prelude;
use ropus_tools_core::{commands, ui};

#[derive(Parser, Debug)]
#[command(
    name = "ropusdec",
    version,
    about = "Decode an Ogg Opus file to WAV or raw PCM using the ropus codec",
    color = clap::ColorChoice::Auto,
)]
struct Args {
    /// Input .opus file. Use `-` for stdin; the entire input is buffered
    /// in memory for Ogg page sniffing, so a multi-GB pipe will use that
    /// much RAM.
    input: PathBuf,

    /// Output path. Defaults to `<input>.wav` (or `<input>.pcm` with `--raw`),
    /// or stdout when input is `-`. Use `-` for stdout; progress/banner
    /// lines route to stderr in that case so the WAV/PCM stream stays clean.
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
    /// range is -128 dB..=32767/256 dB; out of range surfaces as a clean
    /// error.
    #[arg(
        long,
        value_name = "DB",
        default_value_t = 0.0,
        allow_negative_numbers = true,
        value_parser = parse_gain_db
    )]
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
    #[arg(
        long = "packet-loss",
        value_name = "PCT",
        default_value_t = 0,
        value_parser = clap::value_parser!(u8).range(0..=100)
    )]
    packet_loss: u8,

    #[arg(short, long, action = ArgAction::SetTrue)]
    quiet: bool,

    #[arg(long, action = ArgAction::SetTrue)]
    no_color: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();
    prelude::configure_color(args.no_color);

    // Clap has already resolved every option value, so the `-` here is the
    // authoritative input/output sentinel rather than a guessed positional.
    let output_is_stdout = prelude::output_is_stdout(&args.input, args.output.as_deref());
    if !args.quiet {
        if output_is_stdout {
            ui::print_banner_stderr(
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION"),
                env!("BUILD_TIMESTAMP"),
                env!("BUILD_GIT_SHA"),
            );
        } else {
            ui::print_banner(
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION"),
                env!("BUILD_TIMESTAMP"),
                env!("BUILD_GIT_SHA"),
            );
        }
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

fn parse_gain_db(raw: &str) -> Result<f32, String> {
    let value = raw
        .parse::<f32>()
        .map_err(|e| format!("gain must be a number ({e})"))?;
    const MAX_GAIN_DB: f32 = 32_767.0 / 256.0;
    if !value.is_finite() {
        return Err("gain must be finite".to_string());
    }
    if !(-128.0..=MAX_GAIN_DB).contains(&value) {
        return Err(format!(
            "gain {value} dB out of range [-128.0, {MAX_GAIN_DB}]"
        ));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_value_option_can_surround_stdin_without_changing_implicit_stdout() {
        let value_options = [
            ("--rate", "44100"),
            ("--gain", "-3.0"),
            ("--packet-loss", "5"),
        ];

        for (option, value) in value_options {
            for argv in [
                vec!["ropusdec", option, value, "-"],
                vec!["ropusdec", "-", option, value],
            ] {
                let args = Args::try_parse_from(&argv)
                    .unwrap_or_else(|error| panic!("{argv:?} should parse: {error}"));
                assert!(
                    prelude::output_is_stdout(&args.input, args.output.as_deref()),
                    "{argv:?} must select implicit stdout"
                );
            }
        }
    }

    #[test]
    fn invalid_packet_loss_and_gain_are_rejected_at_cli_boundary() {
        assert!(Args::try_parse_from(["ropusdec", "input.opus", "--packet-loss", "101"]).is_err());
        assert!(parse_gain_db("128").is_err());
        assert!(parse_gain_db("NaN").is_err());
        assert!(parse_gain_db("-128").is_ok());
    }
}
