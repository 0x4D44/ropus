//! ropusplay — play any audio file via the default output device.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{ArgAction, ColorChoice, CommandFactory, FromArgMatches, Parser};
use ropus_tools_core::options::{LoopMode, PlayOptions};
use ropus_tools_core::prelude;
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
    #[arg(long, value_parser = parse_volume)]
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

    /// dB gain applied during decode before playback. Opus combines it with
    /// OpusHead.output_gain in the decoder; other codecs use a linear f32
    /// multiplier.
    /// Range `[-128.0, 32767/256]` (the representable decoder Q8 range);
    /// 0.0 is a no-op. NaN / ±∞ are rejected.
    #[arg(
        long,
        value_name = "DB",
        default_value_t = 0.0,
        allow_hyphen_values = true,
        value_parser = parse_gain_db
    )]
    gain: f32,
}

fn parse_volume(raw: &str) -> Result<f32, String> {
    let value = raw
        .parse::<f32>()
        .map_err(|e| format!("volume must be a number ({e})"))?;
    if !value.is_finite() {
        return Err("volume must be finite".to_string());
    }
    if !(0.0..=1.0).contains(&value) {
        return Err(format!("volume {value} out of range [0.0, 1.0]"));
    }
    Ok(value)
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

fn command_with_color(color: ColorChoice) -> clap::Command {
    Args::command().color(color)
}

fn parse_args() -> Args {
    let color = if prelude::no_color_requested() {
        ColorChoice::Never
    } else {
        ColorChoice::Auto
    };
    let matches = command_with_color(color).get_matches();
    Args::from_arg_matches(&matches).expect("Clap already validated the command line")
}

fn main() -> ExitCode {
    let args = parse_args();
    prelude::configure_color(args.no_color);
    if !args.quiet && !args.list_devices {
        ui::print_banner(
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION"),
            env!("BUILD_TIMESTAMP"),
            env!("BUILD_GIT_SHA"),
        );
    }
    // `input` is only optional at the clap layer — `required_unless_present`
    // guarantees it is `Some` on every non-`--list-devices` invocation, and
    // the `--list-devices` branch in `commands::play` returns before ever
    // looking at `input`. A `PathBuf::new()` fallback keeps the type shape
    // simple without introducing an Option into the library surface.
    let opts = PlayOptions {
        input: args.input.unwrap_or_default(),
        volume: args.volume,
        loop_mode: args.loop_mode.into(),
        quiet: args.quiet,
        device: args.device,
        list_devices: args.list_devices,
        gain_db: args.gain,
    };
    prelude::run(commands::play(opts))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_rejects_nonfinite_or_out_of_range_volume() {
        assert!(parse_volume("NaN").is_err());
        assert!(parse_volume("-0.1").is_err());
        assert!(parse_volume("1.1").is_err());
        assert_eq!(parse_volume("0.5").unwrap(), 0.5);
    }

    #[test]
    fn cli_rejects_unrepresentable_gain() {
        assert!(parse_gain_db("128").is_err());
        assert!(parse_gain_db("Infinity").is_err());
        assert!(parse_gain_db("-128").is_ok());
    }

    #[test]
    fn no_color_disables_clap_ansi_for_help_and_errors() {
        let help = command_with_color(ColorChoice::Never)
            .render_help()
            .to_string();
        assert!(
            !help.contains('\x1b'),
            "help unexpectedly contains ANSI: {help:?}"
        );

        let error = command_with_color(ColorChoice::Never)
            .try_get_matches_from(["ropusplay", "--no-color", "--unknown"])
            .expect_err("unknown flag must fail parsing");
        assert!(
            !error.to_string().contains('\x1b'),
            "error unexpectedly contains ANSI: {error}"
        );
    }
}
