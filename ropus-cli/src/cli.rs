//! Clap CLI definitions: top-level `Cli`, subcommand enum, per-subcommand
//! argument structs, and the application-kind value enum that maps to
//! `ropus::Application`.

use std::path::PathBuf;

use clap::{ArgAction, Parser, Subcommand, ValueEnum};
use ropus::Application;

#[derive(Parser, Debug)]
#[command(
    name = "ropus-cli",
    version,
    about = "Encode/decode/transcode/play audio with the ropus Opus codec",
    color = clap::ColorChoice::Auto,
)]
pub(crate) struct Cli {
    /// Suppress the banner line (useful when piping output).
    #[arg(short, long, global = true, action = ArgAction::SetTrue)]
    pub(crate) quiet: bool,

    /// Disable ANSI colour even on a TTY.
    #[arg(long, global = true, action = ArgAction::SetTrue)]
    pub(crate) no_color: bool,

    #[command(subcommand)]
    pub(crate) cmd: Command,
}

#[derive(Subcommand, Debug)]
pub(crate) enum Command {
    /// Encode an audio file to .opus (Ogg container).
    Encode(EncodeArgs),
    /// Decode an .opus file to a 16-bit PCM .wav.
    Decode(DecodeArgs),
    /// Alias for `encode`; takes any symphonia-supported input.
    Transcode(EncodeArgs),
    /// Play an audio file via the default output device.
    Play(PlayArgs),
    /// Print stream info for an .opus file.
    Info(InfoArgs),
}

#[derive(clap::Args, Debug)]
pub(crate) struct EncodeArgs {
    /// Input file (any format symphonia can decode).
    pub(crate) input: PathBuf,

    /// Output .opus file. Defaults to <input>.opus next to the input.
    #[arg(short = 'o', long)]
    pub(crate) output: Option<PathBuf>,

    /// Target bitrate in bits per second (e.g. 64000). Default: codec auto.
    #[arg(long)]
    pub(crate) bitrate: Option<u32>,

    /// Encoder complexity 0..=10 (higher = better quality, more CPU).
    #[arg(long)]
    pub(crate) complexity: Option<u8>,

    /// Application hint.
    #[arg(long, value_enum, default_value_t = AppKind::Audio)]
    pub(crate) application: AppKind,

    /// Use variable bitrate (default).
    #[arg(long, conflicts_with = "cbr")]
    pub(crate) vbr: bool,

    /// Use constant bitrate.
    #[arg(long)]
    pub(crate) cbr: bool,
}

#[derive(clap::Args, Debug)]
pub(crate) struct DecodeArgs {
    /// Input .opus file.
    pub(crate) input: PathBuf,

    /// Output .wav file. Defaults to <input>.wav next to the input.
    #[arg(short = 'o', long)]
    pub(crate) output: Option<PathBuf>,
}

#[derive(clap::Args, Debug)]
pub(crate) struct PlayArgs {
    /// Input audio file.
    pub(crate) input: PathBuf,

    /// Playback volume in [0.0, 1.0]. Defaults to 1.0.
    #[arg(long)]
    pub(crate) volume: Option<f32>,
}

#[derive(clap::Args, Debug)]
pub(crate) struct InfoArgs {
    /// Input .opus file.
    pub(crate) input: PathBuf,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum AppKind {
    /// VoIP / videoconference; biased toward intelligibility.
    Voip,
    /// General music and high-fidelity content.
    Audio,
    /// Lowest algorithmic delay.
    Lowdelay,
}

impl From<AppKind> for Application {
    fn from(a: AppKind) -> Application {
        match a {
            AppKind::Voip => Application::Voip,
            AppKind::Audio => Application::Audio,
            AppKind::Lowdelay => Application::RestrictedLowDelay,
        }
    }
}
