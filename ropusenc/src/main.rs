//! ropusenc — encode any symphonia-supported input to Ogg Opus.
//!
//! Single flat-argument surface (`ropusenc INPUT [-o OUTPUT] ...`). The
//! earlier `transcode` subcommand was removed because clap's double-flatten
//! (top-level + subcommand) silently dropped top-level flags when the verb
//! was present; the flat path already handles MP3/FLAC/OGG/AAC inputs
//! transparently via symphonia, so the verb was pure footgun.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{ArgAction, Parser, ValueEnum};
use ropus_tools_core::Application;
use ropus_tools_core::options::EncodeOptions;
use ropus_tools_core::prelude::{self, PreludeFlags};
use ropus_tools_core::{commands, ui};

#[derive(Parser, Debug)]
#[command(
    name = "ropusenc",
    version,
    about = "Encode an audio file to Ogg Opus using the ropus codec",
    color = clap::ColorChoice::Auto,
)]
struct Cli {
    /// Input file (any format symphonia can decode).
    input: PathBuf,

    /// Output .opus file. Defaults to <input>.opus next to the input.
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,

    /// Target bitrate in bits per second (e.g. 64000). Default: codec auto.
    #[arg(long)]
    bitrate: Option<u32>,

    /// Encoder complexity 0..=10 (higher = better quality, more CPU).
    #[arg(long)]
    complexity: Option<u8>,

    /// Application hint.
    #[arg(long, value_enum, default_value_t = AppKind::Audio)]
    application: AppKind,

    /// Use variable bitrate (default).
    #[arg(long, conflicts_with = "cbr")]
    vbr: bool,

    /// Use constant bitrate.
    #[arg(long)]
    cbr: bool,

    #[arg(short, long, action = ArgAction::SetTrue)]
    quiet: bool,

    #[arg(long, action = ArgAction::SetTrue)]
    no_color: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum AppKind {
    Voip,
    Audio,
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

fn main() -> ExitCode {
    // Sniff --quiet/--no-color from raw argv before clap runs so the banner
    // decision is still honoured when the user passes --help / --version
    // (clap exits before our main body would otherwise see them).
    let PreludeFlags { quiet, no_color: _ } = prelude::run_prelude();
    if !quiet {
        ui::print_banner(
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION"),
            env!("BUILD_TIMESTAMP"),
            env!("BUILD_GIT_SHA"),
        );
    }

    let cli = Cli::parse();

    let opts = EncodeOptions {
        input: cli.input,
        output: cli.output,
        bitrate: cli.bitrate,
        complexity: cli.complexity,
        application: cli.application.into(),
        vbr: !cli.cbr,
        vendor: env!("CARGO_PKG_NAME").to_string(),
        comments: Vec::new(),
    };

    prelude::run(commands::encode(opts))
}
