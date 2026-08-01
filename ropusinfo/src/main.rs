//! ropusinfo — print stream info for an Ogg Opus file.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{ArgAction, Parser};
use ropus_tools_core::options::InfoOptions;
use ropus_tools_core::prelude;
use ropus_tools_core::{commands, ui};

#[derive(Parser, Debug)]
#[command(
    name = "ropusinfo",
    version,
    about = "Print stream info for an Ogg Opus file",
    color = clap::ColorChoice::Auto,
)]
struct Args {
    /// Input .opus file.
    input: PathBuf,

    /// Suppress banner. Long-form only — `-q` is reserved for `--query` to
    /// match opus-tools' `opusinfo -q` muscle memory. All *other* ropus
    /// binaries (`ropusenc`, `ropusdec`, `ropusplay`) still accept `-q` for
    /// quiet; this divergence is localised to `ropusinfo` on purpose.
    #[arg(long, action = ArgAction::SetTrue)]
    quiet: bool,

    /// Per-packet TOC decode. Adds a `Packets:` section and a granule-gap
    /// list after the default human-readable block.
    #[arg(short = 'e', long, action = ArgAction::SetTrue)]
    extended: bool,

    /// Print one named value (bare, no banner, no colour). Keys:
    /// `channels`, `samplerate`, `preskip`, `gain`, `duration`, `bitrate`,
    /// `vendor`, or `comment:KEY` for a case-insensitive tag lookup.
    #[arg(short = 'q', long)]
    query: Option<String>,

    #[arg(long, action = ArgAction::SetTrue)]
    no_color: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();
    prelude::configure_color(args.no_color);

    // Query mode is authoritative only after Clap has handled attached short
    // values and `--` end-of-options semantics.
    if banner_enabled(&args) {
        ui::print_banner(
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION"),
            env!("BUILD_TIMESTAMP"),
            env!("BUILD_GIT_SHA"),
        );
    }
    let opts = InfoOptions {
        input: args.input,
        extended: args.extended,
        query: args.query,
    };
    prelude::run(commands::info(opts))
}

fn banner_enabled(args: &Args) -> bool {
    !args.quiet && args.query.is_none()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_query_controls_banner_for_all_value_spellings() {
        for argv in [
            vec!["ropusinfo", "-q", "duration", "input.opus"],
            vec!["ropusinfo", "-q=duration", "input.opus"],
            vec!["ropusinfo", "--query", "duration", "input.opus"],
            vec!["ropusinfo", "--query=duration", "input.opus"],
            vec!["ropusinfo", "input.opus", "-q", "duration"],
        ] {
            let args = Args::try_parse_from(&argv)
                .unwrap_or_else(|error| panic!("{argv:?} should parse: {error}"));
            assert!(!banner_enabled(&args), "{argv:?} must suppress the banner");
        }
    }

    #[test]
    fn end_of_options_keeps_query_like_input_positional() {
        let args = Args::try_parse_from(["ropusinfo", "--", "-q=duration"])
            .expect("query-like filename after -- should parse as input");
        assert_eq!(args.input, PathBuf::from("-q=duration"));
        assert!(args.query.is_none());
        assert!(banner_enabled(&args));
    }
}
