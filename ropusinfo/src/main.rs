//! ropusinfo — print stream info for an Ogg Opus file.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{ArgAction, Parser};
use ropus_tools_core::options::InfoOptions;
use ropus_tools_core::prelude::{self, PreludeFlags};
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
    let PreludeFlags { quiet, no_color: _ } = prelude::run_prelude();
    // `--query` implicitly disables the banner regardless of `--quiet`, per
    // the HLD. Sniff argv for `--query` or `-q …` here so we don't print the
    // banner before clap has had a chance to parse. We explicitly exclude a
    // bare `-q` (which, under opus-tools' convention, needs an argument) —
    // only `-q <value>` or `--query <value>` / `--query=<value>` disable it.
    //
    // Known limitation: we don't recognise the `-q=VALUE` short-flag-with-equals
    // form (clap accepts it). If a user passes that, the sniff falls through
    // and the banner will print — clap still parses correctly, so the query
    // value is honoured; the banner is cosmetic overspill, not a correctness
    // issue.
    let raw: Vec<String> = std::env::args().collect();
    let has_query = raw
        .iter()
        .enumerate()
        .any(|(i, a)| a == "--query" || a.starts_with("--query=") || (a == "-q" && i + 1 < raw.len()));

    if !quiet && !has_query {
        ui::print_banner(
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION"),
            env!("BUILD_TIMESTAMP"),
            env!("BUILD_GIT_SHA"),
        );
    }
    let args = Args::parse();
    let opts = InfoOptions {
        input: args.input,
        extended: args.extended,
        query: args.query,
    };
    prelude::run(commands::info(opts))
}
