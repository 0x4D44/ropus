//! CLI parsing for `full-test`.
//!
//! Five flags, exactly as the HLD specifies:
//! - `--quick`
//! - `--skip-quality`
//! - `--skip-coverage`
//! - `--skip-benchmarks`
//! - `--skip-ambisonics`
//!
//! `--help` / `-h` prints usage and exits 0. Any other argument is an error
//! (exit 2). The surface here is tiny enough that a hand parser keeps the
//! binary lean and the tests trivial; later phases can pull in clap if the
//! flag set grows.

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Options {
    pub quick: bool,
    pub skip_quality: bool,
    pub skip_coverage: bool,
    pub skip_benchmarks: bool,
    pub skip_ambisonics: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ParseOutcome {
    Options(Options),
    HelpRequested,
    Error(String),
}

/// Parse an argv tail (program name already stripped).
pub fn parse(args: &[String]) -> ParseOutcome {
    let mut opts = Options::default();
    for a in args {
        match a.as_str() {
            "--help" | "-h" => return ParseOutcome::HelpRequested,
            "--quick" => opts.quick = true,
            "--skip-quality" => opts.skip_quality = true,
            "--skip-coverage" => opts.skip_coverage = true,
            "--skip-benchmarks" => opts.skip_benchmarks = true,
            "--skip-ambisonics" => opts.skip_ambisonics = true,
            other => return ParseOutcome::Error(format!("unknown argument: {other}")),
        }
    }
    ParseOutcome::Options(opts)
}

pub fn print_help() {
    println!("full-test — ropus validation runner (Phase 1)");
    println!();
    println!("USAGE:");
    println!("    cargo run --release -p full-test -- [FLAGS]");
    println!();
    println!("FLAGS:");
    println!("    --quick             Skip stages 1 (quality) and 4 (benchmarks).");
    println!("    --skip-quality      Skip stage 1 (cargo fmt + clippy).");
    println!("    --skip-coverage     Downgrade stage 2 to plain `cargo test` (Phase 2+).");
    println!("    --skip-benchmarks   Skip stage 4 (Phase 4+).");
    println!("    --skip-ambisonics   Skip stage 3 (Phase 3+).");
    println!("    -h, --help          Show this help and exit.");
    println!();
    println!("Phase 1 only exercises stage 0 (setup capture) and stage 1 (quality).");
    println!("Other --skip-* flags are recorded for later phases.");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn empty_argv_yields_defaults() {
        assert_eq!(parse(&[]), ParseOutcome::Options(Options::default()));
    }

    #[test]
    fn each_flag_flips_its_bit() {
        type FlagCase = (&'static str, fn(&Options) -> bool);
        let cases: &[FlagCase] = &[
            ("--quick", |o| o.quick),
            ("--skip-quality", |o| o.skip_quality),
            ("--skip-coverage", |o| o.skip_coverage),
            ("--skip-benchmarks", |o| o.skip_benchmarks),
            ("--skip-ambisonics", |o| o.skip_ambisonics),
        ];
        for (flag, getter) in cases {
            let parsed = parse(&args(&[flag]));
            match parsed {
                ParseOutcome::Options(o) => assert!(getter(&o), "flag {flag} did not set its bit"),
                _ => panic!("flag {flag} should parse as Options, got {parsed:?}"),
            }
        }
    }

    #[test]
    fn all_flags_together() {
        let parsed = parse(&args(&[
            "--quick",
            "--skip-quality",
            "--skip-coverage",
            "--skip-benchmarks",
            "--skip-ambisonics",
        ]));
        let expected = Options {
            quick: true,
            skip_quality: true,
            skip_coverage: true,
            skip_benchmarks: true,
            skip_ambisonics: true,
        };
        assert_eq!(parsed, ParseOutcome::Options(expected));
    }

    #[test]
    fn help_short_and_long() {
        assert_eq!(parse(&args(&["-h"])), ParseOutcome::HelpRequested);
        assert_eq!(parse(&args(&["--help"])), ParseOutcome::HelpRequested);
        // --help short-circuits even if other flags precede or follow.
        assert_eq!(
            parse(&args(&["--quick", "--help"])),
            ParseOutcome::HelpRequested
        );
    }

    #[test]
    fn unknown_flag_is_error() {
        match parse(&args(&["--wat"])) {
            ParseOutcome::Error(msg) => assert!(msg.contains("--wat")),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[test]
    fn positional_argument_is_error() {
        match parse(&args(&["some-file.wav"])) {
            ParseOutcome::Error(msg) => assert!(msg.contains("some-file.wav")),
            other => panic!("expected Error, got {other:?}"),
        }
    }
}
