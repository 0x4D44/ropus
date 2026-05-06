//! CLI parsing for `full-test`.
//!
//! The five HLD flags plus Phase 4/release-gate additions:
//! - `--quick`
//! - `--skip-quality`
//! - `--skip-coverage`
//! - `--skip-benchmarks`
//! - `--skip-ambisonics`
//! - `--emit-json` (Phase 4: keep the old JSON envelope on stdout in addition
//!   to the HTML report — opt-in so the default shell output is the one-line
//!   summary plus report path, not a 200 KiB JSON dump).
//! - `--release-preflight` (asset policy depends on whether `--quick` is set).
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
    pub emit_json: bool,
    pub release_preflight: bool,
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
            "--emit-json" => opts.emit_json = true,
            "--release-preflight" => opts.release_preflight = true,
            other => return ParseOutcome::Error(format!("unknown argument: {other}")),
        }
    }
    ParseOutcome::Options(opts)
}

pub fn help_text() -> &'static str {
    concat!(
        "full-test — ropus validation runner (Phase 4)\n",
        "\n",
        "USAGE:\n",
        "    cargo run --release -p full-test -- [FLAGS]\n",
        "\n",
        "FLAGS:\n",
        "    --quick             Skip stages 1 and 4; with --release-preflight, use a core smoke profile with no neural/DRED or corpus claim.\n",
        "    --skip-quality      Skip stage 1 (cargo fmt + clippy).\n",
        "    --skip-coverage     Downgrade stage 2 to plain `cargo test`.\n",
        "    --skip-benchmarks   Skip stage 4 (bench sweep).\n",
        "    --skip-ambisonics   Skip stage 3 (projection roundtrip).\n",
        "    --emit-json         Also print the JSON envelope on stdout.\n",
        "    --release-preflight Check release assets; non-quick claims core + neural/DRED + generated corpus gates.\n",
        "    -h, --help          Show this help and exit.\n",
        "\n",
        "Report lands at tests/results/full_test_<YYYYMMDD_HHMMSS>.html.\n",
        "Exit code: 0 on PASS/WARN, 1 on FAIL (see HLD § PASS / FAIL / WARN).\n",
    )
}

pub fn print_help() {
    print!("{}", help_text());
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
            ("--emit-json", |o| o.emit_json),
            ("--release-preflight", |o| o.release_preflight),
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
            "--emit-json",
            "--release-preflight",
        ]));
        let expected = Options {
            quick: true,
            skip_quality: true,
            skip_coverage: true,
            skip_benchmarks: true,
            skip_ambisonics: true,
            emit_json: true,
            release_preflight: true,
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
    fn help_text_documents_release_preflight_claim_profiles() {
        assert!(help_text().contains(
            "    --quick             Skip stages 1 and 4; with --release-preflight, use a core smoke profile with no neural/DRED or corpus claim."
        ));
        assert!(help_text().contains(
            "    --release-preflight Check release assets; non-quick claims core + neural/DRED + generated corpus gates."
        ));
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
