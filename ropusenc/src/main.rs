//! ropusenc — encode any symphonia-supported input to Ogg Opus.
//!
//! Single flat-argument surface (`ropusenc INPUT [-o OUTPUT] ...`). The
//! earlier `transcode` subcommand was removed because clap's double-flatten
//! (top-level + subcommand) silently dropped top-level flags when the verb
//! was present; the flat path already handles MP3/FLAC/OGG/AAC inputs
//! transparently via symphonia, so the verb was pure footgun.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{ArgAction, ColorChoice, CommandFactory, FromArgMatches, Parser, ValueEnum};
use ropus_tools_core::options::{EncodeOptions, OutputPolicy};
use ropus_tools_core::prelude;
use ropus_tools_core::{Application, FrameDuration, Signal, commands, ui};

#[derive(Parser, Debug)]
#[command(
    name = "ropusenc",
    version,
    about = "Encode an audio file to Ogg Opus using the ropus codec",
    color = clap::ColorChoice::Auto,
)]
struct Cli {
    /// Input file (any format symphonia can decode). Use `-` for stdin; the
    /// entire input is buffered in memory for format probing, so a multi-GB
    /// pipe will use that much RAM.
    input: PathBuf,

    /// Output .opus file. Defaults to <input>.opus next to the input
    /// (or stdout when input is `-`). Use `-` for stdout; progress/banner
    /// lines route to stderr in that case so the bitstream stays clean.
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,

    /// Target bitrate in bits per second (e.g. 64000). Default: 160 kbps
    /// stereo / 96 kbps mono (≈ MP3 320 kbps quality, with the 50 % efficiency
    /// edge Opus has over MP3). The mono tier kicks in only when `--downmix
    /// mono` is set; mono inputs without `--downmix` get the stereo-tier
    /// default since channel count isn't known until decode.
    #[arg(long, value_parser = parse_bitrate)]
    bitrate: Option<u32>,

    /// Encoder complexity 0..=10 (higher = better quality, more CPU).
    /// `--comp` is a shorter alias matching opus-tools' opusenc.
    #[arg(long, alias = "comp", value_parser = parse_complexity)]
    complexity: Option<u8>,

    /// Application hint.
    #[arg(long, value_enum, default_value_t = AppKind::Audio)]
    application: AppKind,

    /// Use variable bitrate (default).
    #[arg(long, conflicts_with_all = ["cbr", "hard_cbr", "cvbr"])]
    vbr: bool,

    /// Use constant bitrate (legacy alias kept for existing muscle memory;
    /// equivalent to --hard-cbr).
    #[arg(long, conflicts_with_all = ["vbr", "cvbr"])]
    cbr: bool,

    /// Use hard constant bitrate — disables VBR and constrained VBR.
    /// Mutually exclusive with --cvbr and --vbr.
    #[arg(long = "hard-cbr", conflicts_with_all = ["cvbr", "vbr", "cbr"])]
    hard_cbr: bool,

    /// Use constrained variable bitrate (CVBR). Mutually exclusive with
    /// --hard-cbr and plain --vbr false.
    #[arg(long, conflicts_with_all = ["hard_cbr", "cbr"])]
    cvbr: bool,

    /// Hint that the input is music content (same as the encoder's
    /// Signal::Music). Mutually exclusive with --speech.
    #[arg(long, conflicts_with = "speech")]
    music: bool,

    /// Hint that the input is speech content (same as the encoder's
    /// Signal::Voice). Mutually exclusive with --music.
    #[arg(long)]
    speech: bool,

    /// Frame size in ms. One of 2.5, 5, 10, 20, 40, 60, 80, 100, 120.
    #[arg(long, value_enum, default_value_t = FrameSizeArg::Ms20)]
    framesize: FrameSizeArg,

    /// Hint the encoder that ~N% of packets will be lost (0..=100).
    /// Trades bitrate for FEC robustness. Default 0 (disabled).
    #[arg(long = "expect-loss", default_value_t = 0, value_parser = clap::value_parser!(u8).range(0..=100))]
    expect_loss: u8,

    /// Mix stereo input to mono before encoding. Only `mono` is accepted;
    /// see the HLD for why surround → stereo is out of scope.
    #[arg(long, value_enum)]
    downmix: Option<DownmixArg>,

    /// Override the Ogg logical stream serial number. Must be non-zero —
    /// RFC 3533 allows 0 but many downstream players treat it as a sentinel
    /// for "uninitialised", so we reject rather than risk silent breakage.
    #[arg(long, value_parser = parse_nonzero_serial)]
    serial: Option<u32>,

    /// Artist name → ARTIST=NAME Vorbis comment.
    #[arg(long)]
    artist: Option<String>,

    /// Track title → TITLE=NAME Vorbis comment.
    #[arg(long)]
    title: Option<String>,

    /// Album name → ALBUM=NAME Vorbis comment.
    #[arg(long)]
    album: Option<String>,

    /// Genre → GENRE=NAME Vorbis comment.
    #[arg(long)]
    genre: Option<String>,

    /// Date → DATE=NAME Vorbis comment.
    #[arg(long)]
    date: Option<String>,

    /// Track number → TRACKNUMBER=NAME Vorbis comment.
    #[arg(long)]
    tracknumber: Option<String>,

    /// Extra Vorbis comment in `KEY=VALUE` form. May be repeated.
    /// Rejects entries that do not contain a `=`.
    #[arg(long = "comment", action = ArgAction::Append, value_parser = parse_comment_kv)]
    comment: Vec<String>,

    /// Attach a PNG or JPEG picture as a METADATA_BLOCK_PICTURE tag
    /// (Front Cover).
    #[arg(long)]
    picture: Option<PathBuf>,

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

/// Frame-size values accepted by `--framesize`. The weird naming (`Ms2_5`)
/// avoids clap's default kebab-case mangling producing a flag spelling like
/// `2-5`; instead we get `2.5` because we override `name` below.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum FrameSizeArg {
    #[value(name = "2.5")]
    Ms2_5,
    #[value(name = "5")]
    Ms5,
    #[value(name = "10")]
    Ms10,
    #[value(name = "20")]
    Ms20,
    #[value(name = "40")]
    Ms40,
    #[value(name = "60")]
    Ms60,
    #[value(name = "80")]
    Ms80,
    #[value(name = "100")]
    Ms100,
    #[value(name = "120")]
    Ms120,
}

impl From<FrameSizeArg> for FrameDuration {
    fn from(f: FrameSizeArg) -> FrameDuration {
        match f {
            FrameSizeArg::Ms2_5 => FrameDuration::Ms2_5,
            FrameSizeArg::Ms5 => FrameDuration::Ms5,
            FrameSizeArg::Ms10 => FrameDuration::Ms10,
            FrameSizeArg::Ms20 => FrameDuration::Ms20,
            FrameSizeArg::Ms40 => FrameDuration::Ms40,
            FrameSizeArg::Ms60 => FrameDuration::Ms60,
            FrameSizeArg::Ms80 => FrameDuration::Ms80,
            FrameSizeArg::Ms100 => FrameDuration::Ms100,
            FrameSizeArg::Ms120 => FrameDuration::Ms120,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum DownmixArg {
    Mono,
}

/// Validate a `--comment KEY=VALUE` argument. Clap calls this once per
/// occurrence; rejecting at parse time gives a clean error rather than a
/// runtime bail deep inside the encode pipeline.
///
/// Vorbis comment spec (§5) constrains field names to ASCII 0x20..=0x7D
/// excluding `=` (0x3D) and forbids empty keys. Enforce here so malformed
/// tags never reach the OpusTags writer.
fn parse_comment_kv(raw: &str) -> Result<String, String> {
    let (key, _value) = raw
        .split_once('=')
        .ok_or_else(|| format!("comment must be KEY=VALUE (missing '=' in {raw:?})"))?;
    if key.is_empty() {
        return Err(format!("comment key must not be empty (in {raw:?})"));
    }
    // §5 allows 0x20..=0x7D except 0x3D ('='). The split above has already
    // removed the first '=', so a stray '=' inside the key half means the
    // user wrote `=value` or `=key=value` — both invalid.
    for &b in key.as_bytes() {
        if !(0x20..=0x7D).contains(&b) || b == b'=' {
            return Err(format!(
                "comment key must be ASCII 0x20..=0x7D excluding '=' (bad byte {b:#04x} in {raw:?})"
            ));
        }
    }
    Ok(raw.to_string())
}

/// Reject `--serial 0`. See the flag doc-comment for the rationale.
fn parse_nonzero_serial(raw: &str) -> Result<u32, String> {
    let v: u32 = raw
        .parse()
        .map_err(|e| format!("--serial must be a non-negative integer ({e})"))?;
    if v == 0 {
        return Err("--serial must be non-zero".to_string());
    }
    Ok(v)
}

fn parse_bitrate(raw: &str) -> Result<u32, String> {
    let value = raw
        .parse::<u32>()
        .map_err(|e| format!("bitrate must be a non-negative integer ({e})"))?;
    if value == 0 {
        return Err("bitrate must be greater than zero".to_string());
    }
    if value > i32::MAX as u32 {
        return Err(format!(
            "bitrate {value} bps exceeds the libopus i32::MAX limit"
        ));
    }
    Ok(value)
}

fn parse_complexity(raw: &str) -> Result<u8, String> {
    let value = raw
        .parse::<u8>()
        .map_err(|e| format!("complexity must be an integer 0..=10 ({e})"))?;
    if value > 10 {
        return Err(format!(
            "complexity {value} out of range (accepted: 0..=10)"
        ));
    }
    Ok(value)
}

fn command_with_color(color: ColorChoice) -> clap::Command {
    Cli::command().color(color)
}

fn parse_cli() -> Cli {
    let color = if prelude::no_color_requested() {
        ColorChoice::Never
    } else {
        ColorChoice::Auto
    };
    let matches = command_with_color(color).get_matches();
    Cli::from_arg_matches(&matches).expect("Clap already validated the command line")
}

fn main() -> ExitCode {
    let cli = parse_cli();
    prelude::configure_color(cli.no_color);

    // Derive banner routing from Clap's typed result. Raw argv cannot safely
    // identify the input sentinel because value-taking options may precede it.
    let output_is_stdout = prelude::output_is_stdout(&cli.input, cli.output.as_deref());
    if !cli.quiet {
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

    // Build comments vector from the flattened metadata flags. Order matches
    // opus-tools' output: artist, title, album, tracknumber, genre, date,
    // then user-supplied --comment entries verbatim.
    let mut comments: Vec<String> = Vec::new();
    if let Some(v) = cli.artist.as_ref() {
        comments.push(format!("ARTIST={v}"));
    }
    if let Some(v) = cli.title.as_ref() {
        comments.push(format!("TITLE={v}"));
    }
    if let Some(v) = cli.album.as_ref() {
        comments.push(format!("ALBUM={v}"));
    }
    if let Some(v) = cli.tracknumber.as_ref() {
        comments.push(format!("TRACKNUMBER={v}"));
    }
    if let Some(v) = cli.genre.as_ref() {
        comments.push(format!("GENRE={v}"));
    }
    if let Some(v) = cli.date.as_ref() {
        comments.push(format!("DATE={v}"));
    }
    comments.extend(cli.comment.iter().cloned());

    // Rate-mode resolution. Default = VBR on, constraint off. --cvbr keeps
    // VBR on but sets constraint. --hard-cbr (or legacy --cbr) disables both.
    // clap's conflicts_with_all prevents contradictory combinations reaching
    // this point, so straightforward booleans suffice here.
    let (vbr, vbr_constraint) = if cli.hard_cbr || cli.cbr {
        (false, false)
    } else if cli.cvbr {
        (true, true)
    } else {
        // Default and plain --vbr both map to plain VBR.
        (true, false)
    };

    let signal = if cli.music {
        Signal::Music
    } else if cli.speech {
        Signal::Voice
    } else {
        Signal::Auto
    };

    let downmix_to_mono = matches!(cli.downmix, Some(DownmixArg::Mono));

    // Default bitrate: 160 kbps stereo / 96 kbps mono. Targets MP3 320 kbps
    // quality (Opus reaches transparency around 160 kbps stereo per
    // Hydrogenaudio listening tests). Mono tier requires `--downmix mono`
    // because the CLI doesn't yet know the input channel count — that's
    // discovered inside `commands::encode` after symphonia decode. A truly
    // mono input run without `--downmix` therefore gets a slightly generous
    // 160 kbps; libopus' VBR will still spend less than the target on
    // simple material, so the cost is small.
    let bitrate = cli
        .bitrate
        .or(Some(if downmix_to_mono { 96_000 } else { 160_000 }));

    let opts = EncodeOptions {
        input: cli.input,
        output: cli.output,
        bitrate,
        complexity: cli.complexity,
        application: cli.application.into(),
        vbr,
        vbr_constraint,
        signal,
        frame_duration: cli.framesize.into(),
        expect_loss: cli.expect_loss,
        downmix_to_mono,
        serial: cli.serial,
        picture_path: cli.picture,
        // opus-tools writes `"libopus VERSION"` into the vendor field so
        // downstream `opusinfo | grep Vendor` workflows see tool + version.
        // Match the NAME-space-VERSION shape with our own crate identity.
        vendor: concat!(env!("CARGO_PKG_NAME"), " ", env!("CARGO_PKG_VERSION")).to_string(),
        comments,
    };

    prelude::run(commands::encode_with_policy(
        opts,
        OutputPolicy { quiet: cli.quiet },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_value_option_can_surround_stdin_without_changing_implicit_stdout() {
        let value_options = [
            ("--bitrate", "64000"),
            ("--complexity", "5"),
            ("--comp", "5"),
            ("--application", "audio"),
            ("--framesize", "20"),
            ("--expect-loss", "5"),
            ("--downmix", "mono"),
            ("--serial", "1"),
            ("--artist", "Artist"),
            ("--title", "Title"),
            ("--album", "Album"),
            ("--genre", "Genre"),
            ("--date", "2026"),
            ("--tracknumber", "1"),
            ("--comment", "KEY=VALUE"),
            ("--picture", "cover.png"),
        ];

        for (option, value) in value_options {
            for argv in [
                vec!["ropusenc", option, value, "-"],
                vec!["ropusenc", "-", option, value],
            ] {
                let cli = Cli::try_parse_from(&argv)
                    .unwrap_or_else(|error| panic!("{argv:?} should parse: {error}"));
                assert!(
                    prelude::output_is_stdout(&cli.input, cli.output.as_deref()),
                    "{argv:?} must select implicit stdout"
                );
            }
        }
    }

    #[test]
    fn invalid_bitrate_and_complexity_are_rejected_at_cli_boundary() {
        assert!(parse_bitrate("0").is_err());
        assert!(parse_bitrate("2147483648").is_err());
        assert!(parse_complexity("11").is_err());
        assert!(parse_complexity("255").is_err());
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
            .try_get_matches_from(["ropusenc", "--no-color", "--unknown"])
            .expect_err("unknown flag must fail parsing");
        assert!(
            !error.to_string().contains('\x1b'),
            "error unexpectedly contains ANSI: {error}"
        );
    }
}
