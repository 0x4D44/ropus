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
use ropus_tools_core::options::EncodeOptions;
use ropus_tools_core::prelude::{self, PreludeFlags};
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

    /// Target bitrate in bits per second (e.g. 64000). Default: codec auto.
    #[arg(long)]
    bitrate: Option<u32>,

    /// Encoder complexity 0..=10 (higher = better quality, more CPU).
    /// `--comp` is a shorter alias matching opus-tools' opusenc.
    #[arg(long, alias = "comp")]
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

fn main() -> ExitCode {
    // Sniff --quiet/--no-color from raw argv before clap runs so the banner
    // decision is still honoured when the user passes --help / --version
    // (clap exits before our main body would otherwise see them).
    // `output_is_stdout` steers the banner to stderr so the bitstream on
    // stdout isn't polluted with text.
    let PreludeFlags {
        quiet,
        no_color: _,
        output_is_stdout,
    } = prelude::run_prelude();
    if !quiet {
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

    let cli = Cli::parse();

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

    let opts = EncodeOptions {
        input: cli.input,
        output: cli.output,
        bitrate: cli.bitrate,
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

    prelude::run(commands::encode(opts))
}
