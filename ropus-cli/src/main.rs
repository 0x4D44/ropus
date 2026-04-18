//! ropus-cli — command-line front-end for the ropus Opus codec.
//!
//! Subcommands:
//!   encode      Decode any audio input (WAV/MP3/FLAC/OGG/AAC/...) and re-encode
//!               to a `.opus` file in the standard Ogg container.
//!   decode      Decode a `.opus` file to a 16-bit PCM `.wav` file.
//!   transcode   Alias for `encode`; familiar verb for ffmpeg users.
//!   play        Play any audio file via the system default audio device.
//!   info        Print stream info for an Opus file.
//!
//! Banner is printed as the first line on every invocation unless `--quiet`
//! / `-q` is supplied (so output remains pipe-friendly).
//!
//! The codec is `ropus`; this binary is a thin glue layer around it: it
//! handles container parsing/writing, sample-rate conversion and audio I/O.

use std::env;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result, anyhow, bail};
use clap::{ArgAction, Parser, Subcommand, ValueEnum};
use colored::*;

use ropus::{
    Application, Bitrate, Channels as RopusChannels, Decoder as RopusDecoder, Encoder, Signal,
};

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, CODEC_TYPE_OPUS, Decoder as SymphoniaDecoder, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use ogg::reading::PacketReader;
use ogg::writing::{PacketWriteEndInfo, PacketWriter};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Opus output sample rate (the codec works natively at 48 kHz).
const OPUS_SR: u32 = 48_000;
/// 20 ms frame at 48 kHz = 960 samples per channel.
const FRAME_SAMPLES_PER_CH: usize = 960;
/// Logical Ogg stream serial. Any non-zero value works for a single stream we
/// write ourselves. NEVER use this for matching against an arbitrary input
/// stream; capture the input's serial from its first OggS page instead.
const OGG_STREAM_SERIAL: u32 = 0xC0DE_C0DE;
/// Maximum bytes for a single Opus frame, per RFC 6716 §3.2.1.
const MAX_OPUS_FRAME_BYTES: usize = 1275;
/// Number of Opus frames per packet for our current encoder config (20 ms = 1 frame).
/// If `--frame-duration` becomes configurable (40/60/80/100/120 ms = 2/3/4/5/6
/// frames per packet), update this constant in lockstep with the encoder so
/// `MAX_PACKET_BYTES` remains correct. The `debug_assert_eq!` at the top of
/// `cmd_encode` enforces the relationship.
const FRAMES_PER_PACKET: usize = 1;
/// Maximum Opus packet bytes for our config: `MAX_OPUS_FRAME_BYTES * FRAMES_PER_PACKET`.
const MAX_PACKET_BYTES: usize = MAX_OPUS_FRAME_BYTES * FRAMES_PER_PACKET;

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "ropus-cli",
    version,
    about = "Encode/decode/transcode/play audio with the ropus Opus codec",
    color = clap::ColorChoice::Auto,
)]
struct Cli {
    /// Suppress the banner line (useful when piping output).
    #[arg(short, long, global = true, action = ArgAction::SetTrue)]
    quiet: bool,

    /// Disable ANSI colour even on a TTY.
    #[arg(long, global = true, action = ArgAction::SetTrue)]
    no_color: bool,

    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
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
struct EncodeArgs {
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
}

#[derive(clap::Args, Debug)]
struct DecodeArgs {
    /// Input .opus file.
    input: PathBuf,

    /// Output .wav file. Defaults to <input>.wav next to the input.
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,
}

#[derive(clap::Args, Debug)]
struct PlayArgs {
    /// Input audio file.
    input: PathBuf,

    /// Playback volume in [0.0, 1.0]. Defaults to 1.0.
    #[arg(long)]
    volume: Option<f32>,
}

#[derive(clap::Args, Debug)]
struct InfoArgs {
    /// Input .opus file.
    input: PathBuf,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum AppKind {
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

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> ExitCode {
    // Detect --quiet / -q and --no-color from raw argv before clap runs, so the
    // banner can be suppressed even when the user passes --help/--version
    // (clap exits before our main body runs after a parse).
    let raw: Vec<String> = env::args().collect();
    let quiet_early = raw.iter().any(|a| a == "-q" || a == "--quiet");
    let no_color_early = raw.iter().any(|a| a == "--no-color");

    if no_color_early {
        colored::control::set_override(false);
    }

    if !quiet_early {
        print_banner();
    }

    let cli = Cli::parse();

    let result = match cli.cmd {
        Command::Encode(a) | Command::Transcode(a) => cmd_encode(a),
        Command::Decode(a) => cmd_decode(a),
        Command::Play(a) => cmd_play(a),
        Command::Info(a) => cmd_info(a),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{} {}", "error:".red().bold(), e);
            for cause in e.chain().skip(1) {
                eprintln!("  {} {}", "caused by:".red(), cause);
            }
            ExitCode::FAILURE
        }
    }
}

fn print_banner() {
    let name = env!("CARGO_PKG_NAME").bright_cyan().bold();
    let version = env!("CARGO_PKG_VERSION").bright_white();
    let timestamp = env!("BUILD_TIMESTAMP");
    let sha = env!("BUILD_GIT_SHA");
    let suffix = format!("(build {timestamp}, sha {sha})").dimmed();
    println!("{name} {version} {suffix}");
}

fn heading(text: &str) {
    println!("{}", text.bright_yellow().bold());
}

fn ok(text: &str) {
    println!("{}", text.green());
}

// ---------------------------------------------------------------------------
// encode / transcode
// ---------------------------------------------------------------------------

fn cmd_encode(args: EncodeArgs) -> Result<()> {
    // Guard the encoder's output buffer sizing. If a future `--frame-duration`
    // flag wires in (40/60/80/100/120 ms = 2/3/4/5/6 frames per packet) and
    // FRAMES_PER_PACKET is bumped without updating MAX_PACKET_BYTES (or vice
    // versa), this fires immediately in debug builds rather than overflowing
    // the buffer at runtime.
    debug_assert_eq!(
        MAX_PACKET_BYTES,
        MAX_OPUS_FRAME_BYTES * FRAMES_PER_PACKET,
        "MAX_PACKET_BYTES must equal MAX_OPUS_FRAME_BYTES * FRAMES_PER_PACKET; \
         update both if --frame-duration becomes configurable"
    );

    heading("encode");
    println!("input    {}", args.input.display().to_string().cyan());

    let output = args
        .output
        .clone()
        .unwrap_or_else(|| with_extension(&args.input, "opus"));
    println!("output   {}", output.display().to_string().cyan());

    // 1. Decode the input to interleaved f32 PCM.
    let DecodedAudio {
        samples,
        sample_rate,
        channels,
    } = decode_to_f32(&args.input).context("decoding input")?;
    println!(
        "decoded  {} samples, {} Hz, {} ch",
        format_num(samples.len() as u64).bright_white(),
        sample_rate.to_string().bright_white(),
        channels.to_string().bright_white(),
    );

    // 2. Resample to 48 kHz if needed.
    let pcm_48k = if sample_rate == OPUS_SR {
        samples
    } else {
        println!("resample {} Hz -> {} Hz", sample_rate, OPUS_SR);
        resample_to_48k(&samples, sample_rate, channels)
            .context("resampling to 48 kHz")?
    };
    println!(
        "resampled {} samples @ 48 kHz",
        format_num(pcm_48k.len() as u64).bright_white(),
    );

    // 3. Build the encoder.
    let opus_channels = channel_count_to_ropus(channels)?;
    let mut builder =
        Encoder::builder(OPUS_SR, opus_channels, Application::from(args.application));
    if let Some(b) = args.bitrate {
        builder = builder.bitrate(Bitrate::Bits(b));
    }
    if let Some(c) = args.complexity {
        builder = builder.complexity(c);
    }
    builder = builder.signal(Signal::Auto);
    if args.cbr {
        builder = builder.vbr(false);
    } else {
        builder = builder.vbr(true);
    }
    let mut encoder = builder
        .build()
        .map_err(|e| anyhow!("encoder build failed: {e}"))?;

    // Query the encoder for its actual lookahead in 48 kHz samples; that is
    // exactly the value RFC 7845 requires in OpusHead.pre_skip (typically
    // 312; 120 in OPUS_APPLICATION_RESTRICTED_LOWDELAY). Real libopus values
    // are always well under 65 535; a value that doesn't fit in u16 means the
    // encoder is in a broken state, so we bail loudly rather than silently
    // capping at u16::MAX and producing an OpusHead with a wrong pre_skip.
    let lookahead = encoder.lookahead();
    let pre_skip = u16::try_from(lookahead).map_err(|_| {
        anyhow!(
            "encoder lookahead {} does not fit in u16 — likely corrupt encoder state",
            lookahead
        )
    })?;

    // 4. Open Ogg writer.
    let file = File::create(&output)
        .with_context(|| format!("creating output file {}", output.display()))?;
    let mut writer = PacketWriter::new(BufWriter::new(file));

    // 5. Emit OpusHead and OpusTags headers (each on its own page per RFC 7845).
    let head = build_opus_head(channels as u8, sample_rate, pre_skip);
    writer
        .write_packet(head, OGG_STREAM_SERIAL, PacketWriteEndInfo::EndPage, 0)
        .context("writing OpusHead page")?;

    let tags = build_opus_tags("ropus-cli");
    writer
        .write_packet(tags, OGG_STREAM_SERIAL, PacketWriteEndInfo::EndPage, 0)
        .context("writing OpusTags page")?;

    // 6. Encode and write data packets in 20 ms frames.
    let frame_interleaved = FRAME_SAMPLES_PER_CH * channels;
    let mut packet_buf = vec![0u8; MAX_PACKET_BYTES];
    let mut samples_written: u64 = 0;
    let mut packet_count: u64 = 0;
    let total_chunks = pcm_48k.len() / frame_interleaved;
    let remainder_len = pcm_48k.len() - total_chunks * frame_interleaved;
    let has_tail = remainder_len > 0;
    let chunks = pcm_48k.chunks_exact(frame_interleaved);
    let remainder_owned: Vec<f32> = chunks.remainder().to_vec();
    for (idx, chunk) in chunks.enumerate() {
        let n = encoder
            .encode_float(chunk, &mut packet_buf)
            .map_err(|e| anyhow!("encode failed: {e}"))?;
        // Advance the granule position by the per-channel frame length;
        // RFC 7845 sec. 4 mandates granule position is in 48 kHz samples.
        samples_written += FRAME_SAMPLES_PER_CH as u64;
        let is_last = idx + 1 == total_chunks && !has_tail;
        let end_info = if is_last {
            PacketWriteEndInfo::EndStream
        } else {
            PacketWriteEndInfo::NormalPacket
        };
        writer
            .write_packet(
                packet_buf[..n].to_vec(),
                OGG_STREAM_SERIAL,
                end_info,
                samples_written,
            )
            .context("writing Opus data page")?;
        packet_count += 1;
    }

    // Pad and encode any tail (silence-pad to a 20 ms frame).
    if has_tail {
        let mut frame_buf = vec![0.0f32; frame_interleaved];
        frame_buf[..remainder_owned.len()].copy_from_slice(&remainder_owned);
        let n = encoder
            .encode_float(&frame_buf, &mut packet_buf)
            .map_err(|e| anyhow!("encode failed (tail): {e}"))?;
        samples_written += FRAME_SAMPLES_PER_CH as u64;
        writer
            .write_packet(
                packet_buf[..n].to_vec(),
                OGG_STREAM_SERIAL,
                PacketWriteEndInfo::EndStream,
                samples_written,
            )
            .context("writing tail Opus packet")?;
        packet_count += 1;
    }

    println!(
        "wrote    {} packets, {} samples (granule)",
        format_num(packet_count).bright_white(),
        format_num(samples_written).bright_white(),
    );
    ok(&format!("encoded -> {}", output.display()));
    Ok(())
}

// ---------------------------------------------------------------------------
// decode
// ---------------------------------------------------------------------------

fn cmd_decode(args: DecodeArgs) -> Result<()> {
    heading("decode");
    println!("input    {}", args.input.display().to_string().cyan());

    let output = args
        .output
        .clone()
        .unwrap_or_else(|| with_extension(&args.input, "wav"));
    println!("output   {}", output.display().to_string().cyan());

    let file = File::open(&args.input)
        .with_context(|| format!("opening {}", args.input.display()))?;
    let mut reader = PacketReader::new(BufReader::new(file));

    // Header packet: OpusHead.
    let head_pkt = reader
        .read_packet()?
        .ok_or_else(|| anyhow!("no packets found in input"))?;
    let head = parse_opus_head(&head_pkt.data)?;
    println!(
        "header   ch={} input_sr={} pre_skip={}",
        head.channels.to_string().bright_white(),
        head.input_sample_rate.to_string().bright_white(),
        head.pre_skip.to_string().bright_white(),
    );

    // Tags packet: OpusTags. Verify magic to catch malformed files (e.g. a
    // stripped tags page would otherwise silently consume the first audio
    // packet).
    read_opus_tags(&mut reader).context("reading OpusTags packet")?;

    let opus_channels = channel_count_to_ropus(head.channels as usize)?;
    let mut decoder = RopusDecoder::new(OPUS_SR, opus_channels)
        .map_err(|e| anyhow!("decoder init failed: {e}"))?;

    // Maximum 120 ms of decoded samples at 48 kHz, per channel.
    let max_per_ch = (OPUS_SR / 1000 * 120) as usize;
    let mut decoded = vec![0i16; max_per_ch * opus_channels.count()];

    // We accumulate decoded samples then write a single WAV when finished.
    let mut all_pcm: Vec<i16> = Vec::with_capacity(1 << 20);
    let mut packet_count: u64 = 0;
    while let Some(pkt) = reader.read_packet()? {
        let n = decoder
            .decode(&pkt.data, &mut decoded, false)
            .map_err(|e| anyhow!("decode failed: {e}"))?;
        // n is samples per channel.
        let total = n * opus_channels.count();
        all_pcm.extend_from_slice(&decoded[..total]);
        packet_count += 1;
    }

    // Trim the leading pre-skip samples.
    let pre_skip = head.pre_skip as usize * opus_channels.count();
    let pre_skip = pre_skip.min(all_pcm.len());
    let trimmed = &all_pcm[pre_skip..];

    write_wav_pcm16(&output, trimmed, OPUS_SR, opus_channels.count() as u16)
        .context("writing WAV")?;

    println!(
        "decoded  {} packets, {} samples ({} after pre-skip)",
        format_num(packet_count).bright_white(),
        format_num(all_pcm.len() as u64).bright_white(),
        format_num(trimmed.len() as u64).bright_white(),
    );
    ok(&format!("decoded -> {}", output.display()));
    Ok(())
}

// ---------------------------------------------------------------------------
// info
// ---------------------------------------------------------------------------

fn cmd_info(args: InfoArgs) -> Result<()> {
    heading("info");
    println!("file     {}", args.input.display().to_string().cyan());

    let file = File::open(&args.input)
        .with_context(|| format!("opening {}", args.input.display()))?;
    let file_len = file
        .metadata()
        .ok()
        .map(|m| m.len())
        .unwrap_or(0);
    let mut reader = PacketReader::new(BufReader::new(file));

    let head_pkt = reader
        .read_packet()?
        .ok_or_else(|| anyhow!("empty file"))?;
    let head = parse_opus_head(&head_pkt.data)?;
    // Capture the OpusHead's stream serial — this identifies the logical Opus
    // bitstream we care about in a multiplexed Ogg file. We need it to filter
    // the reverse-scan in `read_last_granule` to the right stream.
    let target_serial = head_pkt.stream_serial();
    read_opus_tags(&mut reader).context("reading OpusTags packet")?;

    let opus_channels = channel_count_to_ropus(head.channels as usize)?;

    // Fast path: locate the last Ogg page belonging to our serial and read
    // its absolute granule position. Per RFC 7845 §4 that is the total stream
    // length in 48 kHz samples, including pre_skip warm-up.
    let mut fast_file = File::open(&args.input)
        .with_context(|| format!("opening {} for granule scan", args.input.display()))?;
    let absgp_opt =
        read_last_granule(&mut fast_file, target_serial).context("scanning for last Ogg page")?;

    let sample_count = if let Some(absgp) = absgp_opt {
        // absgp is the cumulative per-channel sample count at end-of-stream
        // (48 kHz). Subtract pre_skip to get the real decoded duration.
        absgp.saturating_sub(head.pre_skip as u64)
    } else {
        // Truncated or unknown end (absgp == 0xFFFF_FFFF_FFFF_FFFF). Fall back
        // to the slow path: decode every packet to recover the sample count.
        let mut decoder = RopusDecoder::new(OPUS_SR, opus_channels)
            .map_err(|e| anyhow!("decoder init failed: {e}"))?;
        let max_per_ch = (OPUS_SR / 1000 * 120) as usize;
        let mut decoded = vec![0i16; max_per_ch * opus_channels.count()];

        let mut packet_idx: u64 = 0;
        let mut sample_count: u64 = 0;
        while let Some(pkt) = reader.read_packet()? {
            // We need to know how many samples the packet decodes to. The
            // simplest correct approach is to actually decode it and count.
            match decoder.decode(&pkt.data, &mut decoded, false) {
                Ok(n) => sample_count += n as u64,
                Err(e) => {
                    eprintln!("{} packet {}: {e}", "warning:".yellow(), packet_idx);
                }
            }
            packet_idx += 1;
        }
        // Subtract pre_skip from the total so the reported duration matches the
        // fast path (which subtracts pre_skip from absgp).
        sample_count.saturating_sub(head.pre_skip as u64)
    };

    let duration_s = sample_count as f64 / OPUS_SR as f64;
    let avg_kbps = if duration_s > 0.0 {
        (file_len as f64 * 8.0) / duration_s / 1000.0
    } else {
        0.0
    };

    println!("container       {}", "Ogg".bright_white());
    println!("opus_version    {}", head.version.to_string().bright_white());
    println!("channels        {}", head.channels.to_string().bright_white());
    println!(
        "input_sr        {} Hz",
        head.input_sample_rate.to_string().bright_white()
    );
    println!(
        "decode_sr       {} Hz",
        OPUS_SR.to_string().bright_white()
    );
    println!("pre_skip        {}", head.pre_skip.to_string().bright_white());
    println!(
        "channel_mapping {}",
        head.channel_mapping.to_string().bright_white()
    );
    println!(
        "total_samples   {} ({} per ch)",
        format_num(sample_count * opus_channels.count() as u64).bright_white(),
        format_num(sample_count).bright_white(),
    );
    println!(
        "duration        {} s",
        format!("{:.3}", duration_s).bright_white()
    );
    println!("file_size       {} bytes", format_num(file_len).bright_white());
    println!(
        "avg_bitrate     {} kbps",
        format!("{:.1}", avg_kbps).bright_white()
    );

    Ok(())
}

/// Scan backwards from EOF for the last Ogg `OggS` page belonging to
/// `target_serial`, and return its absolute granule position. Returns
/// `Ok(None)` if the last matching page has the unknown-granule sentinel
/// (`0xFFFF_FFFF_FFFF_FFFF`) or if no matching page can be found in the
/// search window — both of which leave the caller responsible for falling
/// back to the slow whole-stream decode.
///
/// Reads at most the trailing 128 KiB of the file. RFC 3533 caps a single
/// Ogg page at 65,307 bytes (27-byte header + 255 lacing entries × 255
/// bytes payload); 128 KiB therefore reliably covers a max-size last page
/// even with a small amount of trailing junk after the OggS frame.
fn read_last_granule<R: Read + Seek>(
    src: &mut R,
    target_serial: u32,
) -> std::io::Result<Option<u64>> {
    const SCAN_WINDOW: u64 = 128 * 1024;
    const HEADER_LEN: usize = 27;
    const UNKNOWN_GRANULE: u64 = 0xFFFF_FFFF_FFFF_FFFF;

    // Use Seek to obtain the source's length without depending on filesystem
    // metadata, so this helper can drive any Read+Seek (including Cursor in
    // unit tests).
    let file_len = src.seek(SeekFrom::End(0))?;
    if file_len < HEADER_LEN as u64 {
        return Ok(None);
    }

    let read_len = SCAN_WINDOW.min(file_len);
    let start = file_len - read_len;
    src.seek(SeekFrom::Start(start))?;

    let mut buf = vec![0u8; read_len as usize];
    src.read_exact(&mut buf)?;

    // Reverse-scan for the b"OggS" capture pattern. For each candidate, validate
    // the fixed-layout header and that the serial number matches. Walk back to
    // the previous candidate if any check fails (e.g. wrong stream in a
    // multiplexed file, or coincidental "OggS" bytes inside packet data).
    let mut i = buf.len().saturating_sub(4);
    loop {
        if i + HEADER_LEN <= buf.len()
            && &buf[i..i + 4] == b"OggS"
            // stream_structure_version must be 0 per RFC 3533 §6
            && buf[i + 4] == 0
        {
            let absgp = u64::from_le_bytes([
                buf[i + 6],
                buf[i + 7],
                buf[i + 8],
                buf[i + 9],
                buf[i + 10],
                buf[i + 11],
                buf[i + 12],
                buf[i + 13],
            ]);
            let serial = u32::from_le_bytes([buf[i + 14], buf[i + 15], buf[i + 16], buf[i + 17]]);
            if serial == target_serial {
                if absgp == UNKNOWN_GRANULE {
                    return Ok(None);
                }
                return Ok(Some(absgp));
            }
        }
        if i == 0 {
            return Ok(None);
        }
        i -= 1;
    }
}

// ---------------------------------------------------------------------------
// play
// ---------------------------------------------------------------------------

fn cmd_play(args: PlayArgs) -> Result<()> {
    heading("play");
    println!("file     {}", args.input.display().to_string().cyan());

    // Decode to interleaved f32 through one unified pipeline: symphonia demuxes
    // every container, and `decode_to_f32` routes Opus tracks through ropus
    // while everything else uses symphonia's native decoders.
    let DecodedAudio {
        samples,
        sample_rate,
        channels,
    } = decode_to_f32(&args.input)?;

    let channels_u16 = u16::try_from(channels).map_err(|_| anyhow!("channel count overflow"))?;
    println!(
        "audio    {} samples, {} Hz, {} ch",
        format_num(samples.len() as u64).bright_white(),
        sample_rate.to_string().bright_white(),
        channels_u16.to_string().bright_white(),
    );

    // Try to open the default audio device. If that fails (no device, headless
    // environment, etc.) print a clear message instead of panicking.
    let (_stream, handle) = match rodio::OutputStream::try_default() {
        Ok(pair) => pair,
        Err(e) => {
            return Err(anyhow!("no default audio output device available: {e}"));
        }
    };
    let sink = rodio::Sink::try_new(&handle).map_err(|e| anyhow!("creating sink failed: {e}"))?;

    if let Some(v) = args.volume {
        sink.set_volume(v.clamp(0.0, 1.0));
    }

    let source = rodio::buffer::SamplesBuffer::new(channels_u16, sample_rate, samples);
    sink.append(source);
    println!("playing  (Ctrl-C to stop)");
    sink.sleep_until_end();
    ok("playback finished");
    Ok(())
}

// ---------------------------------------------------------------------------
// audio decoding helpers
// ---------------------------------------------------------------------------

struct DecodedAudio {
    samples: Vec<f32>,
    sample_rate: u32,
    channels: usize,
}

/// Internal codec pipeline: either symphonia's native decoder for the track,
/// or ropus driven by symphonia's Ogg demuxer for Opus tracks. Centralising
/// the routing here means `cmd_play`, `cmd_encode` and any future caller goes
/// through the same demuxer/decoder for every supported input format.
enum CodecPipeline {
    Native(Box<dyn SymphoniaDecoder>),
    Opus {
        dec: RopusDecoder,
        pre_skip: usize,
        channels: usize,
        /// Samples already trimmed off the head. cmd_play does not seek, so
        /// this only ever monotonically increases until pre_skip is consumed.
        skipped: usize,
    },
}

fn decode_to_f32(path: &Path) -> Result<DecodedAudio> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .with_context(|| format!("probing {}", path.display()))?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow!("no decodable audio track"))?;

    let track_id = track.id;
    let codec_params = track.codec_params.clone();

    let sample_rate = codec_params
        .sample_rate
        .ok_or_else(|| anyhow!("track has no sample rate"))?;
    let channels = codec_params
        .channels
        .map(|c| c.count())
        .ok_or_else(|| anyhow!("track has no channel info"))?;

    // Branch on codec: Opus goes through ropus (we deliberately don't enable
    // symphonia's stub Opus decoder). Everything else uses the native
    // symphonia decoder for that codec.
    let mut pipeline = if codec_params.codec == CODEC_TYPE_OPUS {
        let opus_channels = channel_count_to_ropus(channels)?;
        let dec = RopusDecoder::new(OPUS_SR, opus_channels)
            .map_err(|e| anyhow!("decoder init failed: {e}"))?;
        // Resolve OpusHead.pre_skip without falling back to a magic constant.
        // Symphonia's Ogg demuxer surfaces it as `codec_params.delay`; failing
        // that, we parse it directly out of the OpusHead bytes that the
        // demuxer hands us in `codec_params.extra_data`. If neither is
        // present the file is malformed (every valid Ogg Opus stream begins
        // with an OpusHead packet), so we bail rather than silently inserting
        // a guess that would shift playback.
        let pre_skip = if let Some(d) = codec_params.delay {
            d as usize
        } else if let Some(buf) = codec_params.extra_data.as_deref() {
            if buf.len() >= 19 && &buf[..8] == b"OpusHead" {
                u16::from_le_bytes([buf[10], buf[11]]) as usize
            } else {
                bail!(
                    "opus track has no pre_skip metadata (no codec delay and \
                     no OpusHead extra_data)"
                );
            }
        } else {
            bail!(
                "opus track has no pre_skip metadata (no codec delay and \
                 no OpusHead extra_data)"
            );
        };
        CodecPipeline::Opus {
            dec,
            pre_skip,
            channels,
            skipped: 0,
        }
    } else {
        let dec = symphonia::default::get_codecs()
            .make(&codec_params, &DecoderOptions::default())
            .context("creating decoder for track")?;
        CodecPipeline::Native(dec)
    };

    let mut interleaved: Vec<f32> = Vec::with_capacity(1 << 20);
    let mut sample_buf: Option<SampleBuffer<f32>> = None;
    let max_per_ch = (OPUS_SR / 1000 * 120) as usize;
    let mut opus_scratch: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                bail!("symphonia: stream reset required (unsupported)")
            }
            Err(e) => return Err(e).context("reading next packet"),
        };
        if packet.track_id() != track_id {
            continue;
        }

        match &mut pipeline {
            CodecPipeline::Native(decoder) => match decoder.decode(&packet) {
                Ok(decoded) => {
                    if sample_buf.is_none() {
                        let spec = *decoded.spec();
                        let dur = decoded.capacity() as u64;
                        sample_buf = Some(SampleBuffer::<f32>::new(dur, spec));
                    }
                    if let Some(buf) = sample_buf.as_mut() {
                        buf.copy_interleaved_ref(decoded);
                        interleaved.extend_from_slice(buf.samples());
                    }
                }
                Err(SymphoniaError::DecodeError(_)) => {
                    // Skip corrupt packets.
                    continue;
                }
                Err(e) => return Err(e).context("decoding packet"),
            },
            CodecPipeline::Opus {
                dec,
                pre_skip,
                channels: ch,
                skipped,
            } => {
                if opus_scratch.len() != max_per_ch * *ch {
                    opus_scratch = vec![0f32; max_per_ch * *ch];
                }
                let n = match dec.decode_float(&packet.data, &mut opus_scratch, false) {
                    Ok(n) => n,
                    Err(e) => {
                        // Match the native path: swallow per-packet decode
                        // failures rather than aborting the whole file.
                        eprintln!("{} opus packet: {e}", "warning:".yellow());
                        continue;
                    }
                };
                let total = n * *ch;
                let frame = &opus_scratch[..total];
                // Trim the leading pre_skip samples (in interleaved units)
                // before they reach the output buffer.
                let want_trim = pre_skip.saturating_mul(*ch).saturating_sub(*skipped);
                if want_trim >= total {
                    *skipped += total;
                } else {
                    interleaved.extend_from_slice(&frame[want_trim..]);
                    *skipped += want_trim;
                }
            }
        }
    }

    Ok(DecodedAudio {
        samples: interleaved,
        sample_rate,
        channels,
    })
}

// ---------------------------------------------------------------------------
// resampling
// ---------------------------------------------------------------------------

fn resample_to_48k(interleaved: &[f32], from_sr: u32, channels: usize) -> Result<Vec<f32>> {
    if channels == 0 {
        bail!("zero-channel input");
    }
    if from_sr == 0 {
        bail!("zero-Hz input");
    }
    let ratio = OPUS_SR as f64 / from_sr as f64;

    let frames_in = interleaved.len() / channels;
    if frames_in == 0 {
        return Ok(Vec::new());
    }

    // Deinterleave into per-channel planar buffers.
    let mut planar: Vec<Vec<f32>> = (0..channels).map(|_| Vec::with_capacity(frames_in)).collect();
    for frame in interleaved.chunks_exact(channels) {
        for (ch, s) in frame.iter().enumerate() {
            planar[ch].push(*s);
        }
    }

    // Choose a chunk size that comfortably fits multiple sinc kernels.
    let chunk = 1024usize;

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::Blackman2,
    };

    let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, chunk, channels)
        .map_err(|e| anyhow!("rubato construction failed: {e}"))?;
    // SincFixedIn warms up by emitting `output_delay()` frames of leading
    // silence before any real output. We must skip those frames or the
    // decoded WAV starts with a click/shifted silence.
    let resampler_delay = resampler.output_delay();

    // Pad each channel to a multiple of the resampler chunk size.
    let pad = chunk - (frames_in % chunk);
    if pad != chunk {
        for ch in &mut planar {
            ch.extend(std::iter::repeat_n(0.0f32, pad));
        }
    }

    let total_in = planar[0].len();
    let mut out_planar: Vec<Vec<f32>> = (0..channels).map(|_| Vec::new()).collect();

    let mut pos = 0;
    while pos < total_in {
        let end = (pos + chunk).min(total_in);
        // Slices for this chunk.
        let chunk_in: Vec<&[f32]> = planar.iter().map(|c| &c[pos..end]).collect();
        let processed = resampler
            .process(&chunk_in, None)
            .map_err(|e| anyhow!("rubato process failed: {e}"))?;
        for (ch, samples) in processed.into_iter().enumerate() {
            out_planar[ch].extend_from_slice(&samples);
        }
        pos = end;
    }

    // Drop the leading resampler delay (silence warm-up), then trim to the
    // expected output length (frames_in * ratio rounded). If the resampler
    // produced fewer frames than expected (input shorter than the warm-up
    // window), cap target_frames to what we actually have.
    let target_frames = (frames_in as f64 * ratio).round() as usize;
    let available = out_planar[0].len().saturating_sub(resampler_delay);
    let out_frames = target_frames.min(available);
    let start = resampler_delay.min(out_planar[0].len());

    // Interleave from out_planar[ch][start .. start + out_frames].
    let mut out = Vec::with_capacity(out_frames * channels);
    for f in 0..out_frames {
        out.extend(out_planar.iter().map(|chan| chan[start + f]));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Opus container helpers
// ---------------------------------------------------------------------------

/// Build the `OpusHead` packet (RFC 7845, section 5.1). 19 bytes for a
/// channel-mapping=0 mono/stereo stream.
fn build_opus_head(channels: u8, input_sample_rate: u32, pre_skip: u16) -> Vec<u8> {
    let mut h = Vec::with_capacity(19);
    h.extend_from_slice(b"OpusHead");
    h.push(1); // version
    h.push(channels);
    h.extend_from_slice(&pre_skip.to_le_bytes());
    h.extend_from_slice(&input_sample_rate.to_le_bytes());
    h.extend_from_slice(&0i16.to_le_bytes()); // output gain (Q7.8 dB)
    h.push(0); // channel mapping family
    debug_assert_eq!(h.len(), 19);
    h
}

/// Build the `OpusTags` packet (RFC 7845, section 5.2).
fn build_opus_tags(vendor: &str) -> Vec<u8> {
    let vendor_bytes = vendor.as_bytes();
    let mut t = Vec::with_capacity(8 + 4 + vendor_bytes.len() + 4);
    t.extend_from_slice(b"OpusTags");
    t.extend_from_slice(&(vendor_bytes.len() as u32).to_le_bytes());
    t.extend_from_slice(vendor_bytes);
    t.extend_from_slice(&0u32.to_le_bytes()); // user-comment count
    t
}

#[derive(Debug, Clone, Copy)]
struct OpusHead {
    version: u8,
    channels: u8,
    pre_skip: u16,
    input_sample_rate: u32,
    #[allow(dead_code)]
    output_gain: i16,
    channel_mapping: u8,
}

fn parse_opus_head(data: &[u8]) -> Result<OpusHead> {
    if data.len() < 19 {
        bail!("OpusHead too short ({} bytes)", data.len());
    }
    if &data[..8] != b"OpusHead" {
        bail!("not an OpusHead packet");
    }
    Ok(OpusHead {
        version: data[8],
        channels: data[9],
        pre_skip: u16::from_le_bytes([data[10], data[11]]),
        input_sample_rate: u32::from_le_bytes([data[12], data[13], data[14], data[15]]),
        output_gain: i16::from_le_bytes([data[16], data[17]]),
        channel_mapping: data[18],
    })
}

/// Read the OpusTags packet from `reader`, verifying its magic. Returns an
/// error if the next packet is missing or doesn't begin with `b"OpusTags"`,
/// which would indicate a malformed file (e.g. tags page stripped) and would
/// otherwise cause the first audio packet to be silently consumed.
fn read_opus_tags<R: std::io::Read + std::io::Seek>(
    reader: &mut PacketReader<R>,
) -> Result<()> {
    let pkt = reader
        .read_packet()?
        .ok_or_else(|| anyhow!("expected OpusTags packet, got end of stream"))?;
    if pkt.data.len() < 8 || &pkt.data[..8] != b"OpusTags" {
        let head = &pkt.data[..8.min(pkt.data.len())];
        bail!("expected OpusTags packet, got {:?}", head);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// WAV writer
// ---------------------------------------------------------------------------

/// Write a 16-bit PCM mono/stereo WAV file. RIFF / fmt / data, fully
/// hand-rolled — avoids pulling in another dependency for ~30 lines.
fn write_wav_pcm16(path: &Path, samples: &[i16], sample_rate: u32, channels: u16) -> Result<()> {
    let f = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut w = BufWriter::new(f);

    let bits_per_sample: u16 = 16;
    let byte_rate: u32 = sample_rate * channels as u32 * (bits_per_sample as u32 / 8);
    let block_align: u16 = channels * (bits_per_sample / 8);
    let data_bytes: u32 = (samples.len() as u64 * 2)
        .try_into()
        .map_err(|_| anyhow!("WAV data exceeds 4 GiB"))?;
    let riff_size: u32 = 36u32
        .checked_add(data_bytes)
        .ok_or_else(|| anyhow!("WAV header size overflow"))?;

    // RIFF header
    w.write_all(b"RIFF")?;
    w.write_all(&riff_size.to_le_bytes())?;
    w.write_all(b"WAVE")?;

    // fmt chunk (PCM)
    w.write_all(b"fmt ")?;
    w.write_all(&16u32.to_le_bytes())?; // chunk size
    w.write_all(&1u16.to_le_bytes())?; // PCM format
    w.write_all(&channels.to_le_bytes())?;
    w.write_all(&sample_rate.to_le_bytes())?;
    w.write_all(&byte_rate.to_le_bytes())?;
    w.write_all(&block_align.to_le_bytes())?;
    w.write_all(&bits_per_sample.to_le_bytes())?;

    // data chunk
    w.write_all(b"data")?;
    w.write_all(&data_bytes.to_le_bytes())?;
    for s in samples {
        w.write_all(&s.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// misc helpers
// ---------------------------------------------------------------------------

fn channel_count_to_ropus(n: usize) -> Result<RopusChannels> {
    match n {
        1 => Ok(RopusChannels::Mono),
        2 => Ok(RopusChannels::Stereo),
        other => bail!("unsupported channel count {other} (ropus supports mono/stereo)"),
    }
}

fn with_extension(path: &Path, ext: &str) -> PathBuf {
    let mut p = path.to_path_buf();
    p.set_extension(ext);
    p
}

/// Format an integer with thousands separators using ASCII commas.
fn format_num(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(bytes.len() + bytes.len() / 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Build a minimal valid 27-byte Ogg page (zero-segment payload) with the
    /// supplied absolute granule position and stream serial. Tests only care
    /// about the fields `read_last_granule` inspects (capture pattern,
    /// version, absgp, serial, segment count).
    fn build_minimal_ogg_page(absgp: u64, serial: u32) -> Vec<u8> {
        let mut page = Vec::with_capacity(27);
        page.extend_from_slice(b"OggS"); // capture pattern
        page.push(0); // stream_structure_version
        page.push(0x04); // header_type_flag (end-of-stream — irrelevant here)
        page.extend_from_slice(&absgp.to_le_bytes());
        page.extend_from_slice(&serial.to_le_bytes());
        page.extend_from_slice(&0u32.to_le_bytes()); // page sequence
        page.extend_from_slice(&0u32.to_le_bytes()); // CRC (read_last_granule does not verify)
        page.push(0); // page_segments = 0 → no lacing bytes follow
        debug_assert_eq!(page.len(), 27);
        page
    }

    #[test]
    fn read_last_granule_empty_file_returns_none() {
        let mut cursor = Cursor::new(Vec::<u8>::new());
        let got = read_last_granule(&mut cursor, 0xC0DE_C0DE).expect("ok");
        assert!(got.is_none(), "empty input must yield None");
    }

    #[test]
    fn read_last_granule_minimal_valid_page() {
        let page = build_minimal_ogg_page(12_345, 0xC0DE_C0DE);
        let mut cursor = Cursor::new(page);
        let got = read_last_granule(&mut cursor, 0xC0DE_C0DE).expect("ok");
        assert_eq!(got, Some(12_345));
    }

    #[test]
    fn read_last_granule_unknown_granule_sentinel() {
        let page = build_minimal_ogg_page(0xFFFF_FFFF_FFFF_FFFF, 0xC0DE_C0DE);
        let mut cursor = Cursor::new(page);
        let got = read_last_granule(&mut cursor, 0xC0DE_C0DE).expect("ok");
        assert!(got.is_none(), "sentinel granule must yield None");
    }

    #[test]
    fn read_last_granule_skips_wrong_serial() {
        // Layout: [target page absgp=42] [other-serial page absgp=999]
        // Reverse scan should walk past the trailing wrong-serial page and
        // pick up the target page's granule.
        let mut buf = Vec::new();
        buf.extend_from_slice(&build_minimal_ogg_page(42, 0xC0DE_C0DE));
        buf.extend_from_slice(&build_minimal_ogg_page(999, 0xDEAD_BEEF));
        let mut cursor = Cursor::new(buf);
        let got = read_last_granule(&mut cursor, 0xC0DE_C0DE).expect("ok");
        assert_eq!(got, Some(42));
    }
}
