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
use std::io::{BufReader, BufWriter, Write};
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
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
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
/// Logical Ogg stream serial. Any non-zero value works for a single stream.
const OGG_STREAM_SERIAL: u32 = 0xC0DE_C0DE;
/// Encoder pre-skip in 48 kHz samples (matches `opusenc` default of 312 +
/// codec lookahead of 312 samples for 20 ms framing — RFC 7845 only requires
/// it to cover the encoder lookahead, so this 312-sample value is safe).
const PRE_SKIP_SAMPLES: u16 = 312;
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

    // 4. Open Ogg writer.
    let file = File::create(&output)
        .with_context(|| format!("creating output file {}", output.display()))?;
    let mut writer = PacketWriter::new(BufWriter::new(file));

    // 5. Emit OpusHead and OpusTags headers (each on its own page per RFC 7845).
    let head = build_opus_head(channels as u8, sample_rate, PRE_SKIP_SAMPLES);
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
    read_opus_tags(&mut reader).context("reading OpusTags packet")?;

    let mut packet_count: u64 = 0;
    let mut sample_count: u64 = 0;
    let opus_channels = channel_count_to_ropus(head.channels as usize)?;
    let mut decoder = RopusDecoder::new(OPUS_SR, opus_channels)
        .map_err(|e| anyhow!("decoder init failed: {e}"))?;
    let max_per_ch = (OPUS_SR / 1000 * 120) as usize;
    let mut decoded = vec![0i16; max_per_ch * opus_channels.count()];

    while let Some(pkt) = reader.read_packet()? {
        // We need to know how many samples the packet decodes to. The
        // simplest correct approach is to actually decode it and count.
        match decoder.decode(&pkt.data, &mut decoded, false) {
            Ok(n) => sample_count += n as u64,
            Err(e) => {
                eprintln!("{} packet {}: {e}", "warning:".yellow(), packet_count);
            }
        }
        packet_count += 1;
    }

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
    println!("packets         {}", format_num(packet_count).bright_white());
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

// ---------------------------------------------------------------------------
// play
// ---------------------------------------------------------------------------

fn cmd_play(args: PlayArgs) -> Result<()> {
    heading("play");
    println!("file     {}", args.input.display().to_string().cyan());

    // Decode to interleaved f32. For .opus we route through ropus; for any
    // other format symphonia handles the decode.
    let DecodedAudio {
        samples,
        sample_rate,
        channels,
    } = if is_opus_path(&args.input) {
        decode_opus_to_f32(&args.input)?
    } else {
        decode_to_f32(&args.input)?
    };

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

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .context("creating decoder for track")?;

    let mut interleaved: Vec<f32> = Vec::with_capacity(1 << 20);
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

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
        match decoder.decode(&packet) {
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
        }
    }

    Ok(DecodedAudio {
        samples: interleaved,
        sample_rate,
        channels,
    })
}

fn is_opus_path(p: &Path) -> bool {
    matches!(
        p.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()),
        Some(ref e) if e == "opus" || e == "opusf"
    )
}

/// Decode a .opus file to interleaved f32 PCM via ropus directly. Used by
/// `play` so we go through the local codec rather than symphonia's vorbis.
fn decode_opus_to_f32(path: &Path) -> Result<DecodedAudio> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut reader = PacketReader::new(BufReader::new(file));

    let head_pkt = reader
        .read_packet()?
        .ok_or_else(|| anyhow!("no packets in {}", path.display()))?;
    let head = parse_opus_head(&head_pkt.data)?;
    read_opus_tags(&mut reader).context("reading OpusTags packet")?;

    let opus_channels = channel_count_to_ropus(head.channels as usize)?;
    let mut decoder = RopusDecoder::new(OPUS_SR, opus_channels)
        .map_err(|e| anyhow!("decoder init failed: {e}"))?;

    let max_per_ch = (OPUS_SR / 1000 * 120) as usize;
    let mut decoded = vec![0f32; max_per_ch * opus_channels.count()];
    let mut all: Vec<f32> = Vec::with_capacity(1 << 20);

    while let Some(pkt) = reader.read_packet()? {
        let n = decoder
            .decode_float(&pkt.data, &mut decoded, false)
            .map_err(|e| anyhow!("decode failed: {e}"))?;
        let total = n * opus_channels.count();
        all.extend_from_slice(&decoded[..total]);
    }

    let pre_skip = (head.pre_skip as usize) * opus_channels.count();
    let pre_skip = pre_skip.min(all.len());
    let trimmed = all.split_off(pre_skip);

    Ok(DecodedAudio {
        samples: trimmed,
        sample_rate: OPUS_SR,
        channels: opus_channels.count(),
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
