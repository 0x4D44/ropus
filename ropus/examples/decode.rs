//! Decode an Ogg-Opus file (RFC 7845) into a 16-bit PCM WAV.
//!
//! Usage: `cargo run --example decode -- input.opus output.wav`
//!
//! Reads the standard Ogg-Opus container produced by `encode.rs` (or by any
//! conforming Opus encoder such as `opusenc` / `ropus-cli encode`). Channel
//! count and sample rate come from the `OpusHead` page, so no flags are
//! needed. The leading `pre_skip` samples are trimmed from the output to
//! match Opus playback semantics.
//!
//! The decoder always runs at the OpusHead `input_sample_rate` (one of
//! 8/12/16/24/48 kHz). The Opus codec supports decoding at any of these
//! rates; we use the rate the original WAV was encoded at so the round-trip
//! WAV matches the source.

use std::env;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use ogg::PacketReader;

use ropus::{Channels, Decoder};

// `#[path]` is needed because Cargo's example auto-discovery treats every
// file under examples/ as a binary target unless it's in a subdirectory.
#[path = "common/wav.rs"]
mod wav;

/// 120 ms at 48 kHz — the largest frame Opus can emit, per channel.
const MAX_FRAME_PER_CHANNEL: usize = 5_760;
/// Sample rates Opus accepts (matches the WAV writer / encoder).
const OPUS_RATES: [u32; 5] = [8_000, 12_000, 16_000, 24_000, 48_000];

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!(
            "usage: {} <input.opus> <output.wav>",
            args.first().map(String::as_str).unwrap_or("decode")
        );
        std::process::exit(2);
    }
    let in_path: &Path = Path::new(&args[1]);
    let out_path: &Path = Path::new(&args[2]);

    let mut reader = PacketReader::new(BufReader::new(File::open(in_path)?));

    // Header packet 1: OpusHead.
    let head_packet = reader.read_packet_expected()?;
    let head = parse_opus_head(&head_packet.data)?;
    let channels = match head.channels {
        1 => Channels::Mono,
        2 => Channels::Stereo,
        n => return Err(format!("OpusHead channel count {n} (need 1 or 2)").into()),
    };
    let ch_count = channels.count();
    let sample_rate = head.input_sample_rate;
    if !OPUS_RATES.contains(&sample_rate) {
        return Err(format!(
            "OpusHead input_sample_rate {sample_rate} Hz not supported by the decoder \
             (need one of {OPUS_RATES:?})"
        )
        .into());
    }

    // Header packet 2: OpusTags. Verify magic to catch malformed files (a
    // missing tags page would otherwise silently consume the first audio
    // packet).
    let tags_packet = reader.read_packet_expected()?;
    if tags_packet.data.len() < 8 || &tags_packet.data[..8] != b"OpusTags" {
        return Err("expected OpusTags packet after OpusHead".into());
    }

    let mut decoder = Decoder::new(sample_rate, channels)?;
    let mut frame = vec![0i16; MAX_FRAME_PER_CHANNEL * ch_count];
    let mut decoded: Vec<i16> = Vec::new();
    let mut packet_count: u64 = 0;

    while let Some(packet) = reader.read_packet()? {
        let n = decoder.decode(&packet.data, &mut frame, false)?;
        decoded.extend_from_slice(&frame[..n * ch_count]);
        packet_count += 1;
    }

    // Trim the leading pre-skip samples (RFC 7845 §4.2 / playback semantics).
    // pre_skip is in 48 kHz samples; scale to the decoder's rate.
    let pre_skip_at_rate = (head.pre_skip as u64 * sample_rate as u64 / 48_000) as usize;
    let pre_skip_interleaved = (pre_skip_at_rate * ch_count).min(decoded.len());
    let trimmed = &decoded[pre_skip_interleaved..];

    let frames = trimmed.len() / ch_count;
    let secs = frames as f64 / sample_rate as f64;
    println!(
        "decoded: {} packets -> {} samples ({:.3} s) at {} Hz, {} ch (pre-skip {} samples/ch)",
        packet_count, frames, secs, sample_rate, ch_count, pre_skip_at_rate
    );

    wav::write(out_path, sample_rate, ch_count as u16, trimmed)?;
    println!("wrote:   {}", out_path.display());
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct OpusHead {
    channels: u8,
    pre_skip: u16,
    input_sample_rate: u32,
}

fn parse_opus_head(data: &[u8]) -> Result<OpusHead, Box<dyn Error>> {
    if data.len() < 19 {
        return Err(format!("OpusHead too short ({} bytes, need >= 19)", data.len()).into());
    }
    if &data[..8] != b"OpusHead" {
        return Err("not an OpusHead packet".into());
    }
    let version = data[8];
    if version != 1 {
        return Err(format!("unsupported OpusHead version {version} (need 1)").into());
    }
    Ok(OpusHead {
        channels: data[9],
        pre_skip: u16::from_le_bytes([data[10], data[11]]),
        input_sample_rate: u32::from_le_bytes([data[12], data[13], data[14], data[15]]),
    })
}
