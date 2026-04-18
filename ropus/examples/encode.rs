//! Encode a 16-bit PCM WAV into an Ogg-Opus file (RFC 7845).
//!
//! Usage: `cargo run --example encode -- input.wav output.opus`
//!
//! Output container: standard Ogg-Opus, suitable for `ffprobe`, `opusinfo`,
//! `mpv`, `vlc` and `ropus-cli decode`. Two header pages (`OpusHead`,
//! `OpusTags`) followed by one Ogg page per encoded packet.
//!
//! Hard-coded encoder config:
//! - sample rate from the WAV (must be one of 8/12/16/24/48 kHz)
//! - mono or stereo per WAV
//! - Application::Audio
//! - 64 kbps CBR
//! - complexity 10
//! - 20 ms frames
//!
//! Limitation: this example does not resample. The encoder runs at the WAV's
//! native rate; the OpusHead `input_sample_rate` field records that rate for
//! informational purposes. Granule positions are written in 48 kHz samples as
//! RFC 7845 §4 requires (independent of the encoder's internal rate).

use std::env;
use std::error::Error;
use std::fs::File;
use std::io::BufWriter;

use ogg::PacketWriter;
use ogg::writing::PacketWriteEndInfo;

use ropus::{Application, Bitrate, Channels, Encoder};

// `#[path]` is needed because Cargo's example auto-discovery treats every
// file under examples/ as a binary target unless it's in a subdirectory.
#[path = "common/wav.rs"]
mod wav;

const FRAME_MS: u32 = 20;
const BITRATE: u32 = 64_000;
const COMPLEXITY: u8 = 10;
const MAX_PACKET: usize = 4000;
/// Granule positions are in 48 kHz samples regardless of encoder rate
/// (RFC 7845 §4). 20 ms = 960 samples at 48 kHz.
const FRAME_SAMPLES_AT_48K: u64 = 960;
/// Pre-skip in 48 kHz samples; 312 covers the encoder lookahead for our
/// 20 ms framing and matches the value used by `opusenc` and `ropus-cli`.
const PRE_SKIP_SAMPLES: u16 = 312;
/// Arbitrary unique logical-stream serial. Any non-zero u32 works for a
/// single-stream file.
const STREAM_SERIAL: u32 = 0xC0DE_C0DE;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: {} <input.wav> <output.opus>", args[0]);
        std::process::exit(2);
    }
    let in_path = &args[1];
    let out_path = &args[2];

    let input = wav::read(in_path)?;
    let channels = match input.channels {
        1 => Channels::Mono,
        2 => Channels::Stereo,
        n => return Err(format!("unsupported channel count {n}").into()),
    };
    let frame_per_channel = (input.sample_rate * FRAME_MS / 1000) as usize;
    let frame_interleaved = frame_per_channel * channels.count();

    println!(
        "input:  {} Hz, {} ch, {} samples ({:.3} s)",
        input.sample_rate,
        channels.count(),
        input.frames(),
        input.duration_secs()
    );

    let mut encoder = Encoder::builder(input.sample_rate, channels, Application::Audio)
        .bitrate(Bitrate::Bits(BITRATE))
        .complexity(COMPLEXITY)
        .build()?;

    let mut writer = PacketWriter::new(BufWriter::new(File::create(out_path)?));

    // Header page 1: OpusHead (RFC 7845 §5.1, 19 bytes for mapping family 0).
    let head = build_opus_head(channels.count() as u8, input.sample_rate, PRE_SKIP_SAMPLES);
    writer.write_packet(head, STREAM_SERIAL, PacketWriteEndInfo::EndPage, 0)?;

    // Header page 2: OpusTags (RFC 7845 §5.2).
    let tags = build_opus_tags("ropus example");
    writer.write_packet(tags, STREAM_SERIAL, PacketWriteEndInfo::EndPage, 0)?;

    // Encode every full 20 ms frame, padding the trailing partial frame with
    // silence. We collect into a Vec first so we know which packet is the
    // last and can mark it `EndStream`; for short example clips the memory
    // cost is trivial.
    let mut packets: Vec<Vec<u8>> = Vec::new();
    let mut packet_buf = vec![0u8; MAX_PACKET];
    let mut pcm_frame = vec![0i16; frame_interleaved];
    let mut idx = 0usize;
    while idx < input.samples.len() {
        let take = (input.samples.len() - idx).min(frame_interleaved);
        pcm_frame[..take].copy_from_slice(&input.samples[idx..idx + take]);
        if take < frame_interleaved {
            pcm_frame[take..].fill(0);
        }
        idx += take;
        let n = encoder.encode(&pcm_frame, &mut packet_buf)?;
        packets.push(packet_buf[..n].to_vec());
    }

    if packets.is_empty() {
        return Err("input produced zero encoded packets".into());
    }

    let mut samples_so_far: u64 = 0;
    let mut total_bytes: usize = 0;
    let last = packets.len() - 1;
    for (i, packet) in packets.iter().enumerate() {
        samples_so_far += FRAME_SAMPLES_AT_48K;
        let info = if i == last {
            PacketWriteEndInfo::EndStream
        } else {
            PacketWriteEndInfo::NormalPacket
        };
        total_bytes += packet.len();
        writer.write_packet(packet.clone(), STREAM_SERIAL, info, samples_so_far)?;
    }

    let frames_encoded = packets.len();
    let secs_encoded = frames_encoded as f64 * (FRAME_MS as f64 / 1000.0);
    let avg_bps = if secs_encoded > 0.0 {
        (total_bytes as f64 * 8.0) / secs_encoded
    } else {
        0.0
    };
    println!(
        "wrote:  {} packets, {} payload bytes, avg bitrate {:.0} bps ({:.1} kbps)",
        frames_encoded,
        total_bytes,
        avg_bps,
        avg_bps / 1000.0
    );

    Ok(())
}

/// Build an `OpusHead` packet (RFC 7845 §5.1). 19 bytes for a
/// channel-mapping-family=0 mono/stereo stream.
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

/// Build an `OpusTags` packet (RFC 7845 §5.2) with a vendor string and zero
/// user comments.
fn build_opus_tags(vendor: &str) -> Vec<u8> {
    let v = vendor.as_bytes();
    let mut t = Vec::with_capacity(8 + 4 + v.len() + 4);
    t.extend_from_slice(b"OpusTags");
    t.extend_from_slice(&(v.len() as u32).to_le_bytes());
    t.extend_from_slice(v);
    t.extend_from_slice(&0u32.to_le_bytes()); // user-comment count
    t
}
