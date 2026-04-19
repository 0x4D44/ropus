//! Play: any symphonia-supported input → default audio output.

use anyhow::{Result, anyhow};
use colored::*;

use crate::audio::decode::{DecodedAudio, decode_to_f32};
use crate::options::PlayOptions;
use crate::ui::{format_num, heading, ok};

pub fn play(opts: PlayOptions) -> Result<()> {
    heading("play");
    println!("file     {}", opts.input.display().to_string().cyan());

    // Decode to interleaved f32 through one unified pipeline: symphonia demuxes
    // every container, and `decode_to_f32` routes Opus tracks through ropus
    // while everything else uses symphonia's native decoders.
    let DecodedAudio {
        samples,
        sample_rate,
        channels,
    } = decode_to_f32(&opts.input)?;

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

    if let Some(v) = opts.volume {
        sink.set_volume(v.clamp(0.0, 1.0));
    }

    let source = rodio::buffer::SamplesBuffer::new(channels_u16, sample_rate, samples);
    sink.append(source);
    println!("playing  (Ctrl-C to stop)");
    sink.sleep_until_end();
    ok("playback finished");
    Ok(())
}
