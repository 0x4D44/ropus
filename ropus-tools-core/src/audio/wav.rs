//! Minimal hand-rolled 16-bit PCM WAV writer (RIFF / fmt / data). Avoids
//! pulling in another dependency for ~30 lines.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, anyhow};

/// Write a 16-bit PCM mono/stereo WAV file. RIFF / fmt / data, fully
/// hand-rolled — avoids pulling in another dependency for ~30 lines.
pub fn write_wav_pcm16(path: &Path, samples: &[i16], sample_rate: u32, channels: u16) -> Result<()> {
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
