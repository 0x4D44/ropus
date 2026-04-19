//! Hand-rolled WAV writers (16-bit PCM and 32-bit IEEE float). Two thin
//! ~40-line functions rather than a dependency — the format is small and the
//! call sites fixed.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, anyhow};

/// WAVE format code for integer PCM (1 = WAVE_FORMAT_PCM).
const WAVE_FORMAT_PCM: u16 = 1;
/// WAVE format code for IEEE-754 float samples (3 = WAVE_FORMAT_IEEE_FLOAT).
const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;

/// Write a 16-bit PCM mono/stereo WAV file. RIFF / fmt / data, fully
/// hand-rolled — avoids pulling in another dependency for ~30 lines.
pub fn write_wav_pcm16(
    path: &Path,
    samples: &[i16],
    sample_rate: u32,
    channels: u16,
) -> Result<()> {
    let f = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut w = BufWriter::new(f);
    write_wav_pcm16_to(&mut w, samples, sample_rate, channels)?;
    w.flush()?;
    Ok(())
}

/// Write a 16-bit PCM WAV into any `Write`. Used by the path-based wrapper
/// and by the stdout-sink branch of `ropusdec -o -`.
pub fn write_wav_pcm16_to<W: Write + ?Sized>(
    w: &mut W,
    samples: &[i16],
    sample_rate: u32,
    channels: u16,
) -> Result<()> {
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
    w.write_all(&WAVE_FORMAT_PCM.to_le_bytes())?;
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
    Ok(())
}

/// Write a 32-bit IEEE-float mono/stereo WAV file. Non-PCM WAV formats require
/// `fmt ` chunk size 18 (trailing `cbSize = 0`) and a `fact` chunk before
/// `data` per the WAVEFORMATEX spec (Microsoft Multimedia Programmer's
/// Reference §§ "fmt", "fact"). Symphonia, ffmpeg, and every WAV reader we
/// care about reject float WAVs without the fact chunk.
pub fn write_wav_float32(
    path: &Path,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
) -> Result<()> {
    let f = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut w = BufWriter::new(f);
    write_wav_float32_to(&mut w, samples, sample_rate, channels)?;
    w.flush()?;
    Ok(())
}

/// Write a 32-bit IEEE-float WAV into any `Write`. Used by the path-based
/// wrapper and by the stdout-sink branch of `ropusdec -o -`.
pub fn write_wav_float32_to<W: Write + ?Sized>(
    w: &mut W,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
) -> Result<()> {
    if channels == 0 {
        return Err(anyhow!("float WAV requires channels >= 1"));
    }

    let bits_per_sample: u16 = 32;
    let block_align: u16 = channels * (bits_per_sample / 8);
    let byte_rate: u32 = sample_rate * u32::from(block_align);
    let data_bytes: u32 = (samples.len() as u64 * 4)
        .try_into()
        .map_err(|_| anyhow!("WAV data exceeds 4 GiB"))?;

    // Sample-frame count for the fact chunk. `samples.len()` counts
    // interleaved samples, so divide by channels.
    let sample_frames: u32 = ((samples.len() as u64) / u64::from(channels))
        .try_into()
        .map_err(|_| anyhow!("WAV sample-frame count exceeds u32"))?;

    // Chunk layout (all sizes in bytes of content, excluding the 8-byte
    // "magic + size" prefix that each chunk header contributes):
    //   "RIFF" + u32 size + "WAVE"                → size counted below
    //   "fmt " + u32(18)  + 18 bytes fmt body     → 8 + 18 = 26
    //   "fact" + u32(4)   + 4 bytes frame count   → 8 +  4 = 12
    //   "data" + u32(n)   + n bytes of samples    → 8 +  n
    // RIFF size = 4 ("WAVE") + 26 (fmt chunk) + 12 (fact chunk) + 8 + data
    //           = 50 + data_bytes.
    let riff_size: u32 = 50u32
        .checked_add(data_bytes)
        .ok_or_else(|| anyhow!("WAV header size overflow"))?;

    // RIFF header
    w.write_all(b"RIFF")?;
    w.write_all(&riff_size.to_le_bytes())?;
    w.write_all(b"WAVE")?;

    // fmt chunk (IEEE float, WAVEFORMATEX with cbSize = 0).
    w.write_all(b"fmt ")?;
    w.write_all(&18u32.to_le_bytes())?; // chunk size (18, not 16 — includes cbSize)
    w.write_all(&WAVE_FORMAT_IEEE_FLOAT.to_le_bytes())?;
    w.write_all(&channels.to_le_bytes())?;
    w.write_all(&sample_rate.to_le_bytes())?;
    w.write_all(&byte_rate.to_le_bytes())?;
    w.write_all(&block_align.to_le_bytes())?;
    w.write_all(&bits_per_sample.to_le_bytes())?;
    w.write_all(&0u16.to_le_bytes())?; // cbSize (no extension)

    // fact chunk (required for non-PCM): sample-frame count as u32.
    w.write_all(b"fact")?;
    w.write_all(&4u32.to_le_bytes())?;
    w.write_all(&sample_frames.to_le_bytes())?;

    // data chunk
    w.write_all(b"data")?;
    w.write_all(&data_bytes.to_le_bytes())?;
    for s in samples {
        w.write_all(&s.to_le_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_wav_header_has_fact_chunk_and_format_code_3() {
        // Write a tiny float WAV and re-read the bytes back. Parse just the
        // header chunks we care about and assert the WAVEFORMATEX shape that
        // downstream tools (symphonia, ffmpeg) require.
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "ropus_float_wav_header_{}_{}.wav",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let samples: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1).collect();
        write_wav_float32(&path, &samples, 44_100, 1).expect("write float wav");
        let bytes = std::fs::read(&path).expect("read back");
        let _ = std::fs::remove_file(&path);

        // RIFF / WAVE header.
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");

        // fmt chunk at offset 12. Size field at 16..20 must be 18 for
        // WAVEFORMATEX.
        assert_eq!(&bytes[12..16], b"fmt ");
        let fmt_size = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        assert_eq!(fmt_size, 18, "float WAV fmt chunk must be 18 bytes");

        // Format code = 3 (WAVE_FORMAT_IEEE_FLOAT).
        let format_code = u16::from_le_bytes([bytes[20], bytes[21]]);
        assert_eq!(format_code, 3, "float WAV format code must be 3");

        // bits_per_sample at offset 34..36.
        let bits = u16::from_le_bytes([bytes[34], bytes[35]]);
        assert_eq!(bits, 32, "float WAV bits_per_sample must be 32");

        // cbSize at offset 36..38 must be 0.
        let cb_size = u16::from_le_bytes([bytes[36], bytes[37]]);
        assert_eq!(cb_size, 0);

        // fact chunk immediately follows fmt (offset 38..46: "fact" + size=4
        // + frame count).
        assert_eq!(&bytes[38..42], b"fact");
        let fact_size = u32::from_le_bytes([bytes[42], bytes[43], bytes[44], bytes[45]]);
        assert_eq!(fact_size, 4);
        let frames = u32::from_le_bytes([bytes[46], bytes[47], bytes[48], bytes[49]]);
        assert_eq!(frames, samples.len() as u32);

        // data chunk next.
        assert_eq!(&bytes[50..54], b"data");
    }

    #[test]
    fn float_wav_stereo_frame_count_divides_by_channels() {
        // 16 interleaved samples, stereo → 8 sample-frames in the fact chunk.
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "ropus_float_wav_stereo_{}_{}.wav",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let samples = vec![0.0_f32; 16];
        write_wav_float32(&path, &samples, 48_000, 2).expect("write");
        let bytes = std::fs::read(&path).expect("read");
        let _ = std::fs::remove_file(&path);
        let frames = u32::from_le_bytes([bytes[46], bytes[47], bytes[48], bytes[49]]);
        assert_eq!(frames, 8, "stereo 16-sample buffer = 8 frames");
    }
}
