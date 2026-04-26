//! Minimal 16-bit PCM WAV reader/writer used by the `ropus` examples.
//!
//! Hand-rolled so the examples have no extra dependencies. Supports only the
//! subset Opus actually accepts: 16-bit PCM, mono or stereo, sample rate one
//! of 8/12/16/24/48 kHz. Anything else is rejected with a clear error.
//!
//! Supported `fmt` chunk formats:
//! - `WAVE_FORMAT_PCM` (tag `0x0001`) — standard 16-bit PCM.
//! - `WAVE_FORMAT_EXTENSIBLE` (tag `0xFFFE`) with `SubFormat` GUID
//!   `KSDATAFORMAT_SUBTYPE_PCM` (`00000001-0000-0010-8000-00aa00389b71`) and
//!   `wValidBitsPerSample == 16`. Audacity exports in this format by default.
//!
//! Pulled in from each example via `#[path = "common/wav.rs"] mod wav;` so it
//! is not itself auto-discovered as an example binary. Each example uses only
//! a subset (encode/roundtrip read; decode/roundtrip write), so silence the
//! dead-code lint here rather than fragmenting the module per consumer.
#![allow(dead_code)]

use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Sample rates Opus accepts.
const OPUS_RATES: [u32; 5] = [8_000, 12_000, 16_000, 24_000, 48_000];

#[derive(Debug, Clone)]
pub struct Wav {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: Vec<i16>, // interleaved
}

impl Wav {
    /// Number of samples per channel.
    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels as usize
    }

    /// Duration in seconds (f64).
    pub fn duration_secs(&self) -> f64 {
        self.frames() as f64 / self.sample_rate as f64
    }
}

/// Read a 16-bit PCM mono/stereo WAV at one of the Opus sample rates.
pub fn read<P: AsRef<Path>>(path: P) -> Result<Wav, Box<dyn Error>> {
    let mut r = BufReader::new(File::open(path)?);
    let mut buf4 = [0u8; 4];

    // RIFF header
    r.read_exact(&mut buf4)?;
    if &buf4 != b"RIFF" {
        return Err("not a RIFF file".into());
    }
    r.read_exact(&mut buf4)?; // file size - 8 (ignored)
    r.read_exact(&mut buf4)?;
    if &buf4 != b"WAVE" {
        return Err("not a WAVE file".into());
    }

    let mut sample_rate: u32 = 0;
    let mut channels: u16 = 0;
    let mut bits_per_sample: u16 = 0;
    let mut format_tag: u16 = 0;
    let mut data: Vec<u8> = Vec::new();

    // Walk chunks until we find both fmt and data.
    loop {
        if r.read(&mut buf4)? == 0 {
            break;
        }
        // If we read fewer than 4 we'd already have errored on read_exact;
        // but with read() we may have a partial. Tolerate clean EOF here.
        let id = buf4;
        let mut size_bytes = [0u8; 4];
        r.read_exact(&mut size_bytes)?;
        let size = u32::from_le_bytes(size_bytes) as usize;

        match &id {
            b"fmt " => {
                let mut fmt = vec![0u8; size];
                r.read_exact(&mut fmt)?;
                if size < 16 {
                    return Err("fmt chunk too small".into());
                }
                format_tag = u16::from_le_bytes([fmt[0], fmt[1]]);
                channels = u16::from_le_bytes([fmt[2], fmt[3]]);
                sample_rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
                bits_per_sample = u16::from_le_bytes([fmt[14], fmt[15]]);

                // WAVE_FORMAT_EXTENSIBLE: parse the extension and validate the
                // SubFormat GUID. We accept it as PCM-equivalent when the GUID
                // is KSDATAFORMAT_SUBTYPE_PCM and the valid bits is 16.
                if format_tag == 0xFFFE {
                    if size < 18 {
                        return Err("WAVE_FORMAT_EXTENSIBLE fmt chunk missing cbSize".into());
                    }
                    let cb_size = u16::from_le_bytes([fmt[16], fmt[17]]);
                    if cb_size < 22 {
                        return Err(format!(
                            "WAVE_FORMAT_EXTENSIBLE cbSize {cb_size} too small (need >= 22)"
                        )
                        .into());
                    }
                    if size < 40 {
                        return Err("WAVE_FORMAT_EXTENSIBLE fmt chunk truncated".into());
                    }
                    let valid_bits = u16::from_le_bytes([fmt[18], fmt[19]]);
                    // dwChannelMask at fmt[20..24] — acknowledged but not enforced.
                    let sub_format: &[u8] = &fmt[24..40];
                    // KSDATAFORMAT_SUBTYPE_PCM
                    // = 00000001-0000-0010-8000-00aa00389b71
                    // On disk (little-endian for the first three fields):
                    const PCM_GUID: [u8; 16] = [
                        0x01, 0x00, 0x00, 0x00, // Data1 = 0x00000001 LE
                        0x00, 0x00, // Data2  = 0x0000 LE
                        0x10, 0x00, // Data3  = 0x0010 LE
                        0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71, // Data4
                    ];
                    if sub_format != PCM_GUID {
                        return Err(format!(
                            "WAVE_FORMAT_EXTENSIBLE SubFormat GUID is not PCM \
                             (KSDATAFORMAT_SUBTYPE_PCM); found {sub_format:02x?}"
                        )
                        .into());
                    }
                    if valid_bits != 16 {
                        return Err(format!(
                            "WAVE_FORMAT_EXTENSIBLE wValidBitsPerSample \
                             {valid_bits} unsupported (need 16)"
                        )
                        .into());
                    }
                    // Treat the rest of the parser as if this were plain PCM.
                    bits_per_sample = valid_bits;
                    format_tag = 1;
                }
            }
            b"data" => {
                data = vec![0u8; size];
                r.read_exact(&mut data)?;
                // Stop after data — we have everything we need.
                break;
            }
            _ => {
                // Skip unknown chunk (incl. LIST/INFO etc.)
                let mut skip = vec![0u8; size];
                r.read_exact(&mut skip)?;
            }
        }
        // WAV chunks are word-aligned; consume one pad byte if size is odd.
        if size % 2 == 1 {
            let mut pad = [0u8; 1];
            // Propagate real I/O errors but tolerate clean EOF (read returns
            // Ok(0), which we deliberately ignore).
            let _ = r.read(&mut pad)?;
        }
    }

    // At this point format_tag is either 1 (plain PCM) or has been normalised
    // to 1 above by the WAVE_FORMAT_EXTENSIBLE PCM branch. Verify by exporting
    // a 16-bit WAV from Audacity (defaults to EXTENSIBLE) and feeding it to
    // any of the examples.
    if format_tag != 1 {
        return Err(format!(
            "unsupported WAV format tag {format_tag} (need 1 = PCM or 0xFFFE = EXTENSIBLE PCM)"
        )
        .into());
    }
    if bits_per_sample != 16 {
        return Err(format!("unsupported sample width {bits_per_sample} bits (need 16)").into());
    }
    if channels != 1 && channels != 2 {
        return Err(format!("unsupported channel count {channels} (need 1 or 2)").into());
    }
    if !OPUS_RATES.contains(&sample_rate) {
        return Err(format!(
            "sample rate {sample_rate} Hz not supported by Opus (need one of {OPUS_RATES:?})"
        )
        .into());
    }
    if data.is_empty() {
        return Err("WAV has no data chunk".into());
    }

    // Decode interleaved 16-bit LE samples.
    let mut samples = Vec::with_capacity(data.len() / 2);
    // chunks_exact drops a trailing odd byte; legal WAV is always even, so this is fine.
    for pair in data.chunks_exact(2) {
        samples.push(i16::from_le_bytes([pair[0], pair[1]]));
    }

    Ok(Wav {
        sample_rate,
        channels,
        samples,
    })
}

/// Write a 16-bit PCM WAV. `samples` are interleaved.
pub fn write<P: AsRef<Path>>(
    path: P,
    sample_rate: u32,
    channels: u16,
    samples: &[i16],
) -> Result<(), Box<dyn Error>> {
    let mut w = BufWriter::new(File::create(path)?);

    let byte_rate = sample_rate * channels as u32 * 2;
    let block_align: u16 = channels * 2;
    let data_size: u32 = (samples.len() * 2) as u32;
    let riff_size: u32 = 36 + data_size;

    w.write_all(b"RIFF")?;
    w.write_all(&riff_size.to_le_bytes())?;
    w.write_all(b"WAVE")?;

    // fmt chunk
    w.write_all(b"fmt ")?;
    w.write_all(&16u32.to_le_bytes())?; // PCM fmt chunk size
    w.write_all(&1u16.to_le_bytes())?; // format tag PCM
    w.write_all(&channels.to_le_bytes())?;
    w.write_all(&sample_rate.to_le_bytes())?;
    w.write_all(&byte_rate.to_le_bytes())?;
    w.write_all(&block_align.to_le_bytes())?;
    w.write_all(&16u16.to_le_bytes())?; // bits per sample

    // data chunk
    w.write_all(b"data")?;
    w.write_all(&data_size.to_le_bytes())?;
    for s in samples {
        w.write_all(&s.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}
