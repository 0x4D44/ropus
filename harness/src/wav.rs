//! Fallible PCM16 RIFF/WAVE parsing shared by the comparison binaries.

use std::fs;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pcm16Wav {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: Vec<i16>,
}

#[derive(Debug, Clone, Copy)]
struct PcmFormat {
    sample_rate: u32,
    channels: u16,
    block_align: u16,
}

/// Read and parse a PCM16 RIFF/WAVE file without panicking on malformed input.
pub fn read_pcm16_wav(path: &Path) -> Result<Pcm16Wav, String> {
    let data = fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    parse_pcm16_wav(&data).map_err(|e| format!("{}: {e}", path.display()))
}

/// Parse a PCM16 RIFF/WAVE byte slice with checked chunk bounds and alignment.
pub fn parse_pcm16_wav(data: &[u8]) -> Result<Pcm16Wav, String> {
    if data.len() < 12 {
        return Err("file too small for a RIFF/WAVE header".to_string());
    }
    if &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".to_string());
    }

    let riff_size = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    if riff_size < 4 {
        return Err(format!("invalid RIFF extent {riff_size}"));
    }
    let riff_end = 8usize
        .checked_add(riff_size)
        .ok_or_else(|| "RIFF extent overflows usize".to_string())?;
    if riff_end > data.len() {
        return Err(format!(
            "RIFF extent ends at {riff_end}, beyond file length {}",
            data.len()
        ));
    }

    let mut pos = 12usize;
    let mut format = None;
    let mut samples = None;

    while pos < riff_end {
        let header_end = pos
            .checked_add(8)
            .ok_or_else(|| "chunk header offset overflows usize".to_string())?;
        if header_end > riff_end {
            return Err(format!("truncated chunk header at offset {pos}"));
        }

        let chunk_id = &data[pos..pos + 4];
        let chunk_size =
            u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]])
                as usize;
        let payload_end = header_end
            .checked_add(chunk_size)
            .ok_or_else(|| format!("chunk {:?} extent overflows usize", chunk_id))?;
        if payload_end > riff_end {
            return Err(format!(
                "chunk {:?} ends at {payload_end}, beyond RIFF extent {riff_end}",
                chunk_id
            ));
        }
        let chunk_end = payload_end
            .checked_add(chunk_size & 1)
            .ok_or_else(|| format!("chunk {:?} alignment overflows usize", chunk_id))?;
        if chunk_end > riff_end {
            return Err(format!(
                "chunk {:?} is missing its alignment byte",
                chunk_id
            ));
        }
        let payload = &data[header_end..payload_end];

        if chunk_id == b"fmt " {
            if format.is_some() {
                return Err("duplicate fmt chunk".to_string());
            }
            format = Some(parse_format(payload)?);
        } else if chunk_id == b"data" {
            let fmt = format.ok_or_else(|| "data chunk appears before fmt chunk".to_string())?;
            if samples.is_some() {
                return Err("duplicate data chunk".to_string());
            }
            if payload.is_empty() {
                return Err("data chunk is empty".to_string());
            }
            let block_align = usize::from(fmt.block_align);
            if !payload.len().is_multiple_of(block_align) {
                return Err(format!(
                    "data chunk length {} is not aligned to {} bytes",
                    payload.len(),
                    block_align
                ));
            }
            let mut pcm = Vec::with_capacity(payload.len() / 2);
            for sample in payload.chunks_exact(2) {
                pcm.push(i16::from_le_bytes([sample[0], sample[1]]));
            }
            samples = Some(pcm);
        }

        pos = chunk_end;
    }

    let fmt = format.ok_or_else(|| "no fmt chunk found".to_string())?;
    let samples = samples.ok_or_else(|| "no data chunk found".to_string())?;
    Ok(Pcm16Wav {
        sample_rate: fmt.sample_rate,
        channels: fmt.channels,
        samples,
    })
}

fn parse_format(payload: &[u8]) -> Result<PcmFormat, String> {
    if payload.len() < 16 {
        return Err(format!(
            "fmt chunk too small: {} bytes, need at least 16",
            payload.len()
        ));
    }

    let audio_format = u16::from_le_bytes([payload[0], payload[1]]);
    if audio_format != 1 {
        return Err(format!("only PCM WAV supported (format {audio_format})"));
    }
    let channels = u16::from_le_bytes([payload[2], payload[3]]);
    if channels == 0 {
        return Err("WAV channel count must be nonzero".to_string());
    }
    let sample_rate = u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]);
    if sample_rate == 0 {
        return Err("WAV sample rate must be nonzero".to_string());
    }
    let byte_rate = u32::from_le_bytes([payload[8], payload[9], payload[10], payload[11]]);
    let block_align = u16::from_le_bytes([payload[12], payload[13]]);
    let bits_per_sample = u16::from_le_bytes([payload[14], payload[15]]);
    if bits_per_sample != 16 {
        return Err(format!("only 16-bit PCM supported (got {bits_per_sample})"));
    }

    let expected_block_align = u32::from(channels)
        .checked_mul(2)
        .ok_or_else(|| "WAV block alignment overflows u16".to_string())?;
    if expected_block_align > u32::from(u16::MAX) || u32::from(block_align) != expected_block_align
    {
        return Err(format!(
            "invalid block alignment {block_align}, expected {expected_block_align}"
        ));
    }
    let expected_byte_rate = sample_rate
        .checked_mul(expected_block_align)
        .ok_or_else(|| "WAV byte rate overflows u32".to_string())?;
    if byte_rate != expected_byte_rate {
        return Err(format!(
            "invalid byte rate {byte_rate}, expected {expected_byte_rate}"
        ));
    }

    Ok(PcmFormat {
        sample_rate,
        channels,
        block_align,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav_bytes(fmt: &[u8], data: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&[0; 4]);
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
        bytes.extend_from_slice(fmt);
        if !fmt.len().is_multiple_of(2) {
            bytes.push(0);
        }
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(data);
        if !data.len().is_multiple_of(2) {
            bytes.push(0);
        }
        let riff_size = (bytes.len() - 8) as u32;
        bytes[4..8].copy_from_slice(&riff_size.to_le_bytes());
        bytes
    }

    fn pcm16_fmt(channels: u16) -> Vec<u8> {
        let block_align = channels * 2;
        let byte_rate = 48_000 * u32::from(block_align);
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&1u16.to_le_bytes());
        fmt.extend_from_slice(&channels.to_le_bytes());
        fmt.extend_from_slice(&48_000u32.to_le_bytes());
        fmt.extend_from_slice(&byte_rate.to_le_bytes());
        fmt.extend_from_slice(&block_align.to_le_bytes());
        fmt.extend_from_slice(&16u16.to_le_bytes());
        fmt
    }

    #[test]
    fn parses_valid_pcm16_wav() {
        let bytes = wav_bytes(&pcm16_fmt(1), &[0x34, 0x12, 0xcc, 0xed]);
        let wav = parse_pcm16_wav(&bytes).expect("valid WAV");
        assert_eq!(wav.sample_rate, 48_000);
        assert_eq!(wav.channels, 1);
        assert_eq!(wav.samples, [0x1234, -0x1234]);
    }

    #[test]
    fn rejects_truncated_fmt_chunk() {
        let mut bytes = wav_bytes(&pcm16_fmt(1)[..12], &[0, 0]);
        bytes[16..20].copy_from_slice(&16u32.to_le_bytes());
        assert!(parse_pcm16_wav(&bytes).is_err());
    }

    #[test]
    fn rejects_oversized_data_chunk() {
        let mut bytes = wav_bytes(&pcm16_fmt(1), &[0, 0]);
        let data_size = 2 + 100;
        let data_header = 12 + 8 + pcm16_fmt(1).len();
        bytes[data_header + 4..data_header + 8].copy_from_slice(&(data_size as u32).to_le_bytes());
        assert!(parse_pcm16_wav(&bytes).is_err());
    }

    #[test]
    fn rejects_zero_channels() {
        assert!(parse_pcm16_wav(&wav_bytes(&pcm16_fmt(0), &[0, 0])).is_err());
    }

    #[test]
    fn rejects_odd_sample_alignment() {
        assert!(parse_pcm16_wav(&wav_bytes(&pcm16_fmt(1), &[0])).is_err());
    }
}
