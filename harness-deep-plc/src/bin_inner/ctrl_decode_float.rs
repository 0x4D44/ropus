//! ctrl_decode_float: Control-experiment decoder that runs classical SILK PLC
//! on the C reference compiled in FLOAT mode.
//!
//! Part of the Stage 7b.3 control experiment. Even though this crate's build
//! links the full DEEP_PLC machinery (compile-time weights, `lpcnet_plc_*`
//! entry points, and the `ENABLE_DEEP_PLC=1` define), the neural PLC path is
//! runtime-gated: `src/opus_decoder.c:443` sets
//! `DecControl.enable_deep_plc = complexity >= 5`, and `silk/PLC.c:401`
//! only runs `lpcnet_plc_conceal` when that flag is set. We request
//! complexity = 4 here so the neural branch stays dormant and classical
//! SILK PLC fills the output — giving us a pure "C-float-classical"
//! decoder to diff against the fixed-point sibling binary.
//!
//! Packets-file format matches `harness/src/bin/ctrl_decode_fixed.rs`. See
//! that file's module doc for the byte-level layout.

#![allow(clippy::needless_range_loop, clippy::collapsible_if)]

use ropus_harness_deep_plc::CRefFloatDecoder;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::process;

const LOST_BIT: u32 = 0x8000_0000;
const CONTROL_COMPLEXITY: i32 = 4; // < 5 → runtime-disables DEEP_PLC
const MAX_PACKET_LEN: usize = 1275;
const MAX_CONTROL_FRAMES: u32 = 1_000_000;
const MAX_OUTPUT_SAMPLES: usize = 50_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DecodeHeader {
    num_frames: u32,
    fs: i32,
    channels: i32,
    frame_size: i32,
}

fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_header<R: Read>(r: &mut R) -> Result<DecodeHeader, String> {
    let num_frames = read_u32(r).map_err(|e| format!("header: num_frames: {e}"))?;
    let fs = read_u32(r).map_err(|e| format!("header: fs: {e}"))?;
    let channels = read_u32(r).map_err(|e| format!("header: channels: {e}"))?;
    let frame_size = read_u32(r).map_err(|e| format!("header: frame_size: {e}"))?;
    validate_header(num_frames, fs, channels, frame_size)
}

fn validate_header(
    num_frames: u32,
    fs: u32,
    channels: u32,
    frame_size: u32,
) -> Result<DecodeHeader, String> {
    if num_frames > MAX_CONTROL_FRAMES {
        return Err(format!(
            "num_frames {num_frames} exceeds maximum {MAX_CONTROL_FRAMES}"
        ));
    }
    if !matches!(fs, 8_000 | 12_000 | 16_000 | 24_000 | 48_000) {
        return Err(format!("unsupported sample rate {fs}"));
    }
    if !matches!(channels, 1 | 2) {
        return Err(format!("unsupported channel count {channels}"));
    }
    let valid_frame_sizes = [fs / 400, fs / 200, fs / 100, fs / 50, fs / 25, fs * 3 / 50];
    if !valid_frame_sizes.contains(&frame_size) {
        return Err(format!(
            "invalid frame_size {frame_size} for sample rate {fs}"
        ));
    }
    let output_samples = usize::try_from(num_frames)
        .ok()
        .and_then(|frames| frames.checked_mul(frame_size as usize))
        .and_then(|samples| samples.checked_mul(channels as usize))
        .ok_or_else(|| "frame/output sample count overflows usize".to_string())?;
    if output_samples > MAX_OUTPUT_SAMPLES {
        return Err(format!(
            "output sample count {output_samples} exceeds maximum {MAX_OUTPUT_SAMPLES}"
        ));
    }
    Ok(DecodeHeader {
        num_frames,
        fs: fs as i32,
        channels: channels as i32,
        frame_size: frame_size as i32,
    })
}

fn skip_exact<R: Read>(r: &mut R, mut len: usize) -> std::io::Result<()> {
    let mut discard = [0u8; 1024];
    while len > 0 {
        let take = len.min(discard.len());
        r.read_exact(&mut discard[..take])?;
        len -= take;
    }
    Ok(())
}

fn validate_flags_len(flags_len: u32, frame: u32) -> Result<(bool, usize), String> {
    let lost = (flags_len & LOST_BIT) != 0;
    let len = (flags_len & !LOST_BIT) as usize;
    if len > MAX_PACKET_LEN {
        return Err(format!(
            "frame {frame}: packet length {len} exceeds maximum {MAX_PACKET_LEN}"
        ));
    }
    if lost && len != 0 {
        return Err(format!(
            "frame {frame}: lost packet must have zero payload length"
        ));
    }
    if !lost && len == 0 {
        return Err(format!("frame {frame}: non-lost packet has zero length"));
    }
    Ok((lost, len))
}

fn validate_packet_records<R: Read>(r: &mut R, num_frames: u32) -> Result<(), String> {
    for frame in 0..num_frames {
        let flags_len = read_u32(r).map_err(|e| format!("frame {frame}: header: {e}"))?;
        let (_, len) = validate_flags_len(flags_len, frame)?;
        skip_exact(r, len).map_err(|e| format!("frame {frame}: payload: {e}"))?;
    }
    let mut trailing = [0u8; 1];
    if r.read(&mut trailing)
        .map_err(|e| format!("trailing data: {e}"))?
        != 0
    {
        return Err("packet file has trailing bytes after the declared frames".to_string());
    }
    Ok(())
}

fn validate_packets_file(path: &PathBuf) -> Result<DecodeHeader, String> {
    let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut reader = BufReader::new(file);
    let header = read_header(&mut reader)?;
    validate_packet_records(&mut reader, header.num_frames)?;
    Ok(header)
}

fn write_pcm<W: Write>(w: &mut W, pcm: &[i16]) -> std::io::Result<()> {
    let mut buf = [0u8; 4096];
    let mut pos = 0;
    for &s in pcm {
        let bytes = s.to_le_bytes();
        buf[pos] = bytes[0];
        buf[pos + 1] = bytes[1];
        pos += 2;
        if pos == buf.len() {
            w.write_all(&buf)?;
            pos = 0;
        }
    }
    if pos > 0 {
        w.write_all(&buf[..pos])?;
    }
    Ok(())
}

pub fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!(
            "usage: ctrl_decode_float <packets_in> <pcm_out>\n\n\
             Decodes the packet stream through the FLOAT C reference\n\
             (complexity = 4 → DEEP_PLC runtime-disabled, classical SILK\n\
             PLC only), writing raw i16 LE PCM."
        );
        process::exit(2);
    }
    let packets_path = PathBuf::from(&args[1]);
    let pcm_path = PathBuf::from(&args[2]);

    // Validate the complete bounded stream before creating the decoder,
    // output file, or any payload/scratch allocation.
    let header = validate_packets_file(&packets_path).unwrap_or_else(|e| {
        eprintln!("invalid packets file {}: {e}", packets_path.display());
        process::exit(1);
    });

    // Reopen the validated stream for the actual decode pass.
    let fin = File::open(&packets_path).unwrap_or_else(|e| {
        eprintln!("open {}: {e}", packets_path.display());
        process::exit(1);
    });
    let mut reader = BufReader::new(fin);
    let decoded_header = read_header(&mut reader).unwrap_or_else(|e| {
        eprintln!("read validated header: {e}");
        process::exit(1);
    });
    debug_assert_eq!(decoded_header, header);

    let mut dec = CRefFloatDecoder::new(header.fs, header.channels).unwrap_or_else(|e| {
        eprintln!("CRefFloatDecoder::new failed: {e}");
        process::exit(1);
    });
    dec.set_complexity(CONTROL_COMPLEXITY).unwrap_or_else(|e| {
        eprintln!("set_complexity({CONTROL_COMPLEXITY}) failed: {e}");
        process::exit(1);
    });

    let fout = File::create(&pcm_path).unwrap_or_else(|e| {
        eprintln!("create {}: {e}", pcm_path.display());
        process::exit(1);
    });
    let mut writer = BufWriter::new(fout);
    let frame_samples = (header.frame_size as usize) * (header.channels as usize);
    let mut scratch = vec![0i16; frame_samples];
    let mut payload = [0u8; MAX_PACKET_LEN];

    for i in 0..header.num_frames {
        let flags_len = read_u32(&mut reader).unwrap_or_else(|e| {
            eprintln!("frame {i}: header: {e}");
            process::exit(1);
        });
        let (lost, len) = validate_flags_len(flags_len, i).unwrap_or_else(|e| {
            eprintln!("frame {i}: {e}");
            process::exit(1);
        });

        if lost {
            let r = dec
                .decode(None, &mut scratch, header.frame_size, false)
                .unwrap_or_else(|e| {
                    eprintln!("frame {i}: PLC decode failed: {e}");
                    process::exit(1);
                });
            assert_eq!(r, header.frame_size, "frame {i}: PLC returned {r} samples");
        } else {
            reader.read_exact(&mut payload[..len]).unwrap_or_else(|e| {
                eprintln!("frame {i}: payload: {e}");
                process::exit(1);
            });
            let r = dec
                .decode(
                    Some(&payload[..len]),
                    &mut scratch,
                    header.frame_size,
                    false,
                )
                .unwrap_or_else(|e| {
                    eprintln!("frame {i}: decode failed: {e}");
                    process::exit(1);
                });
            assert_eq!(
                r, header.frame_size,
                "frame {i}: decoder returned {r} samples"
            );
        }
        write_pcm(&mut writer, &scratch).expect("write pcm");
    }

    writer.flush().expect("flush");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_validation_rejects_unsafe_dimensions() {
        assert!(validate_header(1, 44_100, 1, 882).is_err());
        assert!(validate_header(1, 48_000, 3, 960).is_err());
        assert!(validate_header(1, 48_000, 1, 1).is_err());
        assert!(validate_header(MAX_CONTROL_FRAMES + 1, 48_000, 1, 960).is_err());
        let too_many_output_samples = (MAX_OUTPUT_SAMPLES / (960 * 2) + 1) as u32;
        assert!(validate_header(too_many_output_samples, 48_000, 2, 960).is_err());
        assert!(validate_header(1, 48_000, 1, 960).is_ok());
    }

    #[test]
    fn packet_validation_bounds_lengths_and_flags() {
        let mut too_long = (MAX_PACKET_LEN as u32 + 1).to_le_bytes().to_vec();
        too_long.extend_from_slice(&vec![0u8; MAX_PACKET_LEN + 1]);
        assert!(validate_packet_records(&mut too_long.as_slice(), 1).is_err());

        let lost_with_payload = (LOST_BIT | 1).to_le_bytes().to_vec();
        assert!(validate_packet_records(&mut lost_with_payload.as_slice(), 1).is_err());

        let zero_non_lost = 0u32.to_le_bytes().to_vec();
        assert!(validate_packet_records(&mut zero_non_lost.as_slice(), 1).is_err());

        let valid = 1u32
            .to_le_bytes()
            .iter()
            .copied()
            .chain([0x2a])
            .collect::<Vec<_>>();
        assert!(validate_packet_records(&mut valid.as_slice(), 1).is_ok());
    }
}
