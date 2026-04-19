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

fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
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

fn main() {
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

    // --- Read the header ---
    let fin = File::open(&packets_path).unwrap_or_else(|e| {
        eprintln!("open {}: {e}", packets_path.display());
        process::exit(1);
    });
    let mut reader = BufReader::new(fin);
    let num_frames = read_u32(&mut reader).expect("header: num_frames");
    let fs = read_u32(&mut reader).expect("header: fs") as i32;
    let channels = read_u32(&mut reader).expect("header: channels") as i32;
    let frame_size = read_u32(&mut reader).expect("header: frame_size") as i32;

    let mut dec = CRefFloatDecoder::new(fs, channels).unwrap_or_else(|e| {
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
    let mut scratch = vec![0i16; (frame_size as usize) * (channels as usize)];

    for i in 0..num_frames {
        let flags_len = read_u32(&mut reader).expect("frame header");
        let lost = (flags_len & LOST_BIT) != 0;
        let len = (flags_len & !LOST_BIT) as usize;

        if lost || len == 0 {
            let r = dec
                .decode(None, &mut scratch, frame_size, false)
                .unwrap_or_else(|e| {
                    eprintln!("frame {i}: PLC decode failed: {e}");
                    process::exit(1);
                });
            assert_eq!(r, frame_size, "frame {i}: PLC returned {r} samples");
        } else {
            let mut payload = vec![0u8; len];
            reader.read_exact(&mut payload).expect("payload");
            let r = dec
                .decode(Some(&payload), &mut scratch, frame_size, false)
                .unwrap_or_else(|e| {
                    eprintln!("frame {i}: decode failed: {e}");
                    process::exit(1);
                });
            assert_eq!(r, frame_size, "frame {i}: decoder returned {r} samples");
        }
        write_pcm(&mut writer, &scratch).expect("write pcm");
    }

    writer.flush().expect("flush");
}
