//! ctrl_decode_fixed: Control-experiment decoder that runs the classical SILK
//! PLC on the C reference compiled in FIXED_POINT mode.
//!
//! Part of the Stage 7b.3 control experiment that measures the fixed-vs-float
//! arithmetic gap in isolation (without DEEP_PLC), so we can decide whether
//! the 60 dB tier-2 SNR gate is ceiling-limited by construction or still has
//! a Rust port bug leaking into it.
//!
//! This binary reads a packets file, decodes every frame through the C
//! reference decoder (fixed-point, classical PLC — DEEP_PLC is not compiled
//! into the `harness` build), and writes raw interleaved i16 PCM to an output
//! file. The sister binary in `harness-deep-plc/` does the same with the
//! float-mode C reference at complexity = 4 so its compiled-in DEEP_PLC stays
//! dormant.
//!
//! Packets-file format (little-endian):
//!   u32 num_frames
//!   u32 fs
//!   u32 channels
//!   u32 frame_size       // samples per channel per frame
//!   repeated num_frames times:
//!     u32 flags_len      // high bit 0x8000_0000 = LOST → trigger PLC
//!                        // low bits = packet length (0 iff LOST)
//!     u8[len] payload    // omitted when LOST
//!
//! Output PCM file: raw i16 little-endian interleaved. Length in samples is
//!   num_frames * frame_size * channels. No header.

#![allow(clippy::needless_range_loop, clippy::collapsible_if)]

#[path = "../bindings.rs"]
mod bindings;

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::os::raw::c_int;
use std::path::PathBuf;
use std::process;

const OPUS_OK: c_int = 0;
const OPUS_SET_COMPLEXITY_REQUEST: c_int = 4010;

const LOST_BIT: u32 = 0x8000_0000;

fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn write_pcm<W: Write>(w: &mut W, pcm: &[i16]) -> std::io::Result<()> {
    // Streaming little-endian i16 write. Keeps memory bounded.
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
            "usage: ctrl_decode_fixed <packets_in> <pcm_out>\n\n\
             Decodes the packet stream through the FIXED_POINT C reference\n\
             (complexity = 4 → classical SILK PLC), writing raw i16 LE PCM."
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

    // --- Create and configure the C decoder ---
    let mut err: c_int = 0;
    let dec = unsafe { bindings::opus_decoder_create(fs, channels, &mut err) };
    if dec.is_null() || err != OPUS_OK {
        eprintln!("opus_decoder_create failed: err={err}");
        process::exit(1);
    }
    // Complexity < 5 to force classical SILK PLC in the float harness. The
    // fixed-point build has no DEEP_PLC regardless, so this knob only affects
    // the float side — we set it here too for uniformity.
    let rc = unsafe {
        bindings::opus_decoder_ctl(dec, OPUS_SET_COMPLEXITY_REQUEST, 4)
    };
    if rc != OPUS_OK {
        eprintln!("opus_decoder_ctl(OPUS_SET_COMPLEXITY, 4) failed: {rc}");
        process::exit(1);
    }

    // --- Decode every frame, streaming PCM to output ---
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

        let (data_ptr, data_len) = if lost || len == 0 {
            (std::ptr::null::<u8>(), 0i32)
        } else {
            let mut payload = vec![0u8; len];
            reader.read_exact(&mut payload).expect("payload");
            // Decode reads from its own slice — copy to a local buffer so the
            // pointer stays valid across the FFI call.
            let r = unsafe {
                bindings::opus_decode(
                    dec,
                    payload.as_ptr(),
                    payload.len() as i32,
                    scratch.as_mut_ptr(),
                    frame_size,
                    0,
                )
            };
            if r < 0 {
                eprintln!("frame {i}: opus_decode failed: {r}");
                process::exit(1);
            }
            assert_eq!(r, frame_size, "frame {i}: decoder returned {r} samples");
            write_pcm(&mut writer, &scratch).expect("write pcm");
            continue;
        };

        let r = unsafe {
            bindings::opus_decode(
                dec,
                data_ptr,
                data_len,
                scratch.as_mut_ptr(),
                frame_size,
                0,
            )
        };
        if r < 0 {
            eprintln!("frame {i}: opus_decode PLC failed: {r}");
            process::exit(1);
        }
        assert_eq!(r, frame_size, "frame {i}: PLC returned {r} samples");
        write_pcm(&mut writer, &scratch).expect("write pcm");
    }

    unsafe { bindings::opus_decoder_destroy(dec) };
    writer.flush().expect("flush");
}
