//! SILK WB decode trace tool — Bug #13 root cause investigation.
//!
//! Reads a fuzz_decode crash file (2-byte header + Opus packet), decodes with
//! both Rust and C, and compares SILK intermediate values to find the FIRST
//! divergent computation in the decode pipeline.
//!
//! Usage:
//!   cargo run --release --bin trace_silk_wb -- <crashfile>

#[path = "../bindings.rs"]
mod bindings;

use ropus::opus::decoder::{
    OpusDecoder, opus_packet_get_nb_frames, opus_packet_get_samples_per_frame,
};
use std::fs;
use std::os::raw::c_int;
use std::path::Path;

const SAMPLE_RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];
const MAX_FRAME: i32 = 5760;

// =========================================================================
// Helpers to extract Rust-side SILK state
// =========================================================================

/// Extract Rust SILK decoder indices after decode.
fn rust_silk_indices(dec: &OpusDecoder) -> RustSilkIndices {
    let ch = &dec.debug_silk_dec().channel_state[0];
    let idx = &ch.indices;
    RustSilkIndices {
        signal_type: idx.signal_type as i32,
        quant_offset_type: idx.quant_offset_type as i32,
        gains_indices: [
            idx.gains_indices[0] as i32,
            idx.gains_indices[1] as i32,
            idx.gains_indices[2] as i32,
            idx.gains_indices[3] as i32,
        ],
        nlsf_indices: {
            let mut arr = [0i32; 17];
            for i in 0..17 {
                arr[i] = idx.nlsf_indices[i] as i32;
            }
            arr
        },
        lag_index: idx.lag_index as i32,
        contour_index: idx.contour_index as i32,
        nlsf_interp_coef_q2: idx.nlsf_interp_coef_q2 as i32,
        per_index: idx.per_index as i32,
        ltp_index: [
            idx.ltp_index[0] as i32,
            idx.ltp_index[1] as i32,
            idx.ltp_index[2] as i32,
            idx.ltp_index[3] as i32,
        ],
        ltp_scale_index: idx.ltp_scale_index as i32,
        seed: idx.seed as i32,
    }
}

struct RustSilkIndices {
    signal_type: i32,
    quant_offset_type: i32,
    gains_indices: [i32; 4],
    nlsf_indices: [i32; 17],
    lag_index: i32,
    contour_index: i32,
    nlsf_interp_coef_q2: i32,
    per_index: i32,
    ltp_index: [i32; 4],
    ltp_scale_index: i32,
    seed: i32,
}

/// Extract C SILK decoder indices after decode.
fn c_silk_indices(dec: *mut bindings::OpusDecoder) -> RustSilkIndices {
    let mut r = RustSilkIndices {
        signal_type: 0,
        quant_offset_type: 0,
        gains_indices: [0; 4],
        nlsf_indices: [0; 17],
        lag_index: 0,
        contour_index: 0,
        nlsf_interp_coef_q2: 0,
        per_index: 0,
        ltp_index: [0; 4],
        ltp_scale_index: 0,
        seed: 0,
    };
    unsafe {
        bindings::debug_silk_trace_get_indices(
            dec,
            &mut r.signal_type,
            &mut r.quant_offset_type,
            r.gains_indices.as_mut_ptr(),
            r.nlsf_indices.as_mut_ptr(),
            &mut r.lag_index,
            &mut r.contour_index,
            &mut r.nlsf_interp_coef_q2,
            &mut r.per_index,
            r.ltp_index.as_mut_ptr(),
            &mut r.ltp_scale_index,
            &mut r.seed,
        );
    }
    r
}

fn compare_indices(label: &str, rust: &RustSilkIndices, c: &RustSilkIndices) -> bool {
    let mut ok = true;
    macro_rules! cmp {
        ($field:ident) => {
            if rust.$field != c.$field {
                println!(
                    "  {} MISMATCH {}: rust={:?} c={:?}",
                    label,
                    stringify!($field),
                    rust.$field,
                    c.$field
                );
                ok = false;
            }
        };
    }
    cmp!(signal_type);
    cmp!(quant_offset_type);
    cmp!(gains_indices);
    cmp!(nlsf_indices);
    cmp!(lag_index);
    cmp!(contour_index);
    cmp!(nlsf_interp_coef_q2);
    cmp!(per_index);
    cmp!(ltp_index);
    cmp!(ltp_scale_index);
    cmp!(seed);
    if ok {
        println!("  {} indices: ALL MATCH", label);
    }
    ok
}

fn compare_i32_arrays(label: &str, name: &str, rust: &[i32], c: &[i32]) -> bool {
    let len = rust.len().min(c.len());
    let first_diff = rust[..len]
        .iter()
        .zip(c[..len].iter())
        .position(|(a, b)| a != b);
    match first_diff {
        None => {
            println!("  {} {}: ALL MATCH ({} elements)", label, name, len);
            true
        }
        Some(i) => {
            let n_diff = rust[..len]
                .iter()
                .zip(c[..len].iter())
                .filter(|(a, b)| a != b)
                .count();
            println!(
                "  {} {} MISMATCH: first_diff@{} rust={} c={} (delta={}) [{} diffs of {}]",
                label,
                name,
                i,
                rust[i],
                c[i],
                rust[i] as i64 - c[i] as i64,
                n_diff,
                len
            );
            // Show context around first diff
            let lo = i.saturating_sub(2);
            let hi = (i + 6).min(len);
            for j in lo..hi {
                let mark = if rust[j] != c[j] { " *" } else { "" };
                println!(
                    "    [{:4}] rust={:12} c={:12} delta={:10}{}",
                    j,
                    rust[j],
                    c[j],
                    rust[j] as i64 - c[j] as i64,
                    mark
                );
            }
            false
        }
    }
}

fn compare_i16_arrays(label: &str, name: &str, rust: &[i16], c: &[i16]) -> bool {
    let len = rust.len().min(c.len());
    let first_diff = rust[..len]
        .iter()
        .zip(c[..len].iter())
        .position(|(a, b)| a != b);
    match first_diff {
        None => {
            println!("  {} {}: ALL MATCH ({} elements)", label, name, len);
            true
        }
        Some(i) => {
            let n_diff = rust[..len]
                .iter()
                .zip(c[..len].iter())
                .filter(|(a, b)| a != b)
                .count();
            println!(
                "  {} {} MISMATCH: first_diff@{} rust={} c={} (delta={}) [{} diffs of {}]",
                label,
                name,
                i,
                rust[i],
                c[i],
                rust[i] as i32 - c[i] as i32,
                n_diff,
                len
            );
            let lo = i.saturating_sub(2);
            let hi = (i + 6).min(len);
            for j in lo..hi {
                let mark = if rust[j] != c[j] { " *" } else { "" };
                println!(
                    "    [{:4}] rust={:8} c={:8} delta={:6}{}",
                    j,
                    rust[j],
                    c[j],
                    rust[j] as i32 - c[j] as i32,
                    mark
                );
            }
            false
        }
    }
}

// =========================================================================
// Phase 1: Decode with both, compare final PCM and post-decode SILK state
// =========================================================================

fn phase1_compare_after_decode(packet: &[u8], sample_rate: i32, channels: i32) {
    println!("\n=== PHASE 1: Full decode + post-decode SILK state comparison ===\n");

    // --- Rust decode ---
    let mut rust_dec = OpusDecoder::new(sample_rate, channels).unwrap();
    let max_pcm = MAX_FRAME as usize * channels as usize;
    let mut rust_pcm = vec![0i16; max_pcm];
    let rust_n = rust_dec
        .decode(Some(packet), &mut rust_pcm, MAX_FRAME, false)
        .unwrap_or_else(|e| {
            println!("Rust decode error: {}", e);
            -1
        });
    rust_pcm.truncate(rust_n as usize * channels as usize);

    // --- C decode ---
    let c_dec = unsafe {
        let mut err: c_int = 0;
        let d = bindings::opus_decoder_create(sample_rate, channels, &mut err);
        assert!(!d.is_null() && err == 0, "C decoder create failed");
        d
    };
    let mut c_pcm = vec![0i16; max_pcm];
    let c_n = unsafe {
        bindings::opus_decode(
            c_dec,
            packet.as_ptr(),
            packet.len() as i32,
            c_pcm.as_mut_ptr(),
            MAX_FRAME,
            0,
        )
    };
    c_pcm.truncate(c_n as usize * channels as usize);

    println!("  PCM: rust_samples={}, c_samples={}", rust_n, c_n);

    // Compare PCM
    if rust_pcm.len() == c_pcm.len() {
        compare_i16_arrays("PCM", "output", &rust_pcm, &c_pcm);
    } else {
        println!(
            "  PCM length mismatch: rust={} c={}",
            rust_pcm.len(),
            c_pcm.len()
        );
    }

    // --- Compare post-decode SILK state ---
    println!();
    println!("  --- Post-decode SILK state ---");

    // Compare indices
    let r_idx = rust_silk_indices(&rust_dec);
    let c_idx = c_silk_indices(c_dec);
    compare_indices("INDICES", &r_idx, &c_idx);

    // Compare prevNLSF_Q15
    let r_prev_nlsf = &rust_dec.debug_silk_dec().channel_state[0].prev_nlsf_q15;
    let mut c_prev_nlsf = [0i16; 16];
    unsafe {
        bindings::debug_silk_trace_get_prev_nlsf(c_dec, c_prev_nlsf.as_mut_ptr());
    }
    compare_i16_arrays("STATE", "prevNLSF_Q15", r_prev_nlsf, &c_prev_nlsf);

    // Compare persistent state
    let r_ch = &rust_dec.debug_silk_dec().channel_state[0];
    let mut c_prev_gain_q16: i32 = 0;
    let mut c_slpc_buf = [0i32; 16];
    let mut c_lag_prev: i32 = 0;
    let mut c_last_gain_idx: i32 = 0;
    let mut c_fs_khz: i32 = 0;
    let mut c_nb_subfr: i32 = 0;
    let mut c_frame_length: i32 = 0;
    let mut c_subfr_length: i32 = 0;
    let mut c_ltp_mem_length: i32 = 0;
    let mut c_lpc_order: i32 = 0;
    let mut c_ffar: i32 = 0;
    let mut c_loss_cnt: i32 = 0;
    let mut c_prev_sig_type: i32 = 0;
    unsafe {
        bindings::debug_silk_trace_get_persistent_state(
            c_dec,
            &mut c_prev_gain_q16,
            c_slpc_buf.as_mut_ptr(),
            &mut c_lag_prev,
            &mut c_last_gain_idx,
            &mut c_fs_khz,
            &mut c_nb_subfr,
            &mut c_frame_length,
            &mut c_subfr_length,
            &mut c_ltp_mem_length,
            &mut c_lpc_order,
            &mut c_ffar,
            &mut c_loss_cnt,
            &mut c_prev_sig_type,
        );
    }

    println!(
        "  STATE config: fs_khz=R:{}/C:{} nb_subfr=R:{}/C:{} frame_len=R:{}/C:{} subfr_len=R:{}/C:{} lpc_order=R:{}/C:{} ltp_mem=R:{}/C:{}",
        r_ch.fs_khz,
        c_fs_khz,
        r_ch.nb_subfr,
        c_nb_subfr as usize,
        r_ch.frame_length,
        c_frame_length as usize,
        r_ch.subfr_length,
        c_subfr_length as usize,
        r_ch.lpc_order,
        c_lpc_order as usize,
        r_ch.ltp_mem_length,
        c_ltp_mem_length as usize
    );

    if r_ch.prev_gain_q16 != c_prev_gain_q16 {
        println!(
            "  STATE prev_gain_Q16 MISMATCH: rust={} c={}",
            r_ch.prev_gain_q16, c_prev_gain_q16
        );
    } else {
        println!("  STATE prev_gain_Q16: MATCH ({})", r_ch.prev_gain_q16);
    }

    compare_i32_arrays("STATE", "sLPC_Q14_buf", &r_ch.s_lpc_q14_buf, &c_slpc_buf);

    if r_ch.lag_prev != c_lag_prev {
        println!(
            "  STATE lag_prev MISMATCH: rust={} c={}",
            r_ch.lag_prev, c_lag_prev
        );
    } else {
        println!("  STATE lag_prev: MATCH ({})", r_ch.lag_prev);
    }

    // Compare excitation buffer
    let frame_len = r_ch.frame_length;
    let mut c_exc = vec![0i32; frame_len];
    unsafe {
        bindings::debug_silk_trace_get_exc(c_dec, c_exc.as_mut_ptr(), frame_len as i32);
    }
    compare_i32_arrays(
        "STATE",
        "exc_Q14[0..frame_len]",
        &r_ch.exc_q14[..frame_len],
        &c_exc,
    );

    // Compare outBuf
    let outbuf_len = r_ch.out_buf.len();
    let mut c_outbuf = vec![0i16; outbuf_len];
    unsafe {
        bindings::debug_silk_trace_get_outbuf(c_dec, c_outbuf.as_mut_ptr(), outbuf_len as i32);
    }
    compare_i16_arrays("STATE", "outBuf", &r_ch.out_buf, &c_outbuf);

    unsafe {
        bindings::opus_decoder_destroy(c_dec);
    }
}

// Phase 2 and 3 removed — Phase 1 identified the root cause.

#[allow(dead_code)]
fn phase2_traced_decode(packet: &[u8], sample_rate: i32, channels: i32) {
    println!("\n=== PHASE 2: Traced SILK decode — parameter comparison ===\n");

    // For SILK-only packets (code 0), the SILK payload starts at byte 1.
    // We need to know the TOC to determine this.
    let toc = packet[0];
    let code = toc & 0x3;
    let config = (toc >> 3) & 0x1f;

    // For SILK WB config 9 code 0, the SILK payload is packet[1..].
    // But the SILK decoder expects the range-coder bitstream that
    // silk_decode_indices/silk_decode_pulses read from.
    // In the Opus decoder, the ec_dec is initialized over the full packet
    // (after TOC), and silk_Decode passes it through to silk_decode_frame.
    //
    // The C traced decode function takes raw SILK bitstream data.
    // For a code-0 packet, the SILK payload is everything after the TOC byte.
    let silk_payload = &packet[1..];

    println!("  TOC=0x{:02x} config={} code={}", toc, config, code);
    println!("  SILK payload: {} bytes", silk_payload.len());

    // First, do a normal opus_decode to initialize the C decoder's SILK state
    // (set up fs_kHz, frame_length, etc.), then reset and replay with tracing.
    let c_dec = unsafe {
        let mut err: c_int = 0;
        let d = bindings::opus_decoder_create(sample_rate, channels, &mut err);
        assert!(!d.is_null() && err == 0);
        d
    };

    // Normal decode to initialize SILK state
    let mut tmp_pcm = vec![0i16; MAX_FRAME as usize * channels as usize];
    let init_n = unsafe {
        bindings::opus_decode(
            c_dec,
            packet.as_ptr(),
            packet.len() as i32,
            tmp_pcm.as_mut_ptr(),
            MAX_FRAME,
            0,
        )
    };
    println!("  C init decode: {} samples", init_n);

    // Check SILK config
    let mut c_fs_khz: i32 = 0;
    let mut c_frame_len: i32 = 0;
    let mut c_n_frames_decoded: i32 = 0;
    unsafe {
        bindings::debug_silk_trace_get_config(
            c_dec,
            &mut c_fs_khz,
            &mut c_frame_len,
            &mut c_n_frames_decoded,
        );
    }
    println!(
        "  C SILK state: fs_kHz={} frame_length={} n_frames_decoded={}",
        c_fs_khz, c_frame_len, c_n_frames_decoded
    );

    // Now reset the SILK state and replay with tracing
    unsafe {
        bindings::debug_silk_trace_reset(c_dec);
    }

    let frame_len = c_frame_len as usize;
    let mut c_gains_q16 = [0i32; 4];
    let mut c_pred_coef_0 = [0i16; 16];
    let mut c_pred_coef_1 = [0i16; 16];
    let mut c_pitch_l = [0i32; 4];
    let mut c_ltp_coef = [0i16; 20];
    let mut c_ltp_scale: i32 = 0;
    let mut c_pulses = vec![0i16; frame_len];
    let mut c_pcm = vec![0i16; frame_len];

    let traced_n = unsafe {
        bindings::debug_silk_traced_decode(
            c_dec,
            silk_payload.as_ptr(),
            silk_payload.len() as i32,
            0, // CODE_INDEPENDENTLY for first frame
            c_gains_q16.as_mut_ptr(),
            c_pred_coef_0.as_mut_ptr(),
            c_pred_coef_1.as_mut_ptr(),
            c_pitch_l.as_mut_ptr(),
            c_ltp_coef.as_mut_ptr(),
            &mut c_ltp_scale,
            c_pulses.as_mut_ptr(),
            c_pcm.as_mut_ptr(),
        )
    };
    println!("  C traced decode: {} samples", traced_n);

    // Now do the same on Rust side by decoding normally
    let mut rust_dec = OpusDecoder::new(sample_rate, channels).unwrap();
    let mut rust_pcm = vec![0i16; MAX_FRAME as usize * channels as usize];
    let rust_n = rust_dec
        .decode(Some(packet), &mut rust_pcm, MAX_FRAME, false)
        .unwrap_or(-1);
    rust_pcm.truncate(rust_n as usize * channels as usize);
    println!("  Rust decode: {} samples", rust_n);

    // On the Rust side, we need to extract what silk_decode_parameters produced.
    // Since the Rust decoder doesn't expose silk_decoder_control directly,
    // we need to replay the decode on the Rust side too.
    // But we CAN read the indices (which are stored persistently).
    // The transient values (gains, LPC coefs, etc.) we need to recompute.
    //
    // Actually, the simplest approach: re-derive the silk_decoder_control from
    // the Rust side by calling the parameter decode functions directly.
    // We have access to the Rust SILK decoder internals.

    // Create a fresh Rust decoder and drive the SILK decode manually
    let mut rust_dec2 = OpusDecoder::new(sample_rate, channels).unwrap();
    let mut rust_pcm2 = vec![0i16; MAX_FRAME as usize * channels as usize];
    let _rust_n2 = rust_dec2
        .decode(Some(packet), &mut rust_pcm2, MAX_FRAME, false)
        .unwrap_or(-1);

    // The Rust decoder stores indices after decode
    let _r_ch = &rust_dec2.debug_silk_dec().channel_state[0];

    // We need to recompute silk_decoder_control from the Rust indices.
    // Rather than calling internal functions, let's just compare what we can:
    // 1. Compare indices (already stored)
    // 2. For the decoder control, use the Rust function directly

    // Actually, let me just re-derive them. The Rust silk_decode_parameters is private,
    // but we can replicate its logic:
    // gains_q16 = silk_gains_dequant(indices.gains_indices, last_gain_index, ...)
    // NLSF = silk_NLSF_decode(indices.nlsf_indices, nlsf_cb)
    // pred_coef = silk_NLSF2A(NLSF)
    // etc.
    //
    // But this is complex. Instead, let me use a different approach:
    // Create a second fresh decoder on both sides and use the traced decode.
    // For the Rust side, I'll add a public trace function.
    //
    // Actually, the most pragmatic approach: the Rust decode already ran through
    // silk_decode_parameters internally. I can't capture those transient values
    // after the fact. But I CAN compare the exc_Q14 buffer, which is written by
    // decode_core using those parameters. If exc_Q14 matches but PCM diverges,
    // the bug is in the LPC/LTP synthesis in decode_core.
    //
    // Let me restructure: compare indices first, then exc_Q14, then outBuf.
    // That narrows it down without needing to capture transient control values.

    println!("\n  --- Comparing C traced values vs C normal decode ---");
    println!("  (Verifying traced decode reproduces normal decode)");

    // The C traced PCM should match the initial C decode
    // (Both started from fresh state, same packet)
    compare_i16_arrays(
        "VERIFY",
        "C_traced_pcm vs C_normal_pcm",
        &c_pcm,
        &tmp_pcm[..frame_len],
    );

    // === Now the main comparison: Rust vs C ===
    println!("\n  --- Comparing Rust vs C (post-decode state) ---");

    // 1. Compare indices (the Rust decoder stored them)
    let r_idx = rust_silk_indices(&rust_dec2);
    let c_traced_idx = c_silk_indices(c_dec);
    compare_indices("TRACED", &r_idx, &c_traced_idx);

    // 2. Compare the C traced control values against the Rust decoder's state.
    //    On the Rust side, we need to recompute them. But first, let's check
    //    what we can compare directly.

    // 3. Compare exc_Q14 (written by the excitation computation in decode_core)
    let mut c_traced_exc = vec![0i32; frame_len];
    unsafe {
        bindings::debug_silk_trace_get_exc(c_dec, c_traced_exc.as_mut_ptr(), frame_len as i32);
    }
    let r_exc = &rust_dec2.debug_silk_dec().channel_state[0].exc_q14[..frame_len];
    compare_i32_arrays("TRACED", "exc_Q14", r_exc, &c_traced_exc);

    // 4. Compare sLPC_Q14_buf (LPC state saved at end of decode_core)
    let r_slpc = &rust_dec2.debug_silk_dec().channel_state[0].s_lpc_q14_buf;
    let mut c_slpc = [0i32; 16];
    unsafe {
        let mut dummy = [0i32; 1];
        bindings::debug_silk_trace_get_persistent_state(
            c_dec,
            &mut dummy[0],
            c_slpc.as_mut_ptr(),
            &mut dummy[0],
            &mut dummy[0],
            &mut dummy[0],
            &mut dummy[0],
            &mut dummy[0],
            &mut dummy[0],
            &mut dummy[0],
            &mut dummy[0],
            &mut dummy[0],
            &mut dummy[0],
            &mut dummy[0],
        );
    }
    compare_i32_arrays("TRACED", "sLPC_Q14_buf", r_slpc, &c_slpc);

    // 5. Compare outBuf
    let r_outbuf = &rust_dec2.debug_silk_dec().channel_state[0].out_buf;
    let mut c_outbuf = vec![0i16; r_outbuf.len()];
    unsafe {
        bindings::debug_silk_trace_get_outbuf(c_dec, c_outbuf.as_mut_ptr(), r_outbuf.len() as i32);
    }
    compare_i16_arrays("TRACED", "outBuf", r_outbuf, &c_outbuf);

    // 6. Compare prevNLSF_Q15
    let r_prev_nlsf = &rust_dec2.debug_silk_dec().channel_state[0].prev_nlsf_q15;
    let mut c_prev_nlsf = [0i16; 16];
    unsafe {
        bindings::debug_silk_trace_get_prev_nlsf(c_dec, c_prev_nlsf.as_mut_ptr());
    }
    compare_i16_arrays("TRACED", "prevNLSF_Q15", r_prev_nlsf, &c_prev_nlsf);

    // 7. Print C traced control values for reference
    println!("\n  --- C traced silk_decoder_control values ---");
    println!("  Gains_Q16: {:?}", &c_gains_q16[..]);
    println!(
        "  PredCoef_Q12[0]: {:?}",
        &c_pred_coef_0[..c_frame_len.min(16) as usize]
    );
    println!(
        "  PredCoef_Q12[1]: {:?}",
        &c_pred_coef_1[..c_frame_len.min(16) as usize]
    );
    println!("  pitchL: {:?}", &c_pitch_l[..]);
    println!("  LTPCoef_Q14: {:?}", &c_ltp_coef[..]);
    println!("  LTP_scale_Q14: {}", c_ltp_scale);

    // 8. Show pulses around the divergence point (sample 173 = subframe 2, i=13)
    println!("\n  --- C traced pulses around divergence (samples 155-185) ---");
    for i in 155..185.min(frame_len) {
        let p = c_pulses[i];
        let mark = if i == 173 { " <-- DIVERGE POINT" } else { "" };
        if p != 0 || i == 173 {
            println!("    pulse[{:3}] = {:3}{}", i, p, mark);
        }
    }

    // 9. Detailed PCM comparison around divergence
    println!("\n  --- PCM around divergence (Rust vs C traced) ---");
    let rust_pcm_silk = &rust_pcm2[..frame_len.min(rust_pcm2.len())];
    let lo = 168.min(frame_len);
    let hi = 185.min(frame_len);
    for i in lo..hi {
        let r = if i < rust_pcm_silk.len() {
            rust_pcm_silk[i]
        } else {
            0
        };
        let c = c_pcm[i];
        let mark = if r != c { " *" } else { "" };
        println!(
            "    xq[{:3}] rust={:6} c={:6} delta={:4}{}",
            i,
            r,
            c,
            r as i32 - c as i32,
            mark
        );
    }

    unsafe {
        bindings::opus_decoder_destroy(c_dec);
    }
}

// =========================================================================
// Phase 3: Deep decode_core trace
// =========================================================================

#[allow(dead_code)]
fn phase3_decode_core_trace(packet: &[u8], sample_rate: i32, channels: i32) {
    println!("\n=== PHASE 3: decode_core deep trace (subframe k=2, around sample 173) ===\n");

    // Re-derive the silk_decoder_control on the Rust side by replaying
    // the decode through the Rust internal functions.
    //
    // The Rust decoder stores exc_Q14 in the decoder state after decode_core.
    // decode_core writes:
    //   exc_Q14[i] = (pulse << 14) adjusted by quant offset and sign
    //   sLPC_Q14[MAX_LPC_ORDER + i] = pres_Q14[i] + (LPC_pred_Q10 << 4)
    //   xq[i] = SAT16(RSHIFT_ROUND(SMULWW(sLPC_Q14[MAX_LPC_ORDER+i], Gain_Q10), 8))
    //
    // The exc_Q14 is the INPUT to the LPC/LTP filter, not the output.
    // If exc_Q14 matches, the divergence must be in the LPC filter computation.
    //
    // To trace decode_core on the Rust side, I would need to instrument the
    // Rust code. Instead, I'll compare the C-side excitation and work backwards.

    // Do a traced decode on C side
    let c_dec = unsafe {
        let mut err: c_int = 0;
        let d = bindings::opus_decoder_create(sample_rate, channels, &mut err);
        assert!(!d.is_null() && err == 0);
        d
    };
    let mut tmp_pcm = vec![0i16; MAX_FRAME as usize * channels as usize];
    unsafe {
        bindings::opus_decode(
            c_dec,
            packet.as_ptr(),
            packet.len() as i32,
            tmp_pcm.as_mut_ptr(),
            MAX_FRAME,
            0,
        );
    }
    unsafe {
        bindings::debug_silk_trace_reset(c_dec);
    }

    let silk_payload = &packet[1..];
    let mut c_gains = [0i32; 4];
    let mut c_pred0 = [0i16; 16];
    let mut c_pred1 = [0i16; 16];
    let mut c_pitch = [0i32; 4];
    let mut c_ltp = [0i16; 20];
    let mut c_ltp_scale: i32 = 0;
    let mut c_pulses = vec![0i16; 320];
    let mut c_pcm = vec![0i16; 320];

    let c_n = unsafe {
        bindings::debug_silk_traced_decode(
            c_dec,
            silk_payload.as_ptr(),
            silk_payload.len() as i32,
            0,
            c_gains.as_mut_ptr(),
            c_pred0.as_mut_ptr(),
            c_pred1.as_mut_ptr(),
            c_pitch.as_mut_ptr(),
            c_ltp.as_mut_ptr(),
            &mut c_ltp_scale,
            c_pulses.as_mut_ptr(),
            c_pcm.as_mut_ptr(),
        )
    };
    println!("  C traced: {} samples", c_n);

    // Do the Rust decode
    let mut rust_dec = OpusDecoder::new(sample_rate, channels).unwrap();
    let mut rust_pcm = vec![0i16; MAX_FRAME as usize * channels as usize];
    let rust_n = rust_dec
        .decode(Some(packet), &mut rust_pcm, MAX_FRAME, false)
        .unwrap_or(-1);
    rust_pcm.truncate(rust_n as usize * channels as usize);

    let frame_len = c_n as usize;
    let r_ch = &rust_dec.debug_silk_dec().channel_state[0];

    println!(
        "  Rust: {} samples, frame_length={}",
        rust_n, r_ch.frame_length
    );

    // Compare exc_Q14 in detail around the divergence point
    let r_exc = &r_ch.exc_q14[..frame_len];
    let mut c_exc = vec![0i32; frame_len];
    unsafe {
        bindings::debug_silk_trace_get_exc(c_dec, c_exc.as_mut_ptr(), frame_len as i32);
    }

    // The excitation is the INPUT to the filter. If it matches, the filter diverges.
    let exc_match = r_exc == c_exc.as_slice();
    println!(
        "\n  exc_Q14 overall: {}",
        if exc_match { "EXACT MATCH" } else { "MISMATCH" }
    );

    if !exc_match {
        // Find first difference
        let first = r_exc.iter().zip(c_exc.iter()).position(|(a, b)| a != b);
        if let Some(i) = first {
            println!(
                "  exc_Q14 first diff at index {}: rust={} c={} delta={}",
                i,
                r_exc[i],
                c_exc[i],
                r_exc[i] as i64 - c_exc[i] as i64
            );
        }
    }

    // Since exc_Q14 likely matches (the bug is ±1 at output), the problem is
    // in decode_core's LPC filter. Let's trace the sLPC_Q14_buf which captures
    // the end-of-frame LPC state.

    let r_slpc = &r_ch.s_lpc_q14_buf;
    let mut c_slpc_state = [0i32; 16];
    unsafe {
        let mut d = [0i32; 1];
        bindings::debug_silk_trace_get_persistent_state(
            c_dec,
            &mut d[0],
            c_slpc_state.as_mut_ptr(),
            &mut d[0],
            &mut d[0],
            &mut d[0],
            &mut d[0],
            &mut d[0],
            &mut d[0],
            &mut d[0],
            &mut d[0],
            &mut d[0],
            &mut d[0],
            &mut d[0],
        );
    }
    println!("\n  sLPC_Q14_buf (end-of-frame LPC state):");
    compare_i32_arrays("CORE", "sLPC_Q14_buf", r_slpc, &c_slpc_state);

    // Now let me manually compute what decode_core does for subframe k=2
    // to trace the exact location of the divergence.
    //
    // For subframe k=2 (the third subframe):
    //   - Uses PredCoef_Q12[1] (since k >> 1 = 1)
    //   - Gain_Q10 = Gains_Q16[2] >> 6
    //   - LPC_pred_Q10 starts with bias = LPC_order/2 = 8 (for order 16)
    //   - The filter is: sum of sLPC_Q14[MAX_LPC_ORDER + i - j] * A_Q12[j-1]
    //
    // The key question is: does the LPC filter produce the same intermediate
    // values? With order 16, the computation involves 16 multiply-accumulates
    // using silk_SMLAWB which is: (a + ((b * c) >> 16))
    //
    // In Rust, this should be: a + ((b as i64 * c as i64) >> 16) as i32
    // The C silk_SMLAWB is: (a) + ((opus_int32)((opus_int64)(b) * (c) >> 16))

    // Print the C traced control values for subframe 2
    let lpc_order = r_ch.lpc_order;
    println!("\n  Subframe k=2 parameters:");
    println!(
        "    Gain_Q16[2] = {} (Gain_Q10 = {})",
        c_gains[2],
        c_gains[2] >> 6
    );
    println!(
        "    PredCoef_Q12[1][0..{}] = {:?}",
        lpc_order,
        &c_pred1[..lpc_order]
    );
    println!("    pitchL[2] = {}", c_pitch[2]);
    println!("    signal_type = {}", {
        let mut st: i32 = 0;
        unsafe {
            let mut d = [0i32; 4];
            bindings::debug_silk_trace_get_indices(
                c_dec,
                &mut st,
                &mut d[0],
                d.as_mut_ptr(),
                d.as_mut_ptr(),
                &mut d[0],
                &mut d[0],
                &mut d[0],
                &mut d[0],
                d.as_mut_ptr(),
                &mut d[0],
                &mut d[0],
            );
        }
        st
    });

    // Check if we can compute the exact values at the divergence point.
    // For that we'd need the sLPC_Q14 state just BEFORE subframe k=2 runs,
    // which is the sLPC_Q14 state at the end of subframe k=1.
    //
    // We can reconstruct this: at the start of decode_core, sLPC_Q14[0..16]
    // is loaded from sLPC_Q14_buf (which we have from the fresh decoder state).
    // Then after each subframe, the last MAX_LPC_ORDER samples become the new
    // state for the next subframe.
    //
    // However, we'd need to simulate all of subframes 0 and 1 to get the state
    // at the start of subframe 2. That's complex.
    //
    // Instead, let me look at the outBuf, which captures the decoded output
    // BEFORE the CNG/PLC-glue post-processing.

    // Actually, outBuf is updated AFTER decode_core runs and BEFORE CNG/glue.
    // outBuf[ltp_mem_length - frame_length .. ltp_mem_length] = output samples
    // after the memmove. Let me check the outBuf around the divergence.

    let ltp_mem = r_ch.ltp_mem_length;
    let r_outbuf = &r_ch.out_buf;
    let mut c_outbuf = vec![0i16; r_outbuf.len()];
    unsafe {
        bindings::debug_silk_trace_get_outbuf(c_dec, c_outbuf.as_mut_ptr(), r_outbuf.len() as i32);
    }

    // The output samples are stored at the END of outBuf after the memmove.
    // outBuf layout after update: [shifted old data | new_frame_output]
    // where new_frame_output starts at index (ltp_mem_length - frame_length)
    let out_start = ltp_mem - frame_len;
    println!(
        "\n  outBuf around divergence (offset {} in outBuf = sample 173):",
        out_start + 173
    );
    let idx_base = out_start + 168;
    let idx_end = (out_start + 185).min(r_outbuf.len());
    for i in idx_base..idx_end {
        let r = r_outbuf[i];
        let c = c_outbuf[i];
        let sample_num = i - out_start;
        let mark = if r != c { " *" } else { "" };
        println!(
            "    outBuf[{}] (sample {:3}) rust={:6} c={:6} delta={:4}{}",
            i,
            sample_num,
            r,
            c,
            r as i32 - c as i32,
            mark
        );
    }

    unsafe {
        bindings::opus_decoder_destroy(c_dec);
    }
}

// =========================================================================
// Main
// =========================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: trace_silk_wb <crashfile>");
        std::process::exit(2);
    }

    let path = Path::new(&args[1]);
    println!("========================================");
    println!("SILK WB Decode Trace — Bug #13 Investigation");
    println!("File: {}", path.display());

    let bytes = fs::read(path).expect("read file");
    if bytes.len() < 3 {
        eprintln!("File too small");
        std::process::exit(1);
    }

    let sr_idx = (bytes[0] as usize) % SAMPLE_RATES.len();
    let sample_rate = SAMPLE_RATES[sr_idx];
    let channels: i32 = if bytes[1] & 1 == 0 { 1 } else { 2 };
    let packet = &bytes[2..];

    println!(
        "Header: sr_byte=0x{:02x} sr={} ch={}",
        bytes[0], sample_rate, channels
    );
    println!("Packet: {} bytes, TOC=0x{:02x}", packet.len(), packet[0]);

    let toc = packet[0];
    let config = (toc >> 3) & 0x1f;
    let stereo = (toc >> 2) & 0x1;
    let code = toc & 0x3;
    println!("TOC: config={} stereo={} code={}", config, stereo, code);

    let spf = opus_packet_get_samples_per_frame(packet, sample_rate);
    let nf = opus_packet_get_nb_frames(packet).unwrap_or(-1);
    println!("SPF={} NF={}", spf, nf);

    // Run Phase 1 (the definitive one)
    phase1_compare_after_decode(packet, sample_rate, channels);

    // Print root cause analysis
    println!("\n=== ROOT CAUSE ANALYSIS ===\n");
    println!("  1. exc_Q14 matches exactly (320 elements) => excitation is identical");
    println!("  2. INDICES differ ONLY on nlsf_interp_coef_q2: Rust=1, C=4");
    println!("  3. sLPC_Q14_buf diverges => LPC filter state diverged");
    println!("  4. PCM diverges at sample 173 (subframe k=2, i=13) with delta=+/-1");
    println!();
    println!("  ROOT CAUSE: silk_decode_parameters does not write back");
    println!("  nlsf_interp_coef_q2 = 4 to dec.indices when first_frame_after_reset.");
    println!();
    println!("  In C (decode_parameters.c line 59-61):");
    println!("    if( psDec->first_frame_after_reset == 1 )");
    println!("        psDec->indices.NLSFInterpCoef_Q2 = 4;");
    println!("  This modifies the PERSISTENT indices in the decoder state.");
    println!();
    println!("  In Rust (decoder.rs silk_decode_parameters line 750-775):");
    println!("    let indices = dec.indices.clone();  // clone, not reference");
    println!(
        "    let interp_coef = if dec.first_frame_after_reset {{ 4 }} else {{ indices.nlsf_interp_coef_q2 }};"
    );
    println!("  The local variable interp_coef=4 is used for NLSF interpolation correctly,");
    println!("  but dec.indices.nlsf_interp_coef_q2 is NEVER updated.");
    println!();
    println!("  Later, decode_core reads dec.indices.nlsf_interp_coef_q2 directly:");
    println!("    let nlsf_interp = (dec.indices.nlsf_interp_coef_q2 as i32) < 4;");
    println!("  In C this would be 4 (written by decode_parameters) => nlsf_interp=false");
    println!("  In Rust this is still 1 (original decoded value) => nlsf_interp=true");
    println!();
    println!("  With nlsf_interp=true, decode_core enters re-whitening at k=2,");
    println!("  which changes the LTP/LPC filter state and causes the output divergence.");
    println!();
    println!("  FIX: In silk_decode_parameters, after the interp_coef computation,");
    println!("  write back: dec.indices.nlsf_interp_coef_q2 = interp_coef as i8;");
    println!(
        "  (or specifically: if dec.first_frame_after_reset {{ dec.indices.nlsf_interp_coef_q2 = 4; }})"
    );
    println!();
    println!("  This only affects WB (order 16) because NB/MB (order 10) uses nb_subfr=2,");
    println!("  and the subframe k=2 re-whitening path is never reached with only 2 subframes.");

    println!("\n========================================");
    println!("Trace complete.");
}
