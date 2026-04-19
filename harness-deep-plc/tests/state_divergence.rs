#![cfg(not(no_reference))]
//! Stage 7b.3 state-divergence diagnostic.
//!
//! Task: with the `fs_khz` sync + sLPC back-copy fix applied (partial — total
//! SNR 16.67 dB, still below the 60 dB target), determine *which* of the
//! state-carrying variables diverges most between the C reference (float
//! mode) and ropus (fixed point) across the loss -> recovery boundary.
//!
//! The tier-2 test runs 16 kbps VoIP at 48 kHz — that's *SILK-only* mode at a
//! 16 kHz internal sample rate. The upsample to 48 kHz happens via the SILK
//! resampler, NOT CELT. Therefore CELT's `decode_mem`, `oldBandE`, and
//! `backgroundLogE` are **untouched** (confirmed by direct inspection: C
//! ref's arrays contain garbage/uninitialised values and Rust's are all
//! zero). The state that actually carries across the loss boundary lives
//! entirely in SILK.
//!
//! Two candidates measured:
//!   (1) SILK `sLPC_Q14_buf`  — 16-tap LPC synthesis continuation state.
//!       Partially addressed by the 8070445 fix (sLPC back-copy after deep
//!       PLC), but might still diverge.
//!   (2) SILK `sPLC.prevGain_Q16[2]` — per-subframe gain smoothing — if C
//!       and Rust disagree on this, every post-recovery subframe is
//!       misamplified.
//!
//! Also recorded for context (cheap, since the peek already fetches them):
//!   - `silk_decoder_state.prev_gain_Q16` (single frame gain)
//!   - `silk_decoder_state.outBuf` last 32 samples (rightmost = most recent)
//!   - `sPLC.pitchL_Q8`, `randScale_Q14`, `last_frame_lost`, `fs_kHz`
//!
//! Snapshots captured at:
//!   - T1: last good frame BEFORE loss (state entering the loss)
//!   - T2: during the lost frame (deep-PLC has just synthesised)
//!   - T3: first good frame AFTER recovery (true test of continuity)
//!
//! Normalised RMS delta formula (task-specified):
//!   delta = sqrt(mean((c - rust)^2)) / (sqrt(mean(c^2)) + 1e-9)
//!
//! Units match on both sides — SILK stays in fixed-point Q-format on both
//! C and Rust, even in the float-mode CELT build.

use ropus::{
    OPUS_APPLICATION_VOIP, OPUS_OK, OpusDecoder as RopusDecoder, OpusEncoder as RopusEncoder,
};
use ropus_harness_deep_plc::CRefFloatDecoder;

const FS: i32 = 48_000;
const CHANNELS: i32 = 1;
const FRAME_MS: i32 = 20;
const FRAME_SIZE: i32 = FS * FRAME_MS / 1000; // 960
const BITRATE: i32 = 16_000;
const COMPLEXITY: i32 = 10;

fn synth_reference_pcm(n_frames: usize) -> Vec<i16> {
    let n_samples = n_frames * FRAME_SIZE as usize;
    let mut pcm = Vec::with_capacity(n_samples);
    let mut rng: u32 = 0xC0FFEE_u32;
    for i in 0..n_samples {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        let noise = ((rng as i32) >> 22) as f64 / 512.0;
        let t = i as f64 / FS as f64;
        let env = 0.5 + 0.5 * (2.0 * std::f64::consts::PI * 2.0 * t).sin().abs();
        let tone1 = (2.0 * std::f64::consts::PI * 220.0 * t).sin();
        let tone2 = (2.0 * std::f64::consts::PI * 880.0 * t).sin();
        let sample = env * (0.6 * tone1 + 0.35 * tone2) + 0.05 * noise;
        let s_i16 = (sample.clamp(-1.0, 1.0) * 28_000.0) as i16;
        pcm.push(s_i16);
    }
    pcm
}

fn encode_with_ropus(pcm: &[i16]) -> Vec<Vec<u8>> {
    let mut enc = RopusEncoder::new(FS, CHANNELS, OPUS_APPLICATION_VOIP).expect("encoder create");
    assert_eq!(enc.set_bitrate(BITRATE), OPUS_OK);
    assert_eq!(enc.set_complexity(COMPLEXITY), OPUS_OK);
    let frame_samples = FRAME_SIZE as usize;
    let n_frames = pcm.len() / frame_samples;
    let mut out = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let start = i * frame_samples;
        let frame = &pcm[start..start + frame_samples];
        let mut buf = vec![0u8; 4000];
        let cap = buf.len() as i32;
        let n = enc
            .encode(frame, FRAME_SIZE, &mut buf, cap)
            .expect("encode ok");
        buf.truncate(n as usize);
        out.push(buf);
    }
    out
}

/// Normalised RMS delta: sqrt(mean((c - r)^2)) / (sqrt(mean(c^2)) + 1e-9).
fn rms_delta(c: &[f64], r: &[f64]) -> f64 {
    assert_eq!(c.len(), r.len());
    let mut sum_c2 = 0.0;
    let mut sum_err2 = 0.0;
    for (ci, ri) in c.iter().zip(r.iter()) {
        sum_c2 += ci * ci;
        let e = ci - ri;
        sum_err2 += e * e;
    }
    let n = c.len() as f64;
    let num = (sum_err2 / n).sqrt();
    let den = (sum_c2 / n).sqrt() + 1e-9;
    num / den
}

fn rms(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let n = v.len() as f64;
    (v.iter().map(|x| x * x).sum::<f64>() / n).sqrt()
}

#[derive(Debug, Default, Clone, Copy)]
struct Snapshot {
    s_lpc_q14_delta: f64,
    s_lpc_q14_c_rms: f64,
    plc_prev_gain_delta: f64,
    dec_prev_gain_delta: f64,
    out_buf_tail_delta: f64,
    out_buf_tail_c_rms: f64,
    plc_pitch_c: i32,
    plc_pitch_r: i32,
    plc_rand_scale_c: i32,
    plc_rand_scale_r: i32,
    last_lost_c: i32,
    last_lost_r: i32,
    c_plc_fs: i32,
    r_plc_fs: i32,
}

fn snapshot(c: &CRefFloatDecoder, r_dec: &RopusDecoder) -> Snapshot {
    let silk_r = r_dec.debug_silk_dec();
    let cs_r = &silk_r.channel_state[0];

    // --- sLPC_Q14_buf: 16 entries, same Q14 fixed-point on both sides ---
    let c_slpc = c.silk_s_lpc_q14();
    let r_slpc = cs_r.s_lpc_q14_buf; // [i32; MAX_LPC_ORDER=16]
    let c_slpc_f: Vec<f64> = c_slpc.iter().map(|&v| v as f64).collect();
    let r_slpc_f: Vec<f64> = r_slpc.iter().map(|&v| v as f64).collect();
    let s_lpc_delta = rms_delta(&c_slpc_f, &r_slpc_f);
    let s_lpc_c_rms = rms(&c_slpc_f);

    // --- sPLC.prevGain_Q16[2]: per-subframe smoothed gain ---
    let c_plc_pg = c.silk_plc_prev_gain_q16();
    let r_plc_pg = cs_r.s_plc.prev_gain_q16;
    let c_pg_f: Vec<f64> = c_plc_pg.iter().map(|&v| v as f64).collect();
    let r_pg_f: Vec<f64> = r_plc_pg.iter().map(|&v| v as f64).collect();
    let plc_pg_delta = rms_delta(&c_pg_f, &r_pg_f);

    // --- prev_gain_Q16 (scalar decoder-state field) ---
    let c_dpg = c.silk_prev_gain_q16();
    let r_dpg = cs_r.prev_gain_q16;
    let dpg_delta = rms_delta(&[c_dpg as f64], &[r_dpg as f64]);

    // --- outBuf tail 32 samples (most recent = right side of the buffer).
    // After the end-of-frame shift (decode_frame.c:105-107), the freshest
    // `frame_length` samples sit in `outBuf[mv_len .. ltp_mem_length]`.
    // So the last 32 samples = [ltp_mem_length-32 .. ltp_mem_length).
    let ltp_mem_c = c.silk_ltp_mem_length();
    let tail_len = 32;
    let offset_c = ltp_mem_c - tail_len;
    let c_out = c.silk_out_buf(offset_c, tail_len);
    // Rust mirror
    let ltp_mem_r = cs_r.ltp_mem_length;
    let offset_r = ltp_mem_r - tail_len as usize;
    let r_out = &cs_r.out_buf[offset_r..offset_r + tail_len as usize];
    let c_out_f: Vec<f64> = c_out.iter().map(|&v| v as f64).collect();
    let r_out_f: Vec<f64> = r_out.iter().map(|&v| v as f64).collect();
    let out_buf_delta = rms_delta(&c_out_f, &r_out_f);
    let out_buf_c_rms = rms(&c_out_f);

    Snapshot {
        s_lpc_q14_delta: s_lpc_delta,
        s_lpc_q14_c_rms: s_lpc_c_rms,
        plc_prev_gain_delta: plc_pg_delta,
        dec_prev_gain_delta: dpg_delta,
        out_buf_tail_delta: out_buf_delta,
        out_buf_tail_c_rms: out_buf_c_rms,
        plc_pitch_c: c.silk_plc_pitch_l_q8(),
        plc_pitch_r: cs_r.s_plc.pitch_l_q8,
        plc_rand_scale_c: c.silk_plc_rand_scale_q14(),
        plc_rand_scale_r: cs_r.s_plc.rand_scale_q14 as i32,
        last_lost_c: c.silk_plc_last_frame_lost(),
        last_lost_r: cs_r.s_plc.last_frame_lost,
        c_plc_fs: c.silk_plc_fs_khz(),
        r_plc_fs: cs_r.s_plc.fs_khz,
    }
}

#[test]
fn stage7b3_silk_state_divergence_at_loss_recovery() {
    const N_FRAMES: usize = 30;
    let pcm = synth_reference_pcm(N_FRAMES);
    let packets = encode_with_ropus(&pcm);
    assert_eq!(packets.len(), N_FRAMES);

    let mut c_dec = CRefFloatDecoder::new(FS, CHANNELS).expect("c dec");
    c_dec.set_complexity(COMPLEXITY).expect("c complexity");
    let mut r_dec = RopusDecoder::new(FS, CHANNELS).expect("r dec");
    r_dec.set_complexity(COMPLEXITY).expect("r complexity");

    let frame_samples = FRAME_SIZE as usize;
    let mut scratch_c = vec![0i16; frame_samples];
    let mut scratch_r = vec![0i16; frame_samples];

    let is_lost = |i: usize| i > 0 && i.is_multiple_of(7);

    struct LossEvent {
        idx: usize,
        t1: Option<Snapshot>, // snapshot AFTER good frame idx-1
        t2: Option<Snapshot>, // snapshot AFTER lost frame idx
        t3: Option<Snapshot>, // snapshot AFTER good frame idx+1
    }
    let mut events: Vec<LossEvent> = Vec::new();

    for (i, pkt) in packets.iter().enumerate() {
        if is_lost(i) {
            // About to lose — snapshot T1 on state as it was after the
            // previous good frame.
            let mut ev = LossEvent {
                idx: i,
                t1: None,
                t2: None,
                t3: None,
            };
            ev.t1 = Some(snapshot(&c_dec, &r_dec));
            // Decode the lost frame on both sides.
            c_dec
                .decode(None, &mut scratch_c, FRAME_SIZE, false)
                .unwrap();
            r_dec
                .decode(None, &mut scratch_r, FRAME_SIZE, false)
                .unwrap();
            ev.t2 = Some(snapshot(&c_dec, &r_dec));
            events.push(ev);
        } else {
            c_dec
                .decode(Some(pkt), &mut scratch_c, FRAME_SIZE, false)
                .unwrap();
            r_dec
                .decode(Some(pkt), &mut scratch_r, FRAME_SIZE, false)
                .unwrap();
            // If previous frame was lost, this is T3 for that loss event.
            if i > 0
                && is_lost(i - 1)
                && let Some(ev) = events.iter_mut().rev().find(|e| e.idx == i - 1)
            {
                ev.t3 = Some(snapshot(&c_dec, &r_dec));
            }
        }
    }

    eprintln!("\n=== Stage 7b.3 SILK state-divergence snapshots ===");
    eprintln!("Normalised RMS delta: sqrt(mean((c-r)^2)) / (sqrt(mean(c^2))+1e-9)\n");
    eprintln!(
        "{:<8} {:<4} {:>14} {:>14} {:>14} {:>14}",
        "loss_i", "snap", "s_lpc_Q14", "plc_prevGain", "decPrevGain", "outBuf_tail"
    );
    let mut worst: (String, f64) = ("none".into(), 0.0);
    for ev in &events {
        for (label, snap_opt) in [("T1", ev.t1), ("T2", ev.t2), ("T3", ev.t3)] {
            if let Some(s) = snap_opt {
                eprintln!(
                    "{:<8} {:<4} {:>14.6} {:>14.6} {:>14.6} {:>14.6}",
                    ev.idx,
                    label,
                    s.s_lpc_q14_delta,
                    s.plc_prev_gain_delta,
                    s.dec_prev_gain_delta,
                    s.out_buf_tail_delta
                );
                for (name, v) in [
                    ("s_lpc_q14", s.s_lpc_q14_delta),
                    ("plc_prev_gain", s.plc_prev_gain_delta),
                    ("dec_prev_gain", s.dec_prev_gain_delta),
                    ("out_buf_tail", s.out_buf_tail_delta),
                ] {
                    if v > worst.1 {
                        worst = (format!("{} @ {} (loss_idx={})", name, label, ev.idx), v);
                    }
                }
            }
        }
    }

    // Also surface scalar metadata for the first event only.
    if let Some(ev) = events.first() {
        for (label, snap_opt) in [("T1", ev.t1), ("T2", ev.t2), ("T3", ev.t3)] {
            if let Some(s) = snap_opt {
                eprintln!(
                    "  {} meta  pitch_Q8 C={} R={}  rand_scale C={} R={}  last_lost C={} R={}  plc_fs C={} R={}",
                    label,
                    s.plc_pitch_c,
                    s.plc_pitch_r,
                    s.plc_rand_scale_c,
                    s.plc_rand_scale_r,
                    s.last_lost_c,
                    s.last_lost_r,
                    s.c_plc_fs,
                    s.r_plc_fs
                );
                eprintln!(
                    "  {} RMS(C)  s_lpc={:.1} outBuf_tail={:.1}",
                    label, s.s_lpc_q14_c_rms, s.out_buf_tail_c_rms
                );
            }
        }
    }

    eprintln!("\nWORST OFFENDER: {} delta = {:.4}", worst.0, worst.1);
    eprintln!("====================================================\n");

    assert!(!events.is_empty(), "no loss events captured");
}

/// Per-loss-event SNR summary — walks through ALL losses in the test
/// sequence and reports recovery-frame SNR for each, plus per-loss-frame
/// SNR. Useful to see whether the residual tier-2 SNR is bottlenecked by
/// the first recovery after a loss or by per-lost-frame neural noise.
#[test]
fn stage7b3_per_loss_snr() {
    const N_FRAMES: usize = 50;
    let pcm = synth_reference_pcm(N_FRAMES);
    let packets = encode_with_ropus(&pcm);
    assert_eq!(packets.len(), N_FRAMES);

    let mut c_dec = CRefFloatDecoder::new(FS, CHANNELS).expect("c dec");
    c_dec.set_complexity(COMPLEXITY).expect("c complexity");
    let mut r_dec = RopusDecoder::new(FS, CHANNELS).expect("r dec");
    r_dec.set_complexity(COMPLEXITY).expect("r complexity");

    let frame_samples = FRAME_SIZE as usize;
    let mut scratch_c = vec![0i16; frame_samples];
    let mut scratch_r = vec![0i16; frame_samples];

    let is_lost = |i: usize| i > 0 && i.is_multiple_of(7);

    let frame_snr = |c: &[i16], r: &[i16]| -> f64 {
        let mut sig = 0.0f64;
        let mut err = 0.0f64;
        for i in 0..c.len() {
            sig += (c[i] as f64).powi(2);
            err += ((r[i] as i32 - c[i] as i32) as f64).powi(2);
        }
        if err == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (sig / err).log10()
        }
    };

    eprintln!("\n=== Stage 7b.3 per-frame SNR ===");
    eprintln!("{:>4} {:>6} {:>10}", "idx", "type", "SNR_dB");
    for (i, pkt) in packets.iter().enumerate() {
        if is_lost(i) {
            c_dec
                .decode(None, &mut scratch_c, FRAME_SIZE, false)
                .unwrap();
            r_dec
                .decode(None, &mut scratch_r, FRAME_SIZE, false)
                .unwrap();
            let s = frame_snr(&scratch_c, &scratch_r);
            eprintln!("{:>4} {:>6} {:>10.2}", i, "LOST", s);
        } else {
            c_dec
                .decode(Some(pkt), &mut scratch_c, FRAME_SIZE, false)
                .unwrap();
            r_dec
                .decode(Some(pkt), &mut scratch_r, FRAME_SIZE, false)
                .unwrap();
            let s = frame_snr(&scratch_c, &scratch_r);
            let label = if i > 0 && is_lost(i - 1) {
                "RCVR"
            } else {
                "good"
            };
            eprintln!("{:>4} {:>6} {:>10.2}", i, label, s);
        }
    }
}

/// Deep dump: walk through the first loss event and print per-entry
/// sLPC_Q14_buf deltas at T1, T2 (after lost frame), and T3 (after first
/// good frame). Each row shows c_value, r_value, delta.
#[test]
fn stage7b3_silk_detail_first_loss() {
    const N_FRAMES: usize = 10; // enough to get past first loss at i=7
    let pcm = synth_reference_pcm(N_FRAMES);
    let packets = encode_with_ropus(&pcm);
    assert_eq!(packets.len(), N_FRAMES);

    let mut c_dec = CRefFloatDecoder::new(FS, CHANNELS).expect("c dec");
    c_dec.set_complexity(COMPLEXITY).expect("c complexity");
    let mut r_dec = RopusDecoder::new(FS, CHANNELS).expect("r dec");
    r_dec.set_complexity(COMPLEXITY).expect("r complexity");

    let frame_samples = FRAME_SIZE as usize;
    let mut scratch_c = vec![0i16; frame_samples];
    let mut scratch_r = vec![0i16; frame_samples];

    let is_lost = |i: usize| i > 0 && i.is_multiple_of(7);

    let dump = |label: &str, c_dec: &CRefFloatDecoder, r_dec: &RopusDecoder| {
        let c_slpc = c_dec.silk_s_lpc_q14();
        let silk_r = r_dec.debug_silk_dec();
        let cs_r = &silk_r.channel_state[0];
        let r_slpc = cs_r.s_lpc_q14_buf;
        eprintln!("\n--- {} sLPC_Q14_buf ---", label);
        eprintln!("{:>3} {:>14} {:>14} {:>12}", "i", "C", "R", "R-C");
        for i in 0..16 {
            let c = c_slpc[i];
            let r = r_slpc[i];
            eprintln!("{:>3} {:>14} {:>14} {:>12}", i, c, r, r - c);
        }
        // Context state
        eprintln!(
            "  prev_gain_Q16 C={} R={}  plc_prevGain_Q16 C={:?} R={:?}",
            c_dec.silk_prev_gain_q16(),
            cs_r.prev_gain_q16,
            c_dec.silk_plc_prev_gain_q16(),
            cs_r.s_plc.prev_gain_q16
        );
        // Dump full outBuf diff summary
        let ltp_mem = c_dec.silk_ltp_mem_length();
        let c_ob = c_dec.silk_out_buf(0, ltp_mem);
        let r_ob = &cs_r.out_buf[..ltp_mem as usize];
        let mut ob_diffs = 0;
        let mut first_diff = None;
        let mut last_diff = None;
        let mut max_abs_delta = 0i32;
        let mut diff_positions = Vec::new();
        for i in 0..(ltp_mem as usize) {
            let d = (r_ob[i] as i32) - (c_ob[i] as i32);
            if d != 0 {
                ob_diffs += 1;
                if first_diff.is_none() {
                    first_diff = Some(i);
                }
                last_diff = Some(i);
                if d.abs() > max_abs_delta {
                    max_abs_delta = d.abs();
                }
                if diff_positions.len() < 16 {
                    diff_positions.push((i, c_ob[i], r_ob[i], d));
                }
            }
        }
        eprintln!(
            "  outBuf[0..{}] diffs={} first={:?} last={:?} max_abs_delta={}",
            ltp_mem, ob_diffs, first_diff, last_diff, max_abs_delta
        );
        for (pos, c, r, d) in &diff_positions {
            eprintln!("    outBuf[{}] C={} R={} delta={}", pos, c, r, d);
        }
    };

    for (i, pkt) in packets.iter().enumerate() {
        if is_lost(i) {
            dump(&format!("T1 before loss@{}", i), &c_dec, &r_dec);
            c_dec
                .decode(None, &mut scratch_c, FRAME_SIZE, false)
                .unwrap();
            r_dec
                .decode(None, &mut scratch_r, FRAME_SIZE, false)
                .unwrap();
            dump(&format!("T2 after lost@{}", i), &c_dec, &r_dec);
            // Capture output frame for comparison
            let mut diffs = 0;
            for j in 0..frame_samples {
                if scratch_c[j] != scratch_r[j] {
                    diffs += 1;
                }
            }
            eprintln!(
                "  PCM output samples diverging: {}/{}",
                diffs, frame_samples
            );
            break; // Only first loss
        } else {
            c_dec
                .decode(Some(pkt), &mut scratch_c, FRAME_SIZE, false)
                .unwrap();
            r_dec
                .decode(Some(pkt), &mut scratch_r, FRAME_SIZE, false)
                .unwrap();
        }
    }

    // Now decode the NEXT frame (first good after loss) and dump T3
    let next_good_idx = 8; // since is_lost(7) = true, 8 is next good
    let pkt = &packets[next_good_idx];
    c_dec
        .decode(Some(pkt), &mut scratch_c, FRAME_SIZE, false)
        .unwrap();
    r_dec
        .decode(Some(pkt), &mut scratch_r, FRAME_SIZE, false)
        .unwrap();
    dump(
        &format!("T3 after recover@{}", next_good_idx),
        &c_dec,
        &r_dec,
    );

    // Output PCM comparison on recovery frame
    let mut diffs = 0;
    let mut first_diff = None;
    let mut max_abs_delta = 0i32;
    let mut sum_err_sq = 0.0_f64;
    let mut sum_sig_sq = 0.0_f64;
    for j in 0..frame_samples {
        if scratch_c[j] != scratch_r[j] {
            diffs += 1;
            if first_diff.is_none() {
                first_diff = Some(j);
            }
        }
        let d = scratch_r[j] as i32 - scratch_c[j] as i32;
        if d.abs() > max_abs_delta {
            max_abs_delta = d.abs();
        }
        sum_err_sq += (d as f64).powi(2);
        sum_sig_sq += (scratch_c[j] as f64).powi(2);
    }
    let snr = if sum_err_sq > 0.0 {
        10.0 * (sum_sig_sq / sum_err_sq).log10()
    } else {
        f64::INFINITY
    };
    eprintln!(
        "  Recovery PCM diverging: {}/{}, first diff at {:?}, max_abs_delta={}, SNR={:.2} dB",
        diffs, frame_samples, first_diff, max_abs_delta, snr
    );
}
