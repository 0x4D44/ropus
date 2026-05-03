//! ropus-fuzz-repro-diff: encoder state-accumulation diagnostic.
//!
//! Runs each of the eight `tests/fuzz/known_failures/` repros (4 multiframe
//! and 4 multistream) through both the Rust and C reference encoders with
//! identical setter sequences, then dumps SILK and top-level Opus state per
//! frame (and pre-/post-setter for the multistream variants) to pinpoint
//! the first divergent field.
//!
//! Stage 1 of the encoder-state-accumulation fix per
//! `wrk_docs/2026.05.02 - HLD - encoder-state-accumulation-fix.md`.
//! The harness is read-only against `ropus/src/**`; production-code edits
//! happen in stage 2 after supervisor review of stage-1 findings.
//!
//! Modes:
//!   --mode multiframe  - replays a fuzz_encode_multiframe Arbitrary input.
//!   --mode multistream - replays a fuzz_multistream Arbitrary input.

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::single_match
)]

use ropus_harness::bindings;

use std::os::raw::{c_int, c_uchar};
use std::path::PathBuf;

use ropus::opus::encoder::{
    OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_APPLICATION_VOIP,
    OpusEncoder as RustOpusEncoder,
};
use ropus::opus::multistream::OpusMSEncoder as RustOpusMSEncoder;

const SAMPLE_RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];
const APPLICATIONS: [i32; 3] = [
    OPUS_APPLICATION_VOIP,
    OPUS_APPLICATION_AUDIO,
    OPUS_APPLICATION_RESTRICTED_LOWDELAY,
];
const MAX_CHANNELS: u8 = 8;

// ===========================================================================
// Minimal `arbitrary::Unstructured` reimplementation
// ---------------------------------------------------------------------------
// We can't add the `arbitrary` crate as a harness dependency in stage 1
// (Cargo.toml is locked). The fuzz targets' Arbitrary impls only use a small
// surface — `int_in_range`, `arbitrary::<u16>()`/`u8`/`bool`, `bytes`,
// `fill_buffer` — and the byte-consumption rules are deterministic. We
// re-implement just that surface so the diagnostic deserialises identically
// to libfuzzer at runtime.
//
// Reference (arbitrary v1.4.2):
//   - `int_in_range(0..=N)` reads from the *front*, consuming the minimum
//     bytes to span the range, then `result = (raw % (delta+1)) + start`.
//   - `<u8 as Arbitrary>::arbitrary` reads 1 byte from the front.
//   - `<u16 as Arbitrary>::arbitrary` reads 2 bytes from the front,
//     `u16::from_le_bytes`.
//   - `<bool as Arbitrary>::arbitrary` is `<u8>::arbitrary & 1 == 1`.
//   - `bytes(n)` errors if fewer than n bytes available; otherwise
//     consumes n from the front.
// ===========================================================================
struct Unstructured<'a> {
    data: &'a [u8],
}

impl<'a> Unstructured<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data }
    }

    fn int_in_range_u8(&mut self, lo: u8, hi: u8) -> Result<u8, &'static str> {
        if lo == hi {
            return Ok(lo);
        }
        let delta = hi.wrapping_sub(lo);
        // For u8, only one byte is needed.
        let raw = if self.data.is_empty() {
            0
        } else {
            let b = self.data[0];
            self.data = &self.data[1..];
            b
        };
        let offset = if delta == u8::MAX {
            raw
        } else {
            raw % (delta + 1)
        };
        Ok(lo.wrapping_add(offset))
    }

    fn arbitrary_u8(&mut self) -> Result<u8, &'static str> {
        let mut buf = [0u8; 1];
        self.fill_buffer(&mut buf)?;
        Ok(buf[0])
    }

    fn arbitrary_u16(&mut self) -> Result<u16, &'static str> {
        let mut buf = [0u8; 2];
        self.fill_buffer(&mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    fn arbitrary_bool(&mut self) -> Result<bool, &'static str> {
        Ok(self.arbitrary_u8()? & 1 == 1)
    }

    /// Fill buffer from available data; remaining bytes are zero-padded
    /// (matches `arbitrary::Unstructured::fill_buffer`).
    fn fill_buffer(&mut self, buffer: &mut [u8]) -> Result<(), &'static str> {
        let n = buffer.len().min(self.data.len());
        buffer[..n].copy_from_slice(&self.data[..n]);
        for byte in buffer[n..].iter_mut() {
            *byte = 0;
        }
        self.data = &self.data[n..];
        Ok(())
    }

    /// Consume exactly `n` bytes from the front; error if not available.
    fn bytes(&mut self, n: usize) -> Result<&'a [u8], &'static str> {
        if self.data.len() < n {
            return Err("not enough bytes");
        }
        let (head, tail) = self.data.split_at(n);
        self.data = tail;
        Ok(head)
    }
}

// ===========================================================================
// MultiframeInput — mirrors fuzz_encode_multiframe.rs:135-180.
// ===========================================================================
#[derive(Debug)]
struct FrameConfig {
    bitrate_raw: u16,
    complexity: u8,
    vbr: bool,
    fec: u8,
    dtx: bool,
    loss_perc: u8,
    pcm: Vec<i16>,
}

#[derive(Debug)]
struct MultiframeInput {
    sample_rate: i32,
    channels: i32,
    application: i32,
    frames: Vec<FrameConfig>,
}

fn parse_multiframe(data: &[u8]) -> Result<MultiframeInput, &'static str> {
    let mut u = Unstructured::new(data);
    let sample_rate_idx = u.int_in_range_u8(0, 4)? as usize;
    let channels_minus_one = u.int_in_range_u8(0, 1)? as i32;
    let application_idx = u.int_in_range_u8(0, 2)? as usize;
    let n_frames = u.int_in_range_u8(2, 16)? as usize;

    let sample_rate = SAMPLE_RATES[sample_rate_idx];
    let channels = channels_minus_one + 1;
    let application = APPLICATIONS[application_idx];
    let frame_size = (sample_rate / 50) as usize;
    let pcm_bytes_needed = frame_size * channels as usize * 2;

    let mut frames = Vec::with_capacity(n_frames);
    for _ in 0..n_frames {
        let bitrate_raw = u.arbitrary_u16()?;
        let complexity = u.int_in_range_u8(0, 9)?;
        let vbr = u.arbitrary_bool()?;
        let fec = u.int_in_range_u8(0, 2)?;
        let dtx = u.arbitrary_bool()?;
        let loss_perc = u.int_in_range_u8(0, 100)?;
        let pcm_bytes = u.bytes(pcm_bytes_needed)?;
        let pcm: Vec<i16> = pcm_bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        frames.push(FrameConfig {
            bitrate_raw,
            complexity,
            vbr,
            fec,
            dtx,
            loss_perc,
            pcm,
        });
    }

    Ok(MultiframeInput {
        sample_rate,
        channels,
        application,
        frames,
    })
}

fn raw_to_bitrate(raw: u16) -> i32 {
    6000 + (raw as i32 % 504_001)
}

// ===========================================================================
// MSInput — mirrors fuzz_multistream.rs:107-148.
// ===========================================================================
#[derive(Debug)]
struct MSInput {
    op: u8,
    sample_rate: i32,
    channels: i32,
    mapping_family: i32,
    application: i32,
    bitrate: i32,
    complexity: i32,
    vbr: i32,
    setter_bytes: [u8; 16],
    payload: Vec<u8>,
}

fn parse_multistream(data: &[u8]) -> Result<MSInput, &'static str> {
    let mut u = Unstructured::new(data);
    let op = u.int_in_range_u8(0, 2)?;
    let sample_rate_idx = u.int_in_range_u8(0, 4)? as usize;
    let channels = u.int_in_range_u8(1, MAX_CHANNELS)? as i32;
    let mapping_family_idx = u.int_in_range_u8(0, 3)?;
    let mapping_family = match mapping_family_idx {
        0 => 0,
        1 => 1,
        2 => 2,
        _ => 255,
    };
    let application_idx = u.int_in_range_u8(0, 2)? as usize;
    let bitrate_raw = u.arbitrary_u16()?;
    let complexity = u.int_in_range_u8(0, 9)? as i32;
    let vbr = if u.arbitrary_bool()? { 1 } else { 0 };
    let mut setter_bytes = [0u8; 16];
    u.fill_buffer(&mut setter_bytes)?;
    // `int_in_range(0..=20480)` consumes 2 bytes (range fits in u16).
    let payload_len = int_in_range_u32(&mut u, 0, 20480)? as usize;
    let payload = u.bytes(payload_len)?.to_vec();

    Ok(MSInput {
        op,
        sample_rate: SAMPLE_RATES[sample_rate_idx],
        channels,
        mapping_family,
        application: APPLICATIONS[application_idx],
        bitrate: raw_to_bitrate(bitrate_raw),
        complexity,
        vbr,
        setter_bytes,
        payload,
    })
}

/// `int_in_range` for ranges that don't fit in u8. Mirrors the same
/// minimum-bytes loop as `arbitrary::Unstructured::int_in_range_impl`.
fn int_in_range_u32(u: &mut Unstructured, lo: u32, hi: u32) -> Result<u32, &'static str> {
    if lo == hi {
        return Ok(lo);
    }
    let delta = hi.wrapping_sub(lo);
    let mut raw: u32 = 0;
    let mut consumed: usize = 0;
    while consumed < 4 && (delta >> (consumed * 8)) > 0 {
        let b = if u.data.is_empty() {
            // Real arbitrary stops here with what's been read so far.
            break;
        } else {
            let b = u.data[0];
            u.data = &u.data[1..];
            b
        };
        raw = (raw << 8) | (b as u32);
        consumed += 1;
    }
    let offset = if delta == u32::MAX {
        raw
    } else {
        raw % (delta + 1)
    };
    Ok(lo.wrapping_add(offset))
}

// ===========================================================================
// State dump structures
// ===========================================================================
#[derive(Default, Debug, PartialEq, Eq, Clone)]
struct SilkModeState {
    use_in_band_fec: i32,
    use_cbr: i32,
    use_dtx: i32,
    lbrr_coded: i32,
    complexity: i32,
    packet_loss_percentage: i32,
    bit_rate: i32,
    payload_size_ms: i32,
    n_channels_internal: i32,
    max_internal_sample_rate: i32,
    min_internal_sample_rate: i32,
    desired_internal_sample_rate: i32,
}

#[derive(Default, Debug, PartialEq, Eq, Clone)]
struct OpusTopState {
    use_vbr: i32,
    vbr_constraint: i32,
    use_dtx: i32,
    fec_config: i32,
    user_bitrate_bps: i32,
    bitrate_bps: i32,
    force_channels: i32,
    signal_type: i32,
    lsb_depth: i32,
    lfe: i32,
    application: i32,
}

fn dump_c_top(enc: *mut bindings::OpusEncoder) -> (SilkModeState, OpusTopState) {
    let mut sm = SilkModeState::default();
    let mut top = OpusTopState::default();
    unsafe {
        bindings::debug_get_opus_silk_mode_state(
            enc,
            &mut sm.use_in_band_fec,
            &mut sm.use_cbr,
            &mut sm.use_dtx,
            &mut sm.lbrr_coded,
            &mut sm.complexity,
            &mut sm.packet_loss_percentage,
            &mut sm.bit_rate,
            &mut sm.payload_size_ms,
            &mut sm.n_channels_internal,
            &mut sm.max_internal_sample_rate,
            &mut sm.min_internal_sample_rate,
            &mut sm.desired_internal_sample_rate,
            &mut top.use_vbr,
            &mut top.vbr_constraint,
            &mut top.use_dtx,
            &mut top.fec_config,
            &mut top.user_bitrate_bps,
            &mut top.bitrate_bps,
            &mut top.force_channels,
            &mut top.signal_type,
            &mut top.lsb_depth,
            &mut top.lfe,
            &mut top.application,
        );
    }
    (sm, top)
}

/// Dump the Rust-reachable portion of `OpusEncoder` top-level state:
/// public getters cover `use_vbr`, `use_dtx`, `fec_config`,
/// `silk_mode.complexity`, and `silk_mode.packet_loss_percentage`. Every
/// other `silk_mode.*` field is private — Phase A relies on the C-side
/// dump (via `debug_get_opus_silk_mode_state`) plus the per-channel
/// SILK encoder state-mirror (`state_fxx[ch].s_cmn.{use_in_band_fec,
/// use_cbr, use_dtx}`) which `silk_control_encoder` writes during encode.
fn dump_rust_top(enc: &RustOpusEncoder) -> OpusTopState {
    OpusTopState {
        use_vbr: enc.get_vbr(),
        vbr_constraint: enc.get_vbr_constraint(),
        use_dtx: enc.get_dtx(),
        fec_config: enc.get_inband_fec(),
        user_bitrate_bps: 0, // not exposed; set by ms_set_user_bitrate at encode time
        bitrate_bps: enc.get_bitrate(),
        force_channels: enc.get_force_channels(),
        signal_type: enc.get_signal(),
        lsb_depth: enc.get_lsb_depth(),
        // lfe and application are not exposed via getters at the
        // OpusEncoder level — left at default 0; not used by Phase A diff.
        lfe: 0,
        application: enc.application,
    }
}

#[derive(Default, Debug, PartialEq, Eq, Clone)]
struct SilkPostEncodeMirror {
    /// `state_fxx[ch].s_cmn.use_in_band_fec` — written by
    /// `silk_control_encoder` from `enc_control.use_in_band_fec` (i.e.
    /// from `silk_mode.use_in_band_fec` on the Opus side). Visible
    /// only AFTER an `encode` call has propagated the silk_mode value.
    use_in_band_fec: i32,
    use_cbr: i32,
    use_dtx: i32,
    lbrr_enabled: i32,
}

fn dump_rust_silk_mirror(enc: &RustOpusEncoder, channel: usize) -> Option<SilkPostEncodeMirror> {
    let silk = enc.silk_encoder()?;
    let st = &silk.state_fxx[channel].s_cmn;
    Some(SilkPostEncodeMirror {
        use_in_band_fec: st.use_in_band_fec,
        use_cbr: st.use_cbr,
        use_dtx: st.use_dtx,
        lbrr_enabled: st.lbrr_enabled,
    })
}

// ===========================================================================
// Silk inter-frame state (mirrored from interframe.rs but lifted here so
// this binary is self-contained — replicates fields, not logic).
// ===========================================================================
#[derive(Default, Debug, Clone, PartialEq, Eq)]
struct SilkInterframe {
    last_gain_index: i32,
    prev_gain_q16: i32,
    variable_hp_smth1_q15: i32,
    variable_hp_smth2_q15: i32,
    harm_shape_gain_smth: i32,
    tilt_smth: i32,
    prev_signal_type: i32,
    prev_lag: i32,
    ec_prev_lag_index: i32,
    ec_prev_signal_type: i32,
    prev_nlsfq_q15: [i16; 16],
    stereo_width_prev_q14: i32,
    stereo_smth_width_q14: i32,
    stereo_pred_prev_q13_0: i32,
    stereo_pred_prev_q13_1: i32,
    n_bits_exceeded: i32,
}

fn dump_c_interframe(enc: *mut bindings::OpusEncoder, channel: i32) -> SilkInterframe {
    let mut s = SilkInterframe::default();
    unsafe {
        bindings::debug_get_silk_interframe_state(
            enc,
            channel as c_int,
            &mut s.last_gain_index,
            &mut s.prev_gain_q16,
            &mut s.variable_hp_smth1_q15,
            &mut s.variable_hp_smth2_q15,
            &mut s.harm_shape_gain_smth,
            &mut s.tilt_smth,
            &mut s.prev_signal_type,
            &mut s.prev_lag,
            &mut s.ec_prev_lag_index,
            &mut s.ec_prev_signal_type,
            s.prev_nlsfq_q15.as_mut_ptr(),
            &mut s.stereo_width_prev_q14,
            &mut s.stereo_smth_width_q14,
            &mut s.stereo_pred_prev_q13_0,
            &mut s.stereo_pred_prev_q13_1,
            &mut s.n_bits_exceeded,
        );
    }
    s
}

fn dump_rust_interframe(enc: &RustOpusEncoder, channel: usize) -> SilkInterframe {
    let silk = enc
        .silk_encoder()
        .expect("SILK encoder must be allocated for SILK / Hybrid frames");
    let ch = &silk.state_fxx[channel];
    let st = &ch.s_cmn;
    let mut prev_nlsfq_q15 = [0i16; 16];
    prev_nlsfq_q15.copy_from_slice(&st.prev_nlsfq_q15);
    SilkInterframe {
        last_gain_index: ch.s_shape.last_gain_index as i32,
        prev_gain_q16: st.s_nsq.prev_gain_q16,
        variable_hp_smth1_q15: st.variable_hp_smth1_q15,
        variable_hp_smth2_q15: st.variable_hp_smth2_q15,
        harm_shape_gain_smth: ch.s_shape.harm_shape_gain_smth,
        tilt_smth: ch.s_shape.tilt_smth,
        prev_signal_type: st.prev_signal_type,
        prev_lag: st.prev_lag,
        ec_prev_lag_index: st.ec_prev_lag_index as i32,
        ec_prev_signal_type: st.ec_prev_signal_type,
        prev_nlsfq_q15,
        stereo_width_prev_q14: silk.s_stereo.width_prev_q14 as i32,
        stereo_smth_width_q14: silk.s_stereo.smth_width_q14 as i32,
        stereo_pred_prev_q13_0: silk.s_stereo.pred_prev_q13[0] as i32,
        stereo_pred_prev_q13_1: silk.s_stereo.pred_prev_q13[1] as i32,
        n_bits_exceeded: silk.n_bits_exceeded,
    }
}

/// Print first divergent field across two `SilkInterframe` snapshots.
fn diff_silk_interframe(c: &SilkInterframe, r: &SilkInterframe) -> Vec<String> {
    let mut hits = Vec::new();
    macro_rules! cmp {
        ($name:ident) => {
            if c.$name != r.$name {
                hits.push(format!(
                    "{}: C={} R={}",
                    stringify!($name),
                    c.$name,
                    r.$name
                ));
            }
        };
    }
    cmp!(last_gain_index);
    cmp!(prev_gain_q16);
    cmp!(variable_hp_smth1_q15);
    cmp!(variable_hp_smth2_q15);
    cmp!(harm_shape_gain_smth);
    cmp!(tilt_smth);
    cmp!(prev_signal_type);
    cmp!(prev_lag);
    cmp!(ec_prev_lag_index);
    cmp!(ec_prev_signal_type);
    cmp!(stereo_width_prev_q14);
    cmp!(stereo_smth_width_q14);
    cmp!(stereo_pred_prev_q13_0);
    cmp!(stereo_pred_prev_q13_1);
    cmp!(n_bits_exceeded);
    for i in 0..16 {
        if c.prev_nlsfq_q15[i] != r.prev_nlsfq_q15[i] {
            hits.push(format!(
                "prev_nlsfq_q15[{i}]: C={} R={}",
                c.prev_nlsfq_q15[i], r.prev_nlsfq_q15[i]
            ));
        }
    }
    hits
}

// ===========================================================================
// Repro drivers
// ===========================================================================

fn run_multiframe_repro(path: &PathBuf) {
    println!("\n========================================================");
    println!("REPRO multiframe: {}", path.display());
    println!("========================================================");
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let input = match parse_multiframe(&data) {
        Ok(i) => i,
        Err(e) => {
            println!("PARSE ERROR: {e}");
            return;
        }
    };
    let n_frames = input.frames.len();
    let frame_size = input.sample_rate / 50;
    println!(
        "Config: sr={} ch={} app={} frames={}",
        input.sample_rate, input.channels, input.application, n_frames
    );

    // Rust encoder
    let mut rust_enc =
        match RustOpusEncoder::new(input.sample_rate, input.channels, input.application) {
            Ok(e) => e,
            Err(e) => {
                println!("Rust encoder create failed: {e}");
                return;
            }
        };

    // C encoder
    let c_enc = unsafe {
        let mut error: c_int = 0;
        let enc = bindings::opus_encoder_create(
            input.sample_rate,
            input.channels,
            input.application,
            &mut error,
        );
        if enc.is_null() || error != bindings::OPUS_OK {
            println!("C encoder create failed: error={error}");
            return;
        }
        enc
    };

    let n_channels = input.channels as usize;
    let mut first_divergent_frame: Option<usize> = None;
    let mut frame_findings: Vec<String> = Vec::new();

    for (frame_idx, fc) in input.frames.iter().enumerate() {
        let bitrate = raw_to_bitrate(fc.bitrate_raw);
        let vbr = if fc.vbr { 1 } else { 0 };
        let dtx = if fc.dtx { 1 } else { 0 };
        let inband_fec = fc.fec as i32;
        let loss_perc = fc.loss_perc as i32;
        let complexity = fc.complexity as i32;

        // Apply identical setters on both sides — same order as
        // fuzz_encode_multiframe.rs:245-251 / c_reference.rs:272-282.
        rust_enc.set_bitrate(bitrate);
        rust_enc.set_vbr(vbr);
        rust_enc.set_vbr_constraint(0);
        rust_enc.set_inband_fec(inband_fec);
        rust_enc.set_dtx(dtx);
        rust_enc.set_packet_loss_perc(loss_perc);
        rust_enc.set_complexity(complexity);

        unsafe {
            bindings::opus_encoder_ctl(c_enc, bindings::OPUS_SET_BITRATE_REQUEST, bitrate);
            bindings::opus_encoder_ctl(c_enc, bindings::OPUS_SET_VBR_REQUEST, vbr);
            bindings::opus_encoder_ctl(
                c_enc,
                bindings::OPUS_SET_VBR_CONSTRAINT_REQUEST,
                0 as c_int,
            );
            bindings::opus_encoder_ctl(c_enc, bindings::OPUS_SET_INBAND_FEC_REQUEST, inband_fec);
            bindings::opus_encoder_ctl(c_enc, bindings::OPUS_SET_DTX_REQUEST, dtx);
            bindings::opus_encoder_ctl(
                c_enc,
                bindings::OPUS_SET_PACKET_LOSS_PERC_REQUEST,
                loss_perc,
            );
            bindings::opus_encoder_ctl(c_enc, bindings::OPUS_SET_COMPLEXITY_REQUEST, complexity);
        }

        // Pre-encode dump: top-level + silk_mode.
        let (c_sm_pre, c_top_pre) = dump_c_top(c_enc);
        let r_top_pre = dump_rust_top(&rust_enc);

        if frame_idx == 0 || first_divergent_frame == Some(frame_idx) {
            println!(
                "\n--- Frame {} pre-encode ---  br={} cx={} vbr={} fec={} dtx={} loss={}",
                frame_idx, bitrate, complexity, vbr, inband_fec, dtx, loss_perc
            );
            print_top_state_diff(
                "C silk_mode (post-setter, pre-encode)",
                &c_sm_pre,
                &c_top_pre,
            );
            print_rust_top_state(&r_top_pre);
            // Asymmetry detector: did Rust's reachable getters match C's
            // reachable top-level fields?
            assert_eq_field(
                "use_vbr (top-level)",
                c_top_pre.use_vbr,
                r_top_pre.use_vbr,
                &mut frame_findings,
                frame_idx,
            );
            assert_eq_field(
                "use_dtx (top-level)",
                c_top_pre.use_dtx,
                r_top_pre.use_dtx,
                &mut frame_findings,
                frame_idx,
            );
            assert_eq_field(
                "fec_config (top-level)",
                c_top_pre.fec_config,
                r_top_pre.fec_config,
                &mut frame_findings,
                frame_idx,
            );
        }

        // Encode both sides. Capture Phase B trace tuples per frame.
        let mut rust_out = vec![0u8; 4000];
        ropus::silk_trace::clear();
        let rust_len = match rust_enc.encode(&fc.pcm, frame_size, &mut rust_out, 4000) {
            Ok(n) => n as usize,
            Err(e) => {
                println!("Rust encode frame {frame_idx} failed: {e}");
                break;
            }
        };
        let rust_trace = ropus::silk_trace::snapshot();
        rust_out.truncate(rust_len);

        let mut c_out = vec![0u8; 4000];
        unsafe { bindings::dbg_silk_trace_clear() };
        let c_len = unsafe {
            bindings::opus_encode(
                c_enc,
                fc.pcm.as_ptr() as *const bindings::opus_int16,
                frame_size,
                c_out.as_mut_ptr() as *mut c_uchar,
                4000,
            )
        };
        let c_trace = read_c_silk_trace();
        if c_len < 0 {
            println!("C encode frame {frame_idx} failed: {c_len}");
            break;
        }
        c_out.truncate(c_len as usize);

        let bytes_match = rust_out == c_out;
        let tag = if bytes_match { "OK" } else { "DIFF" };
        let first_diff_byte = if bytes_match {
            None
        } else {
            (0..rust_out.len().min(c_out.len())).find(|&i| rust_out[i] != c_out[i])
        };
        println!(
            "[frame {frame_idx}] BYTES-{tag}  Rust={}B C={}B{}",
            rust_out.len(),
            c_out.len(),
            match first_diff_byte {
                Some(b) => format!(" first-diff-at={}", b),
                None =>
                    if bytes_match {
                        String::new()
                    } else {
                        " (length differs)".to_string()
                    },
            }
        );

        if !bytes_match && first_divergent_frame.is_none() {
            first_divergent_frame = Some(frame_idx);
        }

        // Phase B trace diff: dump only on divergent frames or frame 0,
        // to keep output bounded for multi-frame repros.
        if !bytes_match || frame_idx == 0 {
            print_phase_b_trace_diff(&rust_trace, &c_trace);
            print_phase_c_trace_diff(&rust_trace, &c_trace);
        }

        // Post-encode state dumps.
        for ch in 0..n_channels {
            let c_if = dump_c_interframe(c_enc, ch as i32);
            let r_if = dump_rust_interframe(&rust_enc, ch);
            let diffs = diff_silk_interframe(&c_if, &r_if);
            if !diffs.is_empty() {
                println!(
                    "[frame {frame_idx} ch{ch}] interframe-state mismatches ({}):",
                    diffs.len()
                );
                for d in &diffs {
                    println!("    {d}");
                    frame_findings.push(format!("frame {frame_idx} ch{ch} {d}"));
                }
            } else if frame_idx == 0 || first_divergent_frame == Some(frame_idx) {
                println!("[frame {frame_idx} ch{ch}] interframe-state OK");
            }
            // Extended state diff at frame 0 only (initial baseline) and
            // at the first divergent frame (the smoking-gun frame). Skip
            // intermediate frames to keep output bounded.
            if frame_idx == 0 || first_divergent_frame == Some(frame_idx) {
                let ext_diffs = compare_extended_state(c_enc, &rust_enc, ch);
                if !ext_diffs.is_empty() {
                    println!(
                        "[frame {frame_idx} ch{ch}] extended-state mismatches ({}); first 8:",
                        ext_diffs.len()
                    );
                    for d in ext_diffs.iter().take(8) {
                        println!("    {d}");
                        frame_findings.push(format!("frame {frame_idx} ch{ch} ext {d}"));
                    }
                } else {
                    println!("[frame {frame_idx} ch{ch}] extended-state OK");
                }
            }

            // Mirror state — silk_mode propagation evidence.
            let r_mirror =
                dump_rust_silk_mirror(&rust_enc, ch).expect("Rust SILK encoder allocated");
            // Read C-side silk_mode again post-encode (it may have been
            // recomputed by silk_control_snr / encode_native gating).
            let (c_sm_post, _c_top_post) = dump_c_top(c_enc);
            // Compare C silk_mode against the per-channel mirror that
            // SILK propagated into state_fxx[ch].s_cmn.
            macro_rules! mirror_check {
                ($field_c:ident, $field_r:ident) => {{
                    let cv = c_sm_post.$field_c;
                    let rv = r_mirror.$field_r;
                    if cv != rv {
                        let s = format!(
                            "frame {frame_idx} ch{ch} silk_mirror {} vs C silk_mode.{}: C={} R(s_cmn.{})={}",
                            stringify!($field_r),
                            stringify!($field_c),
                            cv,
                            stringify!($field_r),
                            rv
                        );
                        println!("    MIRROR {s}");
                        frame_findings.push(s);
                    }
                }};
            }
            mirror_check!(use_in_band_fec, use_in_band_fec);
            mirror_check!(use_cbr, use_cbr);
            mirror_check!(use_dtx, use_dtx);
        }

        if !bytes_match && first_divergent_frame == Some(frame_idx) {
            // Stop after the first divergent frame: subsequent frames
            // are downstream effects, not new findings.
            println!("(stopping at first divergent frame)");
            break;
        }
    }

    unsafe {
        bindings::opus_encoder_destroy(c_enc);
    }

    println!("\n=== SUMMARY {} ===", path.display());
    if let Some(f) = first_divergent_frame {
        println!("First divergent frame: {f}");
    } else {
        println!("All frames matched (no divergence reproduced)");
    }
    if frame_findings.is_empty() {
        println!("No state-field findings.");
    } else {
        println!("Findings ({}):", frame_findings.len());
        for f in &frame_findings {
            println!("  - {f}");
        }
    }
}

/// Read the C-side Phase B trace FIFO into a Vec.
///
/// Drains BOTH the V1 7-tuple ring (`dbg_silk_trace_read`) and the V2
/// payload ring (`dbg_silk_trace_read_payload`) into a single combined
/// `Vec<Tuple>`. V1 records have `boundary_id ∈ 1..=7` and `payload = []`;
/// V2 records have `boundary_id ∈ 100..=109` and `payload` populated per
/// the §4.1 schema. The two are interleaved here in V1-first order, but
/// the diff-side filters by `boundary_id` so the order is irrelevant for
/// correctness — it just keeps the sequential dump readable.
fn read_c_silk_trace() -> Vec<ropus::silk_trace::Tuple> {
    let v1_count = unsafe { bindings::dbg_silk_trace_count_get() } as usize;
    let v2_count = unsafe { bindings::dbg_silk_trace_payload_count_get() } as usize;
    let mut out = Vec::with_capacity(v1_count + v2_count);

    // V1 records (boundary 1..=7).
    for i in 0..v1_count {
        let mut boundary_id: c_int = 0;
        let mut channel: c_int = 0;
        let mut ec_tell: i32 = 0;
        let mut rng: u32 = 0;
        let mut target_rate_bps: i32 = 0;
        let mut n_bits_exceeded: i32 = 0;
        let mut curr_n_bits_used_lbrr: i32 = 0;
        let mut n_bits_used_lbrr: i32 = 0;
        let mut mid_only_flag: i32 = 0;
        let mut prev_decode_only_middle: i32 = 0;
        let r = unsafe {
            bindings::dbg_silk_trace_read(
                i as c_int,
                &mut boundary_id,
                &mut channel,
                &mut ec_tell,
                &mut rng,
                &mut target_rate_bps,
                &mut n_bits_exceeded,
                &mut curr_n_bits_used_lbrr,
                &mut n_bits_used_lbrr,
                &mut mid_only_flag,
                &mut prev_decode_only_middle,
            )
        };
        if r != 0 {
            break;
        }
        out.push(ropus::silk_trace::Tuple {
            boundary_id,
            channel,
            ec_tell,
            rng,
            target_rate_bps,
            n_bits_exceeded,
            curr_n_bits_used_lbrr,
            n_bits_used_lbrr,
            mid_only_flag,
            prev_decode_only_middle,
            // V1 records leave V2 fields at defaults (iter=-1 per the
            // custom `Default` impl, payload=[]).
            ..Default::default()
        });
    }

    // V2 records (boundary 100..=109). Each record's payload can be up
    // to 384 i32s (DBG_SILK_TRACE_PAYLOAD_MAX in `harness/debug_silk_trace.c`).
    // We allocate a 512-i32 scratch buffer per call: well above the
    // 384-i32 cap and matches the Rust-side push call sites in
    // `silk/encoder.rs` that build payloads of comparable size.
    const V2_BUF_CAP: usize = 512;
    let mut buf = vec![0i32; V2_BUF_CAP];
    for i in 0..v2_count {
        let mut boundary_id: c_int = 0;
        let mut channel: c_int = 0;
        let mut iter: c_int = 0;
        let mut out_len: c_int = 0;
        let r = unsafe {
            bindings::dbg_silk_trace_read_payload(
                i as c_int,
                &mut boundary_id,
                &mut channel,
                &mut iter,
                &mut out_len,
                buf.as_mut_ptr(),
                V2_BUF_CAP as c_int,
            )
        };
        if r != 0 {
            break;
        }
        let len = (out_len as usize).min(V2_BUF_CAP);
        let payload = buf[..len].to_vec();
        // V2 records: scalar header fields aren't meaningful (V2 boundaries
        // emit state-vector payloads, not the V1 (ec_tell, rng, target_rate)
        // tuple). Default everything except the four V2-defining fields.
        out.push(ropus::silk_trace::Tuple {
            boundary_id,
            channel,
            iter,
            payload,
            ..Default::default()
        });
    }

    out
}

/// Print Phase B trace tuples side-by-side with first-divergence call-out.
/// Each side's tuples are aligned by `(boundary_id, channel)` ordering. If
/// the boundary sequence diverges at any step, the per-side stream from
/// that point on is shown but no further alignment is attempted.
///
/// V1 only — both sides are filtered to `boundary_id < 100` so the V1
/// 7-boundary api-level grid prints cleanly even when the V2 inner-function
/// grid is also populated. The V2 grid is dumped separately by
/// [`print_phase_c_trace_diff`].
fn print_phase_b_trace_diff(
    rust_trace: &[ropus::silk_trace::Tuple],
    c_trace: &[ropus::silk_trace::Tuple],
) {
    let rust_v1: Vec<&ropus::silk_trace::Tuple> = rust_trace
        .iter()
        .filter(|t| t.boundary_id > 0 && t.boundary_id < 100)
        .collect();
    let c_v1: Vec<&ropus::silk_trace::Tuple> = c_trace
        .iter()
        .filter(|t| t.boundary_id > 0 && t.boundary_id < 100)
        .collect();

    println!(
        "\n--- Phase B trace tuples (boundary, ec_tell, rng, target_rate, n_bits_exceeded, curr_lbrr, n_lbrr, mid_only, prev_d_only_mid) ---"
    );
    println!("  count: Rust={} C={}", rust_v1.len(), c_v1.len());
    let n = rust_v1.len().min(c_v1.len());
    let mut first_diff_at: Option<usize> = None;
    for i in 0..n {
        let r = rust_v1[i];
        let c = c_v1[i];
        let same = r == c;
        if !same && first_diff_at.is_none() {
            first_diff_at = Some(i);
        }
        let marker = if same { " " } else { "*" };
        println!(
            "  {marker} [{i:3}] b{}/ch{:2} R: tell={:5} rng={:08x} tr={:6} nx={:5} cl={:5} nl={:5} mo={:2} pdm={:2}",
            r.boundary_id,
            r.channel,
            r.ec_tell,
            r.rng,
            r.target_rate_bps,
            r.n_bits_exceeded,
            r.curr_n_bits_used_lbrr,
            r.n_bits_used_lbrr,
            r.mid_only_flag,
            r.prev_decode_only_middle
        );
        if !same {
            println!(
                "         b{}/ch{:2} C: tell={:5} rng={:08x} tr={:6} nx={:5} cl={:5} nl={:5} mo={:2} pdm={:2}",
                c.boundary_id,
                c.channel,
                c.ec_tell,
                c.rng,
                c.target_rate_bps,
                c.n_bits_exceeded,
                c.curr_n_bits_used_lbrr,
                c.n_bits_used_lbrr,
                c.mid_only_flag,
                c.prev_decode_only_middle
            );
        }
    }
    if rust_v1.len() != c_v1.len() {
        println!(
            "  count mismatch: Rust extra {} / C extra {}",
            rust_v1.len().saturating_sub(c_v1.len()),
            c_v1.len().saturating_sub(rust_v1.len())
        );
    }
    if let Some(idx) = first_diff_at {
        let r = rust_v1[idx];
        println!(
            "\n  >>> FIRST DIVERGENCE at boundary {} (channel {}), index {} <<<",
            r.boundary_id, r.channel, idx
        );
        let label = match r.boundary_id {
            1 => "L1.4.1 — after LBRR encoding",
            2 => "L1.4.2 — after HP variable cutoff",
            3 => "L1.4.3 — after target-rate computation [B2 candidate]",
            4 => "L1.4.4 — after stereo LR->MS [B1 candidate]",
            5 => "L1.4.5 — per-channel after silk_control_SNR",
            6 => "L1.4.6 — per-channel after silk_encode_frame_Fxx [B1 candidate]",
            7 => "L1.4.7 — after bit-reservoir update [B2 candidate]",
            _ => "(unknown)",
        };
        println!("  {label}");
    } else if rust_v1.len() == c_v1.len() {
        println!("\n  >>> trace tuples match across all {n} boundaries <<<");
    }
}

// ===========================================================================
// Phase C (HLD V2) trace decode + diff
// ---------------------------------------------------------------------------
// V2 boundaries 100..=109 carry full state-vector payloads (`Vec<i32>`). The
// schema differs per boundary (HLD V2 §4.1, mirrored in
// `harness/silk_encode_frame_FIX_traced.c` boundaries 100..=109). The diff
// strategy:
//   1. Match Rust against C using `(boundary_id, iter)` as the key. Channel
//      is ignored for V2 because Rust emits `channel = -1` (sentinel) while
//      C emits the per-channel global; aligning by channel would create
//      false negatives (per HLD V2 §4 spec for stage 4).
//   2. For setup boundaries (100..=105), iter is `-1` on both sides; one
//      tuple per encode per boundary.
//   3. For per-iter boundaries (106..=109), the rate-control loop may take
//      different iter counts on each side (G8 candidate). A tuple present
//      on one side but not the other IS a divergence finding.
//   4. Pretty-print each boundary's payload as named columns; highlight the
//      first per-element divergence within the payload.
// ===========================================================================

/// Format a u16-pair (lo, hi) as the original u32 hex value. Boundaries
/// 108/109 split `psRangeEnc->rng` into `rng & 0xFFFF` and `rng >> 16`.
fn rng_pair_to_hex(lo: i32, hi: i32) -> String {
    let lo_u = (lo as u32) & 0xFFFF;
    let hi_u = (hi as u32) & 0xFFFF;
    let v = (hi_u << 16) | lo_u;
    format!("0x{:08x}", v)
}

/// Map a V2 boundary id to a coarse (mnemonic, hypothesis-letter) for
/// display. This is the boundary-level label — for boundaries whose
/// payload mixes hypothesis territories (b103 in particular), prefer
/// `v2_boundary_label_for_index` which disambiguates by the divergent
/// sub-field.
fn v2_boundary_label(boundary_id: i32) -> (&'static str, &'static str) {
    match boundary_id {
        100 => ("F1 post-LP_variable_cutoff", "G7"),
        101 => ("F3 post-find_pitch_lags", "G6"),
        102 => ("F4 post-noise_shape_analysis", "G5"),
        103 => ("F5 post-find_pred_coefs", "G4"),
        104 => ("F6 post-process_gains", "G3"),
        105 => ("F7 post-LBRR_encode", "G9"),
        106 => ("F8 pre-NSQ (per iter)", "G8"),
        107 => ("F8a post-NSQ (per iter)", "G2"),
        108 => ("F8b post-encode_indices (per iter)", "G1a"),
        109 => ("F8c post-encode_pulses (per iter)", "G1b"),
        _ => ("(unknown V2 boundary)", "?"),
    }
}

/// Recover `nb_subfr` from a b103 payload length.
///
/// b103 schema (see `silk/encoder.rs` near `boundary_id: 103` push):
///   2*MAX_LPC_ORDER + LTP_ORDER*nb + MAX_SHAPE_LPC_ORDER*nb
///     + 1 + (MAX_LPC_ORDER+1) + nb + 2
/// = 32 + 5*nb + 24*nb + 1 + 17 + nb + 2
/// = 30*nb + 52.
/// Solve `nb = (len - 52) / 30`, validate `nb in 2..=4` and `30*nb+52 == len`.
fn b103_nb_subfr(payload_len: usize) -> Option<usize> {
    if payload_len < 52 + 30 * 2 {
        return None;
    }
    let rem = payload_len.checked_sub(52)?;
    if rem % 30 != 0 {
        return None;
    }
    let nb = rem / 30;
    if (2..=4).contains(&nb) {
        Some(nb)
    } else {
        None
    }
}

/// Sub-field-aware label/hypothesis for a V2 boundary payload. Falls
/// back to the coarse `v2_boundary_label` when the divergent index
/// doesn't change the call. Currently only b103 has cross-hypothesis
/// territory (the F5 push includes ar_q13, which is computed in F4
/// noise_shape_analysis — so an ar_q13 divergence at b103 actually
/// localizes to G5, not G4). b105 sub-fields are all G9-related, so
/// no disambiguation is needed there.
fn v2_boundary_label_for_index(
    boundary_id: i32,
    first_divergent_index: usize,
    payload_len: usize,
) -> (&'static str, &'static str) {
    if boundary_id == 103 {
        const LTP_ORDER: usize = 5;
        const MAX_SHAPE_LPC_ORDER: usize = 24;
        const PRED_COEF_END: usize = 32; // 2 * MAX_LPC_ORDER

        if let Some(nb) = b103_nb_subfr(payload_len) {
            let ltp_end = PRED_COEF_END + LTP_ORDER * nb;
            let ar_end = ltp_end + MAX_SHAPE_LPC_ORDER * nb;
            let scale_end = ar_end + 1;
            let i = first_divergent_index;
            if i < PRED_COEF_END {
                return ("F5 post-find_pred_coefs (pred_coef_q12)", "G4");
            } else if i < ltp_end {
                return ("F5 post-find_pred_coefs (ltp_coef_q14)", "G4 sub-LTP");
            } else if i < ar_end {
                return (
                    "F4 noise_shape_analysis (ar_q13) — surfaces at b103 because b102 schema omits ar_q13",
                    "G5",
                );
            } else if i < scale_end {
                return ("F5 sub-LTP-scale", "G4");
            } else {
                return ("F5 sub-NLSF/LTP-index", "G4");
            }
        }
        // Fall through if length is malformed.
    }
    v2_boundary_label(boundary_id)
}

/// Format an i32 slice as `[a, b, c, ...]` truncated to `max` elements.
fn fmt_vec(p: &[i32], max: usize) -> String {
    if p.len() <= max {
        let parts: Vec<String> = p.iter().map(|v| v.to_string()).collect();
        format!("[{}]", parts.join(", "))
    } else {
        let parts: Vec<String> = p[..max].iter().map(|v| v.to_string()).collect();
        format!("[{}, ... ({} more)]", parts.join(", "), p.len() - max)
    }
}

/// Format an i32 slice as `[head..., (...), ...tail]` for very long
/// slices (e.g. `ar_q13` with up to 96 elts). When the slice fits in
/// `head + tail`, prints in full.
fn fmt_vec_head_tail(p: &[i32], head: usize, tail: usize) -> String {
    if p.len() <= head + tail {
        let parts: Vec<String> = p.iter().map(|v| v.to_string()).collect();
        return format!("[{}]", parts.join(", "));
    }
    let head_parts: Vec<String> = p[..head].iter().map(|v| v.to_string()).collect();
    let tail_parts: Vec<String> = p[p.len() - tail..].iter().map(|v| v.to_string()).collect();
    format!(
        "[{}, (... {} elided ...), {}]",
        head_parts.join(", "),
        p.len() - head - tail,
        tail_parts.join(", ")
    )
}

/// Decode a payload into a one-line named-column summary per the §4.1
/// schema. `nb_subfr` is inferred from payload length where possible
/// (boundary-specific). For unknown boundaries / malformed payloads,
/// fall back to a hex dump preview.
fn format_v2_payload(boundary_id: i32, payload: &[i32]) -> String {
    match boundary_id {
        100 => {
            // input_buf[1..=frame_length], up to 480 i32s. Just show length
            // + first 8 elements.
            format!(
                "input_buf[len={}] first8={}",
                payload.len(),
                fmt_vec(payload, 8)
            )
        }
        101 => {
            // pitchL[0..nb_subfr] ‖ [lagIndex, contourIndex] = (nb_subfr + 2) elts.
            // For NB nb_subfr=4 → 6 elts; for SWB nb_subfr=2 → 4 elts.
            if payload.len() < 2 {
                return format!("(malformed: len={})", payload.len());
            }
            let n = payload.len() - 2;
            let pitch_l = &payload[..n];
            let lag_idx = payload[n];
            let contour = payload[n + 1];
            format!(
                "pitch_l={} lag_index={} contour_index={}",
                fmt_vec(pitch_l, 8),
                lag_idx,
                contour
            )
        }
        102 => {
            // 4*nb_subfr + 2 elts. Recover nb_subfr.
            if payload.len() < 2 || !(payload.len() - 2).is_multiple_of(4) {
                return format!("(malformed: len={})", payload.len());
            }
            let n = (payload.len() - 2) / 4;
            let harm = &payload[0..n];
            let tilt = &payload[n..2 * n];
            let lf_shp = &payload[2 * n..3 * n];
            let gains_unq = &payload[3 * n..4 * n];
            let harm_boost = payload[4 * n];
            let tilt_smth = payload[4 * n + 1];
            format!(
                "harm_shape_gain_q14={} tilt_q14={} lf_shp_q14={} gains_unq_q16={} harm_boost_smth={} tilt_smth={}",
                fmt_vec(harm, 8),
                fmt_vec(tilt, 8),
                fmt_vec(lf_shp, 8),
                fmt_vec(gains_unq, 8),
                harm_boost,
                tilt_smth,
            )
        }
        103 => {
            // 2*MAX_LPC_ORDER + LTP_ORDER*nb_subfr + MAX_SHAPE_LPC_ORDER*nb_subfr
            //  + 1 + (MAX_LPC_ORDER+1) + nb_subfr + 2 = 30*nb_subfr + 52.
            const MAX_LPC_ORDER: usize = 16;
            const LTP_ORDER: usize = 5;
            const MAX_SHAPE_LPC_ORDER: usize = 24;
            let Some(nb) = b103_nb_subfr(payload.len()) else {
                return format!(
                    "(malformed b103: len={} not 30*nb+52 for nb in 2..=4)",
                    payload.len()
                );
            };
            let p0_end = MAX_LPC_ORDER;
            let p1_end = 2 * MAX_LPC_ORDER;
            let ltp_end = p1_end + LTP_ORDER * nb;
            let ar_end = ltp_end + MAX_SHAPE_LPC_ORDER * nb;
            let scale_end = ar_end + 1;
            let nlsf_end = scale_end + (MAX_LPC_ORDER + 1);
            let ltpidx_end = nlsf_end + nb;
            let pred0 = &payload[0..p0_end];
            let pred1 = &payload[p0_end..p1_end];
            let ltp_coef = &payload[p1_end..ltp_end];
            let ar_q13 = &payload[ltp_end..ar_end];
            let ltp_scale_q14 = payload[ar_end];
            let nlsf_indices = &payload[scale_end..nlsf_end];
            let ltp_index = &payload[nlsf_end..ltpidx_end];
            let ltp_scale_index = payload[ltpidx_end];
            let nlsf_interp = payload[ltpidx_end + 1];
            format!(
                "pred_coef_q12[0]={} pred_coef_q12[1]={} ltp_coef_q14={} ar_q13={} ltp_scale_q14={} nlsf_indices={} ltp_index={} ltp_scale_index={} nlsf_interp_coef_q2={}",
                fmt_vec(pred0, MAX_LPC_ORDER),
                fmt_vec(pred1, MAX_LPC_ORDER),
                fmt_vec(ltp_coef, LTP_ORDER * nb),
                fmt_vec_head_tail(ar_q13, 8, 4),
                ltp_scale_q14,
                fmt_vec(nlsf_indices, MAX_LPC_ORDER + 1),
                fmt_vec(ltp_index, nb),
                ltp_scale_index,
                nlsf_interp,
            )
        }
        104 => {
            // Gains_Q16[0..nb_subfr] ‖ [Lambda_Q10, lastGainIndexPrev]
            //  ‖ GainsIndices[0..nb_subfr]. Length = 2*nb_subfr + 2.
            if payload.len() < 2 || !(payload.len() - 2).is_multiple_of(2) {
                return format!("(malformed: len={})", payload.len());
            }
            let n = (payload.len() - 2) / 2;
            let gains_q16 = &payload[0..n];
            let lambda_q10 = payload[n];
            let last_gain_index_prev = payload[n + 1];
            let gains_indices = &payload[n + 2..2 * n + 2];
            format!(
                "gains_q16={} lambda_q10={} last_gain_index_prev={} gains_indices={}",
                fmt_vec(gains_q16, 8),
                lambda_q10,
                last_gain_index_prev,
                fmt_vec(gains_indices, 8),
            )
        }
        105 => {
            // Gains_Q16[0..nb_subfr] ‖ [LBRR_flag] ‖ GainsIndices_LBRR[0..nb_subfr]
            //  ‖ [signalType, quantOffsetType]. Length = 2*nb_subfr + 3.
            //
            // No sub-field hypothesis disambiguation here: every field on
            // b105 lives in G9 (LBRR_encode_FIX) territory, so the coarse
            // `v2_boundary_label(105) = ("F7 post-LBRR_encode", "G9")`
            // already attributes correctly. b103 is the only V2 boundary
            // that mixes hypothesis territories — see
            // `v2_boundary_label_for_index`.
            if payload.len() < 3 || !(payload.len() - 3).is_multiple_of(2) {
                return format!("(malformed: len={})", payload.len());
            }
            let n = (payload.len() - 3) / 2;
            let gains_q16 = &payload[0..n];
            let lbrr_flag = payload[n];
            let gains_idx_lbrr = &payload[n + 1..2 * n + 1];
            let signal_type = payload[2 * n + 1];
            let quant_offset_type = payload[2 * n + 2];
            format!(
                "gains_q16={} LBRR_flag={} indices_LBRR.gains_indices={} signalType={} quantOffsetType={}",
                fmt_vec(gains_q16, 8),
                lbrr_flag,
                fmt_vec(gains_idx_lbrr, 8),
                signal_type,
                quant_offset_type,
            )
        }
        106 => {
            // [iter, gainsID, gainsID_lower, gainsID_upper, found_lower, found_upper, gainMult_Q8].
            if payload.len() != 7 {
                return format!("(malformed: len={}, expected 7)", payload.len());
            }
            format!(
                "iter={} gainsID={} gainsID_lower={} gainsID_upper={} found_lower={} found_upper={} gainMult_Q8={}",
                payload[0], payload[1], payload[2], payload[3], payload[4], payload[5], payload[6]
            )
        }
        107 => {
            // [iter] ‖ pulses[0..frame_length] ‖ [signalType, quantOffsetType, Seed, lagPrev].
            if payload.len() < 5 {
                return format!("(malformed: len={})", payload.len());
            }
            let n = payload.len() - 5;
            let iter = payload[0];
            let pulses = &payload[1..1 + n];
            let signal_type = payload[1 + n];
            let quant_offset_type = payload[1 + n + 1];
            let seed = payload[1 + n + 2];
            let lag_prev = payload[1 + n + 3];
            format!(
                "iter={} pulses[len={}] first16={} signalType={} quantOffsetType={} Seed={} lagPrev={}",
                iter,
                pulses.len(),
                fmt_vec(pulses, 16),
                signal_type,
                quant_offset_type,
                seed,
                lag_prev,
            )
        }
        108 | 109 => {
            // [iter, ec_tell, rng_lo, rng_hi].
            if payload.len() != 4 {
                return format!("(malformed: len={}, expected 4)", payload.len());
            }
            let rng_hex = rng_pair_to_hex(payload[2], payload[3]);
            format!("iter={} ec_tell={} rng={}", payload[0], payload[1], rng_hex)
        }
        _ => format!(
            "(unknown boundary) payload[len={}] {}",
            payload.len(),
            fmt_vec(payload, 8)
        ),
    }
}

/// Find the first divergent index between two payloads. Returns
/// `min(R.len, C.len)` when one is a strict prefix of the other.
fn first_divergent_index(rust_payload: &[i32], c_payload: &[i32]) -> usize {
    let n_min = rust_payload.len().min(c_payload.len());
    (0..n_min)
        .find(|&i| rust_payload[i] != c_payload[i])
        .unwrap_or(n_min)
}

/// Print a per-element divergence highlight when a payload differs.
/// Shows up to `WINDOW` elements before and after the first divergent
/// index, formatted side-by-side.
fn print_payload_divergence(boundary_id: i32, rust_payload: &[i32], c_payload: &[i32]) {
    const WINDOW: usize = 4;
    if rust_payload == c_payload {
        return;
    }
    let first_div = first_divergent_index(rust_payload, c_payload);

    let lo = first_div.saturating_sub(WINDOW);
    let hi_r = (first_div + WINDOW + 1).min(rust_payload.len());
    let hi_c = (first_div + WINDOW + 1).min(c_payload.len());

    println!("        first divergent element at index {}:", first_div);
    println!(
        "          R[{}..{}] = {:?}",
        lo,
        hi_r,
        &rust_payload[lo..hi_r]
    );
    println!("          C[{}..{}] = {:?}", lo, hi_c, &c_payload[lo..hi_c]);
    if rust_payload.len() != c_payload.len() {
        println!(
            "          (length mismatch: R.len={} C.len={})",
            rust_payload.len(),
            c_payload.len()
        );
    }
    // Annotate the bug class hint per HLD V2 §3, using the
    // sub-field-aware mapping where applicable (b103 mixes G4 and G5
    // territory because the F5 push includes ar_q13 from F4).
    let payload_len = rust_payload.len().max(c_payload.len());
    let (label, hyp) = v2_boundary_label_for_index(boundary_id, first_div, payload_len);
    println!(
        "          → b{} {} divergence — hypothesis {}",
        boundary_id, label, hyp
    );
}

/// Match key for V2 tuples. Channel is intentionally omitted (Rust emits
/// `-1` sentinel; C emits the per-channel global). Boundaries 100..=105
/// fire once per encode (`iter = -1`); boundaries 106..=109 fire per
/// rate-control iter (`iter` 0..maxIter).
type V2Key = (i32, i32); // (boundary_id, iter)

fn v2_key(t: &ropus::silk_trace::Tuple) -> V2Key {
    (t.boundary_id, t.iter)
}

/// Print the Phase C (HLD V2) trace diff for boundaries 100..=109.
/// Returns the `(boundary_id, iter)` of the first divergent tuple, if any,
/// and a free-form descriptor — used by the caller to print a top-level
/// "FIRST DIVERGENT V2 BOUNDARY" call-out.
fn print_phase_c_trace_diff(
    rust_trace: &[ropus::silk_trace::Tuple],
    c_trace: &[ropus::silk_trace::Tuple],
) {
    let rust_v2: Vec<&ropus::silk_trace::Tuple> = rust_trace
        .iter()
        .filter(|t| t.boundary_id >= 100 && t.boundary_id <= 109)
        .collect();
    let c_v2: Vec<&ropus::silk_trace::Tuple> = c_trace
        .iter()
        .filter(|t| t.boundary_id >= 100 && t.boundary_id <= 109)
        .collect();

    println!("\n--- Phase C (HLD V2) inner-function trace tuples (boundaries 100..=109) ---");
    println!("  count: Rust={} C={}", rust_v2.len(), c_v2.len());

    if rust_v2.is_empty() && c_v2.is_empty() {
        println!("  (no V2 tuples emitted — non-SILK or pre-Stage-3 build)");
        return;
    }

    // Build keyed maps. V2 setup boundaries (100..=105) are unique by
    // boundary_id alone (iter=-1). V2 per-iter boundaries (106..=109) may
    // appear multiple times per encode (one per rate-control iter).
    use std::collections::BTreeMap;
    let rust_map: BTreeMap<V2Key, &ropus::silk_trace::Tuple> =
        rust_v2.iter().map(|t| (v2_key(t), *t)).collect();
    let c_map: BTreeMap<V2Key, &ropus::silk_trace::Tuple> =
        c_v2.iter().map(|t| (v2_key(t), *t)).collect();

    // Union of keys, sorted: setup boundaries first (100..=105 with iter=-1),
    // then per-iter (106..=109 with iter 0, 1, 2 ...). BTreeMap natural
    // ordering of (boundary_id, iter) gives close-enough output: all
    // boundary 100 tuples (one), then 101 (one), ..., then 106 iter=0,
    // 106 iter=1, etc — but since iters are interleaved across boundaries
    // within the rate-control loop, we re-sort by (iter, boundary_id) for
    // the per-iter group so iter 0 of 106/107/108/109 appears together.
    let mut all_keys: Vec<V2Key> = rust_map
        .keys()
        .chain(c_map.keys())
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    all_keys.sort_by(|a, b| {
        let (ba, ia) = *a;
        let (bb, ib) = *b;
        // Setup boundaries first (sorted by id), then per-iter (sorted by
        // iter, then id).
        let setup_a = ba < 106;
        let setup_b = bb < 106;
        match (setup_a, setup_b) {
            (true, true) => ba.cmp(&bb),
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            (false, false) => ia.cmp(&ib).then(ba.cmp(&bb)),
        }
    });

    // (boundary, channel, iter, first_divergent_index, payload_len, kind).
    // `first_divergent_index` and `payload_len` are used to disambiguate
    // hypothesis territory at boundaries whose payload spans hypotheses
    // (b103: G4 pred_coef vs G5 ar_q13). For Rust-only / C-only cases
    // we record the index as 0 and the available payload length.
    let mut first_divergent: Option<(i32, i32, i32, usize, usize, String)> = None;

    // Rate-control loop iter-count summary (per HLD V2 §3 G8 hypothesis).
    let mut rust_iter_max: i32 = -1;
    let mut c_iter_max: i32 = -1;
    for t in &rust_v2 {
        if t.boundary_id >= 106 && t.boundary_id <= 109 {
            rust_iter_max = rust_iter_max.max(t.iter);
        }
    }
    for t in &c_v2 {
        if t.boundary_id >= 106 && t.boundary_id <= 109 {
            c_iter_max = c_iter_max.max(t.iter);
        }
    }
    let rust_iter_count = (rust_iter_max + 1).max(0);
    let c_iter_count = (c_iter_max + 1).max(0);
    if rust_iter_count != c_iter_count {
        println!(
            "  rate-control iter counts differ: Rust={} C={} (G8 candidate)",
            rust_iter_count, c_iter_count
        );
    } else if rust_iter_count > 0 {
        println!("  rate-control iter count: {} (matches)", rust_iter_count);
    }

    for key in all_keys {
        let (bid, iter) = key;
        let r = rust_map.get(&key);
        let c = c_map.get(&key);
        let (label, hyp) = v2_boundary_label(bid);
        let iter_label = if iter < 0 {
            "setup".to_string()
        } else {
            format!("iter={}", iter)
        };
        match (r, c) {
            (Some(r), Some(c)) => {
                let same = r.payload == c.payload;
                let marker = if same { " " } else { "*" };
                println!(
                    "  {marker} b{:3} ({})  [{:>5}]  ({})",
                    bid, label, iter_label, hyp
                );
                println!("        R: {}", format_v2_payload(bid, &r.payload));
                if !same {
                    println!("        C: {}", format_v2_payload(bid, &c.payload));
                    print_payload_divergence(bid, &r.payload, &c.payload);
                    if first_divergent.is_none() {
                        let idx = first_divergent_index(&r.payload, &c.payload);
                        let plen = r.payload.len().max(c.payload.len());
                        first_divergent = Some((
                            bid,
                            r.channel,
                            iter,
                            idx,
                            plen,
                            "payload-mismatch".to_string(),
                        ));
                    }
                }
            }
            (Some(r), None) => {
                println!(
                    "  * b{:3} ({})  [{:>5}]  ({})  Rust-only (C did not emit)",
                    bid, label, iter_label, hyp
                );
                println!("        R: {}", format_v2_payload(bid, &r.payload));
                if first_divergent.is_none() {
                    first_divergent = Some((
                        bid,
                        r.channel,
                        iter,
                        0,
                        r.payload.len(),
                        "Rust emitted but C did not (rate-control iter-count divergence?)"
                            .to_string(),
                    ));
                }
            }
            (None, Some(c)) => {
                println!(
                    "  * b{:3} ({})  [{:>5}]  ({})  C-only (Rust did not emit)",
                    bid, label, iter_label, hyp
                );
                println!("        C: {}", format_v2_payload(bid, &c.payload));
                if first_divergent.is_none() {
                    first_divergent = Some((
                        bid,
                        c.channel,
                        iter,
                        0,
                        c.payload.len(),
                        "C emitted but Rust did not (rate-control iter-count divergence?)"
                            .to_string(),
                    ));
                }
            }
            (None, None) => unreachable!(),
        }
    }

    if let Some((bid, channel, iter, first_idx, payload_len, kind)) = first_divergent {
        let (label, hyp) = v2_boundary_label_for_index(bid, first_idx, payload_len);
        let iter_label = if iter < 0 {
            "setup".to_string()
        } else {
            format!("{}", iter)
        };
        println!(
            "\n  >>> FIRST DIVERGENT V2 BOUNDARY: b{} (channel={}, iter={}) → hypothesis {} ({}) <<<",
            bid, channel, iter_label, hyp, label
        );
        println!("      kind: {}", kind);
        if bid == 106 && rust_iter_count != c_iter_count {
            println!(
                "      G8 SIGNAL: rate-control loop iter counts differ: Rust={}, C={}",
                rust_iter_count, c_iter_count
            );
        }
    } else {
        println!("\n  >>> V2 trace tuples match across all boundaries <<<");
    }
}

fn run_multistream_repro(path: &PathBuf) {
    println!("\n========================================================");
    println!("REPRO multistream: {}", path.display());
    println!("========================================================");
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let input = match parse_multistream(&data) {
        Ok(i) => i,
        Err(e) => {
            println!("PARSE ERROR: {e}");
            return;
        }
    };
    println!(
        "Config: op={} sr={} ch={} family={} app={} br={} cx={} vbr={}",
        input.op,
        input.sample_rate,
        input.channels,
        input.mapping_family,
        input.application,
        input.bitrate,
        input.complexity,
        input.vbr
    );

    if input.op % 3 != 0 {
        println!("(repro op != encode; skipping — encoder-state diagnostic only)");
        return;
    }
    let frame_size = input.sample_rate / 50;
    let samples_needed = (frame_size as usize) * (input.channels as usize);
    let bytes_needed = samples_needed * 2;
    if input.payload.len() < bytes_needed {
        println!(
            "payload too short ({} < {})",
            input.payload.len(),
            bytes_needed
        );
        return;
    }
    let pcm: Vec<i16> = input.payload[..bytes_needed]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    // Rust MS encoder
    let (mut rust_enc, streams, coupled_streams, mapping) = match RustOpusMSEncoder::new_surround(
        input.sample_rate,
        input.channels,
        input.mapping_family,
        input.application,
    ) {
        Ok(t) => t,
        Err(e) => {
            println!("Rust new_surround failed: {e}");
            return;
        }
    };
    if streams + coupled_streams > 16 {
        println!("streams+coupled exceeds cap (Rust); skipping");
        return;
    }
    println!(
        "Mapping: streams={} coupled={} mapping={:?}",
        streams, coupled_streams, mapping
    );

    // C MS encoder via surround create — match the Rust path used in
    // the fuzz target.
    let (c_enc, _c_streams, _c_coupled) = unsafe {
        let mut err: c_int = 0;
        let mut c_streams: c_int = 0;
        let mut c_coupled: c_int = 0;
        let mut c_mapping = vec![0u8; input.channels.max(1) as usize];
        // Forward declaration: link the ambisonics-aware surround create.
        unsafe extern "C" {
            fn opus_multistream_surround_encoder_create(
                fs: i32,
                channels: c_int,
                mapping_family: c_int,
                streams: *mut c_int,
                coupled_streams: *mut c_int,
                mapping: *mut u8,
                application: c_int,
                error: *mut c_int,
            ) -> *mut bindings::OpusMSEncoder;
        }
        let enc = opus_multistream_surround_encoder_create(
            input.sample_rate,
            input.channels,
            input.mapping_family,
            &mut c_streams,
            &mut c_coupled,
            c_mapping.as_mut_ptr(),
            input.application,
            &mut err,
        );
        if enc.is_null() || err != bindings::OPUS_OK {
            println!("C surround encoder create failed: err={err}");
            return;
        }
        (enc, c_streams, c_coupled)
    };

    // Apply baseline + setter shuffle and dump pre-/post-state at the
    // per-call granularity. Mirrors fuzz_multistream.rs:693-700 (baseline)
    // and lines 157-187 (shuffle).
    type BaselineSetter = (&'static str, fn(&mut RustOpusMSEncoder, i32) -> i32, i32);
    let baseline: [BaselineSetter; 6] = [
        ("set_bitrate", RustOpusMSEncoder::set_bitrate, input.bitrate),
        ("set_vbr", RustOpusMSEncoder::set_vbr, input.vbr),
        ("set_inband_fec", RustOpusMSEncoder::set_inband_fec, 0),
        ("set_dtx", RustOpusMSEncoder::set_dtx, 0),
        (
            "set_packet_loss_perc",
            RustOpusMSEncoder::set_packet_loss_perc,
            0,
        ),
        (
            "set_complexity",
            RustOpusMSEncoder::set_complexity,
            input.complexity,
        ),
    ];

    let inner_c = unsafe { bindings::debug_get_inner_opus_encoder(c_enc, 0) };
    if inner_c.is_null() {
        println!("debug_get_inner_opus_encoder returned NULL — skipping");
        return;
    }

    println!("\n--- Baseline setters ---");
    for (name, f, val) in baseline {
        let _ = f(&mut rust_enc, val);
        // Drive C through the matching CTL on the MS encoder.
        let req = match name {
            "set_bitrate" => bindings::OPUS_SET_BITRATE_REQUEST,
            "set_vbr" => bindings::OPUS_SET_VBR_REQUEST,
            "set_inband_fec" => bindings::OPUS_SET_INBAND_FEC_REQUEST,
            "set_dtx" => bindings::OPUS_SET_DTX_REQUEST,
            "set_packet_loss_perc" => bindings::OPUS_SET_PACKET_LOSS_PERC_REQUEST,
            "set_complexity" => bindings::OPUS_SET_COMPLEXITY_REQUEST,
            _ => unreachable!(),
        };
        unsafe {
            unsafe extern "C" {
                fn opus_multistream_encoder_ctl(
                    st: *mut bindings::OpusMSEncoder,
                    request: c_int,
                    ...
                ) -> c_int;
            }
            opus_multistream_encoder_ctl(c_enc, req, val);
        }
        let inner_r = rust_enc
            .get_encoder(0)
            .expect("MS encoder must have stream 0");
        let r_top = dump_rust_top(inner_r);
        let (c_sm, c_top) = dump_c_top(inner_c);
        println!(
            "  baseline {name}({val})  Rust top: use_vbr={} use_dtx={} fec_config={} bitrate_bps={}",
            r_top.use_vbr, r_top.use_dtx, r_top.fec_config, r_top.bitrate_bps
        );
        println!(
            "                              C top: use_vbr={} use_dtx={} fec_config={} bitrate_bps={}  silk_mode useInBandFEC={} useCBR={} useDTX={}",
            c_top.use_vbr,
            c_top.use_dtx,
            c_top.fec_config,
            c_top.bitrate_bps,
            c_sm.use_in_band_fec,
            c_sm.use_cbr,
            c_sm.use_dtx
        );
    }

    // Setter shuffle. We mirror apply_ms_encoder_setter_sequence /
    // apply_c_ms_setter_sequence symbol-for-symbol. Stage 6a aligned both
    // sides to the same complexity modulus (see the `1 => { ... }` arm).
    println!(
        "\n--- Setter shuffle ({} chunks) ---",
        input.setter_bytes.len() / 2
    );
    for (i, chunk) in input.setter_bytes.chunks_exact(2).take(8).enumerate() {
        let sel = chunk[0] % 7;
        let arg_byte = chunk[1];
        let (_rust_label, c_req, val): (&str, c_int, i32) = match sel {
            0 => {
                let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                let rate = 6000 + (raw as i32 % 504_001);
                let _ = rust_enc.set_bitrate(rate);
                ("set_bitrate", bindings::OPUS_SET_BITRATE_REQUEST, rate)
            }
            1 => {
                // Stage 6a fix: align Rust and C to the same modulus (10),
                // matching the Rust-side `% 10` cap at
                // `tests/fuzz/fuzz_targets/fuzz_multistream.rs:165` (which
                // intentionally dodges the analysis.c divergence class for
                // complexity=10, see lines 121-126 there). The previous
                // mismatch (Rust % 10, C % 11) systematically delivered
                // different complexities whenever arg_byte >= 10,
                // producing the `multistream-encode-bytes-divergence`
                // crash cluster — those are harness artefacts, not codec
                // bugs. The matching fix to the fuzz-target C wrapper
                // lives in `tests/fuzz/fuzz_targets/c_reference.rs:756`.
                let cx = (arg_byte % 10) as i32;
                let _ = rust_enc.set_complexity(cx);
                ("set_complexity", bindings::OPUS_SET_COMPLEXITY_REQUEST, cx)
            }
            2 => {
                let v = (arg_byte & 1) as i32;
                let _ = rust_enc.set_vbr(v);
                ("set_vbr", bindings::OPUS_SET_VBR_REQUEST, v)
            }
            3 => {
                let v = (arg_byte % 3) as i32;
                let _ = rust_enc.set_inband_fec(v);
                ("set_inband_fec", bindings::OPUS_SET_INBAND_FEC_REQUEST, v)
            }
            4 => {
                let v = (arg_byte & 1) as i32;
                let _ = rust_enc.set_dtx(v);
                ("set_dtx", bindings::OPUS_SET_DTX_REQUEST, v)
            }
            5 => {
                let v = (arg_byte % 101) as i32;
                let _ = rust_enc.set_packet_loss_perc(v);
                (
                    "set_packet_loss_perc",
                    bindings::OPUS_SET_PACKET_LOSS_PERC_REQUEST,
                    v,
                )
            }
            6 => {
                let v = (arg_byte & 1) as i32;
                let _ = rust_enc.set_lfe(v);
                ("set_lfe", 4030 as c_int, v)
            }
            _ => unreachable!(),
        };
        unsafe {
            unsafe extern "C" {
                fn opus_multistream_encoder_ctl(
                    st: *mut bindings::OpusMSEncoder,
                    request: c_int,
                    ...
                ) -> c_int;
            }
            opus_multistream_encoder_ctl(c_enc, c_req, val);
        }
        let inner_r = rust_enc
            .get_encoder(0)
            .expect("MS encoder must have stream 0");
        let r_top = dump_rust_top(inner_r);
        let (c_sm, c_top) = dump_c_top(inner_c);
        println!(
            "  shuffle[{i}] sel={sel} val={val:>6}  Rust top: use_vbr={} use_dtx={} fec_config={} bitrate_bps={}",
            r_top.use_vbr, r_top.use_dtx, r_top.fec_config, r_top.bitrate_bps
        );
        println!(
            "                              C top: use_vbr={} use_dtx={} fec_config={} bitrate_bps={}  silk_mode useInBandFEC={} useCBR={} useDTX={}",
            c_top.use_vbr,
            c_top.use_dtx,
            c_top.fec_config,
            c_top.bitrate_bps,
            c_sm.use_in_band_fec,
            c_sm.use_cbr,
            c_sm.use_dtx
        );
    }

    // Pre-encode SILK extended-state diff: are the encoders in matching
    // initial states immediately before the first encode? If anything
    // differs here it's an init-time bug, not encode-time.
    {
        let inner_r = rust_enc
            .get_encoder(0)
            .expect("MS encoder must have stream 0");
        let pre_diffs = compare_extended_state(inner_c, inner_r, 0);
        if !pre_diffs.is_empty() {
            println!(
                "  pre-encode SILK extended-state mismatches ({}):",
                pre_diffs.len()
            );
            for d in pre_diffs.iter().take(20) {
                println!("    {d}");
            }
            if pre_diffs.len() > 20 {
                println!("    ... ({} more)", pre_diffs.len() - 20);
            }
        } else {
            println!("  pre-encode SILK extended-state OK");
        }
        let pre_if = diff_silk_interframe(
            &dump_c_interframe(inner_c, 0),
            &dump_rust_interframe(inner_r, 0),
        );
        if !pre_if.is_empty() {
            println!(
                "  pre-encode SILK interframe-state mismatches ({}):",
                pre_if.len()
            );
            for d in &pre_if {
                println!("    {d}");
            }
        } else {
            println!("  pre-encode SILK interframe-state OK");
        }
    }

    // Encode one frame on each side. Clear and capture Phase B trace
    // tuples around each encode so we can diff the per-boundary stream.
    let max_data_bytes = 4000 * streams.max(1);
    let mut rust_out = vec![0u8; max_data_bytes as usize];
    ropus::silk_trace::clear();
    let rust_ret = rust_enc.encode(&pcm, frame_size, &mut rust_out, max_data_bytes);
    let rust_trace = ropus::silk_trace::snapshot();

    let mut c_out = vec![0u8; max_data_bytes as usize];
    unsafe { bindings::dbg_silk_trace_clear() };
    let c_len = unsafe {
        unsafe extern "C" {
            fn opus_multistream_encode(
                st: *mut bindings::OpusMSEncoder,
                pcm: *const bindings::opus_int16,
                frame_size: c_int,
                data: *mut c_uchar,
                max_data_bytes: i32,
            ) -> i32;
        }
        opus_multistream_encode(
            c_enc,
            pcm.as_ptr() as *const bindings::opus_int16,
            frame_size,
            c_out.as_mut_ptr() as *mut c_uchar,
            max_data_bytes,
        )
    };
    let c_trace = read_c_silk_trace();

    println!("\n--- Encode result ---");
    let rust_len = match rust_ret {
        Ok(n) => n as usize,
        Err(e) => {
            println!("Rust encode failed: {e}");
            unsafe {
                unsafe extern "C" {
                    fn opus_multistream_encoder_destroy(st: *mut bindings::OpusMSEncoder);
                }
                opus_multistream_encoder_destroy(c_enc);
            }
            return;
        }
    };
    rust_out.truncate(rust_len);
    if c_len < 0 {
        println!("C encode failed: {c_len}");
        unsafe {
            unsafe extern "C" {
                fn opus_multistream_encoder_destroy(st: *mut bindings::OpusMSEncoder);
            }
            opus_multistream_encoder_destroy(c_enc);
        }
        return;
    }
    c_out.truncate(c_len as usize);
    let bytes_match = rust_out == c_out;
    let first_diff = if bytes_match {
        None
    } else {
        (0..rust_out.len().min(c_out.len())).find(|&i| rust_out[i] != c_out[i])
    };
    println!(
        "  Rust={} bytes  C={} bytes  match={}  first-diff-at={:?}",
        rust_out.len(),
        c_out.len(),
        bytes_match,
        first_diff
    );

    print_phase_b_trace_diff(&rust_trace, &c_trace);
    print_phase_c_trace_diff(&rust_trace, &c_trace);

    // Family=0/ch=1 = single-stream wrapper. Inspect that sub-encoder's
    // state on the Rust side; on the C side we don't have a stable accessor
    // for the inner OpusEncoder pointer through the MS wrapper, so the
    // most useful diagnostic is the post-encode mirror state on the Rust
    // side combined with the top-level state we recorded during the
    // setter shuffle above.
    if input.channels == 1 && input.mapping_family == 0 {
        let inner = rust_enc.get_encoder(0).unwrap();
        let r_mirror = dump_rust_silk_mirror(inner, 0);
        let r_top = dump_rust_top(inner);
        let (c_sm_post, c_top_post) = dump_c_top(inner_c);
        // Pre-encode interframe state diff: a sanity check that initial
        // state lined up. (Frame 0 in multiframe shows C=0/R=10/65536/100
        // for last_gain_index/prev_gain_q16/prev_lag — that's an
        // init-time *cosmetic* difference because C's silk_init_encoder
        // does a memset(0) then control_codec sets the non-zero defaults
        // at first SILK encode, while Rust writes them at init. Worth
        // recording.)
        println!(
            "  Rust post-encode top:    use_vbr={} use_dtx={} fec_config={} bitrate_bps={}",
            r_top.use_vbr, r_top.use_dtx, r_top.fec_config, r_top.bitrate_bps
        );
        println!(
            "  C    post-encode top:    use_vbr={} use_dtx={} fec_config={} bitrate_bps={}  silk_mode useInBandFEC={} useCBR={} useDTX={} LBRR_coded={} bitRate={}",
            c_top_post.use_vbr,
            c_top_post.use_dtx,
            c_top_post.fec_config,
            c_top_post.bitrate_bps,
            c_sm_post.use_in_band_fec,
            c_sm_post.use_cbr,
            c_sm_post.use_dtx,
            c_sm_post.lbrr_coded,
            c_sm_post.bit_rate
        );
        // Diff per-frame interframe state on the inner C encoder vs Rust.
        let c_if = dump_c_interframe(inner_c, 0);
        let r_if = dump_rust_interframe(inner, 0);
        let if_diffs = diff_silk_interframe(&c_if, &r_if);
        if !if_diffs.is_empty() {
            println!(
                "  post-encode SILK interframe-state mismatches ({}):",
                if_diffs.len()
            );
            for d in &if_diffs {
                println!("    {d}");
            }
        } else {
            println!("  post-encode SILK interframe-state OK");
        }

        // Also diff the extended SILK state (NSQ, VAD, LP, x_buf hash).
        let ext_diffs = compare_extended_state(inner_c, inner, 0);
        if !ext_diffs.is_empty() {
            println!(
                "  post-encode SILK extended-state mismatches ({}):",
                ext_diffs.len()
            );
            for d in &ext_diffs {
                println!("    {d}");
            }
        } else {
            println!("  post-encode SILK extended-state OK");
        }
        if let Some(m) = r_mirror {
            println!(
                "  Rust post-encode SILK mirror (state_fxx[0].s_cmn): use_in_band_fec={} use_cbr={} use_dtx={} lbrr_enabled={}",
                m.use_in_band_fec, m.use_cbr, m.use_dtx, m.lbrr_enabled
            );
            // The H1/H2 smoking gun: top-level fec_config / use_vbr say one
            // thing, the SILK mirror (which was populated by silk_mode.* via
            // silk_control_encoder during encode_native) says another. If
            // they disagree, an `ms_set_*` setter wrote the top-level field
            // but failed to propagate into silk_mode, and the encoder ran
            // with the stale silk_mode value.
            if r_top.fec_config != 0 && m.use_in_band_fec == 0 {
                println!(
                    "  *** H1 SIGNAL: top-level fec_config={} but s_cmn.use_in_band_fec={} ***",
                    r_top.fec_config, m.use_in_band_fec
                );
            }
            if r_top.use_vbr == 0 && m.use_cbr == 0 {
                println!(
                    "  *** H2 SIGNAL: top-level use_vbr=0 (CBR requested) but s_cmn.use_cbr=0 ***"
                );
            }
            if r_top.use_vbr == 1 && m.use_cbr == 1 {
                println!(
                    "  *** H2 SIGNAL: top-level use_vbr=1 (VBR requested) but s_cmn.use_cbr=1 ***"
                );
            }
        }
    }

    unsafe {
        unsafe extern "C" {
            fn opus_multistream_encoder_destroy(st: *mut bindings::OpusMSEncoder);
        }
        opus_multistream_encoder_destroy(c_enc);
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

fn print_top_state_diff(label: &str, sm: &SilkModeState, top: &OpusTopState) {
    println!(
        "  {label}: silk_mode useInBandFEC={} useCBR={} useDTX={} LBRR_coded={} complexity={} packet_loss_perc={} bitRate={}",
        sm.use_in_band_fec,
        sm.use_cbr,
        sm.use_dtx,
        sm.lbrr_coded,
        sm.complexity,
        sm.packet_loss_percentage,
        sm.bit_rate
    );
    println!(
        "    top: use_vbr={} use_dtx={} fec_config={} user_bitrate={} bitrate_bps={} app={}",
        top.use_vbr,
        top.use_dtx,
        top.fec_config,
        top.user_bitrate_bps,
        top.bitrate_bps,
        top.application,
    );
}

fn print_rust_top_state(top: &OpusTopState) {
    println!(
        "  Rust top (post-setter, pre-encode): use_vbr={} use_dtx={} fec_config={} bitrate_bps={} app={}",
        top.use_vbr, top.use_dtx, top.fec_config, top.bitrate_bps, top.application
    );
}

fn assert_eq_field<T: PartialEq + std::fmt::Debug>(
    name: &str,
    c: T,
    r: T,
    findings: &mut Vec<String>,
    frame_idx: usize,
) {
    if c != r {
        findings.push(format!("frame {frame_idx} {name}: C={c:?} R={r:?}"));
        println!("    *** DIFF {name}: C={c:?} R={r:?}");
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
struct ExtendedState {
    nsq_rand_seed: i32,
    nsq_slf_ar_shp_q14: i32,
    nsq_lag_prev: i32,
    nsq_sdiff_shp_q14: i32,
    nsq_sltp_buf_idx: i32,
    nsq_sltp_shp_buf_idx: i32,
    nsq_rewhite_flag: i32,
    nsq_slpc_q14: [i32; 16],
    nsq_sar2_q14: [i32; 16],
    nsq_sltp_shp_q14: [i32; 32],
    vad_hp_state: i32,
    vad_counter: i32,
    vad_noise_level_bias: [i32; 4],
    vad_ana_state: [i32; 2],
    vad_ana_state1: [i32; 2],
    vad_ana_state2: [i32; 2],
    vad_nrg_ratio_smth_q8: [i32; 4],
    vad_nl: [i32; 4],
    vad_inv_nl: [i32; 4],
    sum_log_gain_q7: i32,
    in_hp_state: [i32; 2],
    input_tilt_q15: i32,
    input_quality_bands_q15: [i32; 4],
    frame_counter: i32,
    no_speech_counter: i32,
    lp_in_lp_state: [i32; 2],
    lp_transition_frame_no: i32,
    shape_harm_boost_smth: i32,
    x_buf_hash: i32,
    ltp_corr_q15: i32,
    res_nrg_smth: i32,
}

fn dump_c_extended(enc: *mut bindings::OpusEncoder, channel: i32) -> ExtendedState {
    let mut s = ExtendedState::default();
    unsafe {
        bindings::debug_get_silk_extended_state(
            enc,
            channel as c_int,
            &mut s.nsq_rand_seed,
            &mut s.nsq_slf_ar_shp_q14,
            &mut s.nsq_lag_prev,
            &mut s.nsq_sdiff_shp_q14,
            &mut s.nsq_sltp_buf_idx,
            &mut s.nsq_sltp_shp_buf_idx,
            &mut s.nsq_rewhite_flag,
            s.nsq_slpc_q14.as_mut_ptr(),
            s.nsq_sar2_q14.as_mut_ptr(),
            s.nsq_sltp_shp_q14.as_mut_ptr(),
            &mut s.vad_hp_state,
            &mut s.vad_counter,
            s.vad_noise_level_bias.as_mut_ptr(),
            s.vad_ana_state.as_mut_ptr(),
            s.vad_ana_state1.as_mut_ptr(),
            s.vad_ana_state2.as_mut_ptr(),
            s.vad_nrg_ratio_smth_q8.as_mut_ptr(),
            s.vad_nl.as_mut_ptr(),
            s.vad_inv_nl.as_mut_ptr(),
            &mut s.sum_log_gain_q7,
            s.in_hp_state.as_mut_ptr(),
            &mut s.input_tilt_q15,
            s.input_quality_bands_q15.as_mut_ptr(),
            &mut s.frame_counter,
            &mut s.no_speech_counter,
            s.lp_in_lp_state.as_mut_ptr(),
            &mut s.lp_transition_frame_no,
            &mut s.shape_harm_boost_smth,
            &mut s.x_buf_hash,
            &mut s.ltp_corr_q15,
            &mut s.res_nrg_smth,
        );
    }
    s
}

fn dump_rust_extended(enc: &RustOpusEncoder, channel: usize) -> ExtendedState {
    let silk = enc.silk_encoder().expect("SILK encoder allocated");
    let ch = &silk.state_fxx[channel];
    let st = &ch.s_cmn;
    let mut s = ExtendedState {
        nsq_rand_seed: st.s_nsq.rand_seed,
        nsq_slf_ar_shp_q14: st.s_nsq.s_lf_ar_shp_q14,
        nsq_lag_prev: st.s_nsq.lag_prev,
        nsq_sdiff_shp_q14: st.s_nsq.s_diff_shp_q14,
        nsq_sltp_buf_idx: st.s_nsq.s_ltp_buf_idx,
        nsq_sltp_shp_buf_idx: st.s_nsq.s_ltp_shp_buf_idx,
        nsq_rewhite_flag: st.s_nsq.rewhite_flag,
        ..Default::default()
    };
    for i in 0..16 {
        s.nsq_slpc_q14[i] = st.s_nsq.s_lpc_q14[i];
        s.nsq_sar2_q14[i] = st.s_nsq.s_ar2_q14[i];
    }
    for i in 0..32 {
        s.nsq_sltp_shp_q14[i] = st.s_nsq.s_ltp_shp_q14[i];
    }
    s.vad_hp_state = st.s_vad.hp_state as i32;
    s.vad_counter = st.s_vad.counter;
    for i in 0..4 {
        s.vad_noise_level_bias[i] = st.s_vad.noise_level_bias[i];
        s.vad_nrg_ratio_smth_q8[i] = st.s_vad.nrg_ratio_smth_q8[i];
        s.vad_nl[i] = st.s_vad.nl[i];
        s.vad_inv_nl[i] = st.s_vad.inv_nl[i];
    }
    for i in 0..2 {
        s.vad_ana_state[i] = st.s_vad.ana_state[i];
        s.vad_ana_state1[i] = st.s_vad.ana_state1[i];
        s.vad_ana_state2[i] = st.s_vad.ana_state2[i];
    }
    s.sum_log_gain_q7 = st.sum_log_gain_q7;
    s.in_hp_state[0] = st.in_hp_state[0];
    s.in_hp_state[1] = st.in_hp_state[1];
    s.input_tilt_q15 = st.input_tilt_q15;
    for i in 0..4 {
        s.input_quality_bands_q15[i] = st.input_quality_bands_q15[i];
    }
    s.frame_counter = st.frame_counter;
    s.no_speech_counter = st.no_speech_counter;
    s.lp_in_lp_state[0] = st.s_lp.in_lp_state[0];
    s.lp_in_lp_state[1] = st.s_lp.in_lp_state[1];
    s.lp_transition_frame_no = st.s_lp.transition_frame_no;
    s.shape_harm_boost_smth = 0; // Rust doesn't track this dead field
    let mut h: i32 = 0;
    for b in ch.x_buf.iter() {
        h = h.wrapping_mul(31).wrapping_add(*b as i32);
    }
    s.x_buf_hash = h;
    s.ltp_corr_q15 = ch.ltp_corr_q15;
    s.res_nrg_smth = ch.res_nrg_smth;
    s
}

fn compare_extended_state(
    c_enc: *mut bindings::OpusEncoder,
    rust_enc: &RustOpusEncoder,
    channel: usize,
) -> Vec<String> {
    let c = dump_c_extended(c_enc, channel as i32);
    let r = dump_rust_extended(rust_enc, channel);
    let mut hits = Vec::new();
    macro_rules! cmp_scalar {
        ($name:ident) => {
            if c.$name != r.$name {
                hits.push(format!(
                    "{}: C={} R={}",
                    stringify!($name),
                    c.$name,
                    r.$name
                ));
            }
        };
    }
    macro_rules! cmp_array {
        ($name:ident) => {
            for i in 0..c.$name.len() {
                if c.$name[i] != r.$name[i] {
                    hits.push(format!(
                        "{}[{}]: C={} R={}",
                        stringify!($name),
                        i,
                        c.$name[i],
                        r.$name[i]
                    ));
                }
            }
        };
    }
    cmp_scalar!(nsq_rand_seed);
    cmp_scalar!(nsq_slf_ar_shp_q14);
    cmp_scalar!(nsq_lag_prev);
    cmp_scalar!(nsq_sdiff_shp_q14);
    cmp_scalar!(nsq_sltp_buf_idx);
    cmp_scalar!(nsq_sltp_shp_buf_idx);
    cmp_scalar!(nsq_rewhite_flag);
    cmp_array!(nsq_slpc_q14);
    cmp_array!(nsq_sar2_q14);
    cmp_array!(nsq_sltp_shp_q14);
    cmp_scalar!(vad_hp_state);
    cmp_scalar!(vad_counter);
    cmp_array!(vad_noise_level_bias);
    cmp_array!(vad_ana_state);
    cmp_array!(vad_ana_state1);
    cmp_array!(vad_ana_state2);
    cmp_array!(vad_nrg_ratio_smth_q8);
    cmp_array!(vad_nl);
    cmp_array!(vad_inv_nl);
    cmp_scalar!(sum_log_gain_q7);
    cmp_array!(in_hp_state);
    cmp_scalar!(input_tilt_q15);
    cmp_array!(input_quality_bands_q15);
    cmp_scalar!(frame_counter);
    cmp_scalar!(no_speech_counter);
    cmp_array!(lp_in_lp_state);
    cmp_scalar!(lp_transition_frame_no);
    cmp_scalar!(x_buf_hash);
    cmp_scalar!(ltp_corr_q15);
    cmp_scalar!(res_nrg_smth);
    hits
}

// ===========================================================================
// CLI
// ===========================================================================

pub fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 1 {
        run_all_known_failures();
        return;
    }
    if args.len() < 5 {
        eprintln!(
            "Usage: {} --mode <multiframe|multistream> --repro <path>\n       {} (no args: run all 8 known failures)",
            args[0], args[0]
        );
        std::process::exit(1);
    }
    let mut mode: Option<String> = None;
    let mut repro: Option<PathBuf> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" => {
                mode = Some(args[i + 1].clone());
                i += 2;
            }
            "--repro" => {
                repro = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    let mode = mode.expect("--mode required");
    let repro = repro.expect("--repro required");
    match mode.as_str() {
        "multiframe" => run_multiframe_repro(&repro),
        "multistream" => run_multistream_repro(&repro),
        _ => panic!("unknown mode: {mode}"),
    }
}

fn run_all_known_failures() {
    let workspace = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf();
    let multiframe_dir =
        workspace.join("tests/fuzz/known_failures/multiframe-cbr-cross-frame-divergence");
    let multistream_dir =
        workspace.join("tests/fuzz/known_failures/multistream-encode-bytes-divergence");
    let mut multiframe: Vec<_> = std::fs::read_dir(&multiframe_dir)
        .expect("read multiframe dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "bin"))
        .collect();
    multiframe.sort();
    let mut multistream: Vec<_> = std::fs::read_dir(&multistream_dir)
        .expect("read multistream dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .map(|n| {
                    let s = n.to_string_lossy();
                    s.starts_with("crash-") && (s.ends_with(".bin") || !s.contains('.'))
                })
                .unwrap_or(false)
        })
        .collect();
    multistream.sort();

    println!(
        "Running {} multiframe + {} multistream repros\n",
        multiframe.len(),
        multistream.len()
    );

    for p in &multiframe {
        run_multiframe_repro(p);
    }
    for p in &multistream {
        run_multistream_repro(p);
    }
}
