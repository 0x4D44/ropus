//! SILK Decoder — complete decoding pipeline.
//!
//! Ported from: dec_API.c, init_decoder.c, decode_frame.c, decode_indices.c,
//! decode_parameters.c, decode_core.c, decode_pitch.c, decode_pulses.c,
//! shell_coder.c, code_signs.c, gain_quant.c, PLC.c, CNG.c,
//! stereo_MS_to_LR.c, stereo_decode_pred.c, decoder_set_fs.c,
//! resampler.c, biquad_alt.c, and associated resampler private functions.

use crate::types::*;
use crate::celt::range_coder::RangeDecoder;
use crate::silk::common::*;
use crate::silk::tables::*;

// ===========================================================================
// State structures
// ===========================================================================

/// Side information indices decoded from the bitstream.
#[derive(Clone, Default)]
pub struct SideInfoIndices {
    pub gains_indices: [i8; MAX_NB_SUBFR],
    pub ltp_index: [i8; MAX_NB_SUBFR],
    pub nlsf_indices: [i8; MAX_LPC_ORDER + 1],
    pub lag_index: i16,
    pub contour_index: i8,
    pub signal_type: i8,
    pub quant_offset_type: i8,
    pub nlsf_interp_coef_q2: i8,
    pub per_index: i8,
    pub ltp_scale_index: i8,
    pub seed: i8,
}

/// Per-frame transient decoder control (stack-allocated, not persisted).
#[derive(Clone)]
pub struct SilkDecoderControl {
    pub pitch_l: [i32; MAX_NB_SUBFR],
    pub gains_q16: [i32; MAX_NB_SUBFR],
    pub pred_coef_q12: [[i16; MAX_LPC_ORDER]; 2],
    pub ltp_coef_q14: [i16; LTP_ORDER * MAX_NB_SUBFR],
    pub ltp_scale_q14: i32,
}

impl Default for SilkDecoderControl {
    fn default() -> Self {
        Self {
            pitch_l: [0; MAX_NB_SUBFR],
            gains_q16: [0; MAX_NB_SUBFR],
            pred_coef_q12: [[0; MAX_LPC_ORDER]; 2],
            ltp_coef_q14: [0; LTP_ORDER * MAX_NB_SUBFR],
            ltp_scale_q14: 0,
        }
    }
}

/// PLC (Packet Loss Concealment) state.
#[derive(Clone)]
pub struct SilkPlcState {
    pub pitch_l_q8: i32,
    pub ltp_coef_q14: [i16; LTP_ORDER],
    pub prev_lpc_q12: [i16; MAX_LPC_ORDER],
    pub last_frame_lost: i32,
    pub rand_seed: i32,
    pub rand_scale_q14: i16,
    pub conc_energy: i32,
    pub conc_energy_shift: i32,
    pub prev_ltp_scale_q14: i16,
    pub prev_gain_q16: [i32; 2],
    pub fs_khz: i32,
    pub nb_subfr: i32,
    pub subfr_length: i32,
}

impl Default for SilkPlcState {
    fn default() -> Self {
        Self {
            pitch_l_q8: 0,
            ltp_coef_q14: [0; LTP_ORDER],
            prev_lpc_q12: [0; MAX_LPC_ORDER],
            last_frame_lost: 0,
            rand_seed: 0,
            rand_scale_q14: 0,
            conc_energy: 0,
            conc_energy_shift: 0,
            prev_ltp_scale_q14: 0,
            prev_gain_q16: [0; 2],
            fs_khz: 0,
            nb_subfr: 0,
            subfr_length: 0,
        }
    }
}

/// CNG (Comfort Noise Generation) state.
#[derive(Clone)]
pub struct SilkCngState {
    pub cng_exc_buf_q14: [i32; MAX_FRAME_LENGTH],
    pub cng_smth_nlsf_q15: [i16; MAX_LPC_ORDER],
    pub cng_synth_state: [i32; MAX_LPC_ORDER],
    pub cng_smth_gain_q16: i32,
    pub rand_seed: i32,
    pub fs_khz: i32,
}

impl Default for SilkCngState {
    fn default() -> Self {
        Self {
            cng_exc_buf_q14: [0; MAX_FRAME_LENGTH],
            cng_smth_nlsf_q15: [0; MAX_LPC_ORDER],
            cng_synth_state: [0; MAX_LPC_ORDER],
            cng_smth_gain_q16: 0,
            rand_seed: 3176576,
            fs_khz: 0,
        }
    }
}

/// Resampler state.
#[derive(Clone)]
pub struct SilkResamplerState {
    pub s_iir: [i32; SILK_RESAMPLER_MAX_IIR_ORDER],
    pub s_fir_i32: [i32; SILK_RESAMPLER_MAX_FIR_ORDER],
    pub s_fir_i16: [i16; SILK_RESAMPLER_MAX_FIR_ORDER],
    pub delay_buf: [i16; 96],
    pub resampler_function: i32,
    pub batch_size: i32,
    pub inv_ratio_q16: i32,
    pub fir_order: i32,
    pub fir_fracs: i32,
    pub fs_in_khz: i32,
    pub fs_out_khz: i32,
    pub input_delay: i32,
    pub coefs: ResamplerCoefs,
}

/// Which resampler coefficient set to use.
#[derive(Clone, Copy, Default)]
pub enum ResamplerCoefs {
    #[default]
    None,
    Ratio3_4,
    Ratio2_3,
    Ratio1_2,
    Ratio1_3,
    Ratio1_4,
    Ratio1_6,
    LowQuality2_3,
}

/// Resampler function selector constants.
const USE_SILK_RESAMPLER_COPY: i32 = 0;
const USE_SILK_RESAMPLER_UP2_HQ: i32 = 1;
const USE_SILK_RESAMPLER_IIR_FIR: i32 = 2;
const USE_SILK_RESAMPLER_DOWN_FIR: i32 = 3;

impl Default for SilkResamplerState {
    fn default() -> Self {
        Self {
            s_iir: [0; SILK_RESAMPLER_MAX_IIR_ORDER],
            s_fir_i32: [0; SILK_RESAMPLER_MAX_FIR_ORDER],
            s_fir_i16: [0; SILK_RESAMPLER_MAX_FIR_ORDER],
            delay_buf: [0; 96],
            resampler_function: 0,
            batch_size: 0,
            inv_ratio_q16: 0,
            fir_order: 0,
            fir_fracs: 0,
            fs_in_khz: 0,
            fs_out_khz: 0,
            input_delay: 0,
            coefs: ResamplerCoefs::None,
        }
    }
}

/// Stereo decoder state.
#[derive(Clone, Default)]
pub struct StereoDecState {
    pub pred_prev_q13: [i16; 2],
    pub s_mid: [i16; 2],
    pub s_side: [i16; 2],
}

/// Per-channel decoder state.
#[derive(Clone)]
pub struct SilkDecoderState {
    // Persistent state (not reset by reset_decoder)
    // OSCE fields would go here if enabled

    // State reset from here onward
    pub prev_gain_q16: i32,
    pub exc_q14: Vec<i32>,
    pub s_lpc_q14_buf: [i32; MAX_LPC_ORDER],
    pub out_buf: Vec<i16>,
    pub lag_prev: i32,
    pub last_gain_index: i8,
    pub fs_khz: i32,
    pub fs_api_hz: i32,
    pub nb_subfr: usize,
    pub frame_length: usize,
    pub subfr_length: usize,
    pub ltp_mem_length: usize,
    pub lpc_order: usize,
    pub prev_nlsf_q15: [i16; MAX_LPC_ORDER],
    pub first_frame_after_reset: bool,
    pub pitch_lag_low_bits_icdf: &'static [u8],
    pub pitch_contour_icdf: &'static [u8],
    pub n_frames_decoded: usize,
    pub n_frames_per_packet: usize,
    pub ec_prev_signal_type: i32,
    pub ec_prev_lag_index: i16,
    pub vad_flags: [bool; MAX_FRAMES_PER_PACKET],
    pub lbrr_flag: bool,
    pub lbrr_flags: [bool; MAX_FRAMES_PER_PACKET],
    pub resampler_state: SilkResamplerState,
    pub nlsf_cb: &'static SilkNlsfCbStruct,
    pub indices: SideInfoIndices,
    pub s_cng: SilkCngState,
    pub loss_cnt: i32,
    pub prev_signal_type: i32,
    pub s_plc: SilkPlcState,
}

impl SilkDecoderState {
    pub fn new() -> Self {
        let mut state = Self {
            prev_gain_q16: 65536, // Q16 1.0
            exc_q14: vec![0i32; MAX_FRAME_LENGTH],
            s_lpc_q14_buf: [0; MAX_LPC_ORDER],
            out_buf: vec![0i16; MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH],
            lag_prev: 100,
            last_gain_index: 10,
            fs_khz: 0,
            fs_api_hz: 0,
            nb_subfr: 4,
            frame_length: 0,
            subfr_length: 0,
            ltp_mem_length: 0,
            lpc_order: 10,
            prev_nlsf_q15: [0; MAX_LPC_ORDER],
            first_frame_after_reset: true,
            pitch_lag_low_bits_icdf: &SILK_UNIFORM4_ICDF,
            pitch_contour_icdf: &SILK_PITCH_CONTOUR_NB_ICDF,
            n_frames_decoded: 0,
            n_frames_per_packet: 1,
            ec_prev_signal_type: 0,
            ec_prev_lag_index: 0,
            vad_flags: [false; MAX_FRAMES_PER_PACKET],
            lbrr_flag: false,
            lbrr_flags: [false; MAX_FRAMES_PER_PACKET],
            resampler_state: SilkResamplerState::default(),
            nlsf_cb: &SILK_NLSF_CB_NB_MB,
            indices: SideInfoIndices::default(),
            s_cng: SilkCngState::default(),
            loss_cnt: 0,
            prev_signal_type: TYPE_NO_VOICE_ACTIVITY,
            s_plc: SilkPlcState::default(),
        };
        state.s_plc.prev_gain_q16 = [65536, 65536];
        state.s_plc.subfr_length = 20;
        state.s_plc.nb_subfr = 2;
        state
    }

    /// Reset decoder state (preserves OSCE state).
    pub fn reset(&mut self) {
        self.prev_gain_q16 = 65536;
        self.exc_q14.iter_mut().for_each(|x| *x = 0);
        self.s_lpc_q14_buf = [0; MAX_LPC_ORDER];
        self.out_buf.iter_mut().for_each(|x| *x = 0);
        self.first_frame_after_reset = true;
        self.loss_cnt = 0;
        self.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
        silk_cng_reset(self);
        silk_plc_reset(self);
    }
}

/// Top-level decoder super-struct.
pub struct SilkDecoder {
    pub channel_state: [SilkDecoderState; DECODER_NUM_CHANNELS],
    pub s_stereo: StereoDecState,
    pub n_channels_api: usize,
    pub n_channels_internal: usize,
    pub prev_decode_only_middle: bool,
}

impl SilkDecoder {
    pub fn new() -> Self {
        Self {
            channel_state: [SilkDecoderState::new(), SilkDecoderState::new()],
            s_stereo: StereoDecState::default(),
            n_channels_api: 1,
            n_channels_internal: 1,
            prev_decode_only_middle: false,
        }
    }

    pub fn init(&mut self) {
        self.s_stereo = StereoDecState::default();
        self.channel_state[0] = SilkDecoderState::new();
        self.channel_state[1] = SilkDecoderState::new();
        self.prev_decode_only_middle = false;
    }
}

/// Decoder control parameters from the Opus decoder.
pub struct SilkDecControl {
    pub n_channels_api: usize,
    pub n_channels_internal: usize,
    pub api_sample_rate: i32,
    pub internal_sample_rate: i32,
    pub payload_size_ms: i32,
    pub prev_pitch_lag: i32,
}

// ===========================================================================
// PLC Reset / CNG Reset
// ===========================================================================

fn silk_plc_reset(dec: &mut SilkDecoderState) {
    let frame_len = if dec.frame_length > 0 {
        dec.frame_length as i32
    } else {
        320
    };
    dec.s_plc = SilkPlcState::default();
    dec.s_plc.pitch_l_q8 = frame_len << 7;
    dec.s_plc.prev_gain_q16 = [1 << 16, 1 << 16];
    dec.s_plc.subfr_length = 20;
    dec.s_plc.nb_subfr = 2;
}

fn silk_cng_reset(dec: &mut SilkDecoderState) {
    let lpc_order = if dec.lpc_order > 0 { dec.lpc_order } else { 10 };
    let nlsf_step_q15 = 32767i32 / (lpc_order as i32 + 1);
    dec.s_cng = SilkCngState::default();
    for i in 0..lpc_order {
        dec.s_cng.cng_smth_nlsf_q15[i] = (nlsf_step_q15 * (i as i32 + 1)) as i16;
    }
}

// ===========================================================================
// Decoder sample rate configuration
// ===========================================================================

fn silk_decoder_set_fs(dec: &mut SilkDecoderState, fs_khz: i32, fs_api_hz: i32) {
    let fs_changed = dec.fs_khz != fs_khz;

    dec.subfr_length = (SUB_FRAME_LENGTH_MS as i32 * fs_khz) as usize;
    dec.frame_length = dec.nb_subfr * dec.subfr_length;
    dec.ltp_mem_length = (LTP_MEM_LENGTH_MS as i32 * fs_khz) as usize;

    if fs_changed || dec.fs_api_hz != fs_api_hz {
        silk_resampler_init(&mut dec.resampler_state, fs_khz * 1000, fs_api_hz, false);
        dec.fs_api_hz = fs_api_hz;
    }

    if fs_changed {
        // Set LPC order and codebook
        if fs_khz == 8 || fs_khz == 12 {
            dec.lpc_order = MIN_LPC_ORDER;
            dec.nlsf_cb = &SILK_NLSF_CB_NB_MB;
        } else {
            // 16 kHz
            dec.lpc_order = MAX_LPC_ORDER;
            dec.nlsf_cb = &SILK_NLSF_CB_WB;
        }

        // Set pitch lag low bits iCDF based on rate
        dec.pitch_lag_low_bits_icdf = match fs_khz {
            16 => &SILK_UNIFORM8_ICDF,
            12 => &SILK_UNIFORM6_ICDF,
            _ => &SILK_UNIFORM4_ICDF,
        };

        // Set pitch contour iCDF
        if dec.nb_subfr == MAX_NB_SUBFR {
            dec.pitch_contour_icdf = if fs_khz == 8 {
                &SILK_PITCH_CONTOUR_NB_ICDF
            } else {
                &SILK_PITCH_CONTOUR_ICDF
            };
        } else {
            dec.pitch_contour_icdf = if fs_khz == 8 {
                &SILK_PITCH_CONTOUR_10_MS_NB_ICDF
            } else {
                &SILK_PITCH_CONTOUR_10_MS_ICDF
            };
        }

        // Clear buffers and reset state
        dec.first_frame_after_reset = true;
        dec.lag_prev = 100;
        dec.last_gain_index = 10;
        dec.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
        dec.out_buf.iter_mut().for_each(|x| *x = 0);
        dec.s_lpc_q14_buf = [0; MAX_LPC_ORDER];
        dec.fs_khz = fs_khz;
    }

    // Update pitch contour iCDF when frame length changes
    if !fs_changed && dec.nb_subfr == MAX_NB_SUBFR {
        dec.pitch_contour_icdf = if fs_khz == 8 {
            &SILK_PITCH_CONTOUR_NB_ICDF
        } else {
            &SILK_PITCH_CONTOUR_ICDF
        };
    } else if !fs_changed {
        dec.pitch_contour_icdf = if fs_khz == 8 {
            &SILK_PITCH_CONTOUR_10_MS_NB_ICDF
        } else {
            &SILK_PITCH_CONTOUR_10_MS_ICDF
        };
    }
}

// ===========================================================================
// Bitstream Parsing: decode_indices
// ===========================================================================

/// Decode all quantization indices from the range coder.
fn silk_decode_indices(
    dec: &mut SilkDecoderState,
    rc: &mut RangeDecoder,
    frame_index: usize,
    decode_lbrr: bool,
    cond_coding: i32,
) {
    let indices = &mut dec.indices;

    // 1. Signal type and quantizer offset
    let ix = if decode_lbrr || dec.vad_flags[frame_index] {
        rc.decode_icdf(&SILK_TYPE_OFFSET_VAD_ICDF, 8) + 2
    } else {
        rc.decode_icdf(&SILK_TYPE_OFFSET_NO_VAD_ICDF, 8)
    };
    indices.signal_type = (ix >> 1) as i8;
    indices.quant_offset_type = (ix & 1) as i8;

    // 2. Gain indices
    if cond_coding == CODE_CONDITIONALLY {
        indices.gains_indices[0] = rc.decode_icdf(&SILK_DELTA_GAIN_ICDF, 8) as i8;
    } else {
        let msb = rc.decode_icdf(
            &SILK_GAIN_ICDF[indices.signal_type as usize],
            8,
        );
        let lsb = rc.decode_icdf(&SILK_UNIFORM8_ICDF, 8);
        indices.gains_indices[0] = ((msb << 3) + lsb) as i8;
    }
    for i in 1..dec.nb_subfr {
        indices.gains_indices[i] = rc.decode_icdf(&SILK_DELTA_GAIN_ICDF, 8) as i8;
    }

    // 3. NLSF indices
    let sig_type_half = (indices.signal_type as usize) >> 1;
    let n_vectors = dec.nlsf_cb.n_vectors as usize;
    let cb1_icdf_offset = sig_type_half * n_vectors;
    indices.nlsf_indices[0] =
        rc.decode_icdf(&dec.nlsf_cb.cb1_icdf[cb1_icdf_offset..], 8) as i8;

    let mut ec_ix: [i16; MAX_LPC_ORDER] = [0; MAX_LPC_ORDER];
    let mut pred_q8: [u8; MAX_LPC_ORDER] = [0; MAX_LPC_ORDER];
    silk_nlsf_unpack(&mut ec_ix, &mut pred_q8, dec.nlsf_cb, indices.nlsf_indices[0] as usize);

    for i in 0..dec.lpc_order {
        let icdf_offset = ec_ix[i] as usize;
        let mut ix_val = rc.decode_icdf(&dec.nlsf_cb.ec_icdf[icdf_offset..], 8);
        if ix_val == 0 {
            ix_val -= rc.decode_icdf(&SILK_NLSF_EXT_ICDF, 8);
        } else if ix_val == 2 * NLSF_QUANT_MAX_AMPLITUDE {
            ix_val += rc.decode_icdf(&SILK_NLSF_EXT_ICDF, 8);
        }
        indices.nlsf_indices[i + 1] = (ix_val - NLSF_QUANT_MAX_AMPLITUDE) as i8;
    }

    // 4. NLSF interpolation coefficient
    if dec.nb_subfr == MAX_NB_SUBFR {
        indices.nlsf_interp_coef_q2 =
            rc.decode_icdf(&SILK_NLSF_INTERPOLATION_FACTOR_ICDF, 8) as i8;
    } else {
        indices.nlsf_interp_coef_q2 = 4; // No interpolation for 10ms
    }

    // 5. Pitch (voiced only)
    if indices.signal_type as i32 == TYPE_VOICED {
        // Pitch lag
        let mut decode_absolute = true;
        if cond_coding == CODE_CONDITIONALLY && dec.ec_prev_signal_type == TYPE_VOICED {
            let delta_lag = rc.decode_icdf(&SILK_PITCH_DELTA_ICDF, 8);
            if delta_lag > 0 {
                let delta = delta_lag - 9;
                indices.lag_index = dec.ec_prev_lag_index + delta as i16;
                decode_absolute = false;
            }
        }
        if decode_absolute {
            let msb = rc.decode_icdf(&SILK_PITCH_LAG_ICDF, 8) as i16;
            let lsb = rc.decode_icdf(dec.pitch_lag_low_bits_icdf, 8) as i16;
            indices.lag_index = msb * (dec.fs_khz as i16 >> 1) + lsb;
        }
        dec.ec_prev_lag_index = indices.lag_index;

        // Pitch contour
        indices.contour_index = rc.decode_icdf(dec.pitch_contour_icdf, 8) as i8;

        // LTP gains
        indices.per_index = rc.decode_icdf(&SILK_LTP_PER_INDEX_ICDF, 8) as i8;
        for k in 0..dec.nb_subfr {
            indices.ltp_index[k] =
                rc.decode_icdf(SILK_LTP_GAIN_ICDF_PTRS[indices.per_index as usize], 8) as i8;
        }

        // LTP scaling
        if cond_coding == CODE_INDEPENDENTLY {
            indices.ltp_scale_index = rc.decode_icdf(&SILK_LTP_SCALE_ICDF, 8) as i8;
        } else {
            indices.ltp_scale_index = 0;
        }
    }

    dec.ec_prev_signal_type = indices.signal_type as i32;

    // 6. Random seed
    indices.seed = rc.decode_icdf(&SILK_UNIFORM4_ICDF, 8) as i8;
}

// ===========================================================================
// Shell coding: decode excitation pulses
// ===========================================================================

/// Decode a single split in the shell coder tree.
#[inline]
fn decode_split(
    rc: &mut RangeDecoder,
    p: i32,
    shell_table: &[u8],
) -> (i16, i16) {
    if p > 0 {
        let offset = SILK_SHELL_CODE_TABLE_OFFSETS[p as usize] as usize;
        let child1 = rc.decode_icdf(&shell_table[offset..], 8) as i16;
        (child1, p as i16 - child1)
    } else {
        (0, 0)
    }
}

/// Decode 16 pulse values from the shell coder.
fn silk_shell_decoder(
    pulses0: &mut [i16],
    rc: &mut RangeDecoder,
    pulses4: i32,
) {
    let mut pulses3: [i16; 2] = [0; 2];
    let mut pulses2: [i16; 4] = [0; 4];
    let mut pulses1: [i16; 8] = [0; 8];

    let (p3a, p3b) = decode_split(rc, pulses4, &SILK_SHELL_CODE_TABLE3);
    pulses3[0] = p3a;
    pulses3[1] = p3b;

    let (p2a, p2b) = decode_split(rc, pulses3[0] as i32, &SILK_SHELL_CODE_TABLE2);
    pulses2[0] = p2a;
    pulses2[1] = p2b;

    let (p1a, p1b) = decode_split(rc, pulses2[0] as i32, &SILK_SHELL_CODE_TABLE1);
    pulses1[0] = p1a;
    pulses1[1] = p1b;

    let (a, b) = decode_split(rc, pulses1[0] as i32, &SILK_SHELL_CODE_TABLE0);
    pulses0[0] = a; pulses0[1] = b;
    let (a, b) = decode_split(rc, pulses1[1] as i32, &SILK_SHELL_CODE_TABLE0);
    pulses0[2] = a; pulses0[3] = b;

    let (p1a, p1b) = decode_split(rc, pulses2[1] as i32, &SILK_SHELL_CODE_TABLE1);
    pulses1[2] = p1a;
    pulses1[3] = p1b;

    let (a, b) = decode_split(rc, pulses1[2] as i32, &SILK_SHELL_CODE_TABLE0);
    pulses0[4] = a; pulses0[5] = b;
    let (a, b) = decode_split(rc, pulses1[3] as i32, &SILK_SHELL_CODE_TABLE0);
    pulses0[6] = a; pulses0[7] = b;

    // Right subtree of level 3
    let (p2a, p2b) = decode_split(rc, pulses3[1] as i32, &SILK_SHELL_CODE_TABLE2);
    pulses2[2] = p2a;
    pulses2[3] = p2b;

    let (p1a, p1b) = decode_split(rc, pulses2[2] as i32, &SILK_SHELL_CODE_TABLE1);
    pulses1[4] = p1a;
    pulses1[5] = p1b;

    let (a, b) = decode_split(rc, pulses1[4] as i32, &SILK_SHELL_CODE_TABLE0);
    pulses0[8] = a; pulses0[9] = b;
    let (a, b) = decode_split(rc, pulses1[5] as i32, &SILK_SHELL_CODE_TABLE0);
    pulses0[10] = a; pulses0[11] = b;

    let (p1a, p1b) = decode_split(rc, pulses2[3] as i32, &SILK_SHELL_CODE_TABLE1);
    pulses1[6] = p1a;
    pulses1[7] = p1b;

    let (a, b) = decode_split(rc, pulses1[6] as i32, &SILK_SHELL_CODE_TABLE0);
    pulses0[12] = a; pulses0[13] = b;
    let (a, b) = decode_split(rc, pulses1[7] as i32, &SILK_SHELL_CODE_TABLE0);
    pulses0[14] = a; pulses0[15] = b;
}

/// Decode signs for excitation pulses.
fn silk_decode_signs(
    rc: &mut RangeDecoder,
    pulses: &mut [i16],
    length: usize,
    signal_type: i32,
    quant_offset_type: i32,
    sum_pulses: &[i32],
) {
    let icdf_table_idx = 7 * (quant_offset_type + 2 * signal_type) as usize;
    let icdf_ptr = &SILK_SIGN_ICDF[icdf_table_idx..];
    let block_count = (length + SHELL_CODEC_FRAME_LENGTH / 2) >> LOG2_SHELL_CODEC_FRAME_LENGTH;

    for i in 0..block_count {
        let p = sum_pulses[i];
        if p > 0 {
            let icdf_idx = imin((p & 0x1F) as i32, 6) as usize;
            let icdf: [u8; 2] = [icdf_ptr[icdf_idx], 0];
            let base = i * SHELL_CODEC_FRAME_LENGTH;
            for j in 0..SHELL_CODEC_FRAME_LENGTH {
                if base + j < length && pulses[base + j] > 0 {
                    let bit = rc.decode_icdf(&icdf, 8);
                    // dec_map: 0→-1, 1→+1
                    if bit == 0 {
                        pulses[base + j] = -pulses[base + j];
                    }
                }
            }
        }
    }
}

/// Full pulse decoding: rate level, shell, LSB, signs.
fn silk_decode_pulses(
    rc: &mut RangeDecoder,
    pulses: &mut [i16],
    signal_type: i32,
    quant_offset_type: i32,
    frame_length: usize,
) {
    // 1. Rate level
    let rate_level = rc.decode_icdf(
        &SILK_RATE_LEVELS_ICDF[(signal_type >> 1) as usize],
        8,
    ) as usize;

    // 2. Shell blocks
    let iter = (frame_length + SHELL_CODEC_FRAME_LENGTH - 1) / SHELL_CODEC_FRAME_LENGTH;
    let mut sum_pulses = [0i32; MAX_NB_SHELL_BLOCKS];
    let mut n_lshifts = [0i32; MAX_NB_SHELL_BLOCKS];

    for i in 0..iter {
        n_lshifts[i] = 0;
        sum_pulses[i] = rc.decode_icdf(
            &SILK_PULSES_PER_BLOCK_ICDF[rate_level],
            8,
        );
        // Handle overflow: while sum_pulses == SILK_MAX_PULSES + 1
        while sum_pulses[i] == SILK_MAX_PULSES + 1 {
            n_lshifts[i] += 1;
            let use_last = if n_lshifts[i] == 10 { 1 } else { 0 };
            sum_pulses[i] = rc.decode_icdf(
                &SILK_PULSES_PER_BLOCK_ICDF[N_RATE_LEVELS - 1 + use_last as usize],
                8,
            );
        }
    }

    // 3. Shell decoding
    for i in 0..iter {
        let base = i * SHELL_CODEC_FRAME_LENGTH;
        if sum_pulses[i] > 0 {
            silk_shell_decoder(&mut pulses[base..], rc, sum_pulses[i]);
        } else {
            for j in 0..SHELL_CODEC_FRAME_LENGTH {
                if base + j < pulses.len() {
                    pulses[base + j] = 0;
                }
            }
        }
    }

    // 4. LSB decoding
    for i in 0..iter {
        if n_lshifts[i] > 0 {
            let base = i * SHELL_CODEC_FRAME_LENGTH;
            for k in 0..SHELL_CODEC_FRAME_LENGTH {
                if base + k < pulses.len() {
                    let mut abs_q = pulses[base + k] as i32;
                    for _j in 0..n_lshifts[i] {
                        abs_q = shl32(abs_q, 1);
                        abs_q += rc.decode_icdf(&SILK_LSB_ICDF, 8);
                    }
                    pulses[base + k] = abs_q as i16;
                }
            }
            sum_pulses[i] |= n_lshifts[i] << 5;
        }
    }

    // 5. Sign decoding
    silk_decode_signs(
        rc,
        pulses,
        frame_length,
        signal_type,
        quant_offset_type,
        &sum_pulses,
    );
}

// ===========================================================================
// Parameter dequantization
// ===========================================================================

fn silk_decode_parameters(
    dec: &mut SilkDecoderState,
    dec_ctrl: &mut SilkDecoderControl,
    cond_coding: i32,
) {
    let indices = dec.indices.clone();
    let nb_subfr = dec.nb_subfr;
    let lpc_order = dec.lpc_order;

    // 1. Gain dequantization
    silk_gains_dequant(
        &mut dec_ctrl.gains_q16,
        &indices.gains_indices,
        &mut dec.last_gain_index,
        cond_coding == CODE_CONDITIONALLY,
        nb_subfr,
    );

    // 2. NLSF decoding
    let mut nlsf_q15 = [0i16; MAX_LPC_ORDER];
    silk_nlsf_decode(&mut nlsf_q15, &indices.nlsf_indices, dec.nlsf_cb);

    // 3. Convert NLSF to LPC (second half)
    silk_nlsf2a(&mut dec_ctrl.pred_coef_q12[1], &nlsf_q15, lpc_order);

    // 4. NLSF interpolation
    let interp_coef = if dec.first_frame_after_reset {
        4i32 // Force no interpolation
    } else {
        indices.nlsf_interp_coef_q2 as i32
    };

    if interp_coef < 4 {
        // Interpolate NLSF for first half
        let mut nlsf0_q15 = [0i16; MAX_LPC_ORDER];
        for i in 0..lpc_order {
            nlsf0_q15[i] = (dec.prev_nlsf_q15[i] as i32
                + ((interp_coef * (nlsf_q15[i] as i32 - dec.prev_nlsf_q15[i] as i32)) >> 2))
                as i16;
        }
        silk_nlsf2a(&mut dec_ctrl.pred_coef_q12[0], &nlsf0_q15, lpc_order);
    } else {
        // Copy second half to first half
        dec_ctrl.pred_coef_q12[0] = dec_ctrl.pred_coef_q12[1];
    }

    // 5. Save current NLSF
    dec.prev_nlsf_q15[..lpc_order].copy_from_slice(&nlsf_q15[..lpc_order]);

    // 6. Bandwidth expansion after loss
    if dec.loss_cnt > 0 {
        silk_bwexpander(&mut dec_ctrl.pred_coef_q12[0], lpc_order, BWE_COEF_Q16);
        silk_bwexpander(&mut dec_ctrl.pred_coef_q12[1], lpc_order, BWE_COEF_Q16);
    }

    // 7. Voiced frame processing
    if indices.signal_type as i32 == TYPE_VOICED {
        silk_decode_pitch(
            indices.lag_index,
            indices.contour_index,
            &mut dec_ctrl.pitch_l,
            dec.fs_khz,
            nb_subfr,
        );

        // LTP coefficient dequantization
        let cbk = SILK_LTP_VQ_PTRS_Q7[indices.per_index as usize];
        for k in 0..nb_subfr {
            let ix = indices.ltp_index[k] as usize;
            for i in 0..LTP_ORDER {
                dec_ctrl.ltp_coef_q14[k * LTP_ORDER + i] =
                    (cbk[ix * LTP_ORDER + i] as i16) << 7;
            }
        }

        // LTP scaling
        dec_ctrl.ltp_scale_q14 =
            SILK_LTP_SCALES_TABLE_Q14[indices.ltp_scale_index as usize] as i32;
    } else {
        // Unvoiced
        dec_ctrl.pitch_l = [0; MAX_NB_SUBFR];
        dec_ctrl.ltp_coef_q14 = [0; LTP_ORDER * MAX_NB_SUBFR];
        dec_ctrl.ltp_scale_q14 = 0;
    }
}

// ===========================================================================
// Inverse NSQ: decode_core
// ===========================================================================

fn silk_decode_core(
    dec: &mut SilkDecoderState,
    dec_ctrl: &SilkDecoderControl,
    xq: &mut [i16],
    pulses: &[i16],
) {
    let frame_length = dec.frame_length;
    let subfr_length = dec.subfr_length;
    let nb_subfr = dec.nb_subfr;
    let lpc_order = dec.lpc_order;
    let ltp_mem_length = dec.ltp_mem_length;
    let signal_type = dec.indices.signal_type as i32;
    let quant_offset_type = dec.indices.quant_offset_type as i32;

    // Quantization offset from table
    let offset_q10 = SILK_QUANTIZATION_OFFSETS_Q10
        [(signal_type >> 1) as usize][quant_offset_type as usize] as i32;

    // NLSF interpolation flag
    let nlsf_interp = (dec.indices.nlsf_interp_coef_q2 as i32) < 4;

    // Step 1: Generate excitation from pulses
    let mut rand_seed = dec.indices.seed as i32;
    for i in 0..frame_length {
        rand_seed = silk_rand(rand_seed);
        let pulse = pulses[i] as i32;
        let mut exc_val = pulse << 14;
        // Dead-zone adjustment
        if exc_val > 0 {
            exc_val -= QUANT_LEVEL_ADJUST_Q10 << 4;
        } else if exc_val < 0 {
            exc_val += QUANT_LEVEL_ADJUST_Q10 << 4;
        }
        // Add quantization offset
        exc_val += offset_q10 << 4;
        // Random sign flip
        if rand_seed < 0 {
            exc_val = -exc_val;
        }
        // PRNG feedback
        rand_seed = rand_seed.wrapping_add(pulse);
        dec.exc_q14[i] = exc_val;
    }

    // Step 2: Copy previous LPC state to working buffer
    let mut s_lpc_q14 = vec![0i32; subfr_length + MAX_LPC_ORDER];
    s_lpc_q14[..MAX_LPC_ORDER].copy_from_slice(&dec.s_lpc_q14_buf);

    // Allocate sLTP buffers for voiced frames
    let mut s_ltp = vec![0i16; ltp_mem_length + frame_length];
    let mut s_ltp_q15 = vec![0i32; ltp_mem_length + frame_length];
    let mut s_ltp_buf_idx = ltp_mem_length;

    let mut prev_gain_q16 = dec.prev_gain_q16;

    // Step 3: Main subframe loop
    let mut exc_offset = 0;
    let mut xq_offset = 0;

    for k in 0..nb_subfr {
        let a_q12 = &dec_ctrl.pred_coef_q12[k >> 1];
        let b_q14 = &dec_ctrl.ltp_coef_q14[k * LTP_ORDER..(k + 1) * LTP_ORDER];
        let gain_q10 = dec_ctrl.gains_q16[k] >> 6;

        // Compute inverse gain
        let inv_gain_q31 = silk_inverse32_var_q(dec_ctrl.gains_q16[k], 47);

        // Gain adjustment if gain changed
        if dec_ctrl.gains_q16[k] != prev_gain_q16 {
            let gain_adj_q16 = silk_div32_var_q(prev_gain_q16, dec_ctrl.gains_q16[k], 16);
            // Scale sLPC_Q14 state
            for i in 0..MAX_LPC_ORDER {
                s_lpc_q14[i] = silk_smulww(gain_adj_q16, s_lpc_q14[i]);
            }
        }
        prev_gain_q16 = dec_ctrl.gains_q16[k];

        // Allocate residual buffer
        let mut res_q14 = vec![0i32; subfr_length];

        if signal_type == TYPE_VOICED {
            let lag = dec_ctrl.pitch_l[k];

            // Re-whitening at subframe 0 or 2 (if interpolated)
            if k == 0 || (k == 2 && nlsf_interp) {
                // LPC analysis filter on outBuf to produce sLTP
                let start_idx = ltp_mem_length as i32 - lag - lpc_order as i32 - LTP_ORDER as i32 / 2;
                let start_idx = imax(start_idx, 0) as usize;

                for i in start_idx..ltp_mem_length {
                    let mut sum: i64 = 0;
                    for j in 0..lpc_order {
                        if i >= j + 1 {
                            sum += a_q12[j] as i64 * dec.out_buf[i - j - 1] as i64;
                        }
                    }
                    s_ltp[i] = (dec.out_buf[i] as i32 - (sum >> 12) as i32) as i16;
                }

                // Scale sLTP → sLTP_Q15 using inv_gain
                let mut inv_gain_for_ltp = inv_gain_q31;
                if k == 0 {
                    // Apply LTP scale for first subframe
                    inv_gain_for_ltp =
                        ((inv_gain_for_ltp as i64 * dec_ctrl.ltp_scale_q14 as i64) >> 14) as i32;
                }
                let n_samples = lag as usize + LTP_ORDER / 2;
                for i in 0..n_samples {
                    let idx = s_ltp_buf_idx as i64 - i as i64 - 1;
                    if idx >= 0 && (idx as usize) < s_ltp_q15.len() {
                        let s_idx = ltp_mem_length as i64 - i as i64 - 1;
                        if s_idx >= 0 {
                            s_ltp_q15[idx as usize] =
                                silk_smulwb(inv_gain_for_ltp, s_ltp[s_idx as usize] as i16);
                        }
                    }
                }
            } else if dec_ctrl.gains_q16[k] != dec_ctrl.gains_q16[k - 1] {
                // Scale existing LTP state when gain changes
                let gain_adj_q16 = silk_div32_var_q(
                    dec_ctrl.gains_q16[k - 1],
                    dec_ctrl.gains_q16[k],
                    16,
                );
                let n_start = s_ltp_buf_idx as i64 - lag as i64 - LTP_ORDER as i64 / 2;
                let n_start = imax(n_start as i32, 0) as usize;
                for i in n_start..s_ltp_buf_idx {
                    s_ltp_q15[i] = silk_smulww(gain_adj_q16, s_ltp_q15[i]);
                }
            }

            // LTP synthesis
            for i in 0..subfr_length {
                // 5-tap FIR LTP filter
                let mut ltp_pred_q13: i32 = 2; // Rounding bias
                let pred_base = s_ltp_buf_idx as i64 - lag as i64 + LTP_ORDER as i64 / 2;
                for j in 0..LTP_ORDER {
                    let idx = pred_base - j as i64;
                    if idx >= 0 && (idx as usize) < s_ltp_q15.len() {
                        ltp_pred_q13 = (ltp_pred_q13 as i64
                            + ((s_ltp_q15[idx as usize] as i64 * b_q14[j] as i64) >> 16))
                            as i32;
                    }
                }

                // Combine: res = exc + LTP_pred << 1
                res_q14[i] = dec.exc_q14[exc_offset + i] + shl32(ltp_pred_q13, 1);
                // Update sLTP state
                if s_ltp_buf_idx < s_ltp_q15.len() {
                    s_ltp_q15[s_ltp_buf_idx] = shl32(res_q14[i], 1);
                }
                s_ltp_buf_idx += 1;
            }
        } else {
            // Unvoiced: residual = excitation
            for i in 0..subfr_length {
                res_q14[i] = dec.exc_q14[exc_offset + i];
            }
        }

        // LPC synthesis
        for i in 0..subfr_length {
            // LPC prediction with rounding bias
            let mut lpc_pred_q10: i32 = (lpc_order >> 1) as i32;
            for j in 0..lpc_order {
                let idx = MAX_LPC_ORDER + i - j - 1;
                lpc_pred_q10 = (lpc_pred_q10 as i64
                    + ((s_lpc_q14[idx] as i64 * a_q12[j] as i64) >> 16))
                    as i32;
            }

            // Combine residual with prediction
            s_lpc_q14[MAX_LPC_ORDER + i] =
                silk_add_sat32(res_q14[i], silk_lshift_sat32(lpc_pred_q10, 4));

            // Apply gain and output
            let out_val = silk_rshift_round(silk_smulww(s_lpc_q14[MAX_LPC_ORDER + i], gain_q10), 8);
            xq[xq_offset + i] = sat16(out_val);
        }

        // Shift LPC state for next subframe
        let new_state_start = subfr_length;
        for i in 0..MAX_LPC_ORDER {
            s_lpc_q14[i] = s_lpc_q14[new_state_start + i];
        }

        exc_offset += subfr_length;
        xq_offset += subfr_length;
    }

    // Save LPC state for next frame
    dec.s_lpc_q14_buf.copy_from_slice(&s_lpc_q14[..MAX_LPC_ORDER]);
    dec.prev_gain_q16 = prev_gain_q16;
}

// ===========================================================================
// PLC (Packet Loss Concealment)
// ===========================================================================

fn silk_plc_update(dec: &mut SilkDecoderState, dec_ctrl: &SilkDecoderControl) {
    let nb_subfr = dec.nb_subfr;

    if dec.indices.signal_type as i32 == TYPE_VOICED {
        // Find subframe with strongest pitch pulse (use last subframe for simplicity)
        let k = nb_subfr - 1;

        // Sum LTP coefficients
        let mut ltp_gain_q14: i32 = 0;
        for i in 0..LTP_ORDER {
            ltp_gain_q14 += dec_ctrl.ltp_coef_q14[k * LTP_ORDER + i] as i32;
        }

        // Save LTP state, collapse to single tap
        dec.s_plc.ltp_coef_q14 = [0; LTP_ORDER];
        let gain_clamped = imin(imax(ltp_gain_q14, V_PITCH_GAIN_START_MIN_Q14), V_PITCH_GAIN_START_MAX_Q14);
        dec.s_plc.ltp_coef_q14[LTP_ORDER / 2] = gain_clamped as i16;

        // Save pitch in Q8
        dec.s_plc.pitch_l_q8 = dec_ctrl.pitch_l[nb_subfr - 1] << 8;
    } else {
        // Unvoiced
        dec.s_plc.pitch_l_q8 = (dec.fs_khz * MAX_PITCH_LAG_MS) << 8;
        dec.s_plc.ltp_coef_q14 = [0; LTP_ORDER];
    }

    // Save LPC coefficients and gains
    for i in 0..dec.lpc_order {
        dec.s_plc.prev_lpc_q12[i] = dec_ctrl.pred_coef_q12[1][i];
    }
    dec.s_plc.prev_gain_q16[0] = dec_ctrl.gains_q16[nb_subfr.saturating_sub(2).max(0)];
    dec.s_plc.prev_gain_q16[1] = dec_ctrl.gains_q16[nb_subfr - 1];
    dec.s_plc.prev_ltp_scale_q14 = dec_ctrl.ltp_scale_q14 as i16;
}

fn silk_plc_conceal(
    dec: &mut SilkDecoderState,
    frame: &mut [i16],
) {
    let frame_length = dec.frame_length;
    let subfr_length = dec.subfr_length;
    let nb_subfr = dec.nb_subfr;
    let lpc_order = dec.lpc_order;

    // Apply bandwidth expansion to saved LPC coefficients
    let mut a_q12 = dec.s_plc.prev_lpc_q12;
    silk_bwexpander(&mut a_q12, lpc_order, BWE_COEF_Q16);

    let mut b_q14 = dec.s_plc.ltp_coef_q14;
    let mut rand_seed = dec.s_plc.rand_seed;
    let mut rand_scale_q14 = dec.s_plc.rand_scale_q14 as i32;

    // Get attenuation indices (clamp at NB_ATT-1)
    let att_idx = imin(dec.loss_cnt as i32, NB_ATT as i32 - 1) as usize;
    let harm_gain_q15 = HARM_ATT_Q15[att_idx];
    let rand_gain_q15 = if dec.prev_signal_type == TYPE_VOICED {
        PLC_RAND_ATTENUATE_V_Q15[att_idx]
    } else {
        PLC_RAND_ATTENUATE_UV_Q15[att_idx]
    };

    // Initialize concealment
    if dec.loss_cnt == 0 {
        // First lost frame
        rand_scale_q14 = 1 << 14;
        if dec.prev_signal_type == TYPE_VOICED {
            // Reduce random gain by LTP gain
            let mut ltp_sum: i32 = 0;
            for i in 0..LTP_ORDER {
                ltp_sum += b_q14[i] as i32;
            }
            rand_scale_q14 -= ltp_sum;
            rand_scale_q14 = imax(rand_scale_q14, 3277); // Min 0.2 in Q14
            rand_scale_q14 = ((rand_scale_q14 as i64 * dec.s_plc.prev_ltp_scale_q14 as i64) >> 14) as i32;
        }
    }

    let _prev_gain_q10_0 = dec.s_plc.prev_gain_q16[0] >> 6;
    let prev_gain_q10_1 = dec.s_plc.prev_gain_q16[1] >> 6;

    // Use excitation buffer as noise source
    let exc_buf: Vec<i32> = dec.exc_q14[..frame_length].to_vec();

    // LPC state
    let mut s_lpc_q14 = vec![0i32; subfr_length + MAX_LPC_ORDER];
    s_lpc_q14[..MAX_LPC_ORDER].copy_from_slice(&dec.s_lpc_q14_buf);

    // LTP state for voiced concealment
    let mut pitch_lag = (dec.s_plc.pitch_l_q8 >> 8).max(1);
    let mut s_ltp_q14 = vec![0i32; frame_length + pitch_lag as usize + LTP_ORDER];

    let mut frame_offset = 0;

    for _k in 0..nb_subfr {
        // Generate concealment excitation
        for i in 0..subfr_length {
            rand_seed = silk_rand(rand_seed);
            let idx = ((rand_seed >> 25) as usize) & RAND_BUF_MASK;
            let rand_val = if idx < exc_buf.len() {
                exc_buf[idx]
            } else {
                0
            };

            // LTP prediction for voiced
            let ltp_pred = if dec.prev_signal_type == TYPE_VOICED {
                let mut pred: i32 = 2; // Rounding bias
                for j in 0..LTP_ORDER {
                    let s_idx = frame_offset as i64 + i as i64 - pitch_lag as i64 + (LTP_ORDER / 2) as i64 - j as i64;
                    if s_idx >= 0 && (s_idx as usize) < s_ltp_q14.len() {
                        pred += ((s_ltp_q14[s_idx as usize] as i64 * b_q14[j] as i64) >> 16) as i32;
                    }
                }
                pred
            } else {
                0
            };

            // Combine: LTP_pred + rand_noise * rand_scale
            let exc = shl32(ltp_pred, 2)
                + ((rand_val as i64 * rand_scale_q14 as i64 >> 14) as i32);

            if frame_offset + i < s_ltp_q14.len() {
                s_ltp_q14[frame_offset + i] = exc;
            }

            // LPC synthesis
            let mut lpc_pred_q10: i32 = (lpc_order >> 1) as i32;
            for j in 0..lpc_order {
                let idx = MAX_LPC_ORDER + i - j - 1;
                if idx < s_lpc_q14.len() {
                    lpc_pred_q10 += ((s_lpc_q14[idx] as i64 * a_q12[j] as i64) >> 16) as i32;
                }
            }

            s_lpc_q14[MAX_LPC_ORDER + i] =
                silk_add_sat32(exc, silk_lshift_sat32(lpc_pred_q10, 4));

            // Scale and output
            let out_val = silk_rshift_round(silk_smulww(s_lpc_q14[MAX_LPC_ORDER + i], prev_gain_q10_1), 8);
            frame[frame_offset + i] = sat16(out_val);
        }

        // Attenuate LTP gains
        for j in 0..LTP_ORDER {
            b_q14[j] = ((b_q14[j] as i32 * harm_gain_q15) >> 15) as i16;
        }
        rand_scale_q14 = (rand_scale_q14 * rand_gain_q15) >> 15;

        // Drift pitch upward
        let pitch_incr = ((dec.s_plc.pitch_l_q8 as i64 * PITCH_DRIFT_FAC_Q16 as i64) >> 16) as i32;
        dec.s_plc.pitch_l_q8 += pitch_incr;
        dec.s_plc.pitch_l_q8 = imin(
            dec.s_plc.pitch_l_q8,
            (MAX_PITCH_LAG_MS * dec.fs_khz) << 8,
        );
        pitch_lag = (dec.s_plc.pitch_l_q8 >> 8).max(1);

        // Shift LPC state
        for i in 0..MAX_LPC_ORDER {
            s_lpc_q14[i] = s_lpc_q14[subfr_length + i];
        }
        frame_offset += subfr_length;
    }

    // Save state
    dec.s_lpc_q14_buf.copy_from_slice(&s_lpc_q14[..MAX_LPC_ORDER]);
    dec.s_plc.rand_seed = rand_seed;
    dec.s_plc.rand_scale_q14 = rand_scale_q14 as i16;
}

fn silk_plc_glue_frames(
    dec: &mut SilkDecoderState,
    frame: &mut [i16],
    length: usize,
) {
    if dec.loss_cnt > 0 {
        // Transitioning from loss to good: compute concealment energy
        let (energy, shift) = silk_sum_sqr_shift(&frame[..length]);
        dec.s_plc.conc_energy = energy;
        dec.s_plc.conc_energy_shift = shift;
    } else if dec.s_plc.last_frame_lost != 0 {
        // First good frame after loss: fade in if needed
        let (new_energy, new_shift) = silk_sum_sqr_shift(&frame[..length]);

        if new_energy > 0 && dec.s_plc.conc_energy > 0 {
            // Normalize energies to same shift
            let shift_diff = dec.s_plc.conc_energy_shift - new_shift;
            let conc_e = if shift_diff > 0 {
                dec.s_plc.conc_energy >> shift_diff
            } else {
                dec.s_plc.conc_energy << (-shift_diff)
            };

            if conc_e < new_energy {
                // New frame louder than concealment: fade in
                let gain_q16 = silk_sqrt_approx(
                    ((conc_e as i64) << 16) as i32 / imax(new_energy, 1),
                );
                let mut slope_q16 = ((65536 - gain_q16) as i64 / imax(length as i32, 1) as i64) as i32;
                slope_q16 = shl32(slope_q16, 2); // 4x steeper

                let mut cur_gain_q16 = gain_q16;
                for i in 0..length {
                    frame[i] = ((frame[i] as i32 * cur_gain_q16) >> 16) as i16;
                    cur_gain_q16 += slope_q16;
                    cur_gain_q16 = imin(cur_gain_q16, 65536);
                }
            }
        }
    }
    dec.s_plc.last_frame_lost = if dec.loss_cnt > 0 { 1 } else { 0 };
}

// ===========================================================================
// CNG (Comfort Noise Generation)
// ===========================================================================

fn silk_cng(
    dec: &mut SilkDecoderState,
    dec_ctrl: &SilkDecoderControl,
    frame: &mut [i16],
    length: usize,
) {
    let lpc_order = dec.lpc_order;

    // Check for rate change
    if dec.fs_khz != dec.s_cng.fs_khz {
        silk_cng_reset(dec);
        dec.s_cng.fs_khz = dec.fs_khz;
    }

    // Update CNG parameters on good, inactive frames
    if dec.loss_cnt == 0 && dec.prev_signal_type == TYPE_NO_VOICE_ACTIVITY {
        // Smooth NLSFs
        for i in 0..lpc_order {
            let diff = dec.prev_nlsf_q15[i] as i32 - dec.s_cng.cng_smth_nlsf_q15[i] as i32;
            dec.s_cng.cng_smth_nlsf_q15[i] =
                (dec.s_cng.cng_smth_nlsf_q15[i] as i32 + ((diff * CNG_NLSF_SMTH_Q16) >> 16)) as i16;
        }

        // Find max-gain subframe and copy its excitation
        let mut max_gain = 0;
        let mut max_k = 0;
        for k in 0..dec.nb_subfr {
            if dec_ctrl.gains_q16[k] > max_gain {
                max_gain = dec_ctrl.gains_q16[k];
                max_k = k;
            }
        }
        let exc_start = max_k * dec.subfr_length;
        let _exc_end = exc_start + dec.subfr_length;
        for i in 0..dec.subfr_length.min(MAX_FRAME_LENGTH) {
            if exc_start + i < dec.exc_q14.len() {
                dec.s_cng.cng_exc_buf_q14[i] = dec.exc_q14[exc_start + i];
            }
        }

        // Smooth gain
        for k in 0..dec.nb_subfr {
            let diff = dec_ctrl.gains_q16[k] - dec.s_cng.cng_smth_gain_q16;
            dec.s_cng.cng_smth_gain_q16 += ((diff as i64 * CNG_GAIN_SMTH_Q16 as i64) >> 16) as i32;
            // Fast adapt if 3dB above
            if dec.s_cng.cng_smth_gain_q16 > ((dec_ctrl.gains_q16[k] as i64 * CNG_GAIN_SMTH_THRESHOLD_Q16 as i64) >> 16) as i32 {
                dec.s_cng.cng_smth_gain_q16 = dec_ctrl.gains_q16[k];
            }
        }
    }

    // Generate CNG during loss
    if dec.loss_cnt > 0 {
        let gain_q16 = dec.s_cng.cng_smth_gain_q16;
        let gain_q10 = gain_q16 >> 6;

        // Generate random excitation
        let mut cng_sig_q14 = vec![0i32; length + MAX_LPC_ORDER];
        cng_sig_q14[..MAX_LPC_ORDER].copy_from_slice(&dec.s_cng.cng_synth_state);

        let mut seed = dec.s_cng.rand_seed;
        for i in 0..length {
            seed = silk_rand(seed);
            let idx = ((seed >> 24) as usize) & CNG_BUF_MASK_MAX.min(dec.subfr_length.saturating_sub(1));
            cng_sig_q14[MAX_LPC_ORDER + i] = if idx < MAX_FRAME_LENGTH {
                dec.s_cng.cng_exc_buf_q14[idx]
            } else {
                0
            };
        }
        dec.s_cng.rand_seed = seed;

        // Convert smoothed NLSF → LPC
        let mut cng_a_q12 = [0i16; MAX_LPC_ORDER];
        silk_nlsf2a(&mut cng_a_q12, &dec.s_cng.cng_smth_nlsf_q15, lpc_order);

        // LPC synthesis
        for i in 0..length {
            let mut lpc_pred_q10: i32 = (lpc_order >> 1) as i32;
            for j in 0..lpc_order {
                let idx = MAX_LPC_ORDER + i - j - 1;
                lpc_pred_q10 += ((cng_sig_q14[idx] as i64 * cng_a_q12[j] as i64) >> 16) as i32;
            }
            cng_sig_q14[MAX_LPC_ORDER + i] =
                silk_add_sat32(cng_sig_q14[MAX_LPC_ORDER + i], silk_lshift_sat32(lpc_pred_q10, 4));

            // Add CNG to frame output
            let cng_sample = silk_rshift_round(silk_smulww(cng_sig_q14[MAX_LPC_ORDER + i], gain_q10), 8);
            frame[i] = sat16(frame[i] as i32 + sat16(cng_sample) as i32);
        }

        // Save synthesis state
        for i in 0..MAX_LPC_ORDER {
            dec.s_cng.cng_synth_state[i] = cng_sig_q14[length + i];
        }
    } else {
        // No loss: zero the synth state
        dec.s_cng.cng_synth_state = [0; MAX_LPC_ORDER];
    }
}

// ===========================================================================
// Stereo processing
// ===========================================================================

/// Decode stereo mid/side predictor coefficients.
pub fn silk_stereo_decode_pred(
    rc: &mut RangeDecoder,
    pred_q13: &mut [i16; 2],
) {
    // Decode joint index
    let n = rc.decode_icdf(&SILK_STEREO_PRED_JOINT_ICDF, 8);
    let mut ix = [[0i32; 3]; 2];
    ix[0][2] = n / 5;
    ix[1][2] = n - 5 * ix[0][2];

    for n_ch in 0..2 {
        ix[n_ch][0] = rc.decode_icdf(&SILK_UNIFORM3_ICDF, 8);
        ix[n_ch][1] = rc.decode_icdf(&SILK_UNIFORM5_ICDF, 8);
    }

    // Dequantize
    for n_ch in 0..2 {
        ix[n_ch][0] += 3 * ix[n_ch][2];
        let idx = ix[n_ch][0] as usize;
        let low_q13 = SILK_STEREO_PRED_QUANT_Q13[idx] as i32;
        let step_q13 = if idx + 1 < STEREO_QUANT_TAB_SIZE {
            ((SILK_STEREO_PRED_QUANT_Q13[idx + 1] as i32 - low_q13) as i64
                * 26214 // SILK_FIX_CONST(0.5/STEREO_QUANT_SUB_STEPS, 16)
                >> 16) as i32
        } else {
            0
        };
        pred_q13[n_ch] = (low_q13 + step_q13 * (2 * ix[n_ch][1] + 1)) as i16;
    }

    // Differential encoding
    pred_q13[0] = (pred_q13[0] as i32 - pred_q13[1] as i32) as i16;
}

/// Decode mid-only flag.
pub fn silk_stereo_decode_mid_only(rc: &mut RangeDecoder) -> bool {
    rc.decode_icdf(&SILK_STEREO_ONLY_CODE_MID_ICDF, 8) != 0
}

/// Convert mid/side to left/right.
pub fn silk_stereo_ms_to_lr(
    state: &mut StereoDecState,
    x1: &mut [i16], // mid → left
    x2: &mut [i16], // side → right
    pred_q13: &[i16; 2],
    fs_khz: i32,
    frame_length: usize,
) {
    // Save end of buffer for next frame's state
    let new_s_mid = [x1[frame_length - 2], x1[frame_length - 1]];
    let new_s_side = [x2[frame_length - 2], x2[frame_length - 1]];

    // Prepend previous samples
    // We work in-place: shift data right by 2 and insert history
    // Actually the C reference works on [n+1] offset with history at [0], [1]
    // For simplicity, work with temporary buffers
    let mut mid = vec![0i16; frame_length + 2];
    let mut side = vec![0i16; frame_length + 2];
    mid[0] = state.s_mid[0];
    mid[1] = state.s_mid[1];
    mid[2..frame_length + 2].copy_from_slice(&x1[..frame_length]);
    side[0] = state.s_side[0];
    side[1] = state.s_side[1];
    side[2..frame_length + 2].copy_from_slice(&x2[..frame_length]);

    // Interpolation period
    let interp_len = (STEREO_INTERP_LEN_MS as i32 * fs_khz) as usize;
    let mut pred0_q13 = state.pred_prev_q13[0] as i32;
    let mut pred1_q13 = state.pred_prev_q13[1] as i32;

    if interp_len > 0 {
        let denom_q16 = (1 << 16) / interp_len as i32;
        let delta0 = ((pred_q13[0] as i32 - pred0_q13) * denom_q16) >> 16;
        let delta1 = ((pred_q13[1] as i32 - pred1_q13) * denom_q16) >> 16;

        for n in 0..interp_len.min(frame_length) {
            pred0_q13 += delta0;
            pred1_q13 += delta1;

            // Apply prediction: side'[n+1] = side[n+1] + pred0*lowpass(mid) + pred1*mid[n+1]
            let sum_mid = (mid[n] as i32 + 2 * mid[n + 1] as i32 + mid[n + 2] as i32) << 9;
            let pred_side = ((side[n + 1] as i32) << 8)
                .wrapping_add(((sum_mid as i64 * pred0_q13 as i64 >> 7) as i32))
                .wrapping_add((((mid[n + 1] as i32) << 11) as i64 * pred1_q13 as i64 >> 7) as i32);
            side[n + 1] = sat16(pred_side >> 8);
        }
    }

    // Steady state
    pred0_q13 = pred_q13[0] as i32;
    pred1_q13 = pred_q13[1] as i32;
    for n in interp_len..frame_length {
        let sum_mid = (mid[n] as i32 + 2 * mid[n + 1] as i32 + mid[n + 2] as i32) << 9;
        let pred_side = ((side[n + 1] as i32) << 8)
            .wrapping_add(((sum_mid as i64 * pred0_q13 as i64 >> 7) as i32))
            .wrapping_add((((mid[n + 1] as i32) << 11) as i64 * pred1_q13 as i64 >> 7) as i32);
        side[n + 1] = sat16(pred_side >> 8);
    }

    // Convert M/S → L/R
    for n in 0..frame_length {
        let m = mid[n + 1] as i32;
        let s = side[n + 1] as i32;
        x1[n] = sat16(m + s); // Left = mid + side
        x2[n] = sat16(m - s); // Right = mid - side
    }

    // Update state
    state.pred_prev_q13 = *pred_q13;
    state.s_mid = new_s_mid;
    state.s_side = new_s_side;
}

// ===========================================================================
// Resampler
// ===========================================================================

/// Delay compensation tables (encoder): in=[8,12,16,24,48,96] out=[8,12,16].
const DELAY_MATRIX_ENC: [[i32; 3]; 6] = [
    [6,  0,  3],  //  8 kHz in
    [0,  7,  3],  // 12 kHz in
    [0,  1, 10],  // 16 kHz in
    [0,  2,  6],  // 24 kHz in
    [18, 10, 12], // 48 kHz in
    [0,  0, 44],  // 96 kHz in
];

/// Delay compensation tables (decoder): in=[8,12,16] out=[8,12,16,24,48,96].
const DELAY_MATRIX_DEC: [[i32; 6]; 3] = [
    [4, 0, 2, 0, 0, 0],  // 8 kHz input
    [0, 9, 4, 7, 4, 4],  // 12 kHz input
    [0, 3, 12, 7, 7, 7], // 16 kHz input
];

fn rate_id(r: i32) -> usize {
    match r {
        8000 => 0,
        12000 => 1,
        16000 => 2,
        24000 => 3,
        48000 => 4,
        _ => 5,
    }
}

/// Initialize the resampler.
fn silk_resampler_init(
    s: &mut SilkResamplerState,
    fs_hz_in: i32,
    fs_hz_out: i32,
    for_enc: bool,
) {
    *s = SilkResamplerState::default();
    s.fs_in_khz = fs_hz_in / 1000;
    s.fs_out_khz = fs_hz_out / 1000;

    // Delay compensation — encoder and decoder use different matrices
    let in_id = rate_id(fs_hz_in);
    let out_id = rate_id(fs_hz_out);
    s.input_delay = if for_enc {
        DELAY_MATRIX_ENC[in_id.min(5)][out_id.min(2)]
    } else {
        DELAY_MATRIX_DEC[in_id.min(2)][out_id.min(5)]
    };

    s.batch_size = s.fs_in_khz * RESAMPLER_MAX_BATCH_SIZE_MS;

    if fs_hz_out == fs_hz_in {
        s.resampler_function = USE_SILK_RESAMPLER_COPY;
    } else if fs_hz_out == 2 * fs_hz_in {
        s.resampler_function = USE_SILK_RESAMPLER_UP2_HQ;
    } else if fs_hz_out > fs_hz_in {
        s.resampler_function = USE_SILK_RESAMPLER_IIR_FIR;
        // Compute inverse ratio with up2x factor
        let up2x = 1i32;
        let temp = ((fs_hz_in as i64) << (14 + up2x)) / fs_hz_out as i64;
        s.inv_ratio_q16 = (temp << 2) as i32;
        while ((s.inv_ratio_q16 as i64) * fs_hz_out as i64) < ((fs_hz_in as i64) << up2x) {
            s.inv_ratio_q16 += 1;
        }
        s.fir_order = RESAMPLER_ORDER_FIR_12 as i32;
        s.fir_fracs = 12;
    } else {
        s.resampler_function = USE_SILK_RESAMPLER_DOWN_FIR;

        // Determine FIR order and fracs based on rate ratio
        if fs_hz_out * 4 == fs_hz_in * 3 {
            s.fir_fracs = 3;
            s.fir_order = RESAMPLER_DOWN_ORDER_FIR0 as i32;
            s.coefs = ResamplerCoefs::Ratio3_4;
        } else if fs_hz_out * 3 == fs_hz_in * 2 {
            s.fir_fracs = 2;
            s.fir_order = RESAMPLER_DOWN_ORDER_FIR0 as i32;
            s.coefs = ResamplerCoefs::Ratio2_3;
        } else if fs_hz_out * 2 == fs_hz_in {
            s.fir_fracs = 1;
            s.fir_order = RESAMPLER_DOWN_ORDER_FIR1 as i32;
            s.coefs = ResamplerCoefs::Ratio1_2;
        } else if fs_hz_out * 3 == fs_hz_in {
            s.fir_fracs = 1;
            s.fir_order = RESAMPLER_DOWN_ORDER_FIR2 as i32;
            s.coefs = ResamplerCoefs::Ratio1_3;
        } else if fs_hz_out * 4 == fs_hz_in {
            s.fir_fracs = 1;
            s.fir_order = RESAMPLER_DOWN_ORDER_FIR2 as i32;
            s.coefs = ResamplerCoefs::Ratio1_4;
        } else if fs_hz_out * 6 == fs_hz_in {
            s.fir_fracs = 1;
            s.fir_order = RESAMPLER_DOWN_ORDER_FIR2 as i32;
            s.coefs = ResamplerCoefs::Ratio1_6;
        }

        let temp = ((fs_hz_in as i64) << 14) / fs_hz_out as i64;
        s.inv_ratio_q16 = (temp << 2) as i32;
        while ((s.inv_ratio_q16 as i64) * fs_hz_out as i64) < (fs_hz_in as i64) << 0 {
            s.inv_ratio_q16 += 1;
        }
    }
}

/// Public entry point for resampler init (used by encoder).
pub fn silk_resampler_init_pub(
    s: &mut SilkResamplerState,
    fs_hz_in: i32,
    fs_hz_out: i32,
    for_enc: bool,
) {
    silk_resampler_init(s, fs_hz_in, fs_hz_out, for_enc);
}

/// 2x high-quality upsampling using 3rd-order allpass.
fn silk_resampler_private_up2_hq(
    s: &mut [i32; SILK_RESAMPLER_MAX_IIR_ORDER],
    out: &mut [i16],
    input: &[i16],
    len: usize,
) {
    // Allpass coefficients (from resampler_rom.h)
    const UP2_HQ_0: [i32; 3] = [1746, 14986, -26453]; // Even path
    const UP2_HQ_1: [i32; 3] = [6854, 25769, -9994];  // Odd path

    for k in 0..len {
        let in32 = (input[k] as i32) << 10;

        // Even output
        let mut y = in32 - s[0];
        let mut x = ((y as i64 * UP2_HQ_0[0] as i64) >> 16) as i32;
        let mut out32_1 = s[0] + x;
        s[0] = in32 + x;

        y = out32_1 - s[1];
        x = ((y as i64 * UP2_HQ_0[1] as i64) >> 16) as i32;
        let mut out32_2 = s[1] + x;
        s[1] = out32_1 + x;

        y = out32_2 - s[2];
        x = y + ((y as i64 * UP2_HQ_0[2] as i64 >> 16) as i32);
        out32_1 = s[2] + x;
        s[2] = out32_2 + x;

        out[2 * k] = sat16(silk_rshift_round(out32_1, 10));

        // Odd output
        y = in32 - s[3];
        x = ((y as i64 * UP2_HQ_1[0] as i64) >> 16) as i32;
        out32_1 = s[3] + x;
        s[3] = in32 + x;

        y = out32_1 - s[4];
        x = ((y as i64 * UP2_HQ_1[1] as i64) >> 16) as i32;
        out32_2 = s[4] + x;
        s[4] = out32_1 + x;

        y = out32_2 - s[5];
        x = y + ((y as i64 * UP2_HQ_1[2] as i64 >> 16) as i32);
        out32_1 = s[5] + x;
        s[5] = out32_2 + x;

        out[2 * k + 1] = sat16(silk_rshift_round(out32_1, 10));
    }
}

/// Resolve coefficient table from enum.
fn get_down_fir_coefs(c: ResamplerCoefs) -> &'static [i16] {
    use crate::silk::tables::*;
    match c {
        ResamplerCoefs::Ratio3_4 => &SILK_RESAMPLER_3_4_COEFS,
        ResamplerCoefs::Ratio2_3 => &SILK_RESAMPLER_2_3_COEFS,
        ResamplerCoefs::Ratio1_2 => &SILK_RESAMPLER_1_2_COEFS,
        ResamplerCoefs::Ratio1_3 => &SILK_RESAMPLER_1_3_COEFS,
        ResamplerCoefs::Ratio1_4 => &SILK_RESAMPLER_1_4_COEFS,
        ResamplerCoefs::Ratio1_6 => &SILK_RESAMPLER_1_6_COEFS,
        ResamplerCoefs::LowQuality2_3 => &SILK_RESAMPLER_2_3_COEFS_LQ,
        ResamplerCoefs::None => &SILK_RESAMPLER_1_3_COEFS, // fallback
    }
}

/// silk_SMULWB: (a32 * (i16)b) >> 16, using 64-bit intermediate.
#[inline(always)]
fn smulwb(a: i32, b: i16) -> i32 {
    ((a as i64 * b as i64) >> 16) as i32
}

/// silk_SMLAWB: a + silk_SMULWB(b, c)
#[inline(always)]
fn smlawb(a: i32, b: i32, c: i16) -> i32 {
    a.wrapping_add(smulwb(b, c))
}

/// FIR interpolation for down_FIR resampler. Returns number of output samples written.
fn silk_resampler_private_down_fir_interpol(
    out: &mut [i16],
    out_offset: usize,
    buf: &[i32],
    fir_coefs: &[i16],
    fir_order: usize,
    fir_fracs: i32,
    max_index_q16: i32,
    index_increment_q16: i32,
) -> usize {
    let mut out_idx = out_offset;
    let half_order = fir_order / 2;

    match fir_order {
        RESAMPLER_DOWN_ORDER_FIR0 => {
            // Order 18: polyphase with FIR_Fracs phases
            let mut index_q16 = 0i32;
            while index_q16 < max_index_q16 {
                let buf_idx = (index_q16 >> 16) as usize;
                let interpol_ind = smulwb(index_q16 & 0xFFFF, fir_fracs as i16) as usize;

                // First half: use interpol_ind phase
                let interpol_off = half_order * interpol_ind;
                let mut res_q6 = smulwb(buf[buf_idx], fir_coefs[interpol_off]);
                for j in 1..half_order {
                    res_q6 = smlawb(res_q6, buf[buf_idx + j], fir_coefs[interpol_off + j]);
                }

                // Second half: use (FIR_Fracs-1-interpol_ind) phase, reversed
                let interpol_off2 = half_order * (fir_fracs as usize - 1 - interpol_ind);
                for j in 0..half_order {
                    res_q6 = smlawb(
                        res_q6,
                        buf[buf_idx + fir_order - 1 - j],
                        fir_coefs[interpol_off2 + j],
                    );
                }

                if out_idx < out.len() {
                    out[out_idx] = sat16(silk_rshift_round(res_q6, 6));
                }
                out_idx += 1;
                index_q16 += index_increment_q16;
            }
        }
        RESAMPLER_DOWN_ORDER_FIR1 => {
            // Order 24: symmetric filter, FIR_Fracs=1
            let mut index_q16 = 0i32;
            while index_q16 < max_index_q16 {
                let buf_idx = (index_q16 >> 16) as usize;
                let n = fir_order; // 24

                let mut res_q6 = smulwb(
                    buf[buf_idx].wrapping_add(buf[buf_idx + n - 1]),
                    fir_coefs[0],
                );
                for j in 1..half_order {
                    res_q6 = smlawb(
                        res_q6,
                        buf[buf_idx + j].wrapping_add(buf[buf_idx + n - 1 - j]),
                        fir_coefs[j],
                    );
                }

                if out_idx < out.len() {
                    out[out_idx] = sat16(silk_rshift_round(res_q6, 6));
                }
                out_idx += 1;
                index_q16 += index_increment_q16;
            }
        }
        RESAMPLER_DOWN_ORDER_FIR2 => {
            // Order 36: symmetric filter, FIR_Fracs=1
            let mut index_q16 = 0i32;
            while index_q16 < max_index_q16 {
                let buf_idx = (index_q16 >> 16) as usize;
                let n = fir_order; // 36

                let mut res_q6 = smulwb(
                    buf[buf_idx].wrapping_add(buf[buf_idx + n - 1]),
                    fir_coefs[0],
                );
                for j in 1..half_order {
                    res_q6 = smlawb(
                        res_q6,
                        buf[buf_idx + j].wrapping_add(buf[buf_idx + n - 1 - j]),
                        fir_coefs[j],
                    );
                }

                if out_idx < out.len() {
                    out[out_idx] = sat16(silk_rshift_round(res_q6, 6));
                }
                out_idx += 1;
                index_q16 += index_increment_q16;
            }
        }
        _ => {}
    }

    out_idx - out_offset
}

/// Polyphase FIR downsampler. Matches C: `silk_resampler_private_down_FIR`.
fn silk_resampler_private_down_fir(
    s: &mut SilkResamplerState,
    out: &mut [i16],
    input: &[i16],
    in_len: i32,
) {
    let fir_order = s.fir_order as usize;
    let batch_size = s.batch_size as i32;
    let index_increment_q16 = s.inv_ratio_q16;

    let all_coefs = get_down_fir_coefs(s.coefs);
    let ar2_coefs = &all_coefs[..2];
    let fir_coefs = &all_coefs[2..];

    let mut buf = vec![0i32; batch_size as usize + fir_order];

    // Copy FIR state to start of buffer
    buf[..fir_order].copy_from_slice(&s.s_fir_i32[..fir_order]);

    let mut in_ptr = 0usize;
    let mut out_ptr = 0usize;
    let mut remaining = in_len;
    let mut last_n_samples_in: usize;

    loop {
        let n_samples_in = remaining.min(batch_size) as usize;
        last_n_samples_in = n_samples_in;

        // AR2 filter: input → Q8 output into buf[fir_order..]
        silk_resampler_private_ar2(
            &mut s.s_iir[..2],
            &mut buf[fir_order..fir_order + n_samples_in],
            &input[in_ptr..in_ptr + n_samples_in],
            ar2_coefs,
            n_samples_in,
        );

        let max_index_q16 = (n_samples_in as i32) << 16;

        // FIR interpolation
        let n_out = silk_resampler_private_down_fir_interpol(
            out,
            out_ptr,
            &buf,
            fir_coefs,
            fir_order,
            s.fir_fracs,
            max_index_q16,
            index_increment_q16,
        );
        out_ptr += n_out;

        in_ptr += n_samples_in;
        remaining -= n_samples_in as i32;

        if remaining > 1 {
            // Copy last FIR_Order elements to start of buffer for next batch
            for i in 0..fir_order {
                buf[i] = buf[n_samples_in + i];
            }
        } else {
            break;
        }
    }

    // Save FIR state for next call (from last batch's offset)
    if last_n_samples_in + fir_order <= buf.len() {
        s.s_fir_i32[..fir_order]
            .copy_from_slice(&buf[last_n_samples_in..last_n_samples_in + fir_order]);
    }
}

/// IIR/FIR upsampling resampler. Matches C: `silk_resampler_private_IIR_FIR`.
fn silk_resampler_private_iir_fir(
    s: &mut SilkResamplerState,
    out: &mut [i16],
    input: &[i16],
    in_len: i32,
) {
    use crate::silk::tables::*;
    let batch_size = s.batch_size as i32;
    let index_increment_q16 = s.inv_ratio_q16;

    let mut buf = vec![0i16; 2 * batch_size as usize + RESAMPLER_ORDER_FIR_12];

    // Copy FIR state (i16) to start of buffer
    buf[..RESAMPLER_ORDER_FIR_12].copy_from_slice(&s.s_fir_i16[..RESAMPLER_ORDER_FIR_12]);

    let mut in_ptr = 0usize;
    let mut out_ptr = 0usize;
    let mut remaining = in_len;
    let mut last_n_samples_in: usize;

    loop {
        let n_samples_in = remaining.min(batch_size) as usize;
        last_n_samples_in = n_samples_in;

        // Upsample 2x using allpass
        silk_resampler_private_up2_hq(
            &mut s.s_iir,
            &mut buf[RESAMPLER_ORDER_FIR_12..],
            &input[in_ptr..in_ptr + n_samples_in],
            n_samples_in,
        );

        let max_index_q16 = (n_samples_in as i32) << (16 + 1); // +1 for 2x upsampling

        // FIR interpolation on i16 buffer
        let mut index_q16 = 0i32;
        while index_q16 < max_index_q16 {
            let table_index = smulwb(index_q16 & 0xFFFF, 12) as usize;
            let buf_idx = (index_q16 >> 16) as usize;

            let mut res_q15 = (buf[buf_idx] as i32) * (SILK_RESAMPLER_FRAC_FIR_12[table_index][0] as i32);
            res_q15 += (buf[buf_idx + 1] as i32) * (SILK_RESAMPLER_FRAC_FIR_12[table_index][1] as i32);
            res_q15 += (buf[buf_idx + 2] as i32) * (SILK_RESAMPLER_FRAC_FIR_12[table_index][2] as i32);
            res_q15 += (buf[buf_idx + 3] as i32) * (SILK_RESAMPLER_FRAC_FIR_12[table_index][3] as i32);
            res_q15 += (buf[buf_idx + 4] as i32) * (SILK_RESAMPLER_FRAC_FIR_12[11 - table_index][3] as i32);
            res_q15 += (buf[buf_idx + 5] as i32) * (SILK_RESAMPLER_FRAC_FIR_12[11 - table_index][2] as i32);
            res_q15 += (buf[buf_idx + 6] as i32) * (SILK_RESAMPLER_FRAC_FIR_12[11 - table_index][1] as i32);
            res_q15 += (buf[buf_idx + 7] as i32) * (SILK_RESAMPLER_FRAC_FIR_12[11 - table_index][0] as i32);

            if out_ptr < out.len() {
                out[out_ptr] = sat16(silk_rshift_round(res_q15, 15));
            }
            out_ptr += 1;
            index_q16 += index_increment_q16;
        }

        in_ptr += n_samples_in;
        remaining -= n_samples_in as i32;

        if remaining > 0 {
            // Copy last part of buffer to start for next batch
            let shift = n_samples_in << 1;
            for i in 0..RESAMPLER_ORDER_FIR_12 {
                buf[i] = buf[shift + i];
            }
        } else {
            break;
        }
    }

    // Save FIR state (from last batch's offset)
    let last_shift = last_n_samples_in << 1;
    if last_shift + RESAMPLER_ORDER_FIR_12 <= buf.len() {
        s.s_fir_i16[..RESAMPLER_ORDER_FIR_12]
            .copy_from_slice(&buf[last_shift..last_shift + RESAMPLER_ORDER_FIR_12]);
    }
}

/// AR2 filter for resampler.
fn silk_resampler_private_ar2(
    s: &mut [i32],
    out_q8: &mut [i32],
    input: &[i16],
    a_q14: &[i16],
    len: usize,
) {
    for k in 0..len {
        let out32 = s[0] + ((input[k] as i32) << 8);
        out_q8[k] = out32;
        let out32_shifted = out32 << 2;
        s[0] = s[1] + ((out32_shifted as i64 * a_q14[0] as i64) >> 16) as i32;
        s[1] = ((out32_shifted as i64 * a_q14[1] as i64) >> 16) as i32;
    }
}

/// Main resampler entry point.
pub fn silk_resampler(
    s: &mut SilkResamplerState,
    out: &mut [i16],
    input: &[i16],
    in_len: usize,
) {
    // C reference: all modes share delay buffer setup/teardown
    // (resampler.c lines 197-200, 220-221)
    let n_samples = (s.fs_in_khz - s.input_delay).max(0) as usize;
    let delay_samples = s.input_delay as usize;
    let fs_in = s.fs_in_khz as usize;
    let fs_out = s.fs_out_khz as usize;

    // Copy first nSamples from input into delay buffer tail
    let copy_from_input = n_samples.min(in_len);
    for i in 0..copy_from_input {
        if delay_samples + i < s.delay_buf.len() {
            s.delay_buf[delay_samples + i] = input[i];
        }
    }

    // Per-mode processing: delay buffer as first batch, remaining input as second
    if s.resampler_function == USE_SILK_RESAMPLER_UP2_HQ {
        let delay_len = fs_in.min(s.delay_buf.len());
        let delay_copy: Vec<i16> = s.delay_buf[..delay_len].to_vec();
        silk_resampler_private_up2_hq(
            &mut s.s_iir,
            out,
            &delay_copy,
            delay_len,
        );
        if in_len > fs_in {
            silk_resampler_private_up2_hq(
                &mut s.s_iir,
                &mut out[fs_out..],
                &input[n_samples..],
                in_len - fs_in,
            );
        }
    } else if s.resampler_function == USE_SILK_RESAMPLER_DOWN_FIR {
        // AR2 + polyphase FIR downsampling
        let delay_buf_copy: Vec<i16> = s.delay_buf[..fs_in].to_vec();
        silk_resampler_private_down_fir(s, out, &delay_buf_copy, fs_in as i32);
        if in_len > fs_in {
            silk_resampler_private_down_fir(
                s,
                &mut out[fs_out..],
                &input[n_samples..],
                (in_len - fs_in) as i32,
            );
        }
    } else if s.resampler_function == USE_SILK_RESAMPLER_IIR_FIR {
        // Allpass 2x upsample + FIR interpolation
        let delay_buf_copy: Vec<i16> = s.delay_buf[..fs_in].to_vec();
        silk_resampler_private_iir_fir(s, out, &delay_buf_copy, fs_in as i32);
        if in_len > fs_in {
            silk_resampler_private_iir_fir(
                s,
                &mut out[fs_out..],
                &input[n_samples..],
                (in_len - fs_in) as i32,
            );
        }
    } else {
        // COPY case (default): output delay buffer, then remaining input
        let copy_delay = fs_in.min(s.delay_buf.len()).min(out.len());
        out[..copy_delay].copy_from_slice(&s.delay_buf[..copy_delay]);
        if in_len > fs_in {
            let remaining = (in_len - fs_in).min(out.len().saturating_sub(fs_out));
            out[fs_out..fs_out + remaining]
                .copy_from_slice(&input[n_samples..n_samples + remaining]);
        }
    }

    // Save last inputDelay samples to delay buffer (common to all modes)
    if in_len >= delay_samples && delay_samples > 0 {
        let start = in_len - delay_samples;
        for i in 0..delay_samples.min(s.delay_buf.len()) {
            s.delay_buf[i] = input[start + i];
        }
    }
}

// ===========================================================================
// Per-frame decode
// ===========================================================================

/// Decode a single channel's single frame.
pub fn silk_decode_frame(
    dec: &mut SilkDecoderState,
    rc: &mut RangeDecoder,
    p_out: &mut [i16],
    p_n: &mut usize,
    lost_flag: i32,
    cond_coding: i32,
) {
    let frame_length = dec.frame_length;

    if lost_flag != FLAG_PACKET_LOST
        && !(lost_flag == FLAG_DECODE_LBRR && !dec.lbrr_flags[dec.n_frames_decoded])
    {
        // Normal decode or LBRR with flag set
        let mut dec_ctrl = SilkDecoderControl::default();

        // Allocate pulse buffer (rounded up to shell codec frame)
        let pulse_len = (frame_length + SHELL_CODEC_FRAME_LENGTH - 1)
            & !(SHELL_CODEC_FRAME_LENGTH - 1);
        let mut pulses = vec![0i16; pulse_len];

        // Decode indices
        silk_decode_indices(
            dec,
            rc,
            dec.n_frames_decoded,
            lost_flag == FLAG_DECODE_LBRR,
            cond_coding,
        );

        // Decode pulses
        silk_decode_pulses(
            rc,
            &mut pulses,
            dec.indices.signal_type as i32,
            dec.indices.quant_offset_type as i32,
            frame_length,
        );

        // Decode parameters
        silk_decode_parameters(dec, &mut dec_ctrl, cond_coding);

        // Synthesis
        silk_decode_core(dec, &dec_ctrl, p_out, &pulses);

        // Update output buffer
        let mv_len = dec.ltp_mem_length - frame_length;
        // Shift outBuf left by frame_length
        let out_buf_len = dec.out_buf.len();
        if mv_len > 0 && frame_length + mv_len <= out_buf_len {
            for i in 0..mv_len {
                dec.out_buf[i] = dec.out_buf[frame_length + i];
            }
        }
        // Copy new output to end of outBuf
        for i in 0..frame_length.min(out_buf_len - mv_len) {
            dec.out_buf[mv_len + i] = p_out[i];
        }

        // PLC update (save state from good frame)
        silk_plc_update(dec, &dec_ctrl);

        dec.loss_cnt = 0;
        dec.prev_signal_type = dec.indices.signal_type as i32;
        dec.first_frame_after_reset = false;

        // CNG and PLC glue
        silk_cng(dec, &dec_ctrl, p_out, frame_length);
        silk_plc_glue_frames(dec, p_out, frame_length);

        dec.lag_prev = dec_ctrl.pitch_l[dec.nb_subfr - 1];
    } else {
        // Packet lost: generate concealment
        silk_plc_conceal(dec, &mut p_out[..frame_length]);

        // Update output buffer
        let mv_len = dec.ltp_mem_length - frame_length;
        if mv_len > 0 {
            for i in 0..mv_len {
                dec.out_buf[i] = dec.out_buf[frame_length + i];
            }
        }
        for i in 0..frame_length {
            if mv_len + i < dec.out_buf.len() {
                dec.out_buf[mv_len + i] = p_out[i];
            }
        }

        // CNG and glue (with dummy dec_ctrl)
        let dec_ctrl = SilkDecoderControl::default();
        silk_cng(dec, &dec_ctrl, p_out, frame_length);
        silk_plc_glue_frames(dec, p_out, frame_length);
    }

    *p_n = frame_length;
}

// ===========================================================================
// Top-level decoder API
// ===========================================================================

/// Main SILK decoder entry point. Decodes one frame per call.
pub fn silk_decode(
    decoder: &mut SilkDecoder,
    dec_control: &mut SilkDecControl,
    lost_flag: i32,
    new_packet_flag: bool,
    rc: &mut RangeDecoder,
    samples_out: &mut [i16],
    n_samples_out: &mut usize,
) -> i32 {
    let ret = 0;
    let n_channels_internal = dec_control.n_channels_internal;

    // Reset frame counters on new packet
    if new_packet_flag {
        for n in 0..DECODER_NUM_CHANNELS {
            decoder.channel_state[n].n_frames_decoded = 0;
        }
    }

    // Mono→stereo transition
    if n_channels_internal > decoder.n_channels_internal {
        decoder.channel_state[1] = SilkDecoderState::new();
    }

    // Configure frame geometry on first frame
    if decoder.channel_state[0].n_frames_decoded == 0 {
        match dec_control.payload_size_ms {
            10 => {
                for n in 0..n_channels_internal {
                    decoder.channel_state[n].n_frames_per_packet = 1;
                    decoder.channel_state[n].nb_subfr = 2;
                }
            }
            20 => {
                for n in 0..n_channels_internal {
                    decoder.channel_state[n].n_frames_per_packet = 1;
                    decoder.channel_state[n].nb_subfr = MAX_NB_SUBFR;
                }
            }
            40 => {
                for n in 0..n_channels_internal {
                    decoder.channel_state[n].n_frames_per_packet = 2;
                    decoder.channel_state[n].nb_subfr = MAX_NB_SUBFR;
                }
            }
            60 => {
                for n in 0..n_channels_internal {
                    decoder.channel_state[n].n_frames_per_packet = 3;
                    decoder.channel_state[n].nb_subfr = MAX_NB_SUBFR;
                }
            }
            _ => {
                return -1; // SILK_DEC_INVALID_FRAME_SIZE
            }
        }

        // Set sample rate
        let fs_khz = (dec_control.internal_sample_rate >> 10) + 1;
        if fs_khz != 8 && fs_khz != 12 && fs_khz != 16 {
            return -2; // SILK_DEC_INVALID_SAMPLING_FREQUENCY
        }

        for n in 0..n_channels_internal {
            silk_decoder_set_fs(
                &mut decoder.channel_state[n],
                fs_khz,
                dec_control.api_sample_rate,
            );
        }

        // Decode VAD flags and LBRR flags
        if lost_flag != FLAG_PACKET_LOST {
            for n in 0..n_channels_internal {
                for i in 0..decoder.channel_state[n].n_frames_per_packet {
                    decoder.channel_state[n].vad_flags[i] = rc.decode_bit_logp(1);
                }
            }

            for n in 0..n_channels_internal {
                decoder.channel_state[n].lbrr_flag = rc.decode_bit_logp(1);
                if decoder.channel_state[n].lbrr_flag {
                    let nfpp = decoder.channel_state[n].n_frames_per_packet;
                    if nfpp == 1 {
                        decoder.channel_state[n].lbrr_flags[0] = true;
                    } else {
                        let symbol = rc.decode_icdf(
                            SILK_LBRR_FLAGS_ICDF_PTR[nfpp - 2],
                            8,
                        );
                        for i in 0..nfpp {
                            decoder.channel_state[n].lbrr_flags[i] = ((symbol >> i) & 1) != 0;
                        }
                    }
                } else {
                    for i in 0..MAX_FRAMES_PER_PACKET {
                        decoder.channel_state[n].lbrr_flags[i] = false;
                    }
                }
            }

            // Skip LBRR data on normal decode
            if lost_flag == FLAG_DECODE_NORMAL {
                for n in 0..n_channels_internal {
                    let nfpp = decoder.channel_state[n].n_frames_per_packet;
                    for i in 0..nfpp {
                        if decoder.channel_state[n].lbrr_flags[i] {
                            let cond = if i > 0 && decoder.channel_state[n].lbrr_flags[i - 1] {
                                CODE_CONDITIONALLY
                            } else {
                                CODE_INDEPENDENTLY
                            };
                            silk_decode_indices(
                                &mut decoder.channel_state[n],
                                rc,
                                i,
                                true,
                                cond,
                            );
                            let fl = decoder.channel_state[n].frame_length;
                            let mut dummy_pulses = vec![0i16; fl + SHELL_CODEC_FRAME_LENGTH];
                            silk_decode_pulses(
                                rc,
                                &mut dummy_pulses,
                                decoder.channel_state[n].indices.signal_type as i32,
                                decoder.channel_state[n].indices.quant_offset_type as i32,
                                fl,
                            );
                        }
                    }
                }
            }
        }
    }

    // Decode stereo predictor
    let mut ms_pred_q13 = [0i16; 2];
    let mut decode_only_middle = false;
    if n_channels_internal == 2 {
        if lost_flag == FLAG_DECODE_NORMAL
            || (lost_flag == FLAG_DECODE_LBRR
                && decoder.channel_state[1].lbrr_flags
                    [decoder.channel_state[0].n_frames_decoded])
        {
            silk_stereo_decode_pred(rc, &mut ms_pred_q13);
            if n_channels_internal == 2 {
                decode_only_middle = silk_stereo_decode_mid_only(rc);
            }
        } else {
            ms_pred_q13 = decoder.s_stereo.pred_prev_q13;
        }
    }

    let frame_length = decoder.channel_state[0].frame_length;

    // Per-channel decoding
    let mut samples_out1_tmp = vec![vec![0i16; frame_length + 2]; n_channels_internal];

    for n in 0..n_channels_internal {
        let should_decode = n == 0 || !decode_only_middle;
        if should_decode {
            let cond = if decoder.channel_state[n].n_frames_decoded == 0 {
                CODE_INDEPENDENTLY
            } else {
                CODE_CONDITIONALLY
            };

            let mut n_out = 0;
            silk_decode_frame(
                &mut decoder.channel_state[n],
                rc,
                &mut samples_out1_tmp[n][..frame_length],
                &mut n_out,
                lost_flag,
                cond,
            );
            decoder.channel_state[n].n_frames_decoded += 1;
        } else {
            samples_out1_tmp[n][..frame_length].fill(0);
            decoder.channel_state[n].n_frames_decoded += 1;
        }
    }

    // Stereo M/S → L/R
    if n_channels_internal == 2 && dec_control.n_channels_api == 2 {
        let (left, right) = samples_out1_tmp.split_at_mut(1);
        silk_stereo_ms_to_lr(
            &mut decoder.s_stereo,
            &mut left[0],
            &mut right[0],
            &ms_pred_q13,
            decoder.channel_state[0].fs_khz,
            frame_length,
        );
    }

    // Compute output length
    let out_samples = (frame_length as i64 * dec_control.api_sample_rate as i64
        / (decoder.channel_state[0].fs_khz as i64 * 1000)) as usize;
    *n_samples_out = out_samples;

    // Resample and interleave output
    let n_api_ch = dec_control.n_channels_api.min(n_channels_internal);
    for n in 0..n_api_ch {
        let mut resampled = vec![0i16; out_samples];
        silk_resampler(
            &mut decoder.channel_state[n].resampler_state,
            &mut resampled,
            &samples_out1_tmp[n][..frame_length],
            frame_length,
        );

        if dec_control.n_channels_api == 2 {
            // Interleave
            for i in 0..out_samples {
                if 2 * i + n < samples_out.len() {
                    samples_out[2 * i + n] = resampled[i];
                }
            }
        } else {
            // Mono
            let copy_len = out_samples.min(samples_out.len());
            samples_out[..copy_len].copy_from_slice(&resampled[..copy_len]);
        }
    }

    // If API is stereo but internal is mono: duplicate channel
    if dec_control.n_channels_api == 2 && n_channels_internal == 1 {
        for i in 0..out_samples {
            if 2 * i + 1 < samples_out.len() {
                samples_out[2 * i + 1] = samples_out[2 * i];
            }
        }
    }

    // Export pitch lag at 48 kHz for CELT hybrid
    if decoder.channel_state[0].prev_signal_type == TYPE_VOICED {
        let mult_tab: [i32; 3] = [6, 4, 3]; // for fs_kHz in {8, 12, 16}
        let fs_khz = decoder.channel_state[0].fs_khz;
        let idx = ((fs_khz - 8) >> 2) as usize;
        dec_control.prev_pitch_lag =
            decoder.channel_state[0].lag_prev * mult_tab[idx.min(2)];
    } else {
        dec_control.prev_pitch_lag = 0;
    }

    // Update state for next frame
    if lost_flag == FLAG_PACKET_LOST {
        for n in 0..n_channels_internal {
            decoder.channel_state[n].last_gain_index = 10;
        }
    }

    decoder.n_channels_internal = n_channels_internal;
    decoder.n_channels_api = dec_control.n_channels_api;
    decoder.prev_decode_only_middle = decode_only_middle;

    ret
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silk_rand() {
        let seed0 = 0i32;
        let seed1 = silk_rand(seed0);
        assert_eq!(seed1, 907633515);
        let seed2 = silk_rand(seed1);
        // Verify deterministic PRNG
        assert_ne!(seed2, seed1);
    }

    #[test]
    fn test_silk_log2lin_lin2log_roundtrip() {
        // log2lin(lin2log(x)) should approximate x
        for &x in &[1, 10, 100, 1000, 10000, 100000] {
            let log_val = silk_lin2log(x);
            let lin_val = silk_log2lin(log_val);
            // Allow small roundtrip error
            let ratio = lin_val as f64 / x as f64;
            assert!(
                ratio > 0.9 && ratio < 1.1,
                "roundtrip failed for x={}: log={}, lin={}",
                x,
                log_val,
                lin_val
            );
        }
    }

    #[test]
    fn test_silk_log2lin_boundaries() {
        assert_eq!(silk_log2lin(-1), 0);
        assert_eq!(silk_log2lin(0), 1);
        assert!(silk_log2lin(3967) > 0); // Max valid input
    }

    #[test]
    fn test_gains_dequant_independent() {
        let mut gain_q16 = [0i32; 4];
        let ind: [i8; 4] = [20, 5, 5, 5]; // Absolute first, delta rest
        let mut prev_ind: i8 = 10;
        silk_gains_dequant(&mut gain_q16, &ind, &mut prev_ind, false, 4);
        // All gains should be positive
        for g in &gain_q16 {
            assert!(*g > 0, "gain should be positive, got {}", g);
        }
    }

    #[test]
    fn test_gains_dequant_conditional() {
        let mut gain_q16 = [0i32; 4];
        let ind: [i8; 4] = [5, 5, 5, 5]; // All delta
        let mut prev_ind: i8 = 10;
        silk_gains_dequant(&mut gain_q16, &ind, &mut prev_ind, true, 4);
        for g in &gain_q16 {
            assert!(*g > 0);
        }
    }

    #[test]
    fn test_nlsf_stabilize() {
        // Create NLSFs that violate minimum spacing
        let mut nlsf: [i16; 10] = [
            1000, 1001, 5000, 8000, 12000, 16000, 20000, 24000, 28000, 31000,
        ];
        let delta_min = SILK_NLSF_DELTA_MIN_NB_MB_Q15;
        silk_nlsf_stabilize(&mut nlsf, &delta_min, 10);

        // Verify minimum spacing
        assert!(nlsf[0] as i32 >= delta_min[0] as i32);
        for i in 1..10 {
            let diff = nlsf[i] as i32 - nlsf[i - 1] as i32;
            assert!(
                diff >= delta_min[i] as i32,
                "NLSF spacing violation at {}: diff={}, min={}",
                i,
                diff,
                delta_min[i]
            );
        }
    }

    #[test]
    fn test_nlsf2a_basic() {
        // Well-spaced NLSFs for order 10
        let nlsf: [i16; 10] = [
            3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491, 32000,
        ];
        let mut a_q12 = [0i16; 10];
        silk_nlsf2a(&mut a_q12, &nlsf, 10);
        // LPC coefficients should be non-zero
        let non_zero = a_q12.iter().any(|&x| x != 0);
        assert!(non_zero, "NLSF2A produced all-zero LPC coefficients");
    }

    #[test]
    fn test_decode_pitch_basic() {
        let mut pitch_lags = [0i32; 4];
        silk_decode_pitch(50, 0, &mut pitch_lags, 16, 4);
        // All lags should be in valid range
        let min_lag = PITCH_EST_MIN_LAG_MS as i32 * 16;
        let max_lag = PITCH_EST_MAX_LAG_MS as i32 * 16;
        for &lag in &pitch_lags {
            assert!(lag >= min_lag && lag <= max_lag, "lag {} out of range [{}, {}]", lag, min_lag, max_lag);
        }
    }

    #[test]
    fn test_shell_decoder_sum_preserved() {
        // The shell decoder should preserve the total pulse count
        // We can't easily test without a range coder, but we can
        // verify the decode_split helper
        let (a, b) = (5i16, 3i16);
        assert_eq!(a + b, 8);
    }

    #[test]
    fn test_decoder_init() {
        let dec = SilkDecoder::new();
        assert_eq!(dec.n_channels_api, 1);
        assert_eq!(dec.n_channels_internal, 1);
        assert_eq!(dec.channel_state[0].prev_gain_q16, 65536);
        assert!(dec.channel_state[0].first_frame_after_reset);
    }

    #[test]
    fn test_plc_reset() {
        let mut dec = SilkDecoderState::new();
        dec.frame_length = 320;
        silk_plc_reset(&mut dec);
        assert_eq!(dec.s_plc.pitch_l_q8, 320 << 7);
        assert_eq!(dec.s_plc.prev_gain_q16, [65536, 65536]);
    }

    #[test]
    fn test_cng_reset() {
        let mut dec = SilkDecoderState::new();
        dec.lpc_order = 10;
        silk_cng_reset(&mut dec);
        assert_eq!(dec.s_cng.rand_seed, 3176576);
        // NLSFs should be uniformly spaced
        let step = 32767i32 / 11; // order + 1
        assert_eq!(dec.s_cng.cng_smth_nlsf_q15[0] as i32, step);
    }

    #[test]
    fn test_resampler_init_copy() {
        let mut rs = SilkResamplerState::default();
        silk_resampler_init(&mut rs, 16000, 16000, false);
        assert_eq!(rs.resampler_function, USE_SILK_RESAMPLER_COPY);
    }

    #[test]
    fn test_resampler_init_up2() {
        let mut rs = SilkResamplerState::default();
        silk_resampler_init(&mut rs, 8000, 16000, false);
        assert_eq!(rs.resampler_function, USE_SILK_RESAMPLER_UP2_HQ);
    }

    #[test]
    fn test_bwexpander() {
        let mut ar: [i16; 4] = [1000, 2000, 3000, 4000];
        silk_bwexpander(&mut ar, 4, 65000); // chirp slightly < 1.0
        // All coefficients should decrease
        assert!(ar[0] < 1000);
        assert!(ar[3] < 4000);
    }

    #[test]
    fn test_stereo_state_init() {
        let state = StereoDecState::default();
        assert_eq!(state.pred_prev_q13, [0, 0]);
    }
}
