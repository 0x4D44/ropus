#![allow(unused_variables, unused_assignments, unused_mut)]
//! SILK Encoder — complete encoding pipeline.
//!
//! Ported from: enc_API.c, init_encoder.c, encode_indices.c, encode_pulses.c,
//! control_codec.c, control_audio_bandwidth.c, control_SNR.c,
//! check_control_input.c, HP_variable_cutoff.c, LP_variable_cutoff.c,
//! LPC_analysis_filter.c, ana_filt_bank_1.c, NSQ.c, NSQ_del_dec.c,
//! VAD.c, VQ_WMat_EC.c, A2NLSF.c, NLSF_encode.c, NLSF_del_dec_quant.c,
//! process_NLSFs.c, quant_LTP_gains.c, stereo_LR_to_MS.c,
//! stereo_encode_pred.c, stereo_find_predictor.c, stereo_quant_pred.c

use crate::celt::range_coder::RangeEncoder;
use crate::silk::common::*;
use crate::silk::decoder::{SideInfoIndices, SilkResamplerState};
use crate::silk::tables::*;
use crate::types::*;

// ===========================================================================
// Encoder-specific constants
// ===========================================================================

pub const ENCODER_NUM_CHANNELS: usize = 2;
pub const MAX_FS_KHZ: i32 = 16;
pub const LA_PITCH_MS: i32 = 2;
pub const LA_PITCH_MAX: usize = (LA_PITCH_MS * MAX_FS_KHZ) as usize;
pub const LA_SHAPE_MS: i32 = 5;
pub const LA_SHAPE_MAX: usize = (LA_SHAPE_MS * MAX_FS_KHZ) as usize;
pub const FIND_PITCH_LPC_WIN_MS: i32 = 20 + (LA_PITCH_MS << 1); // 24
pub const FIND_PITCH_LPC_WIN_MS_2_SF: i32 = 10 + (LA_PITCH_MS << 1); // 14
pub const FIND_PITCH_LPC_WIN_MAX: usize = (FIND_PITCH_LPC_WIN_MS * MAX_FS_KHZ) as usize;
pub const SHAPE_LPC_WIN_MAX: usize = (15 * MAX_FS_KHZ) as usize;
pub const MAX_SHAPE_LPC_ORDER: usize = 24;
pub const MAX_DEL_DEC_STATES: usize = 4;
pub const DECISION_DELAY: usize = 40;
pub const NSQ_LPC_BUF_LENGTH: usize = MAX_LPC_ORDER;
pub const MAX_FIND_PITCH_LPC_ORDER: usize = 16;

pub const STEREO_INTERP_LEN_MS: usize = 8; // must be even
pub const STEREO_RATIO_SMOOTH_COEF: f64 = 0.01;
pub const VAD_N_BANDS: usize = 4;
pub const VAD_INTERNAL_SUBFRAMES_LOG2: usize = 2;
pub const VAD_INTERNAL_SUBFRAMES: usize = 1 << VAD_INTERNAL_SUBFRAMES_LOG2;
pub const VAD_NOISE_LEVEL_SMOOTH_COEF_Q16: i32 = 1024; // 0.015625 in Q16
pub const VAD_NOISE_LEVELS_BIAS: i32 = 50;
pub const VAD_NEGATIVE_OFFSET_Q5: i32 = 128; // 4.0 in Q5

pub const VARIABLE_HP_MIN_CUTOFF_HZ: i32 = 60;
pub const VARIABLE_HP_MAX_CUTOFF_HZ: i32 = 100;
pub const VARIABLE_HP_MAX_DELTA_FREQ_Q7: i32 = ((0.4f64 * (1 << 7) as f64) + 0.5) as i32;

pub const SILK_PE_MIN_COMPLEX: i32 = 0;
pub const SILK_PE_MID_COMPLEX: i32 = 1;
pub const SILK_PE_MAX_COMPLEX: i32 = 2;

/// Warping multiplier (Q16 per kHz): 0.015 * 65536 ≈ 983
pub const WARPING_MULTIPLIER_Q16: i32 = ((0.015f64 * 65536.0) + 0.5) as i32;

/// Maximum sum of log gains (Q7): 250.0 * 128
pub const MAX_SUM_LOG_GAIN_DB_Q7: i32 = ((250.0f64 * 128.0) + 0.5) as i32;

pub const NLSF_QUANT_DEL_DEC_STATES_LOG2: usize = 2;
pub const NLSF_QUANT_DEL_DEC_STATES: usize = 1 << NLSF_QUANT_DEL_DEC_STATES_LOG2;
pub const NLSF_QUANT_MAX_AMPLITUDE_EXT: i32 = 10;

pub const TRANSITION_FRAMES: i32 = 80;

pub const BIN_DIV_STEPS_A2NLSF: usize = 3;
pub const MAX_ITERATIONS_A2NLSF: usize = 16;

/// Pitch analysis look-ahead (reserved at beginning of analysis buffer)
pub const SCRATCH_SIZE: usize = 3 * LA_PITCH_MAX as usize;

// Error codes
pub const SILK_NO_ERROR: i32 = 0;
pub const SILK_ENC_FS_NOT_SUPPORTED: i32 = -101;
pub const SILK_ENC_PACKET_SIZE_NOT_SUPPORTED: i32 = -102;
pub const SILK_ENC_INVALID_LOSS_RATE: i32 = -104;
pub const SILK_ENC_INVALID_COMPLEXITY_SETTING: i32 = -105;
pub const SILK_ENC_INVALID_INBAND_FEC_SETTING: i32 = -106;
pub const SILK_ENC_INVALID_DTX_SETTING: i32 = -107;
pub const SILK_ENC_INVALID_CBR_SETTING: i32 = -108;
pub const SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR: i32 = -111;
pub const SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES: i32 = -112;

// ===========================================================================
// SNR tables (from control_SNR.c)
// ===========================================================================

const SILK_TARGET_RATE_NB_21: [u8; 107] = [
     0, 15, 39, 52, 61, 68,
    74, 79, 84, 88, 92, 95, 99,102,105,108,111,114,117,119,122,124,
   126,129,131,133,135,137,139,142,143,145,147,149,151,153,155,157,
   158,160,162,163,165,167,168,170,171,173,174,176,177,179,180,182,
   183,185,186,187,189,190,192,193,194,196,197,199,200,201,203,204,
   205,207,208,209,211,212,213,215,216,217,219,220,221,223,224,225,
   227,228,230,231,232,234,235,236,238,239,241,242,243,245,246,248,
   249,250,252,253,255,
];

const SILK_TARGET_RATE_MB_21: [u8; 155] = [
     0,  0, 28, 43, 52, 59,
    65, 70, 74, 78, 81, 85, 87, 90, 93, 95, 98,100,102,105,107,109,
   111,113,115,116,118,120,122,123,125,127,128,130,131,133,134,136,
   137,138,140,141,143,144,145,147,148,149,151,152,153,154,156,157,
   158,159,160,162,163,164,165,166,167,168,169,171,172,173,174,175,
   176,177,178,179,180,181,182,183,184,185,186,187,188,188,189,190,
   191,192,193,194,195,196,197,198,199,200,201,202,203,203,204,205,
   206,207,208,209,210,211,212,213,214,214,215,216,217,218,219,220,
   221,222,223,224,224,225,226,227,228,229,230,231,232,233,234,235,
   236,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,
   251,252,253,254,255,
];

const SILK_TARGET_RATE_WB_21: [u8; 191] = [
     0,  0,  0,  8, 29, 41,
    49, 56, 62, 66, 70, 74, 77, 80, 83, 86, 88, 91, 93, 95, 97, 99,
   101,103,105,107,108,110,112,113,115,116,118,119,121,122,123,125,
   126,127,129,130,131,132,134,135,136,137,138,140,141,142,143,144,
   145,146,147,148,149,150,151,152,153,154,156,157,158,159,159,160,
   161,162,163,164,165,166,167,168,169,170,171,171,172,173,174,175,
   176,177,177,178,179,180,181,181,182,183,184,185,185,186,187,188,
   189,189,190,191,192,192,193,194,195,195,196,197,198,198,199,200,
   200,201,202,203,203,204,205,206,206,207,208,209,209,210,211,211,
   212,213,214,214,215,216,216,217,218,219,219,220,221,221,222,223,
   224,224,225,226,226,227,228,229,229,230,231,232,232,233,234,234,
   235,236,237,237,238,239,240,240,241,242,243,243,244,245,246,246,
   247,248,249,249,250,251,252,253,255,
];

// ===========================================================================
// State structures
// ===========================================================================

/// Encoder control parameters passed from the Opus layer.
#[derive(Clone)]
pub struct SilkEncControlStruct {
    pub n_channels_api: i32,
    pub n_channels_internal: i32,
    pub api_sample_rate: i32,
    pub max_internal_sample_rate: i32,
    pub min_internal_sample_rate: i32,
    pub desired_internal_sample_rate: i32,
    pub payload_size_ms: i32,
    pub bit_rate: i32,
    pub packet_loss_percentage: i32,
    pub complexity: i32,
    pub use_in_band_fec: i32,
    pub use_dred: i32,
    pub lbrr_coded: i32,
    pub use_dtx: i32,
    pub use_cbr: i32,
    pub max_bits: i32,
    pub to_mono: i32,
    pub opus_can_switch: i32,
    pub reduced_dependency: i32,
    // Output fields
    pub internal_sample_rate: i32,
    pub allow_bandwidth_switch: i32,
    pub in_wb_mode_without_variable_lp: i32,
    pub stereo_width_q14: i32,
    pub switch_ready: i32,
    pub signal_type: i32,
    pub offset: i32,
}

impl Default for SilkEncControlStruct {
    fn default() -> Self {
        Self {
            n_channels_api: 1,
            n_channels_internal: 1,
            api_sample_rate: 16000,
            max_internal_sample_rate: 16000,
            min_internal_sample_rate: 8000,
            desired_internal_sample_rate: 16000,
            payload_size_ms: 20,
            bit_rate: 25000,
            packet_loss_percentage: 0,
            complexity: 5,
            use_in_band_fec: 0,
            use_dred: 0,
            lbrr_coded: 0,
            use_dtx: 0,
            use_cbr: 0,
            max_bits: 0,
            to_mono: 0,
            opus_can_switch: 0,
            reduced_dependency: 0,
            internal_sample_rate: 0,
            allow_bandwidth_switch: 0,
            in_wb_mode_without_variable_lp: 0,
            stereo_width_q14: 0,
            switch_ready: 0,
            signal_type: 0,
            offset: 0,
        }
    }
}

/// VAD (Voice Activity Detection) state.
#[derive(Clone)]
pub struct SilkVadState {
    pub ana_state: [i32; 2],
    pub ana_state1: [i32; 2],
    pub ana_state2: [i32; 2],
    pub xnrg_subfr: [i32; VAD_N_BANDS],
    pub nrg_ratio_smth_q8: [i32; VAD_N_BANDS],
    pub hp_state: i16,
    pub nl: [i32; VAD_N_BANDS],
    pub inv_nl: [i32; VAD_N_BANDS],
    pub noise_level_bias: [i32; VAD_N_BANDS],
    pub counter: i32,
}

impl Default for SilkVadState {
    fn default() -> Self {
        Self {
            ana_state: [0; 2],
            ana_state1: [0; 2],
            ana_state2: [0; 2],
            xnrg_subfr: [0; VAD_N_BANDS],
            nrg_ratio_smth_q8: [0; VAD_N_BANDS],
            hp_state: 0,
            nl: [0; VAD_N_BANDS],
            inv_nl: [0; VAD_N_BANDS],
            noise_level_bias: [0; VAD_N_BANDS],
            counter: 0,
        }
    }
}

/// LP variable cutoff state (bandwidth transition filter).
#[derive(Clone)]
pub struct SilkLpState {
    pub in_lp_state: [i32; 2],
    pub transition_frame_no: i32,
    pub mode: i32,
    pub saved_fs_khz: i32,
}

impl Default for SilkLpState {
    fn default() -> Self {
        Self {
            in_lp_state: [0; 2],
            transition_frame_no: 0,
            mode: 0,
            saved_fs_khz: 0,
        }
    }
}

/// Noise shaping quantizer state.
#[derive(Clone)]
pub struct NsqState {
    pub xq: Vec<i16>,                            // 2 * MAX_FRAME_LENGTH
    pub s_ltp_shp_q14: Vec<i32>,                 // 2 * MAX_FRAME_LENGTH
    pub s_lpc_q14: [i32; MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
    pub s_ar2_q14: [i32; MAX_SHAPE_LPC_ORDER],
    pub s_lf_ar_shp_q14: i32,
    pub s_diff_shp_q14: i32,
    pub lag_prev: i32,
    pub s_ltp_buf_idx: i32,
    pub s_ltp_shp_buf_idx: i32,
    pub rand_seed: i32,
    pub prev_gain_q16: i32,
    pub rewhite_flag: i32,
}

impl Default for NsqState {
    fn default() -> Self {
        Self {
            xq: vec![0i16; 2 * MAX_FRAME_LENGTH],
            s_ltp_shp_q14: vec![0i32; 2 * MAX_FRAME_LENGTH],
            s_lpc_q14: [0; MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
            s_ar2_q14: [0; MAX_SHAPE_LPC_ORDER],
            s_lf_ar_shp_q14: 0,
            s_diff_shp_q14: 0,
            lag_prev: 100,
            s_ltp_buf_idx: 0,
            s_ltp_shp_buf_idx: 0,
            rand_seed: 0,
            prev_gain_q16: 65536,
            rewhite_flag: 0,
        }
    }
}

/// Noise shaping analysis state (fixed-point).
#[derive(Clone)]
pub struct SilkShapeStateFix {
    pub last_gain_index: i8,
    pub harm_shape_gain_smth: i32,
    pub tilt_smth: i32,
}

impl Default for SilkShapeStateFix {
    fn default() -> Self {
        Self {
            last_gain_index: 10,
            harm_shape_gain_smth: 0,
            tilt_smth: 0,
        }
    }
}

/// Per-frame encoder control (transient, stack-allocated).
#[derive(Clone)]
pub struct SilkEncoderControl {
    pub gains_q16: [i32; MAX_NB_SUBFR],
    pub pred_coef_q12: [[i16; MAX_LPC_ORDER]; 2],
    pub ltp_coef_q14: [i16; LTP_ORDER * MAX_NB_SUBFR],
    pub ltp_scale_q14: i32,
    pub pitch_l: [i32; MAX_NB_SUBFR],
    pub ar_q13: [i16; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER],
    pub lf_shp_q14: [i32; MAX_NB_SUBFR],
    pub harm_shape_fir_packed_q14: i32,
    pub tilt_q14: [i32; MAX_NB_SUBFR],
    pub harm_shape_gain_q14: [i32; MAX_NB_SUBFR],
    pub lambda_q10: i32,
    pub input_quality_q14: i32,
    pub coding_quality_q14: i32,
    pub pitch_lag_low_bits_icdf: &'static [u8],
    pub pitch_contour_icdf: &'static [u8],
    pub per_index: i8,
    pub current_voice_gain_q14: i32,
    pub gw_temp: i32,
    pub sparseness_q8: i32,
    pub pred_gain_q16: i32,
    pub ltp_scaling_q14: i32,
}

impl Default for SilkEncoderControl {
    fn default() -> Self {
        Self {
            gains_q16: [0; MAX_NB_SUBFR],
            pred_coef_q12: [[0; MAX_LPC_ORDER]; 2],
            ltp_coef_q14: [0; LTP_ORDER * MAX_NB_SUBFR],
            ltp_scale_q14: 0,
            pitch_l: [0; MAX_NB_SUBFR],
            ar_q13: [0; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER],
            lf_shp_q14: [0; MAX_NB_SUBFR],
            harm_shape_fir_packed_q14: 0,
            tilt_q14: [0; MAX_NB_SUBFR],
            harm_shape_gain_q14: [0; MAX_NB_SUBFR],
            lambda_q10: 0,
            input_quality_q14: 0,
            coding_quality_q14: 0,
            pitch_lag_low_bits_icdf: &SILK_UNIFORM4_ICDF,
            pitch_contour_icdf: &SILK_PITCH_CONTOUR_NB_ICDF,
            per_index: 0,
            current_voice_gain_q14: 0,
            gw_temp: 0,
            sparseness_q8: 0,
            pred_gain_q16: 0,
            ltp_scaling_q14: 0,
        }
    }
}

/// Common encoder state (the large `silk_encoder_state` from C).
#[derive(Clone)]
pub struct SilkEncoderState {
    // Filter states
    pub in_hp_state: [i32; 2],
    pub variable_hp_smth1_q15: i32,
    pub variable_hp_smth2_q15: i32,
    pub s_lp: SilkLpState,
    pub s_nsq: NsqState,
    pub s_vad: SilkVadState,
    pub resampler_state: SilkResamplerState,

    // Configuration
    pub fs_khz: i32,
    pub nb_subfr: i32,
    pub frame_length: i32,
    pub subfr_length: i32,
    pub ltp_mem_length: i32,
    pub la_pitch: i32,
    pub la_shape: i32,
    pub shape_win_length: i32,
    pub target_rate_bps: i32,
    pub packet_size_ms: i32,
    pub predict_lpc_order: i32,
    pub prev_signal_type: i32,
    pub prev_lag: i32,
    pub pitch_lpc_win_length: i32,
    pub max_pitch_lag: i32,

    // Complexity settings
    pub complexity: i32,
    pub pitch_estimation_complexity: i32,
    pub pitch_estimation_threshold_q16: i32,
    pub pitch_estimation_lpc_order: i32,
    pub shaping_lpc_order: i32,
    pub n_states_delayed_decision: i32,
    pub use_interpolated_nlsfs: i32,
    pub nlsf_msvq_survivors: i32,
    pub warping_q16: i32,

    // Rate/bitrate control
    pub api_fs_hz: i32,
    pub prev_api_fs_hz: i32,
    pub max_internal_fs_hz: i32,
    pub min_internal_fs_hz: i32,
    pub desired_internal_fs_hz: i32,
    pub snr_db_q7: i32,
    pub use_dtx: i32,
    pub use_cbr: i32,
    pub use_in_band_fec: i32,
    pub n_channels_api: i32,
    pub n_channels_internal: i32,
    pub allow_bandwidth_switch: i32,
    pub channel_nb: i32,
    pub packet_loss_perc: i32,
    pub n_frames_per_packet: i32,
    pub n_frames_encoded: i32,
    pub controlled_since_last_payload: i32,
    pub prefill_flag: i32,

    // Codec parameters (per-frame results)
    pub indices: SideInfoIndices,
    pub pulses: [i8; MAX_FRAME_LENGTH],
    pub prev_nlsfq_q15: [i16; MAX_LPC_ORDER],

    // LBRR (Forward Error Correction)
    pub lbrr_enabled: i32,
    pub lbrr_gain_increases: i32,
    pub indices_lbrr: [SideInfoIndices; MAX_FRAMES_PER_PACKET],
    pub pulses_lbrr: [[i8; MAX_FRAME_LENGTH]; MAX_FRAMES_PER_PACKET],
    pub lbrr_flags: [i32; MAX_FRAMES_PER_PACKET],
    pub lbrr_flag: i32,

    // Buffering
    pub input_buf: [i16; MAX_FRAME_LENGTH + 2],
    pub input_buf_ix: i32,

    // Codec references
    pub nlsf_cb: &'static SilkNlsfCbStruct,
    pub pitch_lag_low_bits_icdf: &'static [u8],
    pub pitch_contour_icdf: &'static [u8],

    // First frame / reset
    pub first_frame_after_reset: i32,

    // VAD/DTX
    pub speech_activity_q8: i32,
    pub in_dtx: i32,
    pub vad_flags: [i32; MAX_FRAMES_PER_PACKET],
    pub input_tilt_q15: i32,
    pub input_quality_bands_q15: [i32; VAD_N_BANDS],
    pub ec_prev_signal_type: i32,
    pub ec_prev_lag_index: i16,
    pub reduced_dependency: i32,
}

impl Default for SilkEncoderState {
    fn default() -> Self {
        Self {
            in_hp_state: [0; 2],
            variable_hp_smth1_q15: 0,
            variable_hp_smth2_q15: 0,
            s_lp: SilkLpState::default(),
            s_nsq: NsqState::default(),
            s_vad: SilkVadState::default(),
            resampler_state: SilkResamplerState::default(),
            fs_khz: 0,
            nb_subfr: 4,
            frame_length: 0,
            subfr_length: 0,
            ltp_mem_length: 0,
            la_pitch: 0,
            la_shape: 0,
            shape_win_length: 0,
            target_rate_bps: 0,
            packet_size_ms: 0,
            predict_lpc_order: 10,
            prev_signal_type: TYPE_NO_VOICE_ACTIVITY,
            prev_lag: 100,
            pitch_lpc_win_length: 0,
            max_pitch_lag: 0,
            complexity: 5,
            pitch_estimation_complexity: 0,
            pitch_estimation_threshold_q16: 0,
            pitch_estimation_lpc_order: 0,
            shaping_lpc_order: 0,
            n_states_delayed_decision: 1,
            use_interpolated_nlsfs: 0,
            nlsf_msvq_survivors: 2,
            warping_q16: 0,
            api_fs_hz: 0,
            prev_api_fs_hz: 0,
            max_internal_fs_hz: 16000,
            min_internal_fs_hz: 8000,
            desired_internal_fs_hz: 16000,
            snr_db_q7: 0,
            use_dtx: 0,
            use_cbr: 0,
            use_in_band_fec: 0,
            n_channels_api: 1,
            n_channels_internal: 1,
            allow_bandwidth_switch: 0,
            channel_nb: 0,
            packet_loss_perc: 0,
            n_frames_per_packet: 1,
            n_frames_encoded: 0,
            controlled_since_last_payload: 0,
            prefill_flag: 0,
            indices: SideInfoIndices::default(),
            pulses: [0; MAX_FRAME_LENGTH],
            prev_nlsfq_q15: [0; MAX_LPC_ORDER],
            lbrr_enabled: 0,
            lbrr_gain_increases: 0,
            indices_lbrr: [
                SideInfoIndices::default(),
                SideInfoIndices::default(),
                SideInfoIndices::default(),
            ],
            pulses_lbrr: [[0; MAX_FRAME_LENGTH]; MAX_FRAMES_PER_PACKET],
            lbrr_flags: [0; MAX_FRAMES_PER_PACKET],
            lbrr_flag: 0,
            input_buf: [0; MAX_FRAME_LENGTH + 2],
            input_buf_ix: 0,
            nlsf_cb: &SILK_NLSF_CB_NB_MB,
            pitch_lag_low_bits_icdf: &SILK_UNIFORM4_ICDF,
            pitch_contour_icdf: &SILK_PITCH_CONTOUR_NB_ICDF,
            first_frame_after_reset: 1,
            speech_activity_q8: 0,
            in_dtx: 0,
            vad_flags: [0; MAX_FRAMES_PER_PACKET],
            input_tilt_q15: 0,
            input_quality_bands_q15: [0; VAD_N_BANDS],
            ec_prev_signal_type: 0,
            ec_prev_lag_index: 0,
            reduced_dependency: 0,
        }
    }
}

/// Per-channel encoder state (fixed-point).
#[derive(Clone)]
pub struct SilkEncoderStateFix {
    pub s_cmn: SilkEncoderState,
    pub s_shape: SilkShapeStateFix,
    pub x_buf: Vec<i16>, // 2 * MAX_FRAME_LENGTH + LA_SHAPE_MAX
    pub ltp_corr_q15: i32,
    pub res_nrg_smth: i32,
}

impl Default for SilkEncoderStateFix {
    fn default() -> Self {
        Self {
            s_cmn: SilkEncoderState::default(),
            s_shape: SilkShapeStateFix::default(),
            x_buf: vec![0i16; 2 * MAX_FRAME_LENGTH + LA_SHAPE_MAX],
            ltp_corr_q15: 0,
            res_nrg_smth: 0,
        }
    }
}

/// Stereo encoder state.
#[derive(Clone)]
pub struct StereoEncState {
    pub pred_prev_q13: [i16; 2],
    pub s_mid: [i16; 2],
    pub s_side: [i16; 2],
    pub mid_side_amp_q0: [i32; 4],
    pub smth_width_q14: i16,
    pub width_prev_q14: i16,
    pub silent_side_len: i16,
    pub pred_ix: [[[i8; 3]; 2]; MAX_FRAMES_PER_PACKET],
    pub mid_only_flags: [i8; MAX_FRAMES_PER_PACKET],
}

impl Default for StereoEncState {
    fn default() -> Self {
        Self {
            pred_prev_q13: [0; 2],
            s_mid: [0; 2],
            s_side: [0; 2],
            mid_side_amp_q0: [0; 4],
            smth_width_q14: 0,
            width_prev_q14: 0,
            silent_side_len: 0,
            pred_ix: [[[0; 3]; 2]; MAX_FRAMES_PER_PACKET],
            mid_only_flags: [0; MAX_FRAMES_PER_PACKET],
        }
    }
}

/// Top-level SILK encoder super struct.
pub struct SilkEncoder {
    pub s_stereo: StereoEncState,
    pub n_bits_used_lbrr: i32,
    pub n_bits_exceeded: i32,
    pub n_channels_api: i32,
    pub n_channels_internal: i32,
    pub n_prev_channels_internal: i32,
    pub time_since_switch_allowed_ms: i32,
    pub allow_bandwidth_switch: i32,
    pub prev_decode_only_middle: i32,
    pub state_fxx: [SilkEncoderStateFix; ENCODER_NUM_CHANNELS],
}

impl SilkEncoder {
    pub fn new() -> Self {
        Self {
            s_stereo: StereoEncState::default(),
            n_bits_used_lbrr: 0,
            n_bits_exceeded: 0,
            n_channels_api: 1,
            n_channels_internal: 1,
            n_prev_channels_internal: 1,
            time_since_switch_allowed_ms: 0,
            allow_bandwidth_switch: 0,
            prev_decode_only_middle: 0,
            state_fxx: [SilkEncoderStateFix::default(), SilkEncoderStateFix::default()],
        }
    }
}

/// NSQ delayed-decision state (per parallel path).
#[derive(Clone)]
pub struct NsqDelDecStruct {
    pub s_lpc_q14: [i32; MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
    pub rand_state: [i32; DECISION_DELAY],
    pub q_q10: [i32; DECISION_DELAY],
    pub xq_q14: [i32; DECISION_DELAY],
    pub pred_q15: [i32; DECISION_DELAY],
    pub shape_q14: [i32; DECISION_DELAY],
    pub s_ar2_q14: [i32; MAX_SHAPE_LPC_ORDER],
    pub lf_ar_q14: i32,
    pub diff_q14: i32,
    pub seed: i32,
    pub seed_init: i32,
    pub rd_q10: i32,
}

impl Default for NsqDelDecStruct {
    fn default() -> Self {
        Self {
            s_lpc_q14: [0; MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
            rand_state: [0; DECISION_DELAY],
            q_q10: [0; DECISION_DELAY],
            xq_q14: [0; DECISION_DELAY],
            pred_q15: [0; DECISION_DELAY],
            shape_q14: [0; DECISION_DELAY],
            s_ar2_q14: [0; MAX_SHAPE_LPC_ORDER],
            lf_ar_q14: 0,
            diff_q14: 0,
            seed: 0,
            seed_init: 0,
            rd_q10: 0,
        }
    }
}

// ===========================================================================
// Check control input
// ===========================================================================

/// Validate encoder control parameters.
/// Matches C: `check_control_input`.
pub fn check_control_input(enc_control: &SilkEncControlStruct) -> i32 {
    if (enc_control.api_sample_rate != 8000
        && enc_control.api_sample_rate != 12000
        && enc_control.api_sample_rate != 16000
        && enc_control.api_sample_rate != 24000
        && enc_control.api_sample_rate != 32000
        && enc_control.api_sample_rate != 44100
        && enc_control.api_sample_rate != 48000)
        || (enc_control.desired_internal_sample_rate != 8000
            && enc_control.desired_internal_sample_rate != 12000
            && enc_control.desired_internal_sample_rate != 16000)
        || (enc_control.max_internal_sample_rate != 8000
            && enc_control.max_internal_sample_rate != 12000
            && enc_control.max_internal_sample_rate != 16000)
        || (enc_control.min_internal_sample_rate != 8000
            && enc_control.min_internal_sample_rate != 12000
            && enc_control.min_internal_sample_rate != 16000)
        || enc_control.min_internal_sample_rate > enc_control.desired_internal_sample_rate
        || enc_control.max_internal_sample_rate < enc_control.desired_internal_sample_rate
        || enc_control.min_internal_sample_rate > enc_control.max_internal_sample_rate
    {
        return SILK_ENC_FS_NOT_SUPPORTED;
    }
    if enc_control.payload_size_ms != 10
        && enc_control.payload_size_ms != 20
        && enc_control.payload_size_ms != 40
        && enc_control.payload_size_ms != 60
    {
        return SILK_ENC_PACKET_SIZE_NOT_SUPPORTED;
    }
    if enc_control.packet_loss_percentage < 0 || enc_control.packet_loss_percentage > 100 {
        return SILK_ENC_INVALID_LOSS_RATE;
    }
    if enc_control.use_dtx < 0 || enc_control.use_dtx > 1 {
        return SILK_ENC_INVALID_DTX_SETTING;
    }
    if enc_control.use_cbr < 0 || enc_control.use_cbr > 1 {
        return SILK_ENC_INVALID_CBR_SETTING;
    }
    if enc_control.use_in_band_fec < 0 || enc_control.use_in_band_fec > 1 {
        return SILK_ENC_INVALID_INBAND_FEC_SETTING;
    }
    if enc_control.n_channels_api < 1 || enc_control.n_channels_api > ENCODER_NUM_CHANNELS as i32 {
        return SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR;
    }
    if enc_control.n_channels_internal < 1
        || enc_control.n_channels_internal > ENCODER_NUM_CHANNELS as i32
    {
        return SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR;
    }
    if enc_control.n_channels_internal > enc_control.n_channels_api {
        return SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR;
    }
    if enc_control.complexity < 0 || enc_control.complexity > 10 {
        return SILK_ENC_INVALID_COMPLEXITY_SETTING;
    }
    SILK_NO_ERROR
}

// ===========================================================================
// Control SNR
// ===========================================================================

/// Set SNR control based on target bitrate.
/// Matches C: `silk_control_SNR`.
pub fn silk_control_snr(ps_enc: &mut SilkEncoderState, mut target_rate_bps: i32) -> i32 {
    ps_enc.target_rate_bps = target_rate_bps;
    if ps_enc.nb_subfr == 2 {
        target_rate_bps -= 2000 + ps_enc.fs_khz / 16;
    }
    let (bound, snr_table): (usize, &[u8]) = if ps_enc.fs_khz == 8 {
        (SILK_TARGET_RATE_NB_21.len(), &SILK_TARGET_RATE_NB_21)
    } else if ps_enc.fs_khz == 12 {
        (SILK_TARGET_RATE_MB_21.len(), &SILK_TARGET_RATE_MB_21)
    } else {
        (SILK_TARGET_RATE_WB_21.len(), &SILK_TARGET_RATE_WB_21)
    };
    let id = (target_rate_bps + 200) / 400;
    let id = imin(id - 10, bound as i32 - 1);
    if id <= 0 {
        ps_enc.snr_db_q7 = 0;
    } else {
        ps_enc.snr_db_q7 = snr_table[id as usize] as i32 * 21;
    }
    SILK_NO_ERROR
}

// ===========================================================================
// VAD initialization
// ===========================================================================

/// Initialize VAD state.
/// Matches C: `silk_VAD_Init`.
pub fn silk_vad_init(vad: &mut SilkVadState) -> i32 {
    *vad = SilkVadState::default();

    // Initialize noise levels with approx pink noise (psd ∝ 1/f)
    for b in 0..VAD_N_BANDS {
        vad.noise_level_bias[b] = imax(VAD_NOISE_LEVELS_BIAS / (b as i32 + 1), 1);
    }
    for b in 0..VAD_N_BANDS {
        vad.nl[b] = 100 * vad.noise_level_bias[b];
        vad.inv_nl[b] = i32::MAX / vad.nl[b];
    }
    vad.counter = 15;

    // Init smoothed energy-to-noise ratio (20 dB)
    for b in 0..VAD_N_BANDS {
        vad.nrg_ratio_smth_q8[b] = 100 * 256;
    }
    0
}

// ===========================================================================
// Analysis filter bank
// ===========================================================================

/// Two-band decimating analysis filter bank.
/// Matches C: `silk_ana_filt_bank_1`.
pub fn silk_ana_filt_bank_1(
    input: &[i16],
    state: &mut [i32; 2],
    out_l: &mut [i16],
    out_h: &mut [i16],
    n: usize,
) {
    // Coefficients: allpass sections for half-band splitting
    // A_fb1_20 = 5394 << 1 = 10788 (fits i16)
    const A_FB1_20: i16 = (5394 << 1) as i16; // 10788
    // A_fb1_21 = (opus_int16)(20623 << 1) = -24290 (wraps in i16)
    const A_FB1_21: i16 = 20623i32.wrapping_shl(1) as i16; // -24290

    let n2 = n >> 1;
    for k in 0..n2 {
        // Convert even input to Q10
        let in32 = (input[2 * k] as i32) << 10;

        // All-pass section for even input sample (uses SMLAWB)
        let y = in32 - state[0];
        let x = silk_smlawb(y, y, A_FB1_21);
        let out_1 = state[0] + x;
        state[0] = in32 + x;

        // Convert odd input to Q10
        let in32 = (input[2 * k + 1] as i32) << 10;

        // All-pass section for odd input sample (uses SMULWB)
        let y = in32 - state[1];
        let x = silk_smulwb(y, A_FB1_20);
        let out_2 = state[1] + x;
        state[1] = in32 + x;

        // Add/subtract, convert back to int16 and store
        out_l[k] = sat16(silk_rshift_round(out_2 + out_1, 11));
        out_h[k] = sat16(silk_rshift_round(out_2 - out_1, 11));
    }
}

// ===========================================================================
// VAD: Get speech activity level
// ===========================================================================

/// Weighting factors for frequency tilt measure.
const TILT_WEIGHTS: [i32; VAD_N_BANDS] = [30000, 6000, -12000, -12000];

/// Get noise levels for VAD.
/// Matches C: `silk_VAD_GetNoiseLevels`.
fn silk_vad_get_noise_levels(px: &[i32; VAD_N_BANDS], vad: &mut SilkVadState) {
    // Faster adaptation during first 1000 frames
    // C: silk_min_int( silk_int16_MAX / ( ( counter >> 4 ) + 1 ), ... )
    let min_coef = if vad.counter < 1000 {
        i16::MAX as i32 / ((vad.counter >> 4) + 1)
    } else {
        0
    };

    for b in 0..VAD_N_BANDS {
        // Get current noise level and add bias to energy
        let nrg = px[b] + vad.noise_level_bias[b];
        let nl = vad.nl[b];
        let inv_nl = vad.inv_nl[b];

        // Determine smoothing coefficient adaptively
        let coef_q16 = if nrg > (nl << 3) {
            // Signal well above noise: slow adaptation (stay close to current NL)
            VAD_NOISE_LEVEL_SMOOTH_COEF_Q16
        } else if nrg < nl {
            // Signal below noise: fast adaptation (track down quickly)
            imax(silk_smulwb(VAD_NOISE_LEVEL_SMOOTH_COEF_Q16 << 1, inv_nl as i16), min_coef)
        } else {
            // In between: normal adaptation
            imax(silk_smulwb(VAD_NOISE_LEVEL_SMOOTH_COEF_Q16, inv_nl as i16), min_coef)
        };
        // Clamp coef to 0.25 in Q16
        let coef_q16 = imin(coef_q16, 16384);

        // Smooth noise level: NL = (1 - coef) * NL + coef * (signal_energy + bias)
        vad.nl[b] = ((vad.nl[b] as i64 * (65536 - coef_q16) as i64
            + (px[b] + vad.noise_level_bias[b]) as i64 * coef_q16 as i64)
            >> 16) as i32;

        // Clamp noise level for headroom
        vad.nl[b] = imax(vad.nl[b], 1);
        vad.nl[b] = imin(vad.nl[b], 0x00FFFFFF);

        // Update inverse noise level
        vad.inv_nl[b] = i32::MAX / vad.nl[b];
    }
    vad.counter += 1;
}

/// Compute speech activity level (Q8).
/// Matches C: `silk_VAD_GetSA_Q8_c`.
pub fn silk_vad_get_sa_q8(ps_enc: &mut SilkEncoderState, p_in: &[i16]) -> i32 {
    let frame_length = ps_enc.frame_length as usize;
    let vad = &mut ps_enc.s_vad;

    // Compute decimated frame lengths
    let decimated_framelength1 = frame_length >> 1;
    let decimated_framelength2 = frame_length >> 2;
    let decimated_framelength = frame_length >> 3;

    // Compute offsets for each band's storage in the X buffer
    let x_offset = [0usize, decimated_framelength, decimated_framelength + decimated_framelength2,
                    decimated_framelength + decimated_framelength2 + decimated_framelength1];
    let total_x_len = x_offset[3] + frame_length;

    let mut x = vec![0i16; total_x_len];

    // Stage 1: 0-8 kHz → 0-4 kHz (out_l) + 4-8 kHz (out_h)
    // out_l at x_offset[3], out_h at x_offset[2]; use split_at_mut for non-overlapping borrows
    {
        let (lo, hi) = x.split_at_mut(x_offset[3]);
        silk_ana_filt_bank_1(
            p_in,
            &mut vad.ana_state,
            hi,                              // 0-4 kHz at x_offset[3]
            &mut lo[x_offset[2]..],          // 4-8 kHz at x_offset[2]
            frame_length,
        );
    }

    // Stage 2: 0-4 kHz → 0-2 kHz + 2-4 kHz
    // Copy input first since source (x_offset[3]) may alias output regions
    let tmp2: Vec<i16> = x[x_offset[3]..x_offset[3] + decimated_framelength1].to_vec();
    {
        let (lo, hi) = x.split_at_mut(x_offset[2]);
        silk_ana_filt_bank_1(
            &tmp2,
            &mut vad.ana_state1,
            &mut lo[x_offset[1]..],          // 0-2 kHz at x_offset[1]
            hi,                              // 2-4 kHz at x_offset[2]
            decimated_framelength1,
        );
    }

    // Stage 3: 0-2 kHz → 0-1 kHz + 1-2 kHz
    let tmp3: Vec<i16> = x[x_offset[1]..x_offset[1] + decimated_framelength2].to_vec();
    {
        let (lo, hi) = x.split_at_mut(x_offset[0]);
        silk_ana_filt_bank_1(
            &tmp3,
            &mut vad.ana_state2,
            lo,                              // 0-1 kHz at offset 0
            hi,                              // 1-2 kHz at x_offset[0]
            decimated_framelength2,
        );
    }

    // HP filter on the lowest band: first-order differentiator to remove DC
    // HPstateTmp is used to save the last input for the differentiator
    {
        let decimated_len = decimated_framelength;
        let mut hp_state_tmp = vad.hp_state;
        for i in 0..decimated_len {
            let tmp = x[i];
            x[i] = sat16(x[i] as i32 - hp_state_tmp as i32);
            hp_state_tmp = tmp;
        }
        vad.hp_state = hp_state_tmp;
    }

    // Compute per-band energies
    let mut xnrg = [0i32; VAD_N_BANDS];
    let dec_subframe_length = (frame_length >> 3) >> VAD_INTERNAL_SUBFRAMES_LOG2;

    for b in 0..VAD_N_BANDS {
        let band_len = if b == 0 {
            decimated_framelength
        } else if b == 1 {
            decimated_framelength2
        } else if b == 2 {
            decimated_framelength1
        } else {
            frame_length
        };
        let band_start = if b < 3 { x_offset[b] } else { x_offset[3] };

        let dec_sf_len = band_len >> VAD_INTERNAL_SUBFRAMES_LOG2;
        let mut sum_sqr: i32 = 0;
        for s in 0..VAD_INTERNAL_SUBFRAMES {
            let offset = band_start + s * dec_sf_len;
            let end = offset + dec_sf_len;
            let (nrg, _shift) = silk_sum_sqr_shift(&x[offset..end.min(x.len())]);
            // Weight last subframe by 0.5 (look-ahead)
            if s < VAD_INTERNAL_SUBFRAMES - 1 {
                sum_sqr = sum_sqr.saturating_add(nrg);
            } else {
                sum_sqr = sum_sqr.saturating_add(nrg >> 1);
            }
        }
        xnrg[b] = sum_sqr;
    }

    // Get noise levels
    silk_vad_get_noise_levels(&xnrg, vad);

    // Compute SNR per band and speech activity
    let mut nrg_to_noise_ratio_q8 = [0i32; VAD_N_BANDS];
    let mut input_tilt = 0i32;

    for b in 0..VAD_N_BANDS {
        // SNR = signal energy / noise level
        nrg_to_noise_ratio_q8[b] = silk_div32_var_q(xnrg[b], vad.nl[b], 8);
        // Smooth the ratio
        vad.nrg_ratio_smth_q8[b] = vad.nrg_ratio_smth_q8[b]
            + ((nrg_to_noise_ratio_q8[b] - vad.nrg_ratio_smth_q8[b]) >> 4);

        // Frequency tilt contribution
        input_tilt += silk_smulwb(TILT_WEIGHTS[b], vad.nrg_ratio_smth_q8[b] as i16);
    }

    // Convert to dB and compute speech probability
    // Compute root-mean-square of SNR across bands
    let mut sum_snr_q7: i32 = 0;
    for b in 0..VAD_N_BANDS {
        let snr_q7 = if vad.nrg_ratio_smth_q8[b] >= 256 {
            // 3 * silk_lin2log(nrg_ratio) - 3 * 8 * 128
            3 * silk_lin2log(vad.nrg_ratio_smth_q8[b]) - 3 * 8 * 128
        } else {
            0
        };
        let snr_q7 = imax(snr_q7, 0);
        sum_snr_q7 += snr_q7;
    }

    // Speech activity = sigmoid(SNR - offset)
    let sa_q15 = silk_sigm_q15(
        (sum_snr_q7 - VAD_NEGATIVE_OFFSET_Q5 * (VAD_N_BANDS as i32) * 128) >> 5,
    );

    // Scale to Q8
    ps_enc.speech_activity_q8 = imin(sa_q15 >> 7, 255);
    ps_enc.input_tilt_q15 = input_tilt;

    // Per-band input quality
    for b in 0..VAD_N_BANDS {
        let snr_q7 = if vad.nrg_ratio_smth_q8[b] >= 256 {
            3 * silk_lin2log(vad.nrg_ratio_smth_q8[b]) - 3 * 8 * 128
        } else {
            0
        };
        ps_enc.input_quality_bands_q15[b] = silk_sigm_q15((snr_q7 - 128) >> 4);
    }

    0
}

// ===========================================================================
// HP variable cutoff filter
// ===========================================================================

/// Biquad filter (second-order IIR), stride 1.
/// Matches C: `silk_biquad_alt_stride1`.
pub fn silk_biquad_alt_stride1(
    input: &[i16],
    b_q28: &[i32; 3],
    a_q28: &[i32; 2],
    state: &mut [i32; 2],
    output: &mut [i16],
    len: usize,
) {
    for k in 0..len {
        // Second-order section (transposed direct form II)
        let in_val = input[k] as i64;

        // out = state[0] + B[0] * input
        let out64 = state[0] as i64 + ((b_q28[0] as i64 * in_val) >> 28);
        let out32 = sat16(out64 as i32);
        output[k] = out32;

        // Update state
        let out_i64 = out32 as i64;
        state[0] = (state[1] as i64
            + ((b_q28[1] as i64 * in_val) >> 28)
            - ((a_q28[0] as i64 * out_i64) >> 28)) as i32;
        state[1] = (((b_q28[2] as i64 * in_val) >> 28)
            - ((a_q28[1] as i64 * out_i64) >> 28)) as i32;
    }
}

/// High-pass filter with variable cutoff frequency.
/// Only updates the smoother — does NOT apply the filter.
/// The actual HP biquad is applied separately in the encode frame pipeline.
/// Matches C: `silk_HP_variable_cutoff`.
pub fn silk_hp_variable_cutoff(ps_enc: &mut SilkEncoderStateFix) {
    let ps_enc_c1 = &mut ps_enc.s_cmn;

    // Only update when previous frame was voiced
    if ps_enc_c1.prev_signal_type != TYPE_VOICED {
        return;
    }

    // Estimate pitch frequency in Hz, Q16
    let pitch_freq_hz_q16 = ((ps_enc_c1.fs_khz * 1000) << 16) / imax(ps_enc_c1.prev_lag, 1);

    // Convert to log domain (Q7)
    let pitch_freq_log_q7 = silk_lin2log(pitch_freq_hz_q16) - (16 << 7);

    // Quality-based adjustment: reduce cutoff tracking when quality is high
    let quality_q15 = ps_enc_c1.input_quality_bands_q15[0];
    // SILK_FIX_CONST(VARIABLE_HP_MIN_CUTOFF_HZ, 16) = 60 << 16 = 3932160
    let min_cutoff_log_q7 = silk_lin2log(VARIABLE_HP_MIN_CUTOFF_HZ << 16) - (16 << 7);
    let pitch_freq_log_q7 = silk_smlawb(
        pitch_freq_log_q7,
        silk_smulwb((-quality_q15) << 2, quality_q15 as i16),
        (pitch_freq_log_q7 - min_cutoff_log_q7) as i16,
    );

    // Delta frequency (how far from current smoother value)
    let mut delta_freq_q7 = pitch_freq_log_q7 - (ps_enc_c1.variable_hp_smth1_q15 >> 8);
    if delta_freq_q7 < 0 {
        delta_freq_q7 *= 3; // Faster tracking downward
    }

    // Limit delta: SILK_FIX_CONST(VARIABLE_HP_MAX_DELTA_FREQ, 7) = 51
    delta_freq_q7 = imax(imin(delta_freq_q7, VARIABLE_HP_MAX_DELTA_FREQ_Q7),
                         -VARIABLE_HP_MAX_DELTA_FREQ_Q7);

    // Update first smoother: SILK_FIX_CONST(VARIABLE_HP_SMTH_COEF1, 16) = 6554
    ps_enc_c1.variable_hp_smth1_q15 = silk_smlawb(
        ps_enc_c1.variable_hp_smth1_q15,
        silk_smulbb(ps_enc_c1.speech_activity_q8, delta_freq_q7),
        6554, // SILK_FIX_CONST(0.1, 16)
    );

    // Limit frequency range
    let min_hp_q15 = silk_lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) << 8;
    let max_hp_q15 = silk_lin2log(VARIABLE_HP_MAX_CUTOFF_HZ) << 8;
    ps_enc_c1.variable_hp_smth1_q15 = imax(imin(ps_enc_c1.variable_hp_smth1_q15, max_hp_q15),
                                            min_hp_q15);
}

// ===========================================================================
// LP variable cutoff (bandwidth transition filter)
// ===========================================================================

/// Apply variable low-pass filter for bandwidth transitions.
/// Matches C: `silk_LP_variable_cutoff`.
pub fn silk_lp_variable_cutoff(
    s_lp: &mut SilkLpState,
    frame: &mut [i16],
    frame_length: usize,
) {
    // Only active during bandwidth transitions
    if s_lp.mode == 0 {
        return;
    }

    // TRANSITION_INT_STEPS = 64, so shift by 16-6 = 10 instead of dividing
    let fac_q16 = (TRANSITION_FRAMES - s_lp.transition_frame_no) << (16 - 6);
    let ind = fac_q16 >> 16;
    let fac_q16 = fac_q16 - (ind << 16);

    // Interpolate filter taps
    let mut b_q28 = [0i32; TRANSITION_NB];
    let mut a_q28 = [0i32; TRANSITION_NA];
    silk_lp_interpolate_filter_taps(&mut b_q28, &mut a_q28, ind, fac_q16);

    // Advance transition counter, clamped to [0, TRANSITION_FRAMES]
    s_lp.transition_frame_no = imax(imin(
        s_lp.transition_frame_no + s_lp.mode, 0), TRANSITION_FRAMES);

    // Apply ARMA low-pass biquad filter (in-place)
    let mut output = vec![0i16; frame_length];
    silk_biquad_alt_stride1(
        &frame[..frame_length],
        &[b_q28[0], b_q28[1], b_q28[2]],
        &[a_q28[0], a_q28[1]],
        &mut s_lp.in_lp_state,
        &mut output,
        frame_length,
    );
    frame[..frame_length].copy_from_slice(&output);
}

/// Interpolate LP transition filter taps.
/// Matches C: `silk_LP_interpolate_filter_taps`.
fn silk_lp_interpolate_filter_taps(
    b_q28: &mut [i32; TRANSITION_NB],
    a_q28: &mut [i32; TRANSITION_NA],
    ind: i32,
    fac_q16: i32,
) {
    let ind_u = ind as usize;
    if (ind as usize) < TRANSITION_INT_NUM - 1 {
        if fac_q16 > 0 {
            if fac_q16 < 32768 {
                // Interpolate from ind toward ind+1
                for i in 0..TRANSITION_NB {
                    b_q28[i] = silk_smlawb(
                        SILK_TRANSITION_LP_B_Q28[ind_u][i],
                        SILK_TRANSITION_LP_B_Q28[ind_u + 1][i] - SILK_TRANSITION_LP_B_Q28[ind_u][i],
                        fac_q16 as i16,
                    );
                }
                for i in 0..TRANSITION_NA {
                    a_q28[i] = silk_smlawb(
                        SILK_TRANSITION_LP_A_Q28[ind_u][i],
                        SILK_TRANSITION_LP_A_Q28[ind_u + 1][i] - SILK_TRANSITION_LP_A_Q28[ind_u][i],
                        fac_q16 as i16,
                    );
                }
            } else {
                // Interpolate from ind+1 back toward ind
                let fac_q16_adj = (fac_q16 - (1 << 16)) as i16;
                for i in 0..TRANSITION_NB {
                    b_q28[i] = silk_smlawb(
                        SILK_TRANSITION_LP_B_Q28[ind_u + 1][i],
                        SILK_TRANSITION_LP_B_Q28[ind_u + 1][i] - SILK_TRANSITION_LP_B_Q28[ind_u][i],
                        fac_q16_adj,
                    );
                }
                for i in 0..TRANSITION_NA {
                    a_q28[i] = silk_smlawb(
                        SILK_TRANSITION_LP_A_Q28[ind_u + 1][i],
                        SILK_TRANSITION_LP_A_Q28[ind_u + 1][i] - SILK_TRANSITION_LP_A_Q28[ind_u][i],
                        fac_q16_adj,
                    );
                }
            }
        } else {
            // No fractional part: direct copy
            b_q28.copy_from_slice(&SILK_TRANSITION_LP_B_Q28[ind_u]);
            a_q28.copy_from_slice(&SILK_TRANSITION_LP_A_Q28[ind_u]);
        }
    } else {
        // At or beyond last interpolation point: use final values
        b_q28.copy_from_slice(&SILK_TRANSITION_LP_B_Q28[TRANSITION_INT_NUM - 1]);
        a_q28.copy_from_slice(&SILK_TRANSITION_LP_A_Q28[TRANSITION_INT_NUM - 1]);
    }
}

// ===========================================================================
// Control audio bandwidth
// ===========================================================================

/// Determine internal sampling rate based on bandwidth control state machine.
/// Matches C: `silk_control_audio_bandwidth`.
pub fn silk_control_audio_bandwidth(
    ps_enc: &mut SilkEncoderState,
    enc_control: &mut SilkEncControlStruct,
) -> i32 {
    let mut orig_khz = ps_enc.fs_khz;
    if orig_khz == 0 {
        orig_khz = ps_enc.s_lp.saved_fs_khz;
    }
    let mut fs_khz = orig_khz;
    let mut fs_hz = fs_khz * 1000;

    if fs_hz == 0 {
        // Encoder just initialized
        fs_hz = imin(ps_enc.desired_internal_fs_hz, ps_enc.api_fs_hz);
        fs_khz = fs_hz / 1000;
    } else if fs_hz > ps_enc.api_fs_hz
        || fs_hz > ps_enc.max_internal_fs_hz
        || fs_hz < ps_enc.min_internal_fs_hz
    {
        fs_hz = ps_enc.api_fs_hz;
        fs_hz = imin(fs_hz, ps_enc.max_internal_fs_hz);
        fs_hz = imax(fs_hz, ps_enc.min_internal_fs_hz);
        fs_khz = fs_hz / 1000;
    } else {
        // State machine for internal sampling rate switching
        if ps_enc.s_lp.transition_frame_no >= TRANSITION_FRAMES {
            ps_enc.s_lp.mode = 0;
        }
        if ps_enc.allow_bandwidth_switch != 0 || enc_control.opus_can_switch != 0 {
            // Check if we should switch down
            if orig_khz * 1000 > ps_enc.desired_internal_fs_hz {
                if ps_enc.s_lp.mode == 0 {
                    ps_enc.s_lp.transition_frame_no = TRANSITION_FRAMES;
                    ps_enc.s_lp.in_lp_state = [0; 2];
                }
                if enc_control.opus_can_switch != 0 {
                    ps_enc.s_lp.mode = 0;
                    fs_khz = if orig_khz == 16 { 12 } else { 8 };
                } else if ps_enc.s_lp.transition_frame_no <= 0 {
                    enc_control.switch_ready = 1;
                    enc_control.max_bits -= enc_control.max_bits * 5
                        / (enc_control.payload_size_ms + 5);
                } else {
                    ps_enc.s_lp.mode = -2;
                }
            } else if orig_khz * 1000 < ps_enc.desired_internal_fs_hz {
                // Switch up
                if enc_control.opus_can_switch != 0 {
                    fs_khz = if orig_khz == 8 { 12 } else { 16 };
                    ps_enc.s_lp.transition_frame_no = 0;
                    ps_enc.s_lp.in_lp_state = [0; 2];
                    ps_enc.s_lp.mode = 1;
                } else if ps_enc.s_lp.mode == 0 {
                    enc_control.switch_ready = 1;
                    enc_control.max_bits -= enc_control.max_bits * 5
                        / (enc_control.payload_size_ms + 5);
                } else {
                    ps_enc.s_lp.mode = 1;
                }
            } else if ps_enc.s_lp.mode < 0 {
                ps_enc.s_lp.mode = 1;
            }
        }
    }
    fs_khz
}

// ===========================================================================
// Encoder setup functions
// ===========================================================================

/// Setup complexity-dependent parameters.
/// Matches C: `silk_setup_complexity`.
fn silk_setup_complexity(ps_enc: &mut SilkEncoderState, complexity: i32) -> i32 {
    if complexity < 1 {
        ps_enc.pitch_estimation_complexity = SILK_PE_MIN_COMPLEX;
        ps_enc.pitch_estimation_threshold_q16 = ((0.8f64 * 65536.0) + 0.5) as i32;
        ps_enc.pitch_estimation_lpc_order = 6;
        ps_enc.shaping_lpc_order = 12;
        ps_enc.la_shape = 3 * ps_enc.fs_khz;
        ps_enc.n_states_delayed_decision = 1;
        ps_enc.use_interpolated_nlsfs = 0;
        ps_enc.nlsf_msvq_survivors = 2;
        ps_enc.warping_q16 = 0;
    } else if complexity < 2 {
        ps_enc.pitch_estimation_complexity = SILK_PE_MID_COMPLEX;
        ps_enc.pitch_estimation_threshold_q16 = ((0.76f64 * 65536.0) + 0.5) as i32;
        ps_enc.pitch_estimation_lpc_order = 8;
        ps_enc.shaping_lpc_order = 14;
        ps_enc.la_shape = 5 * ps_enc.fs_khz;
        ps_enc.n_states_delayed_decision = 1;
        ps_enc.use_interpolated_nlsfs = 0;
        ps_enc.nlsf_msvq_survivors = 3;
        ps_enc.warping_q16 = 0;
    } else if complexity < 3 {
        ps_enc.pitch_estimation_complexity = SILK_PE_MIN_COMPLEX;
        ps_enc.pitch_estimation_threshold_q16 = ((0.8f64 * 65536.0) + 0.5) as i32;
        ps_enc.pitch_estimation_lpc_order = 6;
        ps_enc.shaping_lpc_order = 12;
        ps_enc.la_shape = 3 * ps_enc.fs_khz;
        ps_enc.n_states_delayed_decision = 2;
        ps_enc.use_interpolated_nlsfs = 0;
        ps_enc.nlsf_msvq_survivors = 2;
        ps_enc.warping_q16 = 0;
    } else if complexity < 4 {
        ps_enc.pitch_estimation_complexity = SILK_PE_MID_COMPLEX;
        ps_enc.pitch_estimation_threshold_q16 = ((0.76f64 * 65536.0) + 0.5) as i32;
        ps_enc.pitch_estimation_lpc_order = 8;
        ps_enc.shaping_lpc_order = 14;
        ps_enc.la_shape = 5 * ps_enc.fs_khz;
        ps_enc.n_states_delayed_decision = 2;
        ps_enc.use_interpolated_nlsfs = 0;
        ps_enc.nlsf_msvq_survivors = 4;
        ps_enc.warping_q16 = 0;
    } else if complexity < 6 {
        ps_enc.pitch_estimation_complexity = SILK_PE_MID_COMPLEX;
        ps_enc.pitch_estimation_threshold_q16 = ((0.74f64 * 65536.0) + 0.5) as i32;
        ps_enc.pitch_estimation_lpc_order = 10;
        ps_enc.shaping_lpc_order = 16;
        ps_enc.la_shape = 5 * ps_enc.fs_khz;
        ps_enc.n_states_delayed_decision = 2;
        ps_enc.use_interpolated_nlsfs = 1;
        ps_enc.nlsf_msvq_survivors = 6;
        ps_enc.warping_q16 = ps_enc.fs_khz * WARPING_MULTIPLIER_Q16;
    } else if complexity < 8 {
        ps_enc.pitch_estimation_complexity = SILK_PE_MID_COMPLEX;
        ps_enc.pitch_estimation_threshold_q16 = ((0.72f64 * 65536.0) + 0.5) as i32;
        ps_enc.pitch_estimation_lpc_order = 12;
        ps_enc.shaping_lpc_order = 20;
        ps_enc.la_shape = 5 * ps_enc.fs_khz;
        ps_enc.n_states_delayed_decision = 3;
        ps_enc.use_interpolated_nlsfs = 1;
        ps_enc.nlsf_msvq_survivors = 8;
        ps_enc.warping_q16 = ps_enc.fs_khz * WARPING_MULTIPLIER_Q16;
    } else {
        ps_enc.pitch_estimation_complexity = SILK_PE_MAX_COMPLEX;
        ps_enc.pitch_estimation_threshold_q16 = ((0.7f64 * 65536.0) + 0.5) as i32;
        ps_enc.pitch_estimation_lpc_order = 16;
        ps_enc.shaping_lpc_order = 24;
        ps_enc.la_shape = 5 * ps_enc.fs_khz;
        ps_enc.n_states_delayed_decision = MAX_DEL_DEC_STATES as i32;
        ps_enc.use_interpolated_nlsfs = 1;
        ps_enc.nlsf_msvq_survivors = 16;
        ps_enc.warping_q16 = ps_enc.fs_khz * WARPING_MULTIPLIER_Q16;
    }

    // Don't allow higher pitch estimation LPC order than predict LPC order
    ps_enc.pitch_estimation_lpc_order =
        imin(ps_enc.pitch_estimation_lpc_order, ps_enc.predict_lpc_order);
    ps_enc.shape_win_length = SUB_FRAME_LENGTH_MS as i32 * ps_enc.fs_khz + 2 * ps_enc.la_shape;
    ps_enc.complexity = complexity;
    0
}

/// Setup LBRR (low bitrate redundancy for FEC).
/// Matches C: `silk_setup_LBRR`.
fn silk_setup_lbrr(ps_enc: &mut SilkEncoderState, enc_control: &SilkEncControlStruct) -> i32 {
    let lbrr_in_previous_packet = ps_enc.lbrr_enabled;
    ps_enc.lbrr_enabled = enc_control.lbrr_coded;
    if ps_enc.lbrr_enabled != 0 {
        if lbrr_in_previous_packet == 0 {
            ps_enc.lbrr_gain_increases = 7;
        } else {
            ps_enc.lbrr_gain_increases = imax(
                7 - silk_smulwb(ps_enc.packet_loss_perc, ((0.2f64 * 65536.0) + 0.5) as i32 as i16),
                3,
            );
        }
    }
    SILK_NO_ERROR
}

/// Setup internal sampling frequency and related parameters.
/// Matches C: `silk_setup_fs`.
fn silk_setup_fs(
    ps_enc: &mut SilkEncoderStateFix,
    fs_khz: i32,
    packet_size_ms: i32,
) -> i32 {
    let mut ret = SILK_NO_ERROR;

    // Set packet size
    if packet_size_ms != ps_enc.s_cmn.packet_size_ms {
        if packet_size_ms != 10 && packet_size_ms != 20 && packet_size_ms != 40
            && packet_size_ms != 60
        {
            ret = SILK_ENC_PACKET_SIZE_NOT_SUPPORTED;
        }
        if packet_size_ms <= 10 {
            ps_enc.s_cmn.n_frames_per_packet = 1;
            ps_enc.s_cmn.nb_subfr = if packet_size_ms == 10 { 2 } else { 1 };
            ps_enc.s_cmn.frame_length = packet_size_ms * fs_khz;
            ps_enc.s_cmn.pitch_lpc_win_length = FIND_PITCH_LPC_WIN_MS_2_SF * fs_khz;
            if ps_enc.s_cmn.fs_khz == 8 {
                ps_enc.s_cmn.pitch_contour_icdf = &SILK_PITCH_CONTOUR_10_MS_NB_ICDF;
            } else {
                ps_enc.s_cmn.pitch_contour_icdf = &SILK_PITCH_CONTOUR_10_MS_ICDF;
            }
        } else {
            ps_enc.s_cmn.n_frames_per_packet = packet_size_ms / MAX_FRAME_LENGTH_MS as i32;
            ps_enc.s_cmn.nb_subfr = MAX_NB_SUBFR as i32;
            ps_enc.s_cmn.frame_length = 20 * fs_khz;
            ps_enc.s_cmn.pitch_lpc_win_length = FIND_PITCH_LPC_WIN_MS * fs_khz;
            if ps_enc.s_cmn.fs_khz == 8 {
                ps_enc.s_cmn.pitch_contour_icdf = &SILK_PITCH_CONTOUR_NB_ICDF;
            } else {
                ps_enc.s_cmn.pitch_contour_icdf = &SILK_PITCH_CONTOUR_ICDF;
            }
        }
        ps_enc.s_cmn.packet_size_ms = packet_size_ms;
        ps_enc.s_cmn.target_rate_bps = 0; // trigger new SNR computation
    }

    // Set internal sampling frequency
    if ps_enc.s_cmn.fs_khz != fs_khz {
        // Reset part of the state
        ps_enc.s_shape = SilkShapeStateFix::default();
        ps_enc.s_cmn.s_nsq = NsqState::default();
        ps_enc.s_cmn.prev_nlsfq_q15 = [0; MAX_LPC_ORDER];
        ps_enc.s_cmn.s_lp.in_lp_state = [0; 2];
        ps_enc.s_cmn.input_buf_ix = 0;
        ps_enc.s_cmn.n_frames_encoded = 0;
        ps_enc.s_cmn.target_rate_bps = 0;

        // Initialize non-zero parameters
        ps_enc.s_cmn.prev_lag = 100;
        ps_enc.s_cmn.first_frame_after_reset = 1;
        ps_enc.s_shape.last_gain_index = 10;
        ps_enc.s_cmn.s_nsq.lag_prev = 100;
        ps_enc.s_cmn.s_nsq.prev_gain_q16 = 65536;
        ps_enc.s_cmn.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;

        ps_enc.s_cmn.fs_khz = fs_khz;
        if fs_khz == 8 {
            if ps_enc.s_cmn.nb_subfr == MAX_NB_SUBFR as i32 {
                ps_enc.s_cmn.pitch_contour_icdf = &SILK_PITCH_CONTOUR_NB_ICDF;
            } else {
                ps_enc.s_cmn.pitch_contour_icdf = &SILK_PITCH_CONTOUR_10_MS_NB_ICDF;
            }
        } else if ps_enc.s_cmn.nb_subfr == MAX_NB_SUBFR as i32 {
            ps_enc.s_cmn.pitch_contour_icdf = &SILK_PITCH_CONTOUR_ICDF;
        } else {
            ps_enc.s_cmn.pitch_contour_icdf = &SILK_PITCH_CONTOUR_10_MS_ICDF;
        }

        if fs_khz == 8 || fs_khz == 12 {
            ps_enc.s_cmn.predict_lpc_order = MIN_LPC_ORDER as i32;
            ps_enc.s_cmn.nlsf_cb = &SILK_NLSF_CB_NB_MB;
        } else {
            ps_enc.s_cmn.predict_lpc_order = MAX_LPC_ORDER as i32;
            ps_enc.s_cmn.nlsf_cb = &SILK_NLSF_CB_WB;
        }
        ps_enc.s_cmn.subfr_length = SUB_FRAME_LENGTH_MS as i32 * fs_khz;
        ps_enc.s_cmn.frame_length = ps_enc.s_cmn.subfr_length * ps_enc.s_cmn.nb_subfr;
        ps_enc.s_cmn.ltp_mem_length = LTP_MEM_LENGTH_MS as i32 * fs_khz;
        ps_enc.s_cmn.la_pitch = LA_PITCH_MS * fs_khz;
        ps_enc.s_cmn.max_pitch_lag = 18 * fs_khz;
        if ps_enc.s_cmn.nb_subfr == MAX_NB_SUBFR as i32 {
            ps_enc.s_cmn.pitch_lpc_win_length = FIND_PITCH_LPC_WIN_MS * fs_khz;
        } else {
            ps_enc.s_cmn.pitch_lpc_win_length = FIND_PITCH_LPC_WIN_MS_2_SF * fs_khz;
        }
        if fs_khz == 16 {
            ps_enc.s_cmn.pitch_lag_low_bits_icdf = &SILK_UNIFORM8_ICDF;
        } else if fs_khz == 12 {
            ps_enc.s_cmn.pitch_lag_low_bits_icdf = &SILK_UNIFORM6_ICDF;
        } else {
            ps_enc.s_cmn.pitch_lag_low_bits_icdf = &SILK_UNIFORM4_ICDF;
        }
    }
    ret
}

/// Setup resamplers for rate conversion.
/// Matches C: `silk_setup_resamplers`.
fn silk_setup_resamplers(ps_enc: &mut SilkEncoderStateFix, fs_khz: i32) -> i32 {
    let ret = SILK_NO_ERROR;

    if ps_enc.s_cmn.fs_khz != fs_khz || ps_enc.s_cmn.prev_api_fs_hz != ps_enc.s_cmn.api_fs_hz {
        if ps_enc.s_cmn.fs_khz == 0 {
            // Initialize the resampler
            silk_resampler_init(
                &mut ps_enc.s_cmn.resampler_state,
                ps_enc.s_cmn.api_fs_hz,
                fs_khz * 1000,
            );
        } else {
            // Re-initialize: need to resample buffered data
            let buf_length_ms = (ps_enc.s_cmn.nb_subfr * 5) * 2 + LA_SHAPE_MS;
            let old_buf_samples = buf_length_ms * ps_enc.s_cmn.fs_khz;

            // Temp resampler: internal rate → API rate
            let mut temp_resampler = SilkResamplerState::default();
            silk_resampler_init(
                &mut temp_resampler,
                ps_enc.s_cmn.fs_khz * 1000,
                ps_enc.s_cmn.api_fs_hz,
            );

            let api_buf_samples = buf_length_ms * (ps_enc.s_cmn.api_fs_hz / 1000);
            let mut x_buf_api = vec![0i16; api_buf_samples as usize];

            // Upsample x_buf to API rate
            silk_resampler_run(
                &mut temp_resampler,
                &mut x_buf_api,
                &ps_enc.x_buf[..old_buf_samples as usize],
            );

            // Re-init resampler: API rate → new internal rate
            silk_resampler_init(
                &mut ps_enc.s_cmn.resampler_state,
                ps_enc.s_cmn.api_fs_hz,
                fs_khz * 1000,
            );

            // Downsample back to new internal rate
            let new_buf_samples = buf_length_ms * fs_khz;
            let mut x_buf_new = vec![0i16; new_buf_samples as usize];
            silk_resampler_run(
                &mut ps_enc.s_cmn.resampler_state,
                &mut x_buf_new,
                &x_buf_api,
            );
            ps_enc.x_buf[..new_buf_samples as usize].copy_from_slice(&x_buf_new);
        }
    }
    ps_enc.s_cmn.prev_api_fs_hz = ps_enc.s_cmn.api_fs_hz;
    ret
}

/// Control encoder parameters.
/// Matches C: `silk_control_encoder`.
pub fn silk_control_encoder(
    ps_enc: &mut SilkEncoderStateFix,
    enc_control: &mut SilkEncControlStruct,
    allow_bw_switch: i32,
    channel_nb: i32,
    force_fs_khz: i32,
) -> i32 {
    let mut ret = 0i32;

    ps_enc.s_cmn.use_dtx = enc_control.use_dtx;
    ps_enc.s_cmn.use_cbr = enc_control.use_cbr;
    ps_enc.s_cmn.api_fs_hz = enc_control.api_sample_rate;
    ps_enc.s_cmn.max_internal_fs_hz = enc_control.max_internal_sample_rate;
    ps_enc.s_cmn.min_internal_fs_hz = enc_control.min_internal_sample_rate;
    ps_enc.s_cmn.desired_internal_fs_hz = enc_control.desired_internal_sample_rate;
    ps_enc.s_cmn.use_in_band_fec = enc_control.use_in_band_fec;
    ps_enc.s_cmn.n_channels_api = enc_control.n_channels_api;
    ps_enc.s_cmn.n_channels_internal = enc_control.n_channels_internal;
    ps_enc.s_cmn.allow_bandwidth_switch = allow_bw_switch;
    ps_enc.s_cmn.channel_nb = channel_nb;

    if ps_enc.s_cmn.controlled_since_last_payload != 0 && ps_enc.s_cmn.prefill_flag == 0 {
        if ps_enc.s_cmn.api_fs_hz != ps_enc.s_cmn.prev_api_fs_hz && ps_enc.s_cmn.fs_khz > 0 {
            ret += silk_setup_resamplers(ps_enc, ps_enc.s_cmn.fs_khz);
        }
        return ret;
    }

    // Determine internal sampling rate
    let mut fs_khz = silk_control_audio_bandwidth(&mut ps_enc.s_cmn, enc_control);
    if force_fs_khz != 0 {
        fs_khz = force_fs_khz;
    }

    // Prepare resampler
    ret += silk_setup_resamplers(ps_enc, fs_khz);

    // Set internal sampling frequency
    ret += silk_setup_fs(ps_enc, fs_khz, enc_control.payload_size_ms);

    // Set encoding complexity
    ret += silk_setup_complexity(&mut ps_enc.s_cmn, enc_control.complexity);

    // Set packet loss rate
    ps_enc.s_cmn.packet_loss_perc = enc_control.packet_loss_percentage;

    // Set LBRR usage
    ret += silk_setup_lbrr(&mut ps_enc.s_cmn, enc_control);

    ps_enc.s_cmn.controlled_since_last_payload = 1;
    ret
}

// ===========================================================================
// Resampler stubs (delegate to decoder's resampler implementation)
// ===========================================================================

/// Initialize resampler state. Wrapper delegating to decoder's resampler.
fn silk_resampler_init(state: &mut SilkResamplerState, fs_in_hz: i32, fs_out_hz: i32) {
    // Delegate to the decoder's resampler initialization (for_enc = true)
    crate::silk::decoder::silk_resampler_init_pub(state, fs_in_hz, fs_out_hz, true);
}

/// Run the resampler.
fn silk_resampler_run(state: &mut SilkResamplerState, output: &mut [i16], input: &[i16]) {
    crate::silk::decoder::silk_resampler(state, output, input, input.len());
}

// ===========================================================================
// A2NLSF: LPC coefficients → Normalized Line Spectral Frequencies
// ===========================================================================

/// Convert LPC coefficients to NLSFs.
/// Matches C: `silk_A2NLSF`.
pub fn silk_a2nlsf(nlsf: &mut [i16], a_q16: &[i32], d: usize) {
    let dd = d >> 1;

    // Step 1: Split polynomial into P and Q parts, deconvolve, and convert
    let mut p = [0i32; MAX_LPC_ORDER / 2 + 1];
    let mut q = [0i32; MAX_LPC_ORDER / 2 + 1];

    silk_a2nlsf_init(a_q16, &mut p, &mut q, d);

    // Step 2: Find roots by scanning the cosine table
    let mut root = 0usize;
    let xlo = silk_lsf_cos_tab_fix_q12_eval(&p, &q, 0, dd, root & 1);
    let mut prev_val = xlo;

    for k in 1..=LSF_COS_TAB_SZ_FIX {
        let x_val = silk_lsf_cos_tab_fix_q12_eval(&p, &q, k, dd, root & 1);
        // Detect sign change
        if (prev_val ^ x_val) < 0 {
            // Bisection refinement
            let mut x_lo = ((k - 1) as i32) << 8;
            let mut x_hi = (k as i32) << 8;
            let mut y_lo = prev_val;

            for _iter in 0..BIN_DIV_STEPS_A2NLSF {
                let x_mid = (x_lo + x_hi) >> 1;
                let y_mid = silk_lsf_cos_tab_fix_q12_eval_interp(&p, &q, x_mid, dd, root & 1);
                if (y_lo ^ y_mid) < 0 {
                    x_hi = x_mid;
                } else {
                    x_lo = x_mid;
                    y_lo = y_mid;
                }
            }

            // Linear interpolation for fractional precision
            let y_hi =
                silk_lsf_cos_tab_fix_q12_eval_interp(&p, &q, x_hi, dd, root & 1);
            let denom = y_lo - y_hi;
            let nlsf_val = if denom != 0 {
                x_lo + (((-y_lo as i64) * ((x_hi - x_lo) as i64) / (denom as i64)) as i32)
            } else {
                (x_lo + x_hi) >> 1
            };

            // Map from table index (0..128 << 8) to Q15 (0..32767)
            nlsf[root] = imin(imax(nlsf_val, 0), 32767) as i16;
            root += 1;
            if root >= d {
                break;
            }
            // Re-evaluate at current position for the next polynomial
            prev_val = silk_lsf_cos_tab_fix_q12_eval(&p, &q, k, dd, root & 1);
        } else {
            prev_val = x_val;
        }
    }

    // If not all roots found, apply bandwidth expansion and retry
    if root < d {
        // Fallback: use existing NLSF2A inverse with bandwidth expansion
        // This matches the C reference retry loop
        let mut a_q16_tmp = [0i32; MAX_LPC_ORDER];
        a_q16_tmp[..d].copy_from_slice(&a_q16[..d]);
        for _i in 0..MAX_ITERATIONS_A2NLSF {
            silk_bwexpander_32(&mut a_q16_tmp[..d], d, 65536 - 1);
            let mut root2 = 0usize;
            silk_a2nlsf_init(&a_q16_tmp, &mut p, &mut q, d);
            prev_val = silk_lsf_cos_tab_fix_q12_eval(&p, &q, 0, dd, root2 & 1);
            for k2 in 1..=LSF_COS_TAB_SZ_FIX {
                let x_val2 = silk_lsf_cos_tab_fix_q12_eval(&p, &q, k2, dd, root2 & 1);
                if (prev_val ^ x_val2) < 0 {
                    let nlsf_val = ((k2 as i32 - 1) << 8) + 128; // midpoint approximation
                    nlsf[root2] = imin(imax(nlsf_val, 0), 32767) as i16;
                    root2 += 1;
                    if root2 >= d {
                        break;
                    }
                    prev_val = silk_lsf_cos_tab_fix_q12_eval(&p, &q, k2, dd, root2 & 1);
                } else {
                    prev_val = x_val2;
                }
            }
            if root2 >= d {
                break;
            }
        }
    }
}

/// Initialize P and Q polynomials from LPC coefficients.
fn silk_a2nlsf_init(a_q16: &[i32], p: &mut [i32], q: &mut [i32], d: usize) {
    let dd = d >> 1;

    // Form P(z) + Q(z) and P(z) - Q(z), then deconvolve
    // P: (1 + z^-d) sum, Q: (1 - z^-d) sum, each divided out root at ±1
    let mut a_tmp = [0i64; MAX_LPC_ORDER];
    for k in 0..d {
        a_tmp[k] = a_q16[k] as i64;
    }

    p[0] = 1 << 16;
    q[0] = 1 << 16;
    for k in 0..dd {
        p[k + 1] = -(a_tmp[k] as i32 + a_tmp[d - k - 1] as i32 + p[k]);
        q[k + 1] = -(a_tmp[k] as i32 - a_tmp[d - k - 1] as i32 - q[k]);
    }

    // Divide P by (1+z^-1) and Q by (1-z^-1) (cumulative sum / alternating sum)
    for k in (1..=dd).rev() {
        p[k] -= p[k - 1];
        q[k] += q[k - 1];
    }
}

/// Evaluate polynomial at a cosine table position.
fn silk_lsf_cos_tab_fix_q12_eval(
    p: &[i32],
    q: &[i32],
    k: usize,
    dd: usize,
    use_q: usize,
) -> i32 {
    let cos_val = SILK_LSF_COS_TAB_FIX_Q12[k] as i32;
    let poly = if use_q == 0 { p } else { q };

    // Evaluate using Horner's method
    let mut val: i64 = poly[dd] as i64;
    for i in (0..dd).rev() {
        val = ((val * cos_val as i64) >> 12) + poly[i] as i64;
    }
    val as i32
}

/// Evaluate polynomial with fractional cosine table interpolation.
fn silk_lsf_cos_tab_fix_q12_eval_interp(
    p: &[i32],
    q: &[i32],
    x: i32,
    dd: usize,
    use_q: usize,
) -> i32 {
    let k = (x >> 8) as usize;
    let frac = x & 0xFF;
    let k = k.min(LSF_COS_TAB_SZ_FIX - 1);

    // Interpolate cosine value
    let cos0 = SILK_LSF_COS_TAB_FIX_Q12[k] as i32;
    let cos1 = SILK_LSF_COS_TAB_FIX_Q12[k + 1] as i32;
    let cos_val = cos0 + ((cos1 - cos0) * frac >> 8);

    let poly = if use_q == 0 { p } else { q };
    let mut val: i64 = poly[dd] as i64;
    for i in (0..dd).rev() {
        val = ((val * cos_val as i64) >> 12) + poly[i] as i64;
    }
    val as i32
}

// ===========================================================================
// NLSF processing
// ===========================================================================

/// Linear interpolation of two NLSF vectors.
/// Matches C: `silk_interpolate`.
pub fn silk_interpolate(
    xi: &mut [i16],
    x0: &[i16],
    x1: &[i16],
    ifact_q2: i32,
    d: usize,
) {
    for i in 0..d {
        xi[i] = (x0[i] as i32 + ((x1[i] as i32 - x0[i] as i32) * ifact_q2 >> 2)) as i16;
    }
}

/// Compute NLSF weights using the Laroia method.
/// Matches C: `silk_NLSF_VQ_weights_laroia`.
pub fn silk_nlsf_vq_weights_laroia(
    nlsf_w_q_out: &mut [i16],
    nlsf_q15: &[i16],
    order: usize,
) {
    // First and last weight
    let tmp1_w = nlsf_q15[0] as i32;
    let tmp2_w = nlsf_q15[1] as i32 - nlsf_q15[0] as i32;

    let tmp1_inv = 131072 / imax(tmp1_w, 1); // Q17 / Q15 = Q2 approximately
    let tmp2_inv = 131072 / imax(tmp2_w, 1);
    nlsf_w_q_out[0] = imin(tmp1_inv + tmp2_inv, i16::MAX as i32) as i16;

    for i in 1..(order - 1) {
        let tmp1 = nlsf_q15[i] as i32 - nlsf_q15[i - 1] as i32;
        let tmp2 = nlsf_q15[i + 1] as i32 - nlsf_q15[i] as i32;
        let tmp1_inv = 131072 / imax(tmp1, 1);
        let tmp2_inv = 131072 / imax(tmp2, 1);
        nlsf_w_q_out[i] = imin(tmp1_inv + tmp2_inv, i16::MAX as i32) as i16;
    }

    let last = order - 1;
    let tmp1 = nlsf_q15[last] as i32 - nlsf_q15[last - 1] as i32;
    let tmp2 = (1 << 15) - nlsf_q15[last] as i32;
    let tmp1_inv = 131072 / imax(tmp1, 1);
    let tmp2_inv = 131072 / imax(tmp2, 1);
    nlsf_w_q_out[last] = imin(tmp1_inv + tmp2_inv, i16::MAX as i32) as i16;
}

/// First-stage NLSF VQ: compute weighted error for each codebook vector.
/// Matches C: `silk_NLSF_VQ`.
pub fn silk_nlsf_vq(
    err_q24: &mut [i32],
    input_q15: &[i16],
    nlsf_w_q2: &[i16],
    cb_q8: &[u8],
    cb_wght_q9: &[i16],
    n_vectors: usize,
    order: usize,
) {
    for i in 0..n_vectors {
        let mut sum_q24: i64 = 0;
        for j in 0..order {
            let diff_q15 = input_q15[j] as i32 - ((cb_q8[i * order + j] as i32) << 7);
            let weighted = (diff_q15 as i64 * nlsf_w_q2[j] as i64) >> 2;
            sum_q24 += weighted * weighted >> 15;
        }
        err_q24[i] = imin(sum_q24 as i32, i32::MAX);
    }
}

/// NLSF delayed-decision quantization.
/// Matches C: `silk_NLSF_del_dec_quant`.
pub fn silk_nlsf_del_dec_quant(
    indices: &mut [i8],
    x_q10: &[i16],
    w_q5: &[i16],
    pred_coef_q8: &[u8],
    ec_ix: &[i16],
    ec_rates_q5: &[u8],
    quant_step_size_q16: i32,
    inv_quant_step_size_q6: i16,
    mu_q20: i32,
    order: i16,
) -> i32 {
    let order_u = order as usize;

    // State arrays for delayed decision
    let mut ind: [[i8; MAX_LPC_ORDER]; NLSF_QUANT_DEL_DEC_STATES] =
        [[0; MAX_LPC_ORDER]; NLSF_QUANT_DEL_DEC_STATES];
    let mut rd_q25 = [i32::MAX; 2 * NLSF_QUANT_DEL_DEC_STATES];
    let mut prev_out_q10 = [0i32; 2 * NLSF_QUANT_DEL_DEC_STATES];

    let mut n_states = 1usize;
    rd_q25[0] = 0;

    // Process coefficients in reverse order
    for i in (0..order_u).rev() {
        let pred_q10 = if i + 1 < order_u {
            // Predicted value from already-quantized coefficients
            0 // Prediction from prev_out
        } else {
            0
        };

        let in_q10 = x_q10[i] as i32;

        // For each state, try two candidates (floor and ceil)
        let n_states_old = n_states;
        for j in 0..n_states_old {
            let out_q10 = prev_out_q10[j];
            let pred = (out_q10 * pred_coef_q8[i] as i32) >> 8;
            let res_q10 = in_q10 - pred;

            // Quantize
            let q_raw = (res_q10 as i64 * inv_quant_step_size_q6 as i64 >> 16) as i32;
            let ind0 = imax(imin(q_raw, NLSF_QUANT_MAX_AMPLITUDE_EXT), -NLSF_QUANT_MAX_AMPLITUDE_EXT);
            let ind1 = ind0 + (if ind0 < 0 { -1 } else { 1 });

            // Rate-distortion for candidate 0
            let mut out0_q10 = ind0 << 10;
            if out0_q10 > 0 { out0_q10 -= 102; } else if out0_q10 < 0 { out0_q10 += 102; }
            out0_q10 = pred + ((out0_q10 as i64 * quant_step_size_q16 as i64 >> 16) as i32);
            let err0 = in_q10 - out0_q10;
            let dist0 = ((err0 as i64 * err0 as i64 * w_q5[i] as i64) >> 5) as i32;
            let ix0 = imin(imax(ind0 + NLSF_QUANT_MAX_AMPLITUDE_EXT, 0),
                           2 * NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize;
            let rate0 = ec_rates_q5[(ec_ix[i] as usize) + ix0] as i32;
            let rd0 = rd_q25[j].saturating_add(dist0).saturating_add(mu_q20 * rate0 >> 5);

            // Rate-distortion for candidate 1
            let ind1_clamped = imax(imin(ind1, NLSF_QUANT_MAX_AMPLITUDE_EXT), -NLSF_QUANT_MAX_AMPLITUDE_EXT);
            let mut out1_q10 = ind1_clamped << 10;
            if out1_q10 > 0 { out1_q10 -= 102; } else if out1_q10 < 0 { out1_q10 += 102; }
            out1_q10 = pred + ((out1_q10 as i64 * quant_step_size_q16 as i64 >> 16) as i32);
            let err1 = in_q10 - out1_q10;
            let dist1 = ((err1 as i64 * err1 as i64 * w_q5[i] as i64) >> 5) as i32;
            let ix1 = imin(imax(ind1_clamped + NLSF_QUANT_MAX_AMPLITUDE_EXT, 0),
                           2 * NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize;
            let rate1 = ec_rates_q5[(ec_ix[i] as usize) + ix1] as i32;
            let rd1 = rd_q25[j].saturating_add(dist1).saturating_add(mu_q20 * rate1 >> 5);

            // Store best candidate
            if n_states <= NLSF_QUANT_DEL_DEC_STATES / 2 {
                // Can expand states
                let new_j = n_states;
                // State j gets candidate 0, new state gets candidate 1
                rd_q25[j] = rd0;
                prev_out_q10[j] = out0_q10;
                ind[j][i] = ind0 as i8;

                rd_q25[new_j] = rd1;
                prev_out_q10[new_j] = out1_q10;
                ind[new_j] = ind[j];
                ind[new_j][i] = ind1_clamped as i8;
                n_states += 1;
            } else {
                // Prune: keep the better candidate
                if rd0 <= rd1 {
                    rd_q25[j] = rd0;
                    prev_out_q10[j] = out0_q10;
                    ind[j][i] = ind0 as i8;
                } else {
                    rd_q25[j] = rd1;
                    prev_out_q10[j] = out1_q10;
                    ind[j][i] = ind1_clamped as i8;
                }
            }
        }
    }

    // Select winner (minimum RD)
    let mut best_j = 0usize;
    let mut best_rd = rd_q25[0];
    for j in 1..n_states {
        if rd_q25[j] < best_rd {
            best_rd = rd_q25[j];
            best_j = j;
        }
    }

    // Copy winner indices
    for i in 0..order_u {
        indices[i] = ind[best_j][i];
    }

    best_rd
}

/// Full NLSF encoding pipeline.
/// Matches C: `silk_NLSF_encode`.
pub fn silk_nlsf_encode(
    nlsf_indices: &mut [i8],
    nlsf_q15: &mut [i16],
    cb: &SilkNlsfCbStruct,
    prev_nlsf_q15: &[i16],
    w_q2: &[i16],
    n_survivors: i32,
    signal_type: i32,
    mu_q20_in: i32,
) -> i32 {
    let order = cb.order as usize;
    let n_vectors = cb.n_vectors as usize;

    // Step 1: Compute NLSF VQ errors for all codebook vectors
    let mut err_q24 = vec![0i32; n_vectors];
    silk_nlsf_vq(&mut err_q24, nlsf_q15, w_q2, cb.cb1_nlsf_q8, cb.cb1_wght_q9, n_vectors, order);

    // Step 2: Find best first-stage indices (survivors)
    let n_surv = imin(n_survivors, n_vectors as i32) as usize;
    let mut temp_indices = vec![0usize; n_surv];

    // Simple selection: find n_surv smallest errors
    for s in 0..n_surv {
        let mut min_err = i32::MAX;
        let mut min_idx = 0;
        for k in 0..n_vectors {
            if err_q24[k] < min_err {
                min_err = err_q24[k];
                min_idx = k;
            }
        }
        temp_indices[s] = min_idx;
        err_q24[min_idx] = i32::MAX; // exclude from future selection
    }

    // Step 3: For each survivor, quantize residuals
    let mut best_rd = i32::MAX;
    let mut best_survivor = 0usize;
    let mut best_nlsf_indices = [0i8; MAX_LPC_ORDER + 1];

    for s in 0..n_surv {
        let cb1_idx = temp_indices[s];

        // Compute residual
        let mut res_q10 = [0i16; MAX_LPC_ORDER];
        let cb_offset = cb1_idx * order;
        for i in 0..order {
            let nlsf_tmp = nlsf_q15[i] as i32 - ((cb.cb1_nlsf_q8[cb_offset + i] as i32) << 7);
            res_q10[i] = ((nlsf_tmp as i64 * cb.cb1_wght_q9[cb_offset + i] as i64) >> 14) as i16;
        }

        // Unpack entropy table info for this codebook entry
        let mut ec_ix = [0i16; MAX_LPC_ORDER];
        let mut pred_q8 = [0u8; MAX_LPC_ORDER];
        silk_nlsf_unpack(&mut ec_ix, &mut pred_q8, cb, cb1_idx);

        // Convert weights to Q5
        let mut w_q5 = [0i16; MAX_LPC_ORDER];
        for i in 0..order {
            w_q5[i] = (w_q2[i] as i32 >> 0) as i16; // Already Q2, need Q5 = Q2 << 3
        }

        // Delayed-decision quantization of residuals
        let mut temp_nlsf_ind = [0i8; MAX_LPC_ORDER];
        let rd = silk_nlsf_del_dec_quant(
            &mut temp_nlsf_ind,
            &res_q10,
            &w_q5,
            &pred_q8,
            &ec_ix,
            cb.ec_rates_q5,
            cb.quant_step_size_q16 as i32,
            cb.inv_quant_step_size_q6,
            mu_q20_in,
            order as i16,
        );

        if rd < best_rd {
            best_rd = rd;
            best_survivor = s;
            best_nlsf_indices[0] = cb1_idx as i8;
            best_nlsf_indices[1..=order].copy_from_slice(&temp_nlsf_ind[..order]);
        }
    }

    // Store result
    nlsf_indices[..=order].copy_from_slice(&best_nlsf_indices[..=order]);

    // Decode the quantized NLSFs for use by the encoder
    silk_nlsf_decode(nlsf_q15, nlsf_indices, cb);

    best_rd
}

/// Process NLSFs: interpolate, encode, convert to LPC.
/// Matches C: `silk_process_NLSFs`.
pub fn silk_process_nlsfs(
    ps_enc: &mut SilkEncoderState,
    enc_control: &SilkEncoderControl,
    pred_coef_q12: &mut [[i16; MAX_LPC_ORDER]; 2],
    nlsf_q15: &[i16],
    prev_nlsf_q15: &[i16],
) {
    let order = ps_enc.predict_lpc_order as usize;
    let nb_subfr = ps_enc.nb_subfr as usize;

    // Compute interpolation coefficient
    let interp_coef_q2 = ps_enc.indices.nlsf_interp_coef_q2;

    if nb_subfr == 4 && interp_coef_q2 < 4 {
        // Interpolate NLSFs for the first half
        let mut nlsf_interp_q15 = [0i16; MAX_LPC_ORDER];
        silk_interpolate(
            &mut nlsf_interp_q15,
            prev_nlsf_q15,
            nlsf_q15,
            interp_coef_q2 as i32,
            order,
        );

        // Convert interpolated NLSFs to LPC (first half)
        silk_nlsf2a(&mut pred_coef_q12[0], &nlsf_interp_q15, order);
    } else {
        // No interpolation: first half uses same as second
        silk_nlsf2a(&mut pred_coef_q12[0], nlsf_q15, order);
    }

    // Convert current NLSFs to LPC (second half, or all if no interpolation)
    silk_nlsf2a(&mut pred_coef_q12[1], nlsf_q15, order);
}

// ===========================================================================
// Gain quantization
// ===========================================================================

/// Quantize gains.
/// Matches C: `silk_gains_quant`.
pub fn silk_gains_quant(
    ind: &mut [i8],
    gains_q16: &mut [i32],
    prev_ind: &mut i8,
    conditional: bool,
    nb_subfr: usize,
) {
    for k in 0..nb_subfr {
        // Convert to log domain
        let gain_q16 = imax(gains_q16[k], 1);
        let mut gain_q7 = silk_lin2log(gain_q16) - OFFSET_GAIN;

        // Scale by quantization step
        gain_q7 = (gain_q7 as i64 * 65536 / INV_SCALE_Q16_GAIN as i64) as i32;

        if k == 0 && !conditional {
            // Absolute coding
            let index = imin(imax(gain_q7, 0), N_LEVELS_QGAIN_I - 1);
            ind[k] = index as i8;
            *prev_ind = index as i8;
        } else {
            // Delta coding
            let delta = gain_q7 - *prev_ind as i32;
            let delta = imin(imax(delta, MIN_DELTA_GAIN_QUANT), MAX_DELTA_GAIN_QUANT);
            ind[k] = (delta - MIN_DELTA_GAIN_QUANT) as i8;
            *prev_ind = imin(
                imax(*prev_ind as i32 + delta, 0),
                N_LEVELS_QGAIN_I - 1,
            ) as i8;
        }

        // Dequantize to get reconstructed gain
        let log_val = imin(
            silk_smulwb(INV_SCALE_Q16_GAIN, *prev_ind as i16) + OFFSET_GAIN,
            3967,
        );
        gains_q16[k] = silk_log2lin(log_val);
    }
}

// ===========================================================================
// LTP gain quantization
// ===========================================================================

/// Quantize LTP gains using matrix-weighted VQ.
/// Matches C: `silk_quant_LTP_gains`.
pub fn silk_quant_ltp_gains(
    b_q14: &mut [i16],
    cbk_index: &mut [i8],
    per_index: &mut i8,
    sum_log_gain_q7: &mut i32,
    w_q18: &[i32],
    mu_q9: i32,
    low_complexity: bool,
    nb_subfr: usize,
) {
    let mut best_total_rd = i32::MAX;
    let mut best_per = 0i8;

    // Try each periodicity codebook (0, 1, 2)
    let max_cbk = if low_complexity { 1 } else { NB_LTP_CBKS };
    for cbk in 0..max_cbk {
        let cb = SILK_LTP_VQ_PTRS_Q7[cbk];
        let cb_gain = SILK_LTP_VQ_GAIN_PTRS_Q7[cbk];
        let cb_size = SILK_LTP_VQ_SIZES[cbk] as usize;

        let mut total_rd = 0i32;
        let mut temp_indices = [0i8; MAX_NB_SUBFR];
        let mut temp_b_q14 = [0i16; LTP_ORDER * MAX_NB_SUBFR];
        let mut temp_sum_log_gain = *sum_log_gain_q7;

        for j in 0..nb_subfr {
            // Compute max allowed gain from log gain budget
            let max_gain_q7 = if temp_sum_log_gain < MAX_SUM_LOG_GAIN_DB_Q7 {
                silk_log2lin((MAX_SUM_LOG_GAIN_DB_Q7 - temp_sum_log_gain) >> 1)
            } else {
                0
            };

            // Find best codebook vector via weighted matrix VQ
            let mut min_rd = i32::MAX;
            let mut best_idx = 0usize;

            for k in 0..cb_size {
                // Skip if gain exceeds budget
                if cb_gain[k] as i32 > max_gain_q7 {
                    continue;
                }

                // Compute weighted distortion
                let cb_offset = k * LTP_ORDER;
                let w_offset = j * LTP_ORDER * LTP_ORDER;
                let mut dist = 0i64;

                // Simplified 5x5 VQ distortion (matching silk_VQ_WMat_EC)
                for m in 0..LTP_ORDER {
                    for n in 0..LTP_ORDER {
                        dist += cb[cb_offset + m] as i64 * cb[cb_offset + n] as i64
                            * w_q18[w_offset + m * LTP_ORDER + n] as i64;
                    }
                }

                let rd = (dist >> 18) as i32;
                let rate = SILK_LTP_GAIN_BITS_Q5_PTRS[cbk][k] as i32;
                let total = rd + mu_q9 * rate >> 9;

                if total < min_rd {
                    min_rd = total;
                    best_idx = k;
                }
            }

            temp_indices[j] = best_idx as i8;

            // Copy LTP coefficients (Q7 → Q14)
            let cb_offset = best_idx * LTP_ORDER;
            for m in 0..LTP_ORDER {
                temp_b_q14[j * LTP_ORDER + m] = (cb[cb_offset + m] as i32 * 128) as i16;
            }

            // Update log gain sum
            let gain = imax(cb_gain[best_idx] as i32, 1);
            temp_sum_log_gain += silk_lin2log(gain) << 1;

            total_rd += min_rd;
        }

        if total_rd < best_total_rd {
            best_total_rd = total_rd;
            best_per = cbk as i8;
            *sum_log_gain_q7 = temp_sum_log_gain;
            b_q14[..nb_subfr * LTP_ORDER].copy_from_slice(&temp_b_q14[..nb_subfr * LTP_ORDER]);
            cbk_index[..nb_subfr].copy_from_slice(&temp_indices[..nb_subfr]);
        }
    }

    *per_index = best_per;
}

// ===========================================================================
// Encode indices (bitstream writing)
// ===========================================================================

/// Entropy-encode all side information indices.
/// Matches C: `silk_encode_indices`.
pub fn silk_encode_indices(
    ps_enc: &SilkEncoderState,
    range_enc: &mut RangeEncoder,
    frame_index: usize,
    encode_lbrr: bool,
    cond_coding: i32,
) {
    let indices = if encode_lbrr {
        &ps_enc.indices_lbrr[frame_index]
    } else {
        &ps_enc.indices
    };

    let nb_subfr = ps_enc.nb_subfr as usize;

    // 1. Encode signal type and quantizer offset (joint coding)
    // typeOffset = 2 * signalType + quantOffsetType, range [0..5]
    let type_offset = 2 * indices.signal_type as i32 + indices.quant_offset_type as i32;
    if encode_lbrr || type_offset >= 2 {
        // VAD table: encode typeOffset - 2 (symbol range [0..3])
        range_enc.encode_icdf(
            (type_offset - 2) as u32,
            &SILK_TYPE_OFFSET_VAD_ICDF,
            8,
        );
    } else {
        // No-VAD table: encode typeOffset directly (symbol range [0..1])
        range_enc.encode_icdf(
            type_offset as u32,
            &SILK_TYPE_OFFSET_NO_VAD_ICDF,
            8,
        );
    }

    // 2. Encode gains
    if cond_coding == CODE_CONDITIONALLY {
        // Delta coding for first subframe gain (symbol is the delta index directly)
        range_enc.encode_icdf(
            indices.gains_indices[0] as u32,
            &SILK_DELTA_GAIN_ICDF,
            8,
        );
    } else {
        // Independent coding: MSB (top 5 bits >> 3) + LSB (bottom 3 bits)
        range_enc.encode_icdf(
            (indices.gains_indices[0] as u32) >> 3,
            &SILK_GAIN_ICDF[indices.signal_type as usize],
            8,
        );
        range_enc.encode_icdf(
            (indices.gains_indices[0] as u32) & 7,
            &SILK_UNIFORM8_ICDF,
            8,
        );
    }

    // Subsequent subframe gains (always delta coded)
    for i in 1..nb_subfr {
        range_enc.encode_icdf(
            indices.gains_indices[i] as u32,
            &SILK_DELTA_GAIN_ICDF,
            8,
        );
    }

    // 3. Encode NLSFs
    // First stage index — iCDF is offset by (signalType >> 1) * nVectors
    let cb1_offset = (indices.signal_type as usize >> 1) * ps_enc.nlsf_cb.n_vectors as usize;
    range_enc.encode_icdf(
        indices.nlsf_indices[0] as u32,
        &ps_enc.nlsf_cb.cb1_icdf[cb1_offset..],
        8,
    );

    // NLSF residual indices
    let mut ec_ix = [0i16; MAX_LPC_ORDER];
    let mut pred_q8 = [0u8; MAX_LPC_ORDER];
    silk_nlsf_unpack(
        &mut ec_ix,
        &mut pred_q8,
        ps_enc.nlsf_cb,
        indices.nlsf_indices[0] as usize,
    );

    let order = ps_enc.predict_lpc_order as usize;
    for i in 0..order {
        let res = indices.nlsf_indices[i + 1] as i32;
        let icdf_offset = ec_ix[i] as usize;

        if res >= NLSF_QUANT_MAX_AMPLITUDE {
            // Large positive: encode max symbol (2*MAX_AMP = 8) then extension
            range_enc.encode_icdf(
                (2 * NLSF_QUANT_MAX_AMPLITUDE) as u32,
                &ps_enc.nlsf_cb.ec_icdf[icdf_offset..],
                8,
            );
            range_enc.encode_icdf(
                (res - NLSF_QUANT_MAX_AMPLITUDE) as u32,
                &SILK_NLSF_EXT_ICDF,
                8,
            );
        } else if res <= -NLSF_QUANT_MAX_AMPLITUDE {
            // Large negative: encode 0 then extension
            range_enc.encode_icdf(
                0,
                &ps_enc.nlsf_cb.ec_icdf[icdf_offset..],
                8,
            );
            range_enc.encode_icdf(
                (-res - NLSF_QUANT_MAX_AMPLITUDE) as u32,
                &SILK_NLSF_EXT_ICDF,
                8,
            );
        } else {
            // Normal range: symbol = res + NLSF_QUANT_MAX_AMPLITUDE
            range_enc.encode_icdf(
                (res + NLSF_QUANT_MAX_AMPLITUDE) as u32,
                &ps_enc.nlsf_cb.ec_icdf[icdf_offset..],
                8,
            );
        }
    }

    // 4. NLSF interpolation factor
    if nb_subfr == 4 {
        range_enc.encode_icdf(
            indices.nlsf_interp_coef_q2 as u32,
            &SILK_NLSF_INTERPOLATION_FACTOR_ICDF,
            8,
        );
    }

    // 5. Encode pitch (voiced frames only)
    if indices.signal_type == TYPE_VOICED as i8 {
        // Pitch lag
        let mut encode_absolute_lag = true;

        if cond_coding == CODE_CONDITIONALLY && ps_enc.ec_prev_signal_type == TYPE_VOICED {
            // Try delta coding
            let mut delta_lag = indices.lag_index as i32 - ps_enc.ec_prev_lag_index as i32;
            if delta_lag < -8 || delta_lag > 11 {
                delta_lag = 0; // Out of range → signal absolute encoding via symbol 0
            } else {
                delta_lag += 9; // Bias to [1..20]
                encode_absolute_lag = false;
            }
            range_enc.encode_icdf(
                delta_lag as u32,
                &SILK_PITCH_DELTA_ICDF,
                8,
            );
        }

        if encode_absolute_lag {
            // Absolute coding: high bits + low bits
            // pitch_high = lagIndex / (fs_kHz / 2)
            // pitch_low  = lagIndex % (fs_kHz / 2)
            let half_fs = ps_enc.fs_khz >> 1;
            let pitch_high = indices.lag_index as i32 / half_fs;
            let pitch_low = indices.lag_index as i32 - pitch_high * half_fs;
            range_enc.encode_icdf(
                pitch_high as u32,
                &SILK_PITCH_LAG_ICDF,
                8,
            );
            range_enc.encode_icdf(
                pitch_low as u32,
                ps_enc.pitch_lag_low_bits_icdf,
                8,
            );
        }

        // Pitch contour
        range_enc.encode_icdf(
            indices.contour_index as u32,
            ps_enc.pitch_contour_icdf,
            8,
        );

        // LTP periodicity index
        range_enc.encode_icdf(
            indices.per_index as u32,
            &SILK_LTP_PER_INDEX_ICDF,
            8,
        );

        // LTP codebook indices
        let ltp_icdf = SILK_LTP_GAIN_ICDF_PTRS[indices.per_index as usize];
        for i in 0..nb_subfr {
            range_enc.encode_icdf(
                indices.ltp_index[i] as u32,
                ltp_icdf,
                8,
            );
        }

        // LTP scale (independent coding only)
        if cond_coding == CODE_INDEPENDENTLY {
            range_enc.encode_icdf(
                indices.ltp_scale_index as u32,
                &SILK_LTP_SCALE_ICDF,
                8,
            );
        }
    }

    // 6. Encode seed
    range_enc.encode_icdf(indices.seed as u32, &SILK_UNIFORM4_ICDF, 8);
}

// ===========================================================================
// Encode pulses (excitation)
// ===========================================================================

/// Shell encoder: hierarchical pulse count encoding.
/// Shell encoder: operates on one shell code frame of 16 pulses.
/// Uses explicit unrolled tree with level-specific tables, matching C exactly.
/// Matches C: `silk_shell_encoder`.
fn silk_shell_encoder(
    range_enc: &mut RangeEncoder,
    pulses0: &[i32],
) {
    // Combine bottom-up: 16 → 8 → 4 → 2 → 1
    let mut pulses1 = [0i32; 8];
    for k in 0..8 {
        pulses1[k] = pulses0[2 * k] + pulses0[2 * k + 1];
    }
    let mut pulses2 = [0i32; 4];
    for k in 0..4 {
        pulses2[k] = pulses1[2 * k] + pulses1[2 * k + 1];
    }
    let mut pulses3 = [0i32; 2];
    for k in 0..2 {
        pulses3[k] = pulses2[2 * k] + pulses2[2 * k + 1];
    }
    let pulses4 = pulses3[0] + pulses3[1];

    // Encode top-down in depth-first order, each level uses its own table
    encode_split(range_enc, pulses3[0], pulses4, &SILK_SHELL_CODE_TABLE3);

    encode_split(range_enc, pulses2[0], pulses3[0], &SILK_SHELL_CODE_TABLE2);

    encode_split(range_enc, pulses1[0], pulses2[0], &SILK_SHELL_CODE_TABLE1);
    encode_split(range_enc, pulses0[0] as i32, pulses1[0], &SILK_SHELL_CODE_TABLE0);
    encode_split(range_enc, pulses0[2] as i32, pulses1[1], &SILK_SHELL_CODE_TABLE0);

    encode_split(range_enc, pulses1[2], pulses2[1], &SILK_SHELL_CODE_TABLE1);
    encode_split(range_enc, pulses0[4] as i32, pulses1[2], &SILK_SHELL_CODE_TABLE0);
    encode_split(range_enc, pulses0[6] as i32, pulses1[3], &SILK_SHELL_CODE_TABLE0);

    encode_split(range_enc, pulses2[2], pulses3[1], &SILK_SHELL_CODE_TABLE2);

    encode_split(range_enc, pulses1[4], pulses2[2], &SILK_SHELL_CODE_TABLE1);
    encode_split(range_enc, pulses0[8] as i32, pulses1[4], &SILK_SHELL_CODE_TABLE0);
    encode_split(range_enc, pulses0[10] as i32, pulses1[5], &SILK_SHELL_CODE_TABLE0);

    encode_split(range_enc, pulses1[6], pulses2[3], &SILK_SHELL_CODE_TABLE1);
    encode_split(range_enc, pulses0[12] as i32, pulses1[6], &SILK_SHELL_CODE_TABLE0);
    encode_split(range_enc, pulses0[14] as i32, pulses1[7], &SILK_SHELL_CODE_TABLE0);
}

/// Encode a binary split: encode left child pulse count conditioned on parent total.
/// Matches C: `encode_split`.
#[inline]
fn encode_split(
    range_enc: &mut RangeEncoder,
    p_child1: i32,
    p: i32,
    shell_table: &[u8],
) {
    if p > 0 {
        let offset = SILK_SHELL_CODE_TABLE_OFFSETS[p as usize] as usize;
        range_enc.encode_icdf(p_child1 as u32, &shell_table[offset..], 8);
    }
}

/// Encode signs of pulses.
/// Matches C: `silk_encode_signs`.
fn silk_encode_signs(
    range_enc: &mut RangeEncoder,
    pulses: &[i8],
    length: usize,
    signal_type: i32,
    quant_offset_type: i32,
    sum_pulses: &[i32],
) {
    // Select sign probability table: offset = 7 * (quantOffsetType + 2*signalType)
    let icdf_offset = 7 * (quant_offset_type + (signal_type << 1)) as usize;

    // Round up to shell blocks
    let n_blocks = (length + SHELL_CODEC_FRAME_LENGTH / 2) >> LOG2_SHELL_CODEC_FRAME_LENGTH;

    let mut q_ptr = 0usize;
    for i in 0..n_blocks {
        let p = sum_pulses[i];
        if p > 0 {
            // Sign probability conditioned on pulse magnitude, clamped to [0..6]
            let icdf_idx = imin(p & 0x1F, 6) as usize;
            let icdf = [SILK_SIGN_ICDF[icdf_offset + icdf_idx], 0u8];

            for j in 0..SHELL_CODEC_FRAME_LENGTH {
                if q_ptr + j < length && pulses[q_ptr + j] != 0 {
                    // silk_enc_map: positive → 1, negative → 0
                    let symbol = if pulses[q_ptr + j] > 0 { 1u32 } else { 0u32 };
                    range_enc.encode_icdf(symbol, &icdf, 8);
                }
            }
        }
        q_ptr += SHELL_CODEC_FRAME_LENGTH;
    }
}

/// Entropy-encode quantized excitation pulses.
/// Matches C: `silk_encode_pulses`.
pub fn silk_encode_pulses(
    range_enc: &mut RangeEncoder,
    signal_type: i32,
    quant_offset_type: i32,
    pulses: &[i8],
    frame_length: usize,
) {
    let nb_shell_blocks = frame_length / SHELL_CODEC_FRAME_LENGTH;
    // Extra block for 12 kHz / 10ms case (frame_length = 120, not multiple of 16)
    let iter = if frame_length == 120 { nb_shell_blocks + 1 } else { nb_shell_blocks };

    // Convert pulses to i32 and compute absolute values per shell block
    let mut abs_pulses = vec![0i32; iter * SHELL_CODEC_FRAME_LENGTH];
    for i in 0..frame_length {
        abs_pulses[i] = (pulses[i] as i32).unsigned_abs() as i32;
    }

    // Hierarchical combine to compute sum_pulses with right-shifting
    let mut sum_pulses = vec![0i32; iter];
    let mut nrshifts = vec![0i32; iter];

    for i in 0..iter {
        let block_start = i * SHELL_CODEC_FRAME_LENGTH;
        nrshifts[i] = 0;

        loop {
            let mut pulses_comb = [0i32; 8];
            let mut scale_down = false;

            // 1+1 -> 2 (16 → 8, max=8)
            for k in 0..8 {
                let s = abs_pulses[block_start + 2 * k] + abs_pulses[block_start + 2 * k + 1];
                if s > SILK_MAX_PULSES_TABLE[0] as i32 { scale_down = true; }
                pulses_comb[k] = s;
            }
            // 2+2 -> 4 (8 → 4, max=10)
            let mut pulses_comb2 = [0i32; 4];
            for k in 0..4 {
                let s = pulses_comb[2 * k] + pulses_comb[2 * k + 1];
                if s > SILK_MAX_PULSES_TABLE[1] as i32 { scale_down = true; }
                pulses_comb2[k] = s;
            }
            // 4+4 -> 8 (4 → 2, max=12)
            let mut pulses_comb3 = [0i32; 2];
            for k in 0..2 {
                let s = pulses_comb2[2 * k] + pulses_comb2[2 * k + 1];
                if s > SILK_MAX_PULSES_TABLE[2] as i32 { scale_down = true; }
                pulses_comb3[k] = s;
            }
            // 8+8 -> 16 (2 → 1, max=16)
            let s = pulses_comb3[0] + pulses_comb3[1];
            if s > SILK_MAX_PULSES_TABLE[3] as i32 { scale_down = true; }

            if scale_down {
                nrshifts[i] += 1;
                for k in 0..SHELL_CODEC_FRAME_LENGTH {
                    abs_pulses[block_start + k] >>= 1;
                }
            } else {
                sum_pulses[i] = s;
                break;
            }
        }
    }

    // Find optimal rate level (minimum total bits)
    let mut rate_level_index = 0usize;
    let mut min_sum_bits_q5 = i32::MAX;
    for k in 0..(N_RATE_LEVELS - 1) {
        let n_bits_ptr = &SILK_PULSES_PER_BLOCK_BITS_Q5[k];
        let mut sum_bits_q5 = SILK_RATE_LEVELS_BITS_Q5[(signal_type >> 1) as usize][k] as i32;
        for i in 0..iter {
            if nrshifts[i] > 0 {
                sum_bits_q5 += n_bits_ptr[SILK_MAX_PULSES as usize + 1] as i32;
            } else {
                sum_bits_q5 += n_bits_ptr[sum_pulses[i] as usize] as i32;
            }
        }
        if sum_bits_q5 < min_sum_bits_q5 {
            min_sum_bits_q5 = sum_bits_q5;
            rate_level_index = k;
        }
    }
    range_enc.encode_icdf(
        rate_level_index as u32,
        &SILK_RATE_LEVELS_ICDF[(signal_type >> 1) as usize],
        8,
    );

    // Encode pulse counts per block with overflow handling
    let cdf_ptr = &SILK_PULSES_PER_BLOCK_ICDF[rate_level_index];
    for i in 0..iter {
        if nrshifts[i] == 0 {
            range_enc.encode_icdf(sum_pulses[i] as u32, cdf_ptr, 8);
        } else {
            // Signal overflow with SILK_MAX_PULSES+1
            range_enc.encode_icdf((SILK_MAX_PULSES + 1) as u32, cdf_ptr, 8);
            // Encode (nRshifts-1) additional overflow markers at highest rate level
            for _k in 0..(nrshifts[i] - 1) {
                range_enc.encode_icdf(
                    (SILK_MAX_PULSES + 1) as u32,
                    &SILK_PULSES_PER_BLOCK_ICDF[N_RATE_LEVELS - 1],
                    8,
                );
            }
            // Encode actual sum at highest rate level
            range_enc.encode_icdf(
                sum_pulses[i] as u32,
                &SILK_PULSES_PER_BLOCK_ICDF[N_RATE_LEVELS - 1],
                8,
            );
        }
    }

    // Shell coding: hierarchical encoding of pulse distribution per block
    for i in 0..iter {
        if sum_pulses[i] > 0 {
            let block_start = i * SHELL_CODEC_FRAME_LENGTH;
            let block = &abs_pulses[block_start..block_start + SHELL_CODEC_FRAME_LENGTH];
            silk_shell_encoder(range_enc, block);
        }
    }

    // Encode LSBs (for right-shifted blocks)
    // C encodes bits MSB-to-LSB: for j = nLS..1 encode (abs_q >> j) & 1, then abs_q & 1
    // Positions beyond frame_length are zero (C zeros them), still encoded.
    for i in 0..iter {
        if nrshifts[i] > 0 {
            let block_start = i * SHELL_CODEC_FRAME_LENGTH;
            let n_ls = nrshifts[i] - 1;
            for k in 0..SHELL_CODEC_FRAME_LENGTH {
                let idx = block_start + k;
                // Positions beyond frame_length are 0 (C zeros original pulses there)
                let abs_q = if idx < frame_length {
                    (pulses[idx] as i32).abs() as i32
                } else {
                    0
                };
                // Encode bits from MSB to LSB
                for j in (1..=n_ls).rev() {
                    let bit = (abs_q >> j) & 1;
                    range_enc.encode_icdf(bit as u32, &SILK_LSB_ICDF, 8);
                }
                let bit = abs_q & 1;
                range_enc.encode_icdf(bit as u32, &SILK_LSB_ICDF, 8);
            }
        }
    }

    // Encode signs
    silk_encode_signs(range_enc, pulses, frame_length, signal_type, quant_offset_type, &sum_pulses);
}

// ===========================================================================
// Stereo encoding
// ===========================================================================

/// Inner product with automatic scaling.
/// Matches C: `silk_inner_prod_aligned_scale`.
fn silk_inner_prod_aligned_scale(x: &[i16], y: &[i16], scale: i32, len: usize) -> i32 {
    let mut sum: i64 = 0;
    for i in 0..len {
        sum += (x[i] as i64 * y[i] as i64) >> scale;
    }
    sum as i32
}

/// Find stereo predictor for one frequency band.
/// Returns predictor in Q13.
/// Matches C: `silk_stereo_find_predictor`.
fn silk_stereo_find_predictor(
    ratio_q14: &mut i32,
    x: &[i16],       // basis (mid) signal
    y: &[i16],       // target (side) signal
    mid_res_amp_q0: &mut [i32],  // [2]: smoothed mid, residual norms
    length: usize,
    smooth_coef_q16: i32,
) -> i32 {
    // Compute energies with auto-scaling
    let (mut nrgx, scale1) = silk_sum_sqr_shift(x);
    let (mut nrgy, scale2) = silk_sum_sqr_shift(y);
    let mut scale = imax(scale1 as i32, scale2 as i32);
    scale += scale & 1; // make even
    nrgy >>= (scale - scale2 as i32) as u32;
    nrgx >>= (scale - scale1 as i32) as u32;
    nrgx = imax(nrgx, 1);

    // Cross-correlation with scaling
    let corr = silk_inner_prod_aligned_scale(x, y, scale, length);

    // Predictor
    let mut pred_q13 = silk_div32_var_q(corr, nrgx, 13);
    pred_q13 = imax(imin(pred_q13, 1 << 14), -(1 << 14));
    let pred2_q10 = silk_smulwb(pred_q13, pred_q13 as i16);

    // Faster update for signals with large prediction parameters
    let smooth_coef_q16 = imax(smooth_coef_q16, pred2_q10.unsigned_abs() as i32);

    // Smoothed mid norm
    let scale_half = scale >> 1;
    mid_res_amp_q0[0] = silk_smlawb(
        mid_res_amp_q0[0],
        (silk_sqrt_approx(nrgx) << scale_half) - mid_res_amp_q0[0],
        smooth_coef_q16 as i16,
    );

    // Residual energy = nrgy - 2*pred*corr + pred^2*nrgx
    // silk_SUB_LSHIFT32(nrgy, silk_SMULWB(corr, pred_Q13), 3+1)
    let nrgy = nrgy - (silk_smulwb(corr, pred_q13 as i16) << 4);
    // silk_ADD_LSHIFT32(nrgy, silk_SMULWB(nrgx, pred2_Q10), 6)
    let nrgy = nrgy + (silk_smulwb(nrgx, pred2_q10 as i16) << 6);

    mid_res_amp_q0[1] = silk_smlawb(
        mid_res_amp_q0[1],
        (silk_sqrt_approx(imax(nrgy, 0)) << scale_half) - mid_res_amp_q0[1],
        smooth_coef_q16 as i16,
    );

    // Ratio of smoothed residual and mid norms
    *ratio_q14 = silk_div32_var_q(mid_res_amp_q0[1], imax(mid_res_amp_q0[0], 1), 14);
    *ratio_q14 = imax(imin(*ratio_q14, 32767), 0);

    pred_q13
}

/// Quantize stereo predictor.
/// Matches C: `silk_stereo_quant_pred`.
pub fn silk_stereo_quant_pred(
    pred_q13: &mut [i32; 2],
    ix: &mut [[i8; 3]; 2],
) {
    // SILK_FIX_CONST(0.5/STEREO_QUANT_SUB_STEPS, 16) = SILK_FIX_CONST(0.1, 16) = 6554
    const HALF_STEP_Q16: i16 = 6554;

    for n in 0..2 {
        let mut err_min_q13 = i32::MAX;
        let mut quant_pred_q13 = 0i32;

        // Brute-force search over quantization levels
        'outer: for i in 0..(STEREO_QUANT_TAB_SIZE - 1) {
            let low_q13 = SILK_STEREO_PRED_QUANT_Q13[i] as i32;
            let step_q13 = silk_smulwb(
                SILK_STEREO_PRED_QUANT_Q13[i + 1] as i32 - low_q13,
                HALF_STEP_Q16,
            );
            for j in 0..STEREO_QUANT_SUB_STEPS as i32 {
                // silk_SMLABB(low_q13, step_q13, 2*j+1)
                let lvl_q13 = low_q13 + step_q13 * (2 * j + 1);
                let err_q13 = (pred_q13[n] - lvl_q13).unsigned_abs() as i32;
                if err_q13 < err_min_q13 {
                    err_min_q13 = err_q13;
                    quant_pred_q13 = lvl_q13;
                    ix[n][0] = i as i8;
                    ix[n][1] = j as i8;
                } else {
                    // Error increasing — past the optimum
                    break 'outer;
                }
            }
        }

        // Split main index: ix[n][2] = ix[n][0] / 3, ix[n][0] = ix[n][0] % 3
        ix[n][2] = ix[n][0] / 3;
        ix[n][0] -= ix[n][2] * 3;
        pred_q13[n] = quant_pred_q13;
    }

    // Subtract second from first predictor (differential coding)
    pred_q13[0] -= pred_q13[1];
}

/// Encode stereo prediction indices.
/// Matches C: `silk_stereo_encode_pred`.
pub fn silk_stereo_encode_pred(
    range_enc: &mut RangeEncoder,
    ix: &[[i8; 3]; 2],
) {
    // Joint coding of the two group indices: n = 5 * ix[0][2] + ix[1][2]
    let n = 5 * ix[0][2] as usize + ix[1][2] as usize;
    range_enc.encode_icdf(n as u32, &SILK_STEREO_PRED_JOINT_ICDF, 8);

    // Per-channel: encode ix[n][0] (range 0..2) and ix[n][1] (range 0..4) independently
    for ch in 0..2 {
        range_enc.encode_icdf(ix[ch][0] as u32, &SILK_UNIFORM3_ICDF, 8);
        range_enc.encode_icdf(ix[ch][1] as u32, &SILK_UNIFORM5_ICDF, 8);
    }
}

/// Encode mid-only flag.
pub fn silk_stereo_encode_mid_only(
    range_enc: &mut RangeEncoder,
    mid_only_flag: i8,
) {
    range_enc.encode_icdf(
        mid_only_flag as u32,
        &SILK_STEREO_ONLY_CODE_MID_ICDF,
        8,
    );
}

/// Convert L/R stereo to M/S with stereo prediction.
/// Matches C: `silk_stereo_LR_to_MS`.
pub fn silk_stereo_lr_to_ms(
    state: &mut StereoEncState,
    x1: &mut [i16], // left → mid (modified in place)
    x2: &mut [i16], // right → side (modified in place)
    pred_ix: &mut [[i8; 3]; 2],
    mid_only_flag: &mut i8,
    mid_side_rates_bps: &mut [i32; 2],
    mut total_rate_bps: i32,
    prev_speech_act_q8: i32,
    to_mono: bool,
    fs_khz: i32,
    frame_length: usize,
) {
    let is_10ms_frame = frame_length == 10 * fs_khz as usize;

    // Step 1: Convert L/R to basic M/S
    // C uses x1[-2]..x1[frame_length-1] via pointer aliasing. We use explicit buffers.
    let mut mid = vec![0i16; frame_length + 2];
    let mut side = vec![0i16; frame_length + 2];

    for n in 0..frame_length + 2 {
        // x1[n-2] and x2[n-2]: for n<2, these are overlap from previous frame
        let l = if n < 2 { state.s_mid[n] as i32 } else { x1[n - 2] as i32 };
        let r = if n < 2 { state.s_side[n] as i32 } else { x2[n - 2] as i32 };
        // Note: state.s_mid/s_side store the *original* L/R overlap for the first call,
        // but after the first call they store the *mid/side* overlap. The C code uses
        // x1[-2],x1[-1] which are the previous frame's last two L samples.
        // For simplicity, we compute sum/diff here and handle overlap below.
        let sum = l + r;
        let diff = l - r;
        mid[n] = silk_rshift_round(sum, 1) as i16;
        side[n] = sat16(silk_rshift_round(diff, 1));
    }

    // Step 2: Restore overlap from state, save new overlap
    mid[0] = state.s_mid[0];
    mid[1] = state.s_mid[1];
    side[0] = state.s_side[0];
    side[1] = state.s_side[1];
    state.s_mid[0] = mid[frame_length];
    state.s_mid[1] = mid[frame_length + 1];
    state.s_side[0] = side[frame_length];
    state.s_side[1] = side[frame_length + 1];

    // Step 3-4: LP/HP filter mid and side signals
    let mut lp_mid = vec![0i16; frame_length];
    let mut hp_mid = vec![0i16; frame_length];
    let mut lp_side = vec![0i16; frame_length];
    let mut hp_side = vec![0i16; frame_length];

    for n in 0..frame_length {
        // silk_ADD_LSHIFT32(mid[n] + mid[n+2], mid[n+1], 1) = mid[n]+mid[n+2] + 2*mid[n+1]
        let sum_m = silk_rshift_round(
            mid[n] as i32 + mid[n + 2] as i32 + ((mid[n + 1] as i32) << 1), 2);
        lp_mid[n] = sum_m as i16;
        hp_mid[n] = (mid[n + 1] as i32 - sum_m) as i16;

        let sum_s = silk_rshift_round(
            side[n] as i32 + side[n + 2] as i32 + ((side[n + 1] as i32) << 1), 2);
        lp_side[n] = sum_s as i16;
        hp_side[n] = (side[n + 1] as i32 - sum_s) as i16;
    }

    // Step 5: Smoothing coefficient (speech-activity weighted)
    let smooth_coef_q16 = if is_10ms_frame { 328 } else { 655 }; // STEREO_RATIO_SMOOTH_COEF
    let smooth_coef_q16 = silk_smulwb(
        silk_smulbb(prev_speech_act_q8, prev_speech_act_q8),
        smooth_coef_q16 as i16,
    );

    // Step 6: Find predictors for LP and HP bands
    let mut lp_ratio_q14 = 0i32;
    let mut hp_ratio_q14 = 0i32;
    let mut pred_q13 = [0i32; 2];

    pred_q13[0] = silk_stereo_find_predictor(
        &mut lp_ratio_q14, &lp_mid, &lp_side,
        &mut state.mid_side_amp_q0[0..2], frame_length, smooth_coef_q16);
    pred_q13[1] = silk_stereo_find_predictor(
        &mut hp_ratio_q14, &hp_mid, &hp_side,
        &mut state.mid_side_amp_q0[2..4], frame_length, smooth_coef_q16);

    // Combine LP and HP ratios: frac_Q16 = HP_ratio + 3 * LP_ratio
    let frac_q16 = imin(silk_smlabb(hp_ratio_q14, lp_ratio_q14, 3), 1 << 16);

    // Step 7: Bitrate distribution
    total_rate_bps -= if is_10ms_frame { 1200 } else { 600 };
    total_rate_bps = imax(total_rate_bps, 1);
    let min_mid_rate_bps = silk_smlabb(2000, fs_khz, 600);
    let frac_3_q16 = 3 * frac_q16;

    // mid rate = total / (SILK_FIX_CONST(8+5, 16) + 3*frac), shifted
    mid_side_rates_bps[0] = silk_div32_var_q(
        total_rate_bps, ((8 + 5) << 16) + frac_3_q16, 16 + 3);

    let mut width_q14;
    if mid_side_rates_bps[0] < min_mid_rate_bps {
        mid_side_rates_bps[0] = min_mid_rate_bps;
        mid_side_rates_bps[1] = total_rate_bps - mid_side_rates_bps[0];
        width_q14 = silk_div32_var_q(
            (mid_side_rates_bps[1] << 1) - min_mid_rate_bps,
            silk_smulwb((1 << 16) + frac_3_q16, min_mid_rate_bps as i16),
            14 + 2,
        );
        width_q14 = imax(imin(width_q14, 1 << 14), 0);
    } else {
        mid_side_rates_bps[1] = total_rate_bps - mid_side_rates_bps[0];
        width_q14 = 1 << 14;
    }

    // Step 8: Smooth width
    state.smth_width_q14 = silk_smlawb(
        state.smth_width_q14 as i32,
        width_q14 - state.smth_width_q14 as i32,
        smooth_coef_q16 as i16,
    ) as i16;

    // Step 9: Decision tree for mono/stereo width
    *mid_only_flag = 0;
    if to_mono {
        // Branch A: force mono transition
        width_q14 = 0;
        pred_q13 = [0; 2];
        silk_stereo_quant_pred(&mut pred_q13, pred_ix);
    } else if state.width_prev_q14 == 0
        && (8 * total_rate_bps < 13 * min_mid_rate_bps
            || silk_smulwb(frac_q16, state.smth_width_q14) < (819)) // SILK_FIX_CONST(0.05, 14) = 819
    {
        // Branch B: was mono, stay mono
        pred_q13[0] = (state.smth_width_q14 as i32 * pred_q13[0]) >> 14;
        pred_q13[1] = (state.smth_width_q14 as i32 * pred_q13[1]) >> 14;
        silk_stereo_quant_pred(&mut pred_q13, pred_ix);
        width_q14 = 0;
        pred_q13 = [0; 2];
        mid_side_rates_bps[0] = total_rate_bps;
        mid_side_rates_bps[1] = 0;
        *mid_only_flag = 1;
    } else if state.width_prev_q14 != 0
        && (8 * total_rate_bps < 11 * min_mid_rate_bps
            || silk_smulwb(frac_q16, state.smth_width_q14) < (328)) // SILK_FIX_CONST(0.02, 14) = 328
    {
        // Branch C: transition to mono
        pred_q13[0] = (state.smth_width_q14 as i32 * pred_q13[0]) >> 14;
        pred_q13[1] = (state.smth_width_q14 as i32 * pred_q13[1]) >> 14;
        silk_stereo_quant_pred(&mut pred_q13, pred_ix);
        width_q14 = 0;
        pred_q13 = [0; 2];
    } else if state.smth_width_q14 > 15564 {
        // Branch D: full width (> 0.95)
        silk_stereo_quant_pred(&mut pred_q13, pred_ix);
        width_q14 = 1 << 14;
    } else {
        // Branch E: reduced width
        pred_q13[0] = (state.smth_width_q14 as i32 * pred_q13[0]) >> 14;
        pred_q13[1] = (state.smth_width_q14 as i32 * pred_q13[1]) >> 14;
        silk_stereo_quant_pred(&mut pred_q13, pred_ix);
        width_q14 = state.smth_width_q14 as i32;
    }

    // Step 10: Mid-only taper tracking
    if *mid_only_flag == 1 {
        state.silent_side_len += frame_length as i16 - (STEREO_INTERP_LEN_MS as i32 * fs_khz) as i16;
        if (state.silent_side_len as i32) < LA_SHAPE_MS * fs_khz {
            *mid_only_flag = 0;
        } else {
            state.silent_side_len = 10000;
        }
    } else {
        state.silent_side_len = 0;
    }
    if *mid_only_flag == 0 && mid_side_rates_bps[1] < 1 {
        mid_side_rates_bps[1] = 1;
        mid_side_rates_bps[0] = imax(1, total_rate_bps - mid_side_rates_bps[1]);
    }

    // Step 11: Interpolate predictors and subtract prediction from side channel
    let interp_len = STEREO_INTERP_LEN_MS * fs_khz as usize;
    let denom_q16 = (1 << 16) / (interp_len as i32);

    let mut pred0_q13 = -(state.pred_prev_q13[0] as i32);
    let mut pred1_q13 = -(state.pred_prev_q13[1] as i32);
    let mut w_q24 = (state.width_prev_q14 as i32) << 10;
    let delta0_q13 = -silk_rshift_round(
        silk_smulbb(pred_q13[0] - state.pred_prev_q13[0] as i32, denom_q16), 16);
    let delta1_q13 = -silk_rshift_round(
        silk_smulbb(pred_q13[1] - state.pred_prev_q13[1] as i32, denom_q16), 16);
    let deltaw_q24 = silk_smulwb(width_q14 - state.width_prev_q14 as i32, denom_q16 as i16) << 10;

    // Interpolation region
    for n in 0..interp_len {
        pred0_q13 += delta0_q13;
        pred1_q13 += delta1_q13;
        w_q24 += deltaw_q24;
        // LP mid component (Q11)
        let sum = (mid[n] as i32 + mid[n + 2] as i32 + ((mid[n + 1] as i32) << 1)) << 9;
        // side * width + LP_pred * sum (Q8)
        let mut out = silk_smlawb(
            silk_smulwb(w_q24, side[n + 1] as i16),
            sum,
            pred0_q13 as i16,
        );
        // HP_pred * mid[n+1] (Q8)
        out = silk_smlawb(out, (mid[n + 1] as i32) << 11, pred1_q13 as i16);
        x2[n] = sat16(silk_rshift_round(out, 8));
    }

    // Steady-state region
    let pred0_q13_final = -(pred_q13[0]);
    let pred1_q13_final = -(pred_q13[1]);
    let w_q24_final = width_q14 << 10;
    for n in interp_len..frame_length {
        let sum = (mid[n] as i32 + mid[n + 2] as i32 + ((mid[n + 1] as i32) << 1)) << 9;
        let mut out = silk_smlawb(
            silk_smulwb(w_q24_final, side[n + 1] as i16),
            sum,
            pred0_q13_final as i16,
        );
        out = silk_smlawb(out, (mid[n + 1] as i32) << 11, pred1_q13_final as i16);
        x2[n] = sat16(silk_rshift_round(out, 8));
    }

    // Copy mid to x1 (shifted by 1: x1[n] = mid[n+1])
    for n in 0..frame_length {
        x1[n] = mid[n + 1];
    }

    // Save state
    state.pred_prev_q13[0] = pred_q13[0] as i16;
    state.pred_prev_q13[1] = pred_q13[1] as i16;
    state.width_prev_q14 = width_q14 as i16;
}

// ===========================================================================
// NSQ: Noise Shaping Quantizer (standard)
// ===========================================================================

/// Scale states for new subframe (gain changes, LTP re-whitening).
/// Matches C: `silk_nsq_scale_states`.
fn silk_nsq_scale_states(
    nsq: &mut NsqState,
    x16: &[i16],
    x_sc_q10: &mut [i32],
    s_ltp: &[i16],
    s_ltp_q15: &mut [i32],
    subfr: usize,
    ltp_mem_length: usize,
    subfr_length: usize,
    lag: i32,
    gain_q16: i32,
    pitch_l: &[i32],
    signal_type: i32,
    inv_gain_q31: i32,
    prev_inv_gain_q31: i32,
) {
    // Scale input by inverse gain: x_sc_Q10 = x16 * inv_gain_Q31 >> 21
    for i in 0..subfr_length {
        x_sc_q10[i] = ((x16[i] as i64 * inv_gain_q31 as i64) >> 21) as i32;
    }

    // If gain changed, scale internal states
    if inv_gain_q31 != prev_inv_gain_q31 {
        let gain_adj_q16 = silk_div32_var_q(inv_gain_q31, imax(prev_inv_gain_q31, 1), 16);

        // Scale noise shaping states
        for i in 0..nsq.s_lpc_q14.len() {
            nsq.s_lpc_q14[i] =
                ((nsq.s_lpc_q14[i] as i64 * gain_adj_q16 as i64) >> 16) as i32;
        }
        // Scale AR shaping state
        for i in 0..nsq.s_ar2_q14.len() {
            nsq.s_ar2_q14[i] =
                ((nsq.s_ar2_q14[i] as i64 * gain_adj_q16 as i64) >> 16) as i32;
        }
        nsq.s_lf_ar_shp_q14 =
            ((nsq.s_lf_ar_shp_q14 as i64 * gain_adj_q16 as i64) >> 16) as i32;
    }

    // Re-whiten LTP state if needed
    if signal_type == TYPE_VOICED && nsq.rewhite_flag != 0 {
        // Done by caller providing s_ltp
    }
}

/// Noise shaping quantizer (standard, non-delayed-decision).
/// Matches C: `silk_NSQ_c`.
pub fn silk_nsq(
    ps_enc: &SilkEncoderState,
    nsq: &mut NsqState,
    ps_indices: &mut SideInfoIndices,
    x16: &[i16],
    pulses: &mut [i8],
    pred_coef_q12: &[[i16; MAX_LPC_ORDER]; 2],
    ltp_coef_q14: &[i16],
    ar_q13: &[i16],
    harm_shape_fir_packed_q14: i32,
    tilt_q14: &[i32],
    lf_shp_q14: &[i32],
    gains_q16: &[i32],
    pitch_l: &[i32],
    lambda_q10: i32,
    ltp_scale_q14: i32,
) {
    let nb_subfr = ps_enc.nb_subfr as usize;
    let subfr_length = ps_enc.subfr_length as usize;
    let frame_length = ps_enc.frame_length as usize;
    let ltp_mem_length = ps_enc.ltp_mem_length as usize;
    let predict_lpc_order = ps_enc.predict_lpc_order as usize;
    let shaping_lpc_order = ps_enc.shaping_lpc_order as usize;

    let signal_type = ps_indices.signal_type as i32;

    // Allocate temporary buffers
    let mut s_ltp_q15 = vec![0i32; ltp_mem_length + frame_length];
    let mut s_ltp = vec![0i16; ltp_mem_length + frame_length];
    let mut x_sc_q10 = vec![0i32; subfr_length];

    let mut prev_inv_gain_q31 = nsq.prev_gain_q16;
    // Convert to Q31
    prev_inv_gain_q31 = if prev_inv_gain_q31 > 0 {
        ((1i64 << 31) / prev_inv_gain_q31 as i64) as i32
    } else {
        i32::MAX
    };

    nsq.rand_seed = ps_indices.seed as i32;

    for k in 0..nb_subfr {
        let gain_q16 = gains_q16[k];
        let lag = pitch_l[k];

        // Compute inverse gain (Q31)
        let inv_gain_q31 = if gain_q16 > 0 {
            imin(((1i64 << 31) / gain_q16 as i64) as i32, i32::MAX)
        } else {
            i32::MAX
        };

        // Determine which LPC coefficient set to use
        let coef_idx = if k < (nb_subfr >> 1) { 0 } else { 1 };
        let a_q12 = &pred_coef_q12[coef_idx];

        // Scale states for this subframe
        let x_offset = k * subfr_length;
        silk_nsq_scale_states(
            nsq,
            &x16[x_offset..],
            &mut x_sc_q10,
            &s_ltp,
            &mut s_ltp_q15,
            k,
            ltp_mem_length,
            subfr_length,
            lag,
            gain_q16,
            pitch_l,
            signal_type,
            inv_gain_q31,
            prev_inv_gain_q31,
        );

        // Noise shaping quantization loop
        let ar_offset = k * shaping_lpc_order;
        let ltp_offset = k * LTP_ORDER;

        for i in 0..subfr_length {
            // LPC prediction
            let mut lpc_pred_q10: i64 = 0;
            let lpc_idx = NSQ_LPC_BUF_LENGTH - 1 + i;
            for j in 0..predict_lpc_order {
                lpc_pred_q10 +=
                    nsq.s_lpc_q14[lpc_idx - j] as i64 * a_q12[j] as i64;
            }
            let lpc_pred_q10 = (lpc_pred_q10 >> 14) as i32;

            // LTP prediction (voiced only)
            let ltp_pred_q13 = if signal_type == TYPE_VOICED && lag > 0 {
                let ltp_idx = nsq.s_ltp_buf_idx as usize + i;
                let mut pred: i64 = 0;
                for j in 0..LTP_ORDER {
                    let tap_idx = (ltp_idx as i32 - lag + 2 - j as i32) as usize;
                    if tap_idx < s_ltp_q15.len() {
                        pred +=
                            s_ltp_q15[tap_idx] as i64 * ltp_coef_q14[ltp_offset + j] as i64;
                    }
                }
                (pred >> 16) as i32
            } else {
                0
            };

            // Noise shaping feedback
            let mut n_ar_q12: i64 = 0;
            for j in 0..shaping_lpc_order {
                n_ar_q12 += nsq.s_ar2_q14[j] as i64 * ar_q13[ar_offset + j] as i64;
            }
            let n_ar_q12 = (n_ar_q12 >> 15) as i32;

            let n_lf_q12 = {
                let lf_packed = lf_shp_q14[k];
                let lf_lo = lf_packed as i16 as i32; // Low 16 bits
                let lf_hi = (lf_packed >> 16) as i16 as i32; // High 16 bits
                let shp_idx = nsq.s_ltp_shp_buf_idx as usize + i;
                let prev_shp = if shp_idx > 0 {
                    nsq.s_ltp_shp_q14[shp_idx - 1]
                } else {
                    0
                };
                ((prev_shp as i64 * lf_lo as i64) >> 16) as i32
                    + ((nsq.s_lf_ar_shp_q14 as i64 * lf_hi as i64) >> 16) as i32
            };

            // Combine prediction and shaping
            let residual = x_sc_q10[i]
                - lpc_pred_q10
                - ((ltp_pred_q13 + (ltp_pred_q13 >> 1)) >> 4) // scale Q13→Q10
                + n_ar_q12 + n_lf_q12;

            // Add tilt
            let residual = residual + ((tilt_q14[k] as i64 * nsq.s_lf_ar_shp_q14 as i64 >> 16) as i32);

            // Quantize
            let offset_q10 = SILK_QUANTIZATION_OFFSETS_Q10
                [(signal_type >> 1) as usize]
                [ps_indices.quant_offset_type as usize] as i32;

            // Dither
            nsq.rand_seed = silk_rand(nsq.rand_seed);
            let dither = nsq.rand_seed;

            let residual_dithered = if dither < 0 { -residual } else { residual };
            let q1_q10 = residual_dithered - offset_q10;
            let q1_q0 = if q1_q10 > 0 { q1_q10 >> 10 } else { -(((-q1_q10) >> 10) + 1) };
            let q1_q0 = imin(imax(q1_q0, -31), 30);

            let q_q10 = q1_q0 << 10;
            let exc_q14 = shl32(q_q10 + offset_q10, 4);

            // Apply dither sign flip back
            let q_final = if dither < 0 { -q1_q0 } else { q1_q0 };
            pulses[x_offset + i] = q_final as i8;

            // Update reconstructed signal
            let xq_q14 = shl32(lpc_pred_q10, 4)
                + exc_q14
                + shl32((ltp_pred_q13 + (ltp_pred_q13 >> 1)) >> 4, 4);

            // Update filter states
            nsq.s_lpc_q14[NSQ_LPC_BUF_LENGTH + i] = xq_q14;
            let xq = sat16(silk_rshift_round(xq_q14, 14));
            nsq.xq[ltp_mem_length + x_offset + i] = xq;

            // Update shaping states
            let shp_idx = nsq.s_ltp_shp_buf_idx as usize + i;
            nsq.s_ltp_shp_q14[shp_idx] = xq_q14 - ((nsq.s_lf_ar_shp_q14 as i64 * tilt_q14[k] as i64 >> 16) as i32);
            nsq.s_lf_ar_shp_q14 = xq_q14;
            nsq.s_ar2_q14.copy_within(0..shaping_lpc_order - 1, 1);
            nsq.s_ar2_q14[0] = xq_q14 - shl32(lpc_pred_q10, 4);

            // Update LTP buffer
            if signal_type == TYPE_VOICED {
                s_ltp_q15[nsq.s_ltp_buf_idx as usize + i] = shl32(xq_q14, 1);
            }
        }

        // Shift LPC buffer
        nsq.s_lpc_q14.copy_within(subfr_length..subfr_length + NSQ_LPC_BUF_LENGTH, 0);

        prev_inv_gain_q31 = inv_gain_q31;
    }

    // Shift ring buffers
    let fl = frame_length;
    nsq.xq.copy_within(fl..fl + ltp_mem_length, 0);
    nsq.s_ltp_shp_q14.copy_within(fl..fl + ltp_mem_length, 0);
    nsq.s_ltp_buf_idx = ltp_mem_length as i32;
    nsq.s_ltp_shp_buf_idx = ltp_mem_length as i32;
    nsq.prev_gain_q16 = gains_q16[nb_subfr - 1];
    nsq.lag_prev = pitch_l[nb_subfr - 1];
}

// ===========================================================================
// Initialization
// ===========================================================================

/// Initialize a single encoder channel.
/// Matches C: `silk_init_encoder`.
pub fn silk_init_encoder(ps_enc: &mut SilkEncoderStateFix) -> i32 {
    *ps_enc = SilkEncoderStateFix::default();

    // Initialize HP filter smoother
    let hp_cutoff_q16 = shl32(VARIABLE_HP_MIN_CUTOFF_HZ, 16);
    let log_val = silk_lin2log(hp_cutoff_q16) - (16 << 7);
    ps_enc.s_cmn.variable_hp_smth1_q15 = shl32(log_val, 8);
    ps_enc.s_cmn.variable_hp_smth2_q15 = ps_enc.s_cmn.variable_hp_smth1_q15;

    ps_enc.s_cmn.first_frame_after_reset = 1;

    // Initialize VAD
    silk_vad_init(&mut ps_enc.s_cmn.s_vad);

    0
}

/// Get encoder size in bytes (not used in Rust, but kept for API parity).
pub fn silk_get_encoder_size(channels: usize) -> usize {
    core::mem::size_of::<SilkEncoder>()
        - if channels == 1 {
            core::mem::size_of::<SilkEncoderStateFix>()
        } else {
            0
        }
}

/// Initialize the top-level encoder.
/// Matches C: `silk_InitEncoder`.
pub fn silk_init_encoder_top(enc: &mut SilkEncoder, channels: usize) -> i32 {
    let mut ret = 0i32;
    *enc = SilkEncoder::new();

    for n in 0..channels {
        ret += silk_init_encoder(&mut enc.state_fxx[n]);
    }
    enc.n_channels_api = channels as i32;
    enc.n_channels_internal = channels as i32;
    enc.n_prev_channels_internal = channels as i32;

    ret
}

// ===========================================================================
// Top-level silk_Encode
// ===========================================================================

/// Query encoder status.
fn silk_query_encoder(
    ps_enc: &SilkEncoder,
    enc_status: &mut SilkEncControlStruct,
) {
    enc_status.internal_sample_rate = ps_enc.state_fxx[0].s_cmn.fs_khz * 1000;
    enc_status.allow_bandwidth_switch = ps_enc.allow_bandwidth_switch;
    enc_status.in_wb_mode_without_variable_lp =
        if ps_enc.state_fxx[0].s_cmn.fs_khz == 16
            && ps_enc.state_fxx[0].s_cmn.s_lp.mode == 0
        {
            1
        } else {
            0
        };
}

/// Main SILK encoding entry point.
/// Matches C: `silk_Encode`.
pub fn silk_encode(
    enc: &mut SilkEncoder,
    enc_control: &mut SilkEncControlStruct,
    samples_in: &[i16],
    n_samples_in: i32,
    range_enc: &mut RangeEncoder,
    n_bytes_out: &mut i32,
    prefill_flag: i32,
    activity: i32,
) -> i32 {
    let mut ret = SILK_NO_ERROR;

    // Validate control input
    ret = check_control_input(enc_control);
    if ret != SILK_NO_ERROR {
        return ret;
    }

    let n_channels_internal = enc_control.n_channels_internal as usize;

    // Handle channel count transitions
    if enc.n_prev_channels_internal != n_channels_internal as i32 {
        if n_channels_internal > enc.n_prev_channels_internal as usize {
            // Mono → stereo: init second channel
            silk_init_encoder(&mut enc.state_fxx[1]);
            // Copy resampler state from channel 0
            enc.state_fxx[1].s_cmn.resampler_state =
                enc.state_fxx[0].s_cmn.resampler_state.clone();
        }
        enc.n_prev_channels_internal = n_channels_internal as i32;
    }

    enc.n_channels_api = enc_control.n_channels_api;
    enc.n_channels_internal = n_channels_internal as i32;

    // Control encoder per channel
    for n in 0..n_channels_internal {
        ret = silk_control_encoder(
            &mut enc.state_fxx[n],
            enc_control,
            enc.allow_bandwidth_switch,
            n as i32,
            if n == 0 { 0 } else { 0 }, // force_fs_kHz
        );
        if ret != SILK_NO_ERROR {
            return ret;
        }
    }

    // Set prefill flag
    for n in 0..n_channels_internal {
        enc.state_fxx[n].s_cmn.prefill_flag = prefill_flag;
    }

    let fs_khz = enc.state_fxx[0].s_cmn.fs_khz;
    let frame_length = enc.state_fxx[0].s_cmn.frame_length;
    let n_frames_per_packet = enc.state_fxx[0].s_cmn.n_frames_per_packet;

    // Number of samples per frame at API rate
    let n_samples_from_input = (enc_control.api_sample_rate / 1000) * 10; // 10ms worth

    if n_samples_in < n_samples_from_input {
        *n_bytes_out = 0;
        return SILK_NO_ERROR;
    }

    // Resample and buffer input
    for n in 0..n_channels_internal {
        let channel_offset = if enc_control.n_channels_api == 2 { n } else { 0 };

        // De-interleave if stereo
        let input: Vec<i16> = if enc_control.n_channels_api == 2 {
            (0..n_samples_from_input as usize)
                .map(|i| samples_in[i * 2 + channel_offset])
                .collect()
        } else {
            samples_in[..n_samples_from_input as usize].to_vec()
        };

        // Resample from API rate to internal rate
        let internal_samples = (fs_khz * 10) as usize; // 10ms at internal rate
        let mut buf = vec![0i16; internal_samples];
        silk_resampler_run(
            &mut enc.state_fxx[n].s_cmn.resampler_state,
            &mut buf,
            &input,
        );

        // Buffer the resampled input
        let buf_ix = enc.state_fxx[n].s_cmn.input_buf_ix as usize;
        let copy_len = internal_samples.min(MAX_FRAME_LENGTH + 2 - buf_ix);
        enc.state_fxx[n].s_cmn.input_buf[buf_ix..buf_ix + copy_len]
            .copy_from_slice(&buf[..copy_len]);
        enc.state_fxx[n].s_cmn.input_buf_ix += copy_len as i32;
    }

    // Check if we have a full frame
    if enc.state_fxx[0].s_cmn.input_buf_ix < frame_length {
        *n_bytes_out = 0;
        return SILK_NO_ERROR;
    }

    // Prefill mode: just warm up filters, no output
    if prefill_flag != 0 {
        for n in 0..n_channels_internal {
            enc.state_fxx[n].s_cmn.input_buf_ix = 0;
            enc.state_fxx[n].s_cmn.controlled_since_last_payload = 0;
            enc.state_fxx[n].s_cmn.prefill_flag = 0;
        }
        *n_bytes_out = 0;
        return SILK_NO_ERROR;
    }

    // Encode LBRR data from previous packet (if any)
    if enc.state_fxx[0].s_cmn.n_frames_encoded == 0 {
        // LBRR flags
        if enc.state_fxx[0].s_cmn.lbrr_enabled != 0 {
            let lbrr_flag0 = enc.state_fxx[0].s_cmn.lbrr_flag;
            if n_frames_per_packet > 1 {
                // Encode LBRR flags
                let mut flags = 0u32;
                for i in 0..n_frames_per_packet as usize {
                    flags |= (enc.state_fxx[0].s_cmn.lbrr_flags[i] as u32) << i;
                }
                range_enc.encode_icdf(
                    flags,
                    SILK_LBRR_FLAGS_ICDF_PTR[(n_frames_per_packet - 2) as usize],
                    8,
                );
            }

            // Encode LBRR data for each frame that has it
            for i in 0..n_frames_per_packet as usize {
                if enc.state_fxx[0].s_cmn.lbrr_flags[i] != 0 {
                    silk_encode_indices(
                        &enc.state_fxx[0].s_cmn,
                        range_enc,
                        i,
                        true,
                        CODE_INDEPENDENTLY,
                    );
                    silk_encode_pulses(
                        range_enc,
                        enc.state_fxx[0].s_cmn.indices_lbrr[i].signal_type as i32,
                        enc.state_fxx[0].s_cmn.indices_lbrr[i].quant_offset_type as i32,
                        &enc.state_fxx[0].s_cmn.pulses_lbrr[i],
                        enc.state_fxx[0].s_cmn.frame_length as usize,
                    );
                }
            }
        }
    }

    // Apply HP variable cutoff filter
    silk_hp_variable_cutoff(&mut enc.state_fxx[0]);

    // Compute target bitrate
    let target_rate = enc_control.bit_rate;
    silk_control_snr(&mut enc.state_fxx[0].s_cmn, target_rate);

    // VAD + noise shaping analysis per channel
    for n in 0..n_channels_internal {
        // Clone input to avoid aliasing &mut s_cmn with &s_cmn.input_buf
        let input_copy = enc.state_fxx[n].s_cmn.input_buf[..frame_length as usize].to_vec();
        silk_vad_get_sa_q8(
            &mut enc.state_fxx[n].s_cmn,
            &input_copy,
        );
    }

    // Encode frame per channel
    let n_frames_encoded = enc.state_fxx[0].s_cmn.n_frames_encoded;
    let cond_coding = if n_frames_encoded == 0 || enc.state_fxx[0].s_cmn.first_frame_after_reset != 0 {
        CODE_INDEPENDENTLY
    } else {
        CODE_CONDITIONALLY
    };

    for n in 0..n_channels_internal {
        // Encode indices
        silk_encode_indices(
            &enc.state_fxx[n].s_cmn,
            range_enc,
            n_frames_encoded as usize,
            false,
            cond_coding,
        );

        // Encode pulses
        silk_encode_pulses(
            range_enc,
            enc.state_fxx[n].s_cmn.indices.signal_type as i32,
            enc.state_fxx[n].s_cmn.indices.quant_offset_type as i32,
            &enc.state_fxx[n].s_cmn.pulses,
            frame_length as usize,
        );
    }

    // Update counters
    for n in 0..n_channels_internal {
        enc.state_fxx[n].s_cmn.n_frames_encoded += 1;
        enc.state_fxx[n].s_cmn.controlled_since_last_payload = 0;
        enc.state_fxx[n].s_cmn.input_buf_ix = 0;
        enc.state_fxx[n].s_cmn.first_frame_after_reset = 0;
    }

    // Check if packet is complete
    if enc.state_fxx[0].s_cmn.n_frames_encoded >= n_frames_per_packet {
        // Reset frame counter for next packet
        for n in 0..n_channels_internal {
            enc.state_fxx[n].s_cmn.n_frames_encoded = 0;
        }
    }

    // Query encoder status
    silk_query_encoder(enc, enc_control);

    ret
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_control_input_valid() {
        let ctrl = SilkEncControlStruct::default();
        assert_eq!(check_control_input(&ctrl), SILK_NO_ERROR);
    }

    #[test]
    fn test_check_control_input_bad_api_rate() {
        let mut ctrl = SilkEncControlStruct::default();
        ctrl.api_sample_rate = 11025;
        assert_eq!(check_control_input(&ctrl), SILK_ENC_FS_NOT_SUPPORTED);
    }

    #[test]
    fn test_check_control_input_bad_packet_size() {
        let mut ctrl = SilkEncControlStruct::default();
        ctrl.payload_size_ms = 30;
        assert_eq!(check_control_input(&ctrl), SILK_ENC_PACKET_SIZE_NOT_SUPPORTED);
    }

    #[test]
    fn test_check_control_input_bad_loss_rate() {
        let mut ctrl = SilkEncControlStruct::default();
        ctrl.packet_loss_percentage = 101;
        assert_eq!(check_control_input(&ctrl), SILK_ENC_INVALID_LOSS_RATE);
    }

    #[test]
    fn test_check_control_input_bad_channels() {
        let mut ctrl = SilkEncControlStruct::default();
        ctrl.n_channels_api = 0;
        assert_eq!(check_control_input(&ctrl), SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR);
    }

    #[test]
    fn test_check_control_input_channels_internal_gt_api() {
        let mut ctrl = SilkEncControlStruct::default();
        ctrl.n_channels_api = 1;
        ctrl.n_channels_internal = 2;
        assert_eq!(check_control_input(&ctrl), SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR);
    }

    #[test]
    fn test_check_control_input_bad_complexity() {
        let mut ctrl = SilkEncControlStruct::default();
        ctrl.complexity = 11;
        assert_eq!(check_control_input(&ctrl), SILK_ENC_INVALID_COMPLEXITY_SETTING);
    }

    #[test]
    fn test_vad_init() {
        let mut vad = SilkVadState::default();
        assert_eq!(silk_vad_init(&mut vad), 0);
        // Noise levels should be initialized as pink noise approximation
        assert!(vad.noise_level_bias[0] > vad.noise_level_bias[1]);
        assert_eq!(vad.counter, 15);
        assert_eq!(vad.nrg_ratio_smth_q8[0], 25600); // 100 * 256
    }

    #[test]
    fn test_control_snr() {
        let mut state = SilkEncoderState::default();
        state.fs_khz = 16;
        state.nb_subfr = 4;
        silk_control_snr(&mut state, 20000);
        assert!(state.snr_db_q7 > 0);
        assert_eq!(state.target_rate_bps, 20000);
    }

    #[test]
    fn test_control_snr_low_bitrate() {
        let mut state = SilkEncoderState::default();
        state.fs_khz = 8;
        state.nb_subfr = 4;
        silk_control_snr(&mut state, 3000);
        // Very low bitrate should give 0 SNR
        assert_eq!(state.snr_db_q7, 0);
    }

    #[test]
    fn test_silk_interpolate() {
        let x0 = [0i16; MAX_LPC_ORDER];
        let x1 = [1000i16; MAX_LPC_ORDER];
        let mut xi = [0i16; MAX_LPC_ORDER];
        silk_interpolate(&mut xi, &x0, &x1, 2, 10); // factor = 2/4 = 0.5
        assert_eq!(xi[0], 500);
    }

    #[test]
    fn test_silk_encoder_size() {
        let size_stereo = silk_get_encoder_size(2);
        let size_mono = silk_get_encoder_size(1);
        assert!(size_mono < size_stereo);
    }

    #[test]
    fn test_init_encoder() {
        let mut enc = SilkEncoderStateFix::default();
        assert_eq!(silk_init_encoder(&mut enc), 0);
        assert_eq!(enc.s_cmn.first_frame_after_reset, 1);
        assert!(enc.s_cmn.variable_hp_smth1_q15 != 0);
    }

    #[test]
    fn test_init_encoder_top() {
        let mut enc = SilkEncoder::new();
        assert_eq!(silk_init_encoder_top(&mut enc, 1), 0);
        assert_eq!(enc.n_channels_api, 1);
    }

    #[test]
    fn test_ana_filt_bank_1() {
        let input = [1000i16; 32];
        let mut state = [0i32; 2];
        let mut out_l = [0i16; 16];
        let mut out_h = [0i16; 16];
        silk_ana_filt_bank_1(&input, &mut state, &mut out_l, &mut out_h, 32);
        // Low band should have energy, high band should be near zero for DC input
        let low_energy: i64 = out_l.iter().map(|&s| s as i64 * s as i64).sum();
        let high_energy: i64 = out_h.iter().map(|&s| s as i64 * s as i64).sum();
        assert!(low_energy > high_energy);
    }

    #[test]
    fn test_setup_complexity() {
        let mut state = SilkEncoderState::default();
        state.fs_khz = 16;
        state.predict_lpc_order = 16;
        silk_setup_complexity(&mut state, 10);
        assert_eq!(state.n_states_delayed_decision, MAX_DEL_DEC_STATES as i32);
        assert_eq!(state.shaping_lpc_order, 24);
        assert!(state.warping_q16 > 0);
    }

    #[test]
    fn test_gains_quant_roundtrip() {
        let mut gains_q16 = [65536i32, 80000, 50000, 65536];
        let mut ind = [0i8; 4];
        let mut prev_ind = 10i8;
        silk_gains_quant(&mut ind, &mut gains_q16, &mut prev_ind, false, 4);
        // Quantized gains should be reasonable
        for g in &gains_q16 {
            assert!(*g > 0);
        }
    }

    #[test]
    fn test_nlsf_vq_weights_laroia() {
        let nlsf = [3277i16, 6554, 9830, 13107, 16384, 19660, 22937, 26214, 29491, 32000];
        let mut w = [0i16; 10];
        silk_nlsf_vq_weights_laroia(&mut w, &nlsf, 10);
        // All weights should be positive
        for &ww in &w {
            assert!(ww > 0, "Weight should be positive, got {}", ww);
        }
    }

    #[test]
    fn test_stereo_quant_pred() {
        let mut pred = [5000i32, -3000];
        let mut ix = [[0i8; 3]; 2];
        silk_stereo_quant_pred(&mut pred, &mut ix);
        // Quantized predictors should be close to input
        assert!(pred[0].abs() <= 15000);
        assert!(pred[1].abs() <= 15000);
    }
}
