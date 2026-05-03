//! Manual FFI bindings to the C reference opus library.
//!
//! These map directly to the public opus API and selected internal functions
//! for module-level comparison testing.

#![allow(non_camel_case_types, dead_code)]

use std::os::raw::{c_int, c_uchar, c_uint};

// ---------------------------------------------------------------------------
// Opus types (matching opus_types.h)
// ---------------------------------------------------------------------------
pub type opus_int16 = i16;
pub type opus_int32 = i32;
pub type opus_uint32 = u32;

// ---------------------------------------------------------------------------
// Opus constants (matching opus_defines.h)
// ---------------------------------------------------------------------------
pub const OPUS_OK: c_int = 0;
pub const OPUS_BAD_ARG: c_int = -1;
pub const OPUS_BUFFER_TOO_SMALL: c_int = -2;
pub const OPUS_INTERNAL_ERROR: c_int = -3;
pub const OPUS_INVALID_PACKET: c_int = -4;
pub const OPUS_UNIMPLEMENTED: c_int = -5;
pub const OPUS_INVALID_STATE: c_int = -6;
pub const OPUS_ALLOC_FAIL: c_int = -7;

pub const OPUS_APPLICATION_VOIP: c_int = 2048;
pub const OPUS_APPLICATION_AUDIO: c_int = 2049;
pub const OPUS_APPLICATION_RESTRICTED_LOWDELAY: c_int = 2051;

pub const OPUS_AUTO: c_int = -1000;

// Bandwidth values
pub const OPUS_BANDWIDTH_NARROWBAND: c_int = 1101;
pub const OPUS_BANDWIDTH_MEDIUMBAND: c_int = 1102;
pub const OPUS_BANDWIDTH_WIDEBAND: c_int = 1103;
pub const OPUS_BANDWIDTH_SUPERWIDEBAND: c_int = 1104;
pub const OPUS_BANDWIDTH_FULLBAND: c_int = 1105;

// Signal types
pub const OPUS_SIGNAL_VOICE: c_int = 3001;
pub const OPUS_SIGNAL_MUSIC: c_int = 3002;

// CTL request codes
pub const OPUS_SET_BITRATE_REQUEST: c_int = 4002;
pub const OPUS_SET_MAX_BANDWIDTH_REQUEST: c_int = 4004;
pub const OPUS_SET_VBR_REQUEST: c_int = 4006;
pub const OPUS_SET_BANDWIDTH_REQUEST: c_int = 4008;
pub const OPUS_SET_COMPLEXITY_REQUEST: c_int = 4010;
pub const OPUS_SET_INBAND_FEC_REQUEST: c_int = 4012;
pub const OPUS_SET_PACKET_LOSS_PERC_REQUEST: c_int = 4014;
pub const OPUS_SET_DTX_REQUEST: c_int = 4016;
pub const OPUS_SET_VBR_CONSTRAINT_REQUEST: c_int = 4020;
pub const OPUS_SET_FORCE_CHANNELS_REQUEST: c_int = 4022;
pub const OPUS_SET_SIGNAL_REQUEST: c_int = 4024;
pub const OPUS_SET_LSB_DEPTH_REQUEST: c_int = 4036;
pub const OPUS_SET_PREDICTION_DISABLED_REQUEST: c_int = 4042;
pub const OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST: c_int = 4046;

// Getter CTL request codes
pub const OPUS_GET_BITRATE_REQUEST: c_int = 4003;
pub const OPUS_GET_MAX_BANDWIDTH_REQUEST: c_int = 4005;
pub const OPUS_GET_VBR_REQUEST: c_int = 4007;
pub const OPUS_GET_BANDWIDTH_REQUEST: c_int = 4009;
pub const OPUS_GET_COMPLEXITY_REQUEST: c_int = 4011;
pub const OPUS_GET_INBAND_FEC_REQUEST: c_int = 4013;
pub const OPUS_GET_PACKET_LOSS_PERC_REQUEST: c_int = 4015;
pub const OPUS_GET_DTX_REQUEST: c_int = 4017;
pub const OPUS_GET_VBR_CONSTRAINT_REQUEST: c_int = 4021;
pub const OPUS_GET_FORCE_CHANNELS_REQUEST: c_int = 4023;
pub const OPUS_GET_SIGNAL_REQUEST: c_int = 4025;
pub const OPUS_GET_LOOKAHEAD_REQUEST: c_int = 4027;
pub const OPUS_GET_SAMPLE_RATE_REQUEST: c_int = 4029;
pub const OPUS_GET_FINAL_RANGE_REQUEST: c_int = 4031;
pub const OPUS_GET_PITCH_REQUEST: c_int = 4033;
pub const OPUS_GET_LSB_DEPTH_REQUEST: c_int = 4037;
pub const OPUS_GET_LAST_PACKET_DURATION_REQUEST: c_int = 4039;
pub const OPUS_GET_PREDICTION_DISABLED_REQUEST: c_int = 4043;
pub const OPUS_GET_GAIN_REQUEST: c_int = 4045;
pub const OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST: c_int = 4047;
pub const OPUS_GET_IN_DTX_REQUEST: c_int = 4049;

// Internal CTL (from opus_private.h)
pub const OPUS_SET_FORCE_MODE_REQUEST: c_int = 11002;

// ---------------------------------------------------------------------------
// Opaque encoder/decoder handles
// ---------------------------------------------------------------------------
#[repr(C)]
pub struct OpusEncoder {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct OpusDecoder {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct OpusRepacketizer {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct OpusMSEncoder {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct OpusMSDecoder {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct OpusProjectionEncoder {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct OpusProjectionDecoder {
    _opaque: [u8; 0],
}

// ---------------------------------------------------------------------------
// Range coder context (matches struct ec_ctx in entcode.h)
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ec_ctx {
    pub buf: *mut c_uchar,
    pub storage: opus_uint32,
    pub end_offs: opus_uint32,
    pub end_window: opus_uint32, // ec_window = opus_uint32
    pub nend_bits: c_int,
    pub nbits_total: c_int,
    pub offs: opus_uint32,
    pub rng: opus_uint32,
    pub val: opus_uint32,
    pub ext: opus_uint32,
    pub rem: c_int,
    pub error: c_int,
}

pub type ec_enc = ec_ctx;
pub type ec_dec = ec_ctx;

// ---------------------------------------------------------------------------
// Public Opus API
// ---------------------------------------------------------------------------
#[link(name = "opus_ref", kind = "static")]
unsafe extern "C" {
    // Encoder
    pub fn opus_encoder_get_size(channels: c_int) -> c_int;
    pub fn opus_encoder_create(
        Fs: opus_int32,
        channels: c_int,
        application: c_int,
        error: *mut c_int,
    ) -> *mut OpusEncoder;
    pub fn opus_encode(
        st: *mut OpusEncoder,
        pcm: *const opus_int16,
        frame_size: c_int,
        data: *mut c_uchar,
        max_data_bytes: opus_int32,
    ) -> opus_int32;
    pub fn opus_encoder_destroy(st: *mut OpusEncoder);
    pub fn opus_encoder_ctl(st: *mut OpusEncoder, request: c_int, ...) -> c_int;

    // Decoder
    pub fn opus_decoder_get_size(channels: c_int) -> c_int;
    pub fn opus_decoder_create(
        Fs: opus_int32,
        channels: c_int,
        error: *mut c_int,
    ) -> *mut OpusDecoder;
    pub fn opus_decode(
        st: *mut OpusDecoder,
        data: *const c_uchar,
        len: opus_int32,
        pcm: *mut opus_int16,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
    pub fn opus_decode24(
        st: *mut OpusDecoder,
        data: *const c_uchar,
        len: opus_int32,
        pcm: *mut opus_int32,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
    pub fn opus_decoder_destroy(st: *mut OpusDecoder);
    pub fn opus_decoder_ctl(st: *mut OpusDecoder, request: c_int, ...) -> c_int;

    // Packet introspection
    pub fn opus_packet_get_bandwidth(data: *const c_uchar) -> c_int;
    pub fn opus_packet_get_nb_channels(data: *const c_uchar) -> c_int;
    pub fn opus_packet_get_nb_frames(data: *const c_uchar, len: opus_int32) -> c_int;
    pub fn opus_packet_get_samples_per_frame(data: *const c_uchar, Fs: opus_int32) -> c_int;
    pub fn opus_packet_get_nb_samples(
        data: *const c_uchar,
        len: opus_int32,
        Fs: opus_int32,
    ) -> c_int;

    // Info
    pub fn opus_strerror(error: c_int) -> *const std::os::raw::c_char;
    pub fn opus_get_version_string() -> *const std::os::raw::c_char;

    // Range coder (internal CELT API)
    pub fn ec_enc_init(this: *mut ec_enc, buf: *mut c_uchar, size: opus_uint32);
    pub fn ec_enc_done(this: *mut ec_enc);
    pub fn ec_enc_uint(this: *mut ec_enc, fl: opus_uint32, ft: opus_uint32);
    pub fn ec_enc_bit_logp(this: *mut ec_enc, val: c_int, logp: c_uint);
    pub fn ec_enc_bits(this: *mut ec_enc, fl: opus_uint32, ftb: c_uint);

    pub fn ec_dec_init(this: *mut ec_dec, buf: *mut c_uchar, storage: opus_uint32);
    pub fn ec_dec_uint(this: *mut ec_dec, ft: opus_uint32) -> opus_uint32;
    pub fn ec_dec_bit_logp(this: *mut ec_dec, logp: c_uint) -> c_int;
    pub fn ec_dec_bits(this: *mut ec_dec, ftb: c_uint) -> opus_uint32;
    pub fn ec_dec_icdf(this: *mut ec_dec, icdf: *const c_uchar, ftb: c_uint) -> c_int;

    // Laplace-distribution entropy coder (DRED)
    pub fn ec_laplace_encode_p0(this: *mut ec_enc, value: c_int, p0: u16, decay: u16);
    pub fn ec_laplace_decode_p0(this: *mut ec_dec, p0: u16, decay: u16) -> c_int;

    // Repacketizer
    pub fn opus_repacketizer_create() -> *mut OpusRepacketizer;
    pub fn opus_repacketizer_destroy(rp: *mut OpusRepacketizer);
    pub fn opus_repacketizer_init(rp: *mut OpusRepacketizer) -> *mut OpusRepacketizer;
    pub fn opus_repacketizer_cat(
        rp: *mut OpusRepacketizer,
        data: *const c_uchar,
        len: opus_int32,
    ) -> c_int;
    pub fn opus_repacketizer_out(
        rp: *mut OpusRepacketizer,
        data: *mut c_uchar,
        maxlen: opus_int32,
    ) -> opus_int32;
    pub fn opus_repacketizer_out_range(
        rp: *mut OpusRepacketizer,
        begin: c_int,
        end: c_int,
        data: *mut c_uchar,
        maxlen: opus_int32,
    ) -> opus_int32;
    pub fn opus_repacketizer_get_nb_frames(rp: *mut OpusRepacketizer) -> c_int;
    pub fn opus_packet_pad(data: *mut c_uchar, len: opus_int32, new_len: opus_int32) -> c_int;
    pub fn opus_packet_unpad(data: *mut c_uchar, len: opus_int32) -> opus_int32;

    // Multistream encoder
    pub fn opus_multistream_encoder_create(
        Fs: opus_int32,
        channels: c_int,
        streams: c_int,
        coupled_streams: c_int,
        mapping: *const c_uchar,
        application: c_int,
        error: *mut c_int,
    ) -> *mut OpusMSEncoder;
    pub fn opus_multistream_encode(
        st: *mut OpusMSEncoder,
        pcm: *const opus_int16,
        frame_size: c_int,
        data: *mut c_uchar,
        max_data_bytes: opus_int32,
    ) -> opus_int32;
    pub fn opus_multistream_encoder_destroy(st: *mut OpusMSEncoder);
    pub fn opus_multistream_encoder_ctl(st: *mut OpusMSEncoder, request: c_int, ...) -> c_int;

    // Multistream decoder
    pub fn opus_multistream_decoder_create(
        Fs: opus_int32,
        channels: c_int,
        streams: c_int,
        coupled_streams: c_int,
        mapping: *const c_uchar,
        error: *mut c_int,
    ) -> *mut OpusMSDecoder;
    pub fn opus_multistream_decode(
        st: *mut OpusMSDecoder,
        data: *const c_uchar,
        len: opus_int32,
        pcm: *mut opus_int16,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
    pub fn opus_multistream_decode24(
        st: *mut OpusMSDecoder,
        data: *const c_uchar,
        len: opus_int32,
        pcm: *mut opus_int32,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
    pub fn opus_multistream_decoder_destroy(st: *mut OpusMSDecoder);

    // Projection encoder (mapping family 3, ambisonics)
    pub fn opus_projection_ambisonics_encoder_create(
        Fs: opus_int32,
        channels: c_int,
        mapping_family: c_int,
        streams: *mut c_int,
        coupled_streams: *mut c_int,
        application: c_int,
        error: *mut c_int,
    ) -> *mut OpusProjectionEncoder;
    pub fn opus_projection_encode(
        st: *mut OpusProjectionEncoder,
        pcm: *const opus_int16,
        frame_size: c_int,
        data: *mut c_uchar,
        max_data_bytes: opus_int32,
    ) -> opus_int32;
    pub fn opus_projection_encoder_destroy(st: *mut OpusProjectionEncoder);
    pub fn opus_projection_encoder_ctl(
        st: *mut OpusProjectionEncoder,
        request: c_int,
        ...
    ) -> c_int;

    // Projection decoder
    pub fn opus_projection_decoder_create(
        Fs: opus_int32,
        channels: c_int,
        streams: c_int,
        coupled_streams: c_int,
        demixing_matrix: *const c_uchar,
        demixing_matrix_size: opus_int32,
        error: *mut c_int,
    ) -> *mut OpusProjectionDecoder;
    pub fn opus_projection_decode(
        st: *mut OpusProjectionDecoder,
        data: *const c_uchar,
        len: opus_int32,
        pcm: *mut opus_int16,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
    pub fn opus_projection_decoder_destroy(st: *mut OpusProjectionDecoder);

    // Debug math comparison helpers
    pub fn debug_c_celt_sqrt32(x: opus_int32) -> opus_int32;
    pub fn debug_c_celt_atan2p_norm(y: opus_int32, x: opus_int32) -> opus_int32;
    pub fn debug_c_celt_atan_norm(x: opus_int32) -> opus_int32;
    pub fn debug_c_frac_div32(a: opus_int32, b: opus_int32) -> opus_int32;
    pub fn debug_c_stereo_itheta(
        x: *const opus_int32,
        y: *const opus_int32,
        stereo: c_int,
        n: c_int,
    ) -> opus_int32;
    pub fn debug_c_celt_inner_prod_norm_shift(
        x: *const opus_int32,
        y: *const opus_int32,
        len: c_int,
    ) -> opus_int32;
    pub fn debug_c_celt_rsqrt_norm32(x: opus_int32) -> opus_int32;
    pub fn debug_c_celt_rsqrt_norm(x: opus_int32) -> opus_int32;
    pub fn debug_c_celt_rcp_norm16(x: opus_int32) -> opus_int32;
    pub fn debug_c_celt_rcp_norm32(x: opus_int32) -> opus_int32;
    pub fn debug_c_celt_rcp(x: opus_int32) -> opus_int32;
    pub fn debug_c_normalise_residual_g(ryy: opus_int32, gain: opus_int32) -> opus_int32;
    pub fn debug_c_opus_fast_int64() -> c_int;
    pub fn debug_get_celt_preemph_mem(dec: *mut OpusDecoder, out_mem: *mut opus_int32);
    pub fn debug_get_celt_decode_mem(
        dec: *mut OpusDecoder,
        offset: c_int,
        count: c_int,
        out: *mut opus_int32,
    );
    pub fn debug_get_celt_old_band_e(
        dec: *mut OpusDecoder,
        out_buf: *mut opus_int32,
        max_len: c_int,
    ) -> c_int;
    pub fn debug_get_celt_postfilter(
        dec: *mut OpusDecoder,
        period: *mut opus_int32,
        period_old: *mut opus_int32,
        gain: *mut opus_int32,
        gain_old: *mut opus_int32,
        tapset: *mut opus_int32,
        tapset_old: *mut opus_int32,
    );
    pub fn debug_get_celt_old_log_e(
        dec: *mut OpusDecoder,
        out_log_e: *mut opus_int32,
        out_log_e2: *mut opus_int32,
        max_len: c_int,
    ) -> c_int;
    pub fn debug_c_decode_energy(
        data: *const c_uchar,
        len: c_int,
        old_bands_inout: *mut opus_int32,
        fine_quant_out: *mut opus_int32,
        cc: c_int,
        lm: c_int,
    );
    pub fn debug_c_hp_cutoff_stereo(
        input: *const opus_int16,
        cutoff_hz: opus_int32,
        output: *mut opus_int16,
        hp_mem: *mut opus_int32,
        len: c_int,
        fs: opus_int32,
    );
    pub fn debug_get_encoder_hp_state(
        enc: *mut OpusEncoder,
        hp_mem_out: *mut opus_int32,
        variable_hp_smth2: *mut opus_int32,
        mode_out: *mut opus_int32,
        stream_channels_out: *mut opus_int32,
        bandwidth_out: *mut opus_int32,
    );

    pub fn debug_c_stereo_lr_to_ms(
        x1_in: *const opus_int16,
        x2_in: *const opus_int16,
        mid_out: *mut opus_int16,
        side_out: *mut opus_int16,
        pred_ix_out: *mut i8,
        mid_only_out: *mut i8,
        mid_side_rates_out: *mut opus_int32,
        total_rate_bps: opus_int32,
        prev_speech_act_q8: c_int,
        to_mono: c_int,
        fs_khz: c_int,
        frame_length: c_int,
    );

    pub fn debug_dump_silk_stereo(enc: *mut OpusEncoder);

    pub fn debug_get_silk_state(
        enc: *mut OpusEncoder,
        fs_khz: *mut opus_int32,
        frame_length: *mut opus_int32,
        nb_subfr: *mut opus_int32,
        input_buf_ix: *mut opus_int32,
        n_frames_per_packet: *mut opus_int32,
        packet_size_ms: *mut opus_int32,
        first_frame_after_reset: *mut opus_int32,
        controlled_since_last_payload: *mut opus_int32,
        prefill_flag: *mut opus_int32,
        n_frames_encoded: *mut opus_int32,
        speech_activity_q8: *mut opus_int32,
        signal_type: *mut opus_int32,
        input_quality_bands_q15: *mut opus_int32,
    );

    pub fn debug_clt_mdct_backward(
        input: *const opus_int32,
        output: *mut opus_int32,
        overlap_buf: *const opus_int32,
        overlap: c_int,
        shift: c_int,
        stride: c_int,
        n_mdct: c_int,
    );

    // INSTRUMENT: SILK LBRR and rate control state extraction
    pub fn debug_get_silk_lbrr_state(
        enc: *mut OpusEncoder,
        n_bits_used_lbrr: *mut opus_int32,
        lbrr_flag: *mut opus_int32,
        n_bits_exceeded: *mut opus_int32,
        signal_type: *mut opus_int32,
        quant_offset_type: *mut opus_int32,
        gains_indices: *mut opus_int32,
        lag_index: *mut opus_int32,
        contour_index: *mut opus_int32,
        seed: *mut opus_int32,
        ltp_scale_index: *mut opus_int32,
        nlsf_interp_coef: *mut opus_int32,
    );

    // INSTRUMENT: SILK NLSF indices, pulses, and additional encoder state
    pub fn debug_get_silk_nlsf_and_pulses(
        enc: *mut OpusEncoder,
        nlsf_indices: *mut opus_int32, // 17 elements
        ltp_indices: *mut opus_int32,  // 4 elements
        per_index: *mut opus_int32,
        prev_signal_type: *mut opus_int32,
        prev_lag: *mut opus_int32,
        frame_counter: *mut opus_int32,
        ec_prev_lag_index: *mut opus_int32,
        ec_prev_signal_type: *mut opus_int32,
        first_frame_after_reset: *mut opus_int32,
        controlled_since_last_payload: *mut opus_int32,
        pulses_sum: *mut opus_int32,
    );

    // INSTRUMENT: CELT encoder state extraction
    pub fn debug_get_celt_encoder_state(
        enc: *mut OpusEncoder,
        delayed_intra: *mut opus_int32,
        loss_rate: *mut opus_int32,
        prefilter_period: *mut opus_int32,
        prefilter_gain: *mut opus_int32,
        prefilter_tapset: *mut opus_int32,
        force_intra: *mut opus_int32,
        spread_decision: *mut opus_int32,
        tonal_average: *mut opus_int32,
        last_coded_bands: *mut opus_int32,
        consec_transient: *mut opus_int32,
    );

    pub fn debug_get_celt_encoder_state_ext(
        enc: *mut OpusEncoder,
        stereo_saving: *mut opus_int32,
        hf_average: *mut opus_int32,
        spec_avg: *mut opus_int32,
        intensity: *mut opus_int32,
        overlap_max: *mut opus_int32,
        vbr_reservoir: *mut opus_int32,
        vbr_drift: *mut opus_int32,
        vbr_offset: *mut opus_int32,
        vbr_count: *mut opus_int32,
        preemph_mem_e_0: *mut opus_int32,
        preemph_mem_e_1: *mut opus_int32,
        preemph_mem_d_0: *mut opus_int32,
        preemph_mem_d_1: *mut opus_int32,
    );

    pub fn debug_dump_silk_plc_state(
        dec: *mut OpusDecoder,
        rand_scale_q14: *mut opus_int16,
        rand_seed: *mut opus_int32,
        pitch_l_q8: *mut opus_int32,
        loss_cnt: *mut opus_int32,
        prev_signal_type: *mut opus_int32,
    );

    pub fn debug_get_silk_interframe_state(
        enc: *mut OpusEncoder,
        channel: c_int,
        last_gain_index: *mut opus_int32,
        prev_gain_q16: *mut opus_int32,
        variable_hp_smth1_q15: *mut opus_int32,
        variable_hp_smth2_q15: *mut opus_int32,
        harm_shape_gain_smth: *mut opus_int32,
        tilt_smth: *mut opus_int32,
        prev_signal_type: *mut opus_int32,
        prev_lag: *mut opus_int32,
        ec_prev_lag_index: *mut opus_int32,
        ec_prev_signal_type: *mut opus_int32,
        prev_nlsfq_q15: *mut opus_int16,
        stereo_width_prev_q14: *mut opus_int32,
        stereo_smth_width_q14: *mut opus_int32,
        stereo_pred_prev_q13_0: *mut opus_int32,
        stereo_pred_prev_q13_1: *mut opus_int32,
        n_bits_exceeded: *mut opus_int32,
    );

    pub fn debug_get_silk_nlsf_indices(
        enc: *mut OpusEncoder,
        channel: c_int,
        nlsf_indices: *mut i8,
        predict_lpc_order: *mut opus_int32,
        signal_type: *mut opus_int32,
        nlsf_interp_coef_q2: *mut opus_int32,
    );

    pub fn debug_get_silk_xbuf_hash(
        enc: *mut OpusEncoder,
        channel: c_int,
        hash_out: *mut opus_int32,
        buf_len: *mut opus_int32,
    );

    /// Multistream wrapper accessor: returns a pointer to the inner
    /// OpusEncoder for `stream_id` within an `OpusMSEncoder`. The returned
    /// pointer is valid as long as `ms` is, and is suitable for the other
    /// `debug_get_*` accessors in this module. Returns NULL if
    /// `stream_id` is out of range.
    pub fn debug_get_inner_opus_encoder(
        ms: *mut OpusMSEncoder,
        stream_id: c_int,
    ) -> *mut OpusEncoder;

    /// Top-level Opus state plus a `silk_mode` subset for the encoder
    /// state-accumulation diagnostic (Cluster A, Findings #7 + #8). Reads
    /// fields that are mutated by `OPUS_SET_*` CTLs but not by every
    /// Rust-side `ms_set_*` (the H1/H2 asymmetry suspects).
    pub fn debug_get_opus_silk_mode_state(
        enc: *mut OpusEncoder,
        sm_use_in_band_fec: *mut opus_int32,
        sm_use_cbr: *mut opus_int32,
        sm_use_dtx: *mut opus_int32,
        sm_lbrr_coded: *mut opus_int32,
        sm_complexity: *mut opus_int32,
        sm_packet_loss_percentage: *mut opus_int32,
        sm_bit_rate: *mut opus_int32,
        sm_payload_size_ms: *mut opus_int32,
        sm_n_channels_internal: *mut opus_int32,
        sm_max_internal_sample_rate: *mut opus_int32,
        sm_min_internal_sample_rate: *mut opus_int32,
        sm_desired_internal_sample_rate: *mut opus_int32,
        use_vbr: *mut opus_int32,
        vbr_constraint: *mut opus_int32,
        use_dtx: *mut opus_int32,
        fec_config: *mut opus_int32,
        user_bitrate_bps: *mut opus_int32,
        bitrate_bps: *mut opus_int32,
        force_channels: *mut opus_int32,
        signal_type: *mut opus_int32,
        lsb_depth: *mut opus_int32,
        lfe: *mut opus_int32,
        application: *mut opus_int32,
    );

    pub fn debug_get_silk_extended_state(
        enc: *mut OpusEncoder,
        channel: c_int,
        /* NSQ state */
        nsq_rand_seed: *mut opus_int32,
        nsq_slf_ar_shp_q14: *mut opus_int32,
        nsq_lag_prev: *mut opus_int32,
        nsq_sdiff_shp_q14: *mut opus_int32,
        nsq_sltp_buf_idx: *mut opus_int32,
        nsq_sltp_shp_buf_idx: *mut opus_int32,
        nsq_rewhite_flag: *mut opus_int32,
        nsq_slpc_q14: *mut opus_int32,
        nsq_sar2_q14: *mut opus_int32,
        nsq_sltp_shp_q14: *mut opus_int32,
        /* VAD state */
        vad_hp_state: *mut opus_int32,
        vad_counter: *mut opus_int32,
        vad_noise_level_bias: *mut opus_int32,
        vad_ana_state: *mut opus_int32,
        vad_ana_state1: *mut opus_int32,
        vad_ana_state2: *mut opus_int32,
        vad_nrg_ratio_smth_q8: *mut opus_int32,
        vad_nl: *mut opus_int32,
        vad_inv_nl: *mut opus_int32,
        /* Common state */
        sum_log_gain_q7: *mut opus_int32,
        in_hp_state: *mut opus_int32,
        input_tilt_q15: *mut opus_int32,
        input_quality_bands_q15: *mut opus_int32,
        frame_counter: *mut opus_int32,
        no_speech_counter: *mut opus_int32,
        /* LP state */
        lp_in_lp_state: *mut opus_int32,
        lp_transition_frame_no: *mut opus_int32,
        /* Shape */
        shape_harm_boost_smth: *mut opus_int32,
        /* x_buf hash */
        x_buf_hash: *mut opus_int32,
        ltp_corr_q15: *mut opus_int32,
        res_nrg_smth: *mut opus_int32,
    );

    // ----- SILK decode trace (debug_silk_trace.c) -----

    pub fn debug_silk_trace_get_persistent_state(
        dec: *mut OpusDecoder,
        prev_gain_q16: *mut opus_int32,
        s_lpc_q14_buf: *mut opus_int32,
        lag_prev: *mut opus_int32,
        last_gain_index: *mut opus_int32,
        fs_khz: *mut opus_int32,
        nb_subfr: *mut opus_int32,
        frame_length: *mut opus_int32,
        subfr_length: *mut opus_int32,
        ltp_mem_length: *mut opus_int32,
        lpc_order: *mut opus_int32,
        first_frame_after_reset: *mut opus_int32,
        loss_cnt: *mut opus_int32,
        prev_signal_type: *mut opus_int32,
    );

    pub fn debug_silk_trace_get_indices(
        dec: *mut OpusDecoder,
        signal_type: *mut opus_int32,
        quant_offset_type: *mut opus_int32,
        gains_indices: *mut opus_int32,
        nlsf_indices: *mut opus_int32,
        lag_index: *mut opus_int32,
        contour_index: *mut opus_int32,
        nlsf_interp_coef_q2: *mut opus_int32,
        per_index: *mut opus_int32,
        ltp_index: *mut opus_int32,
        ltp_scale_index: *mut opus_int32,
        seed: *mut opus_int32,
    );

    pub fn debug_silk_trace_get_prev_nlsf(dec: *mut OpusDecoder, prev_nlsf_q15: *mut opus_int16);

    pub fn debug_silk_trace_get_exc(
        dec: *mut OpusDecoder,
        exc_q14: *mut opus_int32,
        count: opus_int32,
    );

    pub fn debug_silk_trace_get_outbuf(
        dec: *mut OpusDecoder,
        out_buf: *mut opus_int16,
        count: opus_int32,
    );

    pub fn debug_silk_traced_decode(
        dec: *mut OpusDecoder,
        silk_data: *const c_uchar,
        silk_len: c_int,
        cond_coding: c_int,
        gains_q16_out: *mut opus_int32,
        pred_coef_q12_0_out: *mut opus_int16,
        pred_coef_q12_1_out: *mut opus_int16,
        pitch_l_out: *mut opus_int32,
        ltp_coef_q14_out: *mut opus_int16,
        ltp_scale_q14_out: *mut opus_int32,
        pulses_out: *mut opus_int16,
        pcm_out: *mut opus_int16,
    ) -> c_int;

    pub fn debug_silk_trace_get_config(
        dec: *mut OpusDecoder,
        fs_khz: *mut opus_int32,
        frame_length: *mut opus_int32,
        n_frames_decoded: *mut opus_int32,
    );

    pub fn debug_silk_trace_reset(dec: *mut OpusDecoder);

    // -- Phase B encode-side trace FIFO (Cluster A stage 2b).
    //    Defined in `harness/debug_silk_trace.c`. The C trace push at
    //    each of the 7 boundaries fires from `harness/silk_enc_api_traced.c`,
    //    which replaces xiph's enc_API.c in the harness build.
    pub fn dbg_silk_trace_clear();
    pub fn dbg_silk_trace_count_get() -> c_int;
    pub fn dbg_silk_trace_read(
        idx: c_int,
        boundary_id: *mut c_int,
        channel: *mut c_int,
        ec_tell: *mut opus_int32,
        rng: *mut u32,
        target_rate_bps: *mut opus_int32,
        n_bits_exceeded: *mut opus_int32,
        curr_n_bits_used_lbrr: *mut opus_int32,
        n_bits_used_lbrr: *mut opus_int32,
        mid_only_flag: *mut opus_int32,
        prev_decode_only_middle: *mut opus_int32,
    ) -> c_int;
}

// ---------------------------------------------------------------------------
// Safe wrappers
// ---------------------------------------------------------------------------

/// Decode an opus error code into a human-readable string.
pub fn error_string(code: c_int) -> &'static str {
    unsafe {
        let ptr = opus_strerror(code);
        if ptr.is_null() {
            return "unknown error";
        }
        std::ffi::CStr::from_ptr(ptr)
            .to_str()
            .unwrap_or("unknown error")
    }
}

/// Get the opus library version string.
pub fn version_string() -> &'static str {
    unsafe {
        let ptr = opus_get_version_string();
        if ptr.is_null() {
            return "unknown";
        }
        std::ffi::CStr::from_ptr(ptr).to_str().unwrap_or("unknown")
    }
}
