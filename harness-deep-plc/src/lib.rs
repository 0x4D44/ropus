//! Minimal FFI bindings to the float-mode xiph C reference linked via the
//! companion `build.rs`. Only the subset needed by the Stage 7b.2 tier-2 PLC
//! acceptance tests is exposed — notably enough to create a decoder, feed it
//! packets (including PLC-trigger null packets), and destroy it.
//!
//! The C reference was compiled in float mode with `ENABLE_DEEP_PLC=1` and
//! compile-time-embedded weights. No `OPUS_SET_DNN_BLOB` call is needed — the
//! decoder auto-activates DEEP_PLC on creation (`reference/dnn/lpcnet_plc.c:58`
//! under `!USE_WEIGHTS_FILE`).

#![allow(non_camel_case_types, dead_code)]
// When `build.rs` can't find `reference/` it sets `cfg(no_reference)` so the
// workspace still builds on a fresh clone. In that mode the whole FFI
// surface compiles to nothing; consumers cfg-gate behind the same flag.
#![cfg(not(no_reference))]

use std::os::raw::{c_int, c_uchar, c_void};

pub type opus_int16 = i16;
pub type opus_int32 = i32;

pub const OPUS_OK: c_int = 0;
pub const OPUS_BAD_ARG: c_int = -1;

const MAX_DECODE_MEM_CAPACITY: usize = 2 * (2048 + 120);
const MAX_ENERGY_MEM_CAPACITY: usize = 2 * 21;
const SILK_OUT_BUF_CAPACITY: usize = 320 + 2 * 80;

#[repr(C)]
pub struct OpusDecoder {
    _opaque: [u8; 0],
}

// CTL request codes used by the tier-2 tests. See
// `reference/include/opus_defines.h`.
pub const OPUS_SET_COMPLEXITY_REQUEST: c_int = 4010;
pub const OPUS_SET_DNN_BLOB_REQUEST: c_int = 4052;

#[link(name = "opus_ref_float", kind = "static")]
unsafe extern "C" {
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

    pub fn opus_decoder_destroy(st: *mut OpusDecoder);

    pub fn opus_decoder_ctl(st: *mut OpusDecoder, request: c_int, ...) -> c_int;

    // --- Stage 8.4 DRED RDOVAE encoder shim ---
    // Defined in `harness-deep-plc/dred_enc_shim.c`. Opaque pointers keep
    // the C struct layouts out of the FFI surface.
    pub fn ropus_test_rdovaeenc_new() -> *mut c_void;
    pub fn ropus_test_rdovaeenc_free(model: *mut c_void);
    pub fn ropus_test_rdovae_enc_state_new() -> *mut c_void;
    pub fn ropus_test_rdovae_enc_state_free(state: *mut c_void);
    pub fn ropus_test_dred_rdovae_encode_dframe(
        state: *mut c_void,
        model: *const c_void,
        latents: *mut f32,
        initial_state: *mut f32,
        input: *const f32,
    );

    // --- Stage 8.5 DRED RDOVAE decoder shim ---
    // Defined in `harness-deep-plc/dred_dec_shim.c`. Same opaque-pointer
    // pattern as the encoder shim above.
    pub fn ropus_test_rdovaedec_new() -> *mut c_void;
    pub fn ropus_test_rdovaedec_free(model: *mut c_void);
    pub fn ropus_test_rdovae_dec_state_new() -> *mut c_void;
    pub fn ropus_test_rdovae_dec_state_free(state: *mut c_void);
    pub fn ropus_test_dred_rdovae_dec_init_states(
        state: *mut c_void,
        model: *const c_void,
        initial_state: *const f32,
    );
    pub fn ropus_test_dred_rdovae_decode_qframe(
        state: *mut c_void,
        model: *const c_void,
        qframe: *mut f32,
        input: *const f32,
    );

    // --- Stage 8.6 DRED full encoder-side pipeline shim ---
    // Defined in `harness-deep-plc/dred_encode_shim.c`. Wraps
    // `dred_encoder_init`, `dred_compute_latents`, and
    // `dred_encode_silk_frame` behind an opaque `DREDEnc *` so the Rust
    // differential test can drive the full payload-emission pipeline
    // without replicating the `DREDEnc` layout across FFI.
    pub fn ropus_test_dredenc_new(fs: c_int, channels: c_int) -> *mut c_void;
    pub fn ropus_test_dredenc_free(enc: *mut c_void);
    pub fn ropus_test_dredenc_input_buffer_fill(enc: *const c_void) -> c_int;
    pub fn ropus_test_dred_compute_latents(
        enc: *mut c_void,
        pcm: *const f32,
        frame_size: c_int,
        extra_delay: c_int,
    );
    pub fn ropus_test_dred_encode_silk_frame(
        enc: *mut c_void,
        buf: *mut c_uchar,
        max_chunks: c_int,
        max_bytes: c_int,
        q0: c_int,
        d_q: c_int,
        qmax: c_int,
        activity_mem: *mut c_uchar,
    ) -> c_int;
    pub fn ropus_test_dredenc_latents_buffer_fill(enc: *const c_void) -> c_int;
    pub fn ropus_test_dredenc_dred_offset(enc: *const c_void) -> c_int;
    pub fn ropus_test_dredenc_latent_offset(enc: *const c_void) -> c_int;
    pub fn ropus_test_dredenc_copy_latents(enc: *const c_void, dst: *mut f32, n: c_int);
    pub fn ropus_test_dredenc_copy_state(enc: *const c_void, dst: *mut f32, n: c_int);
    pub fn ropus_test_dredenc_copy_input_buffer(enc: *const c_void, dst: *mut f32, n: c_int);
    pub fn ropus_test_dredenc_copy_resample_mem(enc: *const c_void, dst: *mut f32, n: c_int);
    pub fn ropus_test_dredenc_copy_lpcnet_features(enc: *const c_void, dst: *mut f32, n: c_int);

    // --- Stage 8.7 payload-level shims: direct buffer poke + C decoder ---
    // Defined in `harness-deep-plc/dred_encode_shim.c`. Let the Rust
    // differential test drive `dred_encode_silk_frame` on hand-synthesised
    // latents/state (no RDOVAE upstream) and cross-check `dred_ec_decode`
    // between C and Rust on the resulting byte buffer.
    pub fn ropus_test_dredenc_set_state_buffer(enc: *mut c_void, src: *const f32, n: c_int);
    pub fn ropus_test_dredenc_set_latents_buffer(enc: *mut c_void, src: *const f32, n: c_int);
    pub fn ropus_test_dredenc_set_bookkeeping(
        enc: *mut c_void,
        latent_offset: c_int,
        latents_buffer_fill: c_int,
        dred_offset: c_int,
        last_extra_dred_offset: c_int,
    );
    pub fn ropus_test_dred_ec_decode(
        bytes: *const c_uchar,
        num_bytes: c_int,
        min_feature_frames: c_int,
        dred_frame_offset: c_int,
        out_state: *mut f32,
        out_latents: *mut f32,
        out_nb_latents: *mut c_int,
        out_process_stage: *mut c_int,
        out_dred_offset: *mut c_int,
    ) -> c_int;

    // Stage 7b.3 diagnostic peek getters (harness-deep-plc/c/peek.c).
    fn peek_decode_mem(
        opus_st: *const OpusDecoder,
        offset: c_int,
        count: c_int,
        out: *mut f32,
    ) -> c_int;
    fn peek_decode_mem_stride(opus_st: *const OpusDecoder) -> c_int;
    fn peek_decode_mem_capacity(opus_st: *const OpusDecoder) -> c_int;
    fn peek_old_band_e(
        opus_st: *const OpusDecoder,
        offset: c_int,
        count: c_int,
        out: *mut f32,
    ) -> c_int;
    fn peek_old_log_e(
        opus_st: *const OpusDecoder,
        offset: c_int,
        count: c_int,
        out: *mut f32,
    ) -> c_int;
    fn peek_background_log_e(
        opus_st: *const OpusDecoder,
        offset: c_int,
        count: c_int,
        out: *mut f32,
    ) -> c_int;
    fn peek_nb_ebands(opus_st: *const OpusDecoder) -> c_int;
    fn peek_energy_mem_capacity(opus_st: *const OpusDecoder) -> c_int;

    // SILK-side peeks
    fn peek_silk_fs_khz_top(opus_st: *const OpusDecoder) -> c_int;
    fn peek_silk_prev_gain(opus_st: *const OpusDecoder) -> opus_int32;
    fn peek_silk_s_lpc_q14(
        opus_st: *const OpusDecoder,
        out: *mut opus_int32,
        max_count: c_int,
    ) -> c_int;
    fn peek_silk_plc_prev_gain_top(opus_st: *const OpusDecoder, out: *mut opus_int32) -> c_int;
    fn peek_silk_plc_pitch(opus_st: *const OpusDecoder) -> opus_int32;
    fn peek_silk_plc_rand_scale(opus_st: *const OpusDecoder) -> opus_int32;
    fn peek_silk_plc_last_lost(opus_st: *const OpusDecoder) -> c_int;
    fn peek_silk_plc_fs(opus_st: *const OpusDecoder) -> c_int;
    fn peek_silk_outbuf(
        opus_st: *const OpusDecoder,
        offset: c_int,
        count: c_int,
        out: *mut i16,
    ) -> c_int;
    fn peek_silk_ltpmem(opus_st: *const OpusDecoder) -> c_int;
    fn peek_silk_framelen(opus_st: *const OpusDecoder) -> c_int;
    fn peek_silk_outbuf_capacity(opus_st: *const OpusDecoder) -> c_int;

    // --- Stage 8.8 full C encoder + C DRED parser shim ---
    // Defined in `harness-deep-plc/dred_encode_shim.c`. Drives the xiph C
    // encoder end-to-end with DRED enabled and exposes a one-shot
    // `opus_dred_parse` helper so the Rust integration test can assert
    // format-level cross-compatibility in both directions.
    pub fn ropus_test_c_encoder_new(
        fs: c_int,
        channels: c_int,
        application: c_int,
        dred_duration: c_int,
    ) -> *mut c_void;
    /// Stage-5 extension: parameterised version of
    /// `ropus_test_c_encoder_new`. Lets the new Tier-1 differential test
    /// pass a non-trivial `(bitrate_bps, use_inband_fec, loss_perc,
    /// use_vbr)` so `compute_dred_bitrate` returns a non-zero
    /// `dred_bitrate_bps` and exercises F33/F33b/F53/F48 + the f32 ops.
    /// Pass `use_vbr = -1` to leave VBR at the encoder default.
    pub fn ropus_test_c_encoder_new_ex(
        fs: c_int,
        channels: c_int,
        application: c_int,
        dred_duration: c_int,
        bitrate_bps: c_int,
        use_inband_fec: c_int,
        loss_perc: c_int,
        use_vbr: c_int,
    ) -> *mut c_void;
    pub fn ropus_test_c_encoder_free(enc: *mut c_void);
    pub fn ropus_test_c_encoder_encode(
        enc: *mut c_void,
        pcm: *const opus_int16,
        frame_size: c_int,
        data: *mut c_uchar,
        max_data_bytes: c_int,
    ) -> c_int;
    pub fn ropus_test_c_dred_parse(
        data: *const c_uchar,
        len: c_int,
        max_dred_samples: c_int,
        sampling_rate: c_int,
        out_nb_latents: *mut c_int,
        out_process_stage: *mut c_int,
        out_dred_offset: *mut c_int,
    ) -> c_int;

    // --- 2026-05-07 burg-cepstrum-pow-fix differential-test thunk ---
    // Defined in `harness-deep-plc/c/burg_thunk.c`. One-line wrapper
    // around the C reference's public `burg_cepstral_analysis` symbol
    // (`reference/dnn/freq.c:183`); accepts a FRAME_SIZE = 160-sample
    // f32 buffer and writes 2 * NB_BANDS = 36 cepstral outputs.
    pub fn ropus_test_burg_cepstral_analysis(x: *const f32, ceps: *mut f32);

    // --- Stage-5 (apply-feedback): direct FFI scalar fixture for the
    // DRED bitrate helpers ---
    //
    // Both functions in C are `static` inside `opus_encoder.c`; the shim
    // file (`harness-deep-plc/dred_encode_shim.c`) holds verbatim copies
    // of the C function bodies and exposes them via these symbols. See
    // the comment block in that file for the rationale and sync rules.
    pub fn ropus_c_estimate_dred_bitrate(
        q0: c_int,
        d_q: c_int,
        qmax: c_int,
        duration: c_int,
        target_bits: c_int,
        target_chunks: *mut c_int,
    ) -> c_int;
    pub fn ropus_c_compute_dred_bitrate(
        use_in_band_fec: c_int,
        packet_loss_perc: c_int,
        fs: c_int,
        dred_duration: c_int,
        bitrate_bps: c_int,
        frame_size: c_int,
        out_q0: *mut c_int,
        out_d_q: *mut c_int,
        out_qmax: *mut c_int,
        out_target_chunks: *mut c_int,
    ) -> c_int;
}

/// Opus application modes (mirrored from `opus_defines.h`). Only the ones
/// Stage 8.8 actually uses — matches ropus's own `opus/encoder.rs` constants.
pub const OPUS_APPLICATION_VOIP: c_int = 2048;
pub const OPUS_APPLICATION_AUDIO: c_int = 2049;

/// Thin RAII wrapper around the C float-mode decoder — used by the tier-2
/// tests so we can just `?` our way through errors and get `Drop` cleanup.
pub struct CRefFloatDecoder {
    ptr: *mut OpusDecoder,
    channels: usize,
    decode_mem_capacity: usize,
    energy_mem_capacity: usize,
    silk_out_buf_capacity: usize,
}

impl CRefFloatDecoder {
    pub fn new(fs: i32, channels: i32) -> Result<Self, i32> {
        let channels = usize::try_from(channels)
            .ok()
            .filter(|&channels| matches!(channels, 1 | 2))
            .ok_or(OPUS_BAD_ARG)?;
        let mut err: c_int = 0;
        let ptr = unsafe { opus_decoder_create(fs, channels as c_int, &mut err) };
        if ptr.is_null() {
            return Err(if err == OPUS_OK { OPUS_BAD_ARG } else { err });
        }
        if err != OPUS_OK {
            unsafe { opus_decoder_destroy(ptr) };
            return Err(err);
        }

        let capacities = unsafe {
            (
                peek_decode_mem_capacity(ptr),
                peek_energy_mem_capacity(ptr),
                peek_silk_outbuf_capacity(ptr),
            )
        };
        let Some((decode_mem_capacity, energy_mem_capacity, silk_out_buf_capacity)) =
            validate_peek_capacities(capacities)
        else {
            unsafe { opus_decoder_destroy(ptr) };
            return Err(OPUS_BAD_ARG);
        };

        Ok(Self {
            ptr,
            channels,
            decode_mem_capacity,
            energy_mem_capacity,
            silk_out_buf_capacity,
        })
    }

    /// Decode one Opus packet to interleaved i16 PCM. Pass `None` for `data`
    /// (or a zero-length slice) to trigger the decoder's PLC path.
    /// Returns the number of samples per channel decoded.
    /// Returns `OPUS_BAD_ARG` without entering C when `frame_size` is not
    /// positive, the packet length does not fit `opus_int32`, or `pcm` cannot
    /// hold `frame_size * channels` samples.
    pub fn decode(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i16],
        frame_size: i32,
        decode_fec: bool,
    ) -> Result<i32, i32> {
        checked_decode_output_len(frame_size, self.channels, pcm.len())?;
        let (data_ptr, data_len) = match data {
            Some(d) if !d.is_empty() => (d.as_ptr(), checked_packet_len(d.len())?),
            _ => (std::ptr::null(), 0),
        };
        let ret = unsafe {
            opus_decode(
                self.ptr,
                data_ptr,
                data_len,
                pcm.as_mut_ptr(),
                frame_size,
                if decode_fec { 1 } else { 0 },
            )
        };
        if ret < 0 { Err(ret) } else { Ok(ret) }
    }

    /// Set the decoder complexity. Needed because xiph gates DEEP_PLC on
    /// `complexity >= 5` (`reference/src/opus_decoder.c:443`); default is 0
    /// on a freshly-created decoder. Matches our ropus contract.
    pub fn set_complexity(&mut self, complexity: i32) -> Result<(), i32> {
        let ret = unsafe { opus_decoder_ctl(self.ptr, OPUS_SET_COMPLEXITY_REQUEST, complexity) };
        if ret == OPUS_OK { Ok(()) } else { Err(ret) }
    }

    /// Stage 7b.3 diagnostic: read `count` samples from the CELT decode_mem
    /// starting at `offset`. Returns samples as f32 (float-mode `celt_sig`).
    /// Invalid or overflowing ranges return `OPUS_BAD_ARG` before entering C.
    pub fn peek_decode_mem(&self, offset: i32, count: i32) -> Result<Vec<f32>, i32> {
        self.copy_f32_peek(offset, count, self.decode_mem_capacity, peek_decode_mem)
    }

    /// Per-channel stride of the CELT decode_mem slab.
    pub fn decode_mem_stride(&self) -> i32 {
        unsafe { peek_decode_mem_stride(self.ptr) }
    }

    /// Stage 7b.3 diagnostic: read oldBandE entries as f32 (celt_glog).
    /// Invalid or overflowing ranges return `OPUS_BAD_ARG` before entering C.
    pub fn peek_old_band_e(&self, offset: i32, count: i32) -> Result<Vec<f32>, i32> {
        self.copy_f32_peek(offset, count, self.energy_mem_capacity, peek_old_band_e)
    }

    /// Stage 7b.3 diagnostic: read oldLogE entries as f32 (celt_glog).
    /// Invalid or overflowing ranges return `OPUS_BAD_ARG` before entering C.
    pub fn peek_old_log_e(&self, offset: i32, count: i32) -> Result<Vec<f32>, i32> {
        self.copy_f32_peek(offset, count, self.energy_mem_capacity, peek_old_log_e)
    }

    /// Stage 7b.3 diagnostic: read backgroundLogE entries as f32 (celt_glog).
    /// Invalid or overflowing ranges return `OPUS_BAD_ARG` before entering C.
    pub fn peek_background_log_e(&self, offset: i32, count: i32) -> Result<Vec<f32>, i32> {
        self.copy_f32_peek(
            offset,
            count,
            self.energy_mem_capacity,
            peek_background_log_e,
        )
    }

    /// nbEBands of the active CELT mode.
    pub fn nb_ebands(&self) -> i32 {
        unsafe { peek_nb_ebands(self.ptr) }
    }

    // --- SILK peeks ---

    pub fn silk_fs_khz(&self) -> i32 {
        unsafe { peek_silk_fs_khz_top(self.ptr) }
    }
    pub fn silk_prev_gain_q16(&self) -> i32 {
        unsafe { peek_silk_prev_gain(self.ptr) }
    }
    /// MAX_LPC_ORDER = 16 entries.
    pub fn silk_s_lpc_q14(&self) -> [i32; 16] {
        let mut out = [0i32; 16];
        unsafe { peek_silk_s_lpc_q14(self.ptr, out.as_mut_ptr(), 16) };
        out
    }
    pub fn silk_plc_prev_gain_q16(&self) -> [i32; 2] {
        let mut out = [0i32; 2];
        unsafe { peek_silk_plc_prev_gain_top(self.ptr, out.as_mut_ptr()) };
        out
    }
    pub fn silk_plc_pitch_l_q8(&self) -> i32 {
        unsafe { peek_silk_plc_pitch(self.ptr) }
    }
    pub fn silk_plc_rand_scale_q14(&self) -> i32 {
        unsafe { peek_silk_plc_rand_scale(self.ptr) }
    }
    pub fn silk_plc_last_frame_lost(&self) -> i32 {
        unsafe { peek_silk_plc_last_lost(self.ptr) }
    }
    pub fn silk_plc_fs_khz(&self) -> i32 {
        unsafe { peek_silk_plc_fs(self.ptr) }
    }
    /// Read a checked range from SILK's fixed-capacity output history.
    pub fn silk_out_buf(&self, offset: i32, count: i32) -> Result<Vec<i16>, i32> {
        let count = checked_peek_range(offset, count, self.silk_out_buf_capacity)?;
        let mut out = vec![0i16; count];
        let ret = unsafe { peek_silk_outbuf(self.ptr, offset, count as c_int, out.as_mut_ptr()) };
        if ret == count as c_int {
            Ok(out)
        } else {
            Err(if ret < 0 { ret } else { OPUS_BAD_ARG })
        }
    }
    pub fn silk_ltp_mem_length(&self) -> i32 {
        unsafe { peek_silk_ltpmem(self.ptr) }
    }
    pub fn silk_frame_length(&self) -> i32 {
        unsafe { peek_silk_framelen(self.ptr) }
    }

    fn copy_f32_peek(
        &self,
        offset: i32,
        count: i32,
        capacity: usize,
        peek: unsafe extern "C" fn(*const OpusDecoder, c_int, c_int, *mut f32) -> c_int,
    ) -> Result<Vec<f32>, i32> {
        let count = checked_peek_range(offset, count, capacity)?;
        let mut out = vec![0.0f32; count];
        let ret = unsafe { peek(self.ptr, offset, count as c_int, out.as_mut_ptr()) };
        if ret == count as c_int {
            Ok(out)
        } else {
            Err(if ret < 0 { ret } else { OPUS_BAD_ARG })
        }
    }
}

fn checked_packet_len(len: usize) -> Result<opus_int32, i32> {
    opus_int32::try_from(len).map_err(|_| OPUS_BAD_ARG)
}

fn checked_decode_output_len(
    frame_size: i32,
    channels: usize,
    pcm_len: usize,
) -> Result<usize, i32> {
    let frame_size = usize::try_from(frame_size)
        .ok()
        .filter(|&frame_size| frame_size > 0)
        .ok_or(OPUS_BAD_ARG)?;
    let required = frame_size.checked_mul(channels).ok_or(OPUS_BAD_ARG)?;
    if pcm_len < required {
        return Err(OPUS_BAD_ARG);
    }
    Ok(required)
}

fn checked_peek_range(offset: i32, count: i32, capacity: usize) -> Result<usize, i32> {
    let offset = usize::try_from(offset).map_err(|_| OPUS_BAD_ARG)?;
    let count = usize::try_from(count)
        .ok()
        .filter(|&count| count > 0)
        .ok_or(OPUS_BAD_ARG)?;
    let end = offset.checked_add(count).ok_or(OPUS_BAD_ARG)?;
    if end > capacity {
        return Err(OPUS_BAD_ARG);
    }
    Ok(count)
}

fn validate_peek_capacities(capacities: (i32, i32, i32)) -> Option<(usize, usize, usize)> {
    let decode_mem = usize::try_from(capacities.0).ok()?;
    let energy_mem = usize::try_from(capacities.1).ok()?;
    let silk_out_buf = usize::try_from(capacities.2).ok()?;
    if !(1..=MAX_DECODE_MEM_CAPACITY).contains(&decode_mem)
        || !(1..=MAX_ENERGY_MEM_CAPACITY).contains(&energy_mem)
        || silk_out_buf != SILK_OUT_BUF_CAPACITY
    {
        return None;
    }
    Some((decode_mem, energy_mem, silk_out_buf))
}

impl Drop for CRefFloatDecoder {
    fn drop(&mut self) {
        unsafe { opus_decoder_destroy(self.ptr) };
    }
}

// The C pointer is confined to this struct; sending it between threads is
// fine as long as the user doesn't clone it (which we don't allow).
unsafe impl Send for CRefFloatDecoder {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::ManuallyDrop;

    fn null_decoder(channels: usize) -> ManuallyDrop<CRefFloatDecoder> {
        ManuallyDrop::new(CRefFloatDecoder {
            ptr: std::ptr::null_mut(),
            channels,
            decode_mem_capacity: MAX_DECODE_MEM_CAPACITY,
            energy_mem_capacity: MAX_ENERGY_MEM_CAPACITY,
            silk_out_buf_capacity: SILK_OUT_BUF_CAPACITY,
        })
    }

    #[test]
    fn constructor_rejects_invalid_channels_before_entering_c() {
        assert!(matches!(
            CRefFloatDecoder::new(48_000, 0),
            Err(OPUS_BAD_ARG)
        ));
        assert!(matches!(
            CRefFloatDecoder::new(48_000, -1),
            Err(OPUS_BAD_ARG)
        ));
        assert!(matches!(
            CRefFloatDecoder::new(48_000, 3),
            Err(OPUS_BAD_ARG)
        ));
    }

    #[test]
    fn decode_rejects_invalid_dimensions_before_entering_c() {
        let mut decoder = null_decoder(2);
        let mut one_sample = [0i16; 1];

        assert_eq!(
            decoder.decode(None, &mut one_sample, 1, false),
            Err(OPUS_BAD_ARG)
        );
        assert_eq!(
            decoder.decode(None, &mut one_sample, 0, false),
            Err(OPUS_BAD_ARG)
        );
        assert_eq!(
            decoder.decode(None, &mut one_sample, -1, false),
            Err(OPUS_BAD_ARG)
        );
    }

    #[test]
    fn oversized_packet_lengths_are_rejected() {
        assert_eq!(checked_packet_len(i32::MAX as usize), Ok(i32::MAX));
        assert_eq!(checked_packet_len(i32::MAX as usize + 1), Err(OPUS_BAD_ARG));
    }

    #[test]
    fn peek_ranges_reject_negative_empty_overflowing_and_past_end() {
        assert_eq!(checked_peek_range(-1, 1, 42), Err(OPUS_BAD_ARG));
        assert_eq!(checked_peek_range(0, -1, 42), Err(OPUS_BAD_ARG));
        assert_eq!(checked_peek_range(0, 0, 42), Err(OPUS_BAD_ARG));
        assert_eq!(checked_peek_range(41, 2, 42), Err(OPUS_BAD_ARG));
        assert_eq!(checked_peek_range(41, 1, 42), Ok(1));
    }

    #[test]
    fn invalid_peeks_return_before_entering_c() {
        let decoder = null_decoder(1);

        assert_eq!(decoder.peek_decode_mem(-1, 1), Err(OPUS_BAD_ARG));
        assert_eq!(decoder.peek_old_band_e(0, 0), Err(OPUS_BAD_ARG));
        assert_eq!(decoder.peek_old_log_e(42, 1), Err(OPUS_BAD_ARG));
        assert_eq!(decoder.peek_background_log_e(41, 2), Err(OPUS_BAD_ARG));
        assert_eq!(decoder.silk_out_buf(480, 1), Err(OPUS_BAD_ARG));
    }

    #[test]
    fn runtime_peek_capacities_are_sanity_checked() {
        assert_eq!(
            validate_peek_capacities((
                MAX_DECODE_MEM_CAPACITY as i32,
                MAX_ENERGY_MEM_CAPACITY as i32,
                SILK_OUT_BUF_CAPACITY as i32,
            )),
            Some((
                MAX_DECODE_MEM_CAPACITY,
                MAX_ENERGY_MEM_CAPACITY,
                SILK_OUT_BUF_CAPACITY,
            ))
        );
        assert_eq!(
            validate_peek_capacities((
                -1,
                MAX_ENERGY_MEM_CAPACITY as i32,
                SILK_OUT_BUF_CAPACITY as i32,
            )),
            None
        );
        assert_eq!(
            validate_peek_capacities((
                MAX_DECODE_MEM_CAPACITY as i32 + 1,
                MAX_ENERGY_MEM_CAPACITY as i32,
                SILK_OUT_BUF_CAPACITY as i32,
            )),
            None
        );
        assert_eq!(
            validate_peek_capacities((
                MAX_DECODE_MEM_CAPACITY as i32,
                MAX_ENERGY_MEM_CAPACITY as i32 + 1,
                SILK_OUT_BUF_CAPACITY as i32,
            )),
            None
        );
        assert_eq!(
            validate_peek_capacities((
                MAX_DECODE_MEM_CAPACITY as i32,
                MAX_ENERGY_MEM_CAPACITY as i32,
                SILK_OUT_BUF_CAPACITY as i32 - 1,
            )),
            None
        );
    }
}
