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

use std::os::raw::{c_int, c_uchar, c_void};

pub type opus_int16 = i16;
pub type opus_int32 = i32;

pub const OPUS_OK: c_int = 0;
pub const OPUS_BAD_ARG: c_int = -1;

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
    pub fn peek_decode_mem(
        opus_st: *const OpusDecoder,
        offset: c_int,
        count: c_int,
        out: *mut f32,
    ) -> c_int;
    pub fn peek_decode_mem_stride(opus_st: *const OpusDecoder) -> c_int;
    pub fn peek_old_band_e(
        opus_st: *const OpusDecoder,
        offset: c_int,
        count: c_int,
        out: *mut f32,
    ) -> c_int;
    pub fn peek_old_log_e(
        opus_st: *const OpusDecoder,
        offset: c_int,
        count: c_int,
        out: *mut f32,
    ) -> c_int;
    pub fn peek_background_log_e(
        opus_st: *const OpusDecoder,
        offset: c_int,
        count: c_int,
        out: *mut f32,
    ) -> c_int;
    pub fn peek_nb_ebands(opus_st: *const OpusDecoder) -> c_int;

    // SILK-side peeks
    pub fn peek_silk_fs_khz_top(opus_st: *const OpusDecoder) -> c_int;
    pub fn peek_silk_prev_gain(opus_st: *const OpusDecoder) -> opus_int32;
    pub fn peek_silk_s_lpc_q14(
        opus_st: *const OpusDecoder,
        out: *mut opus_int32,
        max_count: c_int,
    ) -> c_int;
    pub fn peek_silk_plc_prev_gain_top(
        opus_st: *const OpusDecoder,
        out: *mut opus_int32,
    ) -> c_int;
    pub fn peek_silk_plc_pitch(opus_st: *const OpusDecoder) -> opus_int32;
    pub fn peek_silk_plc_rand_scale(opus_st: *const OpusDecoder) -> opus_int32;
    pub fn peek_silk_plc_last_lost(opus_st: *const OpusDecoder) -> c_int;
    pub fn peek_silk_plc_fs(opus_st: *const OpusDecoder) -> c_int;
    pub fn peek_silk_outbuf(
        opus_st: *const OpusDecoder,
        offset: c_int,
        count: c_int,
        out: *mut i16,
    ) -> c_int;
    pub fn peek_silk_ltpmem(opus_st: *const OpusDecoder) -> c_int;
    pub fn peek_silk_framelen(opus_st: *const OpusDecoder) -> c_int;
}

/// Thin RAII wrapper around the C float-mode decoder — used by the tier-2
/// tests so we can just `?` our way through errors and get `Drop` cleanup.
pub struct CRefFloatDecoder {
    ptr: *mut OpusDecoder,
}

impl CRefFloatDecoder {
    pub fn new(fs: i32, channels: i32) -> Result<Self, i32> {
        let mut err: c_int = 0;
        let ptr = unsafe { opus_decoder_create(fs, channels, &mut err) };
        if ptr.is_null() || err != OPUS_OK {
            return Err(err);
        }
        Ok(Self { ptr })
    }

    /// Decode one Opus packet to interleaved i16 PCM. Pass `None` for `data`
    /// (or a zero-length slice) to trigger the decoder's PLC path.
    /// Returns the number of samples per channel decoded.
    pub fn decode(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i16],
        frame_size: i32,
        decode_fec: bool,
    ) -> Result<i32, i32> {
        let (data_ptr, data_len) = match data {
            Some(d) if !d.is_empty() => (d.as_ptr(), d.len() as opus_int32),
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
        let ret =
            unsafe { opus_decoder_ctl(self.ptr, OPUS_SET_COMPLEXITY_REQUEST, complexity) };
        if ret == OPUS_OK { Ok(()) } else { Err(ret) }
    }

    /// Stage 7b.3 diagnostic: read `count` samples from the CELT decode_mem
    /// starting at `offset`. Returns samples as f32 (float-mode `celt_sig`).
    pub fn peek_decode_mem(&self, offset: i32, count: i32) -> Vec<f32> {
        let mut out = vec![0.0f32; count as usize];
        unsafe { peek_decode_mem(self.ptr, offset, count, out.as_mut_ptr()) };
        out
    }

    /// Per-channel stride of the CELT decode_mem slab.
    pub fn decode_mem_stride(&self) -> i32 {
        unsafe { peek_decode_mem_stride(self.ptr) }
    }

    /// Stage 7b.3 diagnostic: read oldBandE entries as f32 (celt_glog).
    pub fn peek_old_band_e(&self, offset: i32, count: i32) -> Vec<f32> {
        let mut out = vec![0.0f32; count as usize];
        unsafe { peek_old_band_e(self.ptr, offset, count, out.as_mut_ptr()) };
        out
    }

    /// Stage 7b.3 diagnostic: read oldLogE entries as f32 (celt_glog).
    pub fn peek_old_log_e(&self, offset: i32, count: i32) -> Vec<f32> {
        let mut out = vec![0.0f32; count as usize];
        unsafe { peek_old_log_e(self.ptr, offset, count, out.as_mut_ptr()) };
        out
    }

    /// Stage 7b.3 diagnostic: read backgroundLogE entries as f32 (celt_glog).
    pub fn peek_background_log_e(&self, offset: i32, count: i32) -> Vec<f32> {
        let mut out = vec![0.0f32; count as usize];
        unsafe { peek_background_log_e(self.ptr, offset, count, out.as_mut_ptr()) };
        out
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
    pub fn silk_out_buf(&self, offset: i32, count: i32) -> Vec<i16> {
        let mut out = vec![0i16; count as usize];
        unsafe { peek_silk_outbuf(self.ptr, offset, count, out.as_mut_ptr()) };
        out
    }
    pub fn silk_ltp_mem_length(&self) -> i32 {
        unsafe { peek_silk_ltpmem(self.ptr) }
    }
    pub fn silk_frame_length(&self) -> i32 {
        unsafe { peek_silk_framelen(self.ptr) }
    }
}

impl Drop for CRefFloatDecoder {
    fn drop(&mut self) {
        unsafe { opus_decoder_destroy(self.ptr) };
    }
}

// The C pointer is confined to this struct; sending it between threads is
// fine as long as the user doesn't clone it (which we don't allow).
unsafe impl Send for CRefFloatDecoder {}
