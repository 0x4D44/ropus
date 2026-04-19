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
}

impl Drop for CRefFloatDecoder {
    fn drop(&mut self) {
        unsafe { opus_decoder_destroy(self.ptr) };
    }
}

// The C pointer is confined to this struct; sending it between threads is
// fine as long as the user doesn't clone it (which we don't allow).
unsafe impl Send for CRefFloatDecoder {}
