#[path = "../fuzz_targets/c_reference.rs"]
mod c_reference;

use ropus::opus::encoder::{OpusEncoder, OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_VOIP};
use std::ptr::NonNull;

const SAMPLE_RATE: i32 = 48_000;
const CHANNELS: i32 = 1;
const FRAME_SIZE: i32 = SAMPLE_RATE / 50;
const MAX_DATA_BYTES: i32 = 4000;

struct CEncoder {
    ptr: NonNull<c_reference::OpusEncoder>,
}

impl CEncoder {
    fn new(application: i32) -> Self {
        unsafe {
            let mut error = 0;
            let ptr =
                c_reference::opus_encoder_create(SAMPLE_RATE, CHANNELS, application, &mut error);
            assert!(!ptr.is_null());
            assert_eq!(error, c_reference::OPUS_OK);
            Self {
                ptr: NonNull::new_unchecked(ptr),
            }
        }
    }

    fn ctl_i32(&mut self, request: i32, value: i32) -> i32 {
        unsafe { c_reference::opus_encoder_ctl(self.ptr.as_ptr(), request, value) }
    }

    fn ctl_no_arg(&mut self, request: i32) -> i32 {
        unsafe { c_reference::opus_encoder_ctl(self.ptr.as_ptr(), request) }
    }

    fn get_i32(&mut self, request: i32) -> Result<i32, i32> {
        let mut value = 0i32;
        let ret = unsafe { c_reference::opus_encoder_ctl(self.ptr.as_ptr(), request, &mut value) };
        if ret == c_reference::OPUS_OK {
            Ok(value)
        } else {
            Err(ret)
        }
    }

    fn encode_silence(&mut self) -> i32 {
        let pcm = [0i16; FRAME_SIZE as usize];
        let mut out = [0u8; MAX_DATA_BYTES as usize];
        unsafe {
            c_reference::opus_encode(
                self.ptr.as_ptr(),
                pcm.as_ptr(),
                FRAME_SIZE,
                out.as_mut_ptr(),
                MAX_DATA_BYTES,
            )
        }
    }
}

impl Drop for CEncoder {
    fn drop(&mut self) {
        unsafe {
            c_reference::opus_encoder_destroy(self.ptr.as_ptr());
        }
    }
}

#[test]
fn phase_inversion_invalid_values_match_c_return_codes() {
    let mut rust_enc =
        OpusEncoder::new(SAMPLE_RATE, CHANNELS, OPUS_APPLICATION_AUDIO).expect("rust encoder");
    let mut c_enc = CEncoder::new(OPUS_APPLICATION_AUDIO);

    for value in [-1, 0, 1, 2] {
        let rust_ret = rust_enc.set_phase_inversion_disabled(value);
        let c_ret = c_enc.ctl_i32(
            c_reference::OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST,
            value,
        );
        assert_eq!(
            rust_ret, c_ret,
            "phase inversion return mismatch for value={value}"
        );
    }
}

#[test]
fn application_change_after_first_encode_matches_c_return_codes() {
    let mut rust_enc =
        OpusEncoder::new(SAMPLE_RATE, CHANNELS, OPUS_APPLICATION_AUDIO).expect("rust encoder");
    let mut c_enc = CEncoder::new(OPUS_APPLICATION_AUDIO);
    let pcm = [0i16; FRAME_SIZE as usize];
    let mut rust_out = [0u8; MAX_DATA_BYTES as usize];

    assert!(
        rust_enc
            .encode(&pcm, FRAME_SIZE, &mut rust_out, MAX_DATA_BYTES)
            .expect("rust encode")
            > 0
    );
    assert!(c_enc.encode_silence() > 0);

    let same_application_rust = rust_enc.set_application(OPUS_APPLICATION_AUDIO);
    let same_application_c = c_enc.ctl_i32(
        c_reference::OPUS_SET_APPLICATION_REQUEST,
        OPUS_APPLICATION_AUDIO,
    );
    assert_eq!(same_application_rust, same_application_c);

    let changed_application_rust = rust_enc.set_application(OPUS_APPLICATION_VOIP);
    let changed_application_c = c_enc.ctl_i32(
        c_reference::OPUS_SET_APPLICATION_REQUEST,
        OPUS_APPLICATION_VOIP,
    );
    assert_eq!(changed_application_rust, changed_application_c);
}

#[test]
fn bitrate_getter_initial_and_after_reset_matches_c() {
    let mut rust_enc =
        OpusEncoder::new(SAMPLE_RATE, CHANNELS, OPUS_APPLICATION_AUDIO).expect("rust encoder");
    let mut c_enc = CEncoder::new(OPUS_APPLICATION_AUDIO);

    assert_eq!(
        rust_enc.get_bitrate(),
        c_enc
            .get_i32(c_reference::OPUS_GET_BITRATE_REQUEST)
            .expect("C initial bitrate getter")
    );

    assert!(
        rust_enc
            .encode(
                &[0i16; FRAME_SIZE as usize],
                FRAME_SIZE,
                &mut [0u8; MAX_DATA_BYTES as usize],
                MAX_DATA_BYTES,
            )
            .expect("rust encode")
            > 0
    );
    assert!(c_enc.encode_silence() > 0);

    rust_enc.reset();
    assert_eq!(
        c_enc.ctl_no_arg(c_reference::OPUS_RESET_STATE),
        c_reference::OPUS_OK
    );

    assert_eq!(
        rust_enc.get_bitrate(),
        c_enc
            .get_i32(c_reference::OPUS_GET_BITRATE_REQUEST)
            .expect("C reset bitrate getter")
    );
}
