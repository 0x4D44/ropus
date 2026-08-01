use std::{ffi::c_void, os::raw::c_int};

use mdopus_capi::ms_decoder::{opus_multistream_decoder_create, opus_multistream_decoder_destroy};

const OPUS_OK: c_int = 0;
const OPUS_BAD_ARG: c_int = -1;
const OPUS_SET_COMPLEXITY_REQUEST: c_int = 4010;
const OPUS_GET_COMPLEXITY_REQUEST: c_int = 4011;

unsafe extern "C" {
    fn opus_multistream_decoder_ctl(st: *mut c_void, request: c_int, ...) -> c_int;
}

#[test]
fn c_ctl_multistream_decoder_complexity_round_trip_and_validation() {
    let mapping = [0u8, 1];
    let mut error = -3;
    let st =
        unsafe { opus_multistream_decoder_create(48_000, 2, 1, 1, mapping.as_ptr(), &mut error) };
    assert!(!st.is_null());
    assert_eq!(error, OPUS_OK);

    let mut complexity = -1;
    assert_eq!(
        unsafe {
            opus_multistream_decoder_ctl(
                st.cast::<c_void>(),
                OPUS_GET_COMPLEXITY_REQUEST,
                &mut complexity,
            )
        },
        OPUS_OK
    );
    assert!((0..=10).contains(&complexity));

    assert_eq!(
        unsafe {
            opus_multistream_decoder_ctl(st.cast::<c_void>(), OPUS_SET_COMPLEXITY_REQUEST, 7)
        },
        OPUS_OK
    );
    assert_eq!(
        unsafe {
            opus_multistream_decoder_ctl(
                st.cast::<c_void>(),
                OPUS_GET_COMPLEXITY_REQUEST,
                &mut complexity,
            )
        },
        OPUS_OK
    );
    assert_eq!(complexity, 7);

    for invalid in [-1, 11] {
        assert_eq!(
            unsafe {
                opus_multistream_decoder_ctl(
                    st.cast::<c_void>(),
                    OPUS_SET_COMPLEXITY_REQUEST,
                    invalid,
                )
            },
            OPUS_BAD_ARG
        );
        assert_eq!(
            unsafe {
                opus_multistream_decoder_ctl(
                    st.cast::<c_void>(),
                    OPUS_GET_COMPLEXITY_REQUEST,
                    &mut complexity,
                )
            },
            OPUS_OK
        );
        assert_eq!(complexity, 7);
    }

    unsafe { opus_multistream_decoder_destroy(st) };
}
