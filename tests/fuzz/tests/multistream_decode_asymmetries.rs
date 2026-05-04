//! Regression tests for documented multistream decode-side asymmetries.
//!
//! These tests replay libFuzzer `Arbitrary` inputs through the same structured
//! input shape used by `fuzz_multistream`, then compare Rust decoder behaviour
//! against the C reference.

#[path = "../fuzz_targets/c_reference.rs"]
mod c_reference;

#[path = "../fuzz_targets/oracle.rs"]
mod oracle;

use libfuzzer_sys::arbitrary::{self, Arbitrary, Unstructured};
use ropus::opus::encoder::OPUS_APPLICATION_AUDIO;
use ropus::opus::decoder::{
    OpusDecoder, opus_packet_get_bandwidth, opus_packet_get_nb_channels,
    opus_packet_get_nb_frames, opus_packet_get_nb_samples, opus_packet_get_samples_per_frame,
};
use ropus::opus::multistream::{OpusMSDecoder, OpusMSEncoder};

const SAMPLE_RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];
const MAX_CHANNELS: u8 = 8;

#[derive(Debug)]
struct MSInput {
    op: u8,
    sample_rate_idx: u8,
    channels: u8,
    mapping_family: u8,
    _application_idx: u8,
    _bitrate_raw: u16,
    _complexity: u8,
    _vbr: bool,
    _setter_bytes: [u8; 16],
    payload: Vec<u8>,
}

impl<'a> Arbitrary<'a> for MSInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let op = u.int_in_range(0..=2)?;
        let sample_rate_idx = u.int_in_range(0..=4)?;
        let channels = u.int_in_range(1..=MAX_CHANNELS)?;
        let mapping_family_idx: u8 = u.int_in_range(0..=3)?;
        let mapping_family = match mapping_family_idx {
            0 => 0,
            1 => 1,
            2 => 2,
            _ => 255,
        };
        let application_idx = u.int_in_range(0..=2)?;
        let bitrate_raw = u.arbitrary()?;
        let complexity = u.int_in_range(0..=9)?;
        let vbr = u.arbitrary()?;
        let mut setter_bytes = [0u8; 16];
        u.fill_buffer(&mut setter_bytes)?;
        let payload_len = u.int_in_range(0..=20480)?;
        let payload = u.bytes(payload_len)?.to_vec();
        Ok(Self {
            op,
            sample_rate_idx,
            channels,
            mapping_family,
            _application_idx: application_idx,
            _bitrate_raw: bitrate_raw,
            _complexity: complexity,
            _vbr: vbr,
            _setter_bytes: setter_bytes,
            payload,
        })
    }
}

struct DecodeObservation {
    sample_rate: i32,
    channels: i32,
    mapping_family: i32,
    streams: i32,
    coupled_streams: i32,
    mapping: Vec<u8>,
    payload: Vec<u8>,
    rust_ret: Result<i32, i32>,
    c_ret: Result<Vec<i16>, i32>,
    rust_pcm: Vec<i16>,
}

impl DecodeObservation {
    fn c_samples_per_channel(&self) -> Option<usize> {
        self.c_ret
            .as_ref()
            .ok()
            .map(|pcm| pcm.len() / self.channels as usize)
    }

    fn is_celt_only(&self) -> bool {
        !self.payload.is_empty() && (self.payload[0] & 0x80) != 0
    }

    fn snr_db(&self) -> Option<f64> {
        let rust_samples = self.rust_ret.ok()? as usize;
        let total = rust_samples * self.channels as usize;
        Some(oracle::snr_db(self.c_ret.as_ref().ok()?, &self.rust_pcm[..total]))
    }
}

fn decode_observation(bytes: &[u8]) -> DecodeObservation {
    let input = MSInput::arbitrary(&mut Unstructured::new(bytes)).expect("valid MSInput");
    assert_eq!(input.op % 3, 1, "fixture must select decode path");

    let sample_rate = SAMPLE_RATES[input.sample_rate_idx as usize];
    let channels = input.channels as i32;
    let mapping_family = input.mapping_family as i32;

    let (_probe_enc, streams, coupled_streams, mapping) =
        OpusMSEncoder::new_surround(sample_rate, channels, mapping_family, OPUS_APPLICATION_AUDIO)
            .expect("fixture surround mapping");
    assert!(streams + coupled_streams <= 16);

    let mut rust_dec =
        OpusMSDecoder::new(sample_rate, channels, streams, coupled_streams, &mapping)
            .expect("fixture Rust decoder");
    let frame_cap = 5760usize;
    let mut rust_pcm = vec![0i16; frame_cap * channels as usize];
    let rust_ret = rust_dec.decode(
        Some(&input.payload),
        input.payload.len() as i32,
        &mut rust_pcm,
        frame_cap as i32,
        false,
    );
    let c_ret = c_reference::c_ms_decode(
        &input.payload,
        sample_rate,
        channels,
        streams,
        coupled_streams,
        &mapping,
        frame_cap as i32,
    );

    DecodeObservation {
        sample_rate,
        channels,
        mapping_family,
        streams,
        coupled_streams,
        mapping,
        payload: input.payload,
        rust_ret,
        c_ret,
        rust_pcm,
    }
}

fn c_decode_with_range(
    data: &[u8],
    sample_rate: i32,
    channels: i32,
) -> (Result<Vec<i16>, i32>, Option<u32>) {
    unsafe {
        let mut error = 0;
        let dec = c_reference::opus_decoder_create(sample_rate, channels, &mut error);
        if dec.is_null() || error != c_reference::OPUS_OK {
            if !dec.is_null() {
                c_reference::opus_decoder_destroy(dec);
            }
            return (Err(error), None);
        }

        let mut pcm = vec![0i16; 5760 * channels as usize];
        let ret = c_reference::opus_decode(
            dec,
            data.as_ptr(),
            data.len() as c_reference::opus_int32,
            pcm.as_mut_ptr(),
            5760,
            0,
        );
        let mut range = 0u32;
        let range_ret = c_reference::opus_decoder_ctl(
            dec,
            c_reference::OPUS_GET_FINAL_RANGE_REQUEST,
            &mut range,
        );
        c_reference::opus_decoder_destroy(dec);

        let c_ret = if ret < 0 {
            Err(ret as i32)
        } else {
            pcm.truncate(ret as usize * channels as usize);
            Ok(pcm)
        };
        (c_ret, (range_ret == c_reference::OPUS_OK).then_some(range))
    }
}

fn c_ms_decode_with_range(
    data: &[u8],
    sample_rate: i32,
    channels: i32,
    streams: i32,
    coupled_streams: i32,
    mapping: &[u8],
    frame_size: i32,
) -> (Result<Vec<i16>, i32>, Option<u32>) {
    unsafe {
        let mut error = 0;
        let dec = c_reference::opus_multistream_decoder_create(
            sample_rate,
            channels,
            streams,
            coupled_streams,
            mapping.as_ptr(),
            &mut error,
        );
        if dec.is_null() || error != c_reference::OPUS_OK {
            if !dec.is_null() {
                c_reference::opus_multistream_decoder_destroy(dec);
            }
            return (Err(error), None);
        }

        let mut pcm = vec![0i16; frame_size as usize * channels as usize];
        let ret = c_reference::opus_multistream_decode(
            dec,
            data.as_ptr(),
            data.len() as c_reference::opus_int32,
            pcm.as_mut_ptr(),
            frame_size,
            0,
        );
        let mut range = 0u32;
        let range_ret = c_reference::opus_multistream_decoder_ctl(
            dec,
            c_reference::OPUS_GET_FINAL_RANGE_REQUEST,
            &mut range,
        );
        c_reference::opus_multistream_decoder_destroy(dec);

        let c_ret = if ret < 0 {
            Err(ret as i32)
        } else {
            pcm.truncate(ret as usize * channels as usize);
            Ok(pcm)
        };
        (c_ret, (range_ret == c_reference::OPUS_OK).then_some(range))
    }
}

#[test]
fn family0_celt_only_decode_artifact_matches_c_reference() {
    let bytes = include_bytes!(
        "../seeds/fuzz_multistream/regression-ms-decode-celt-plc-neural-gate.bin"
    )
    .as_slice();
    let obs = decode_observation(bytes);
    assert_eq!(obs.sample_rate, 12000);
    assert_eq!(obs.channels, 1);
    assert_eq!(obs.mapping_family, 0);
    assert_eq!(obs.streams, 1);
    assert_eq!(obs.coupled_streams, 0);
    assert_eq!(obs.mapping, [0]);
    assert_eq!(
        obs.payload,
        [
            0xe2, 0x11, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x00,
            0x05, 0x35, 0x04, 0xff, 0x00, 0x00, 0x00,
        ]
    );

    let c_bandwidth =
        unsafe { c_reference::opus_packet_get_bandwidth(obs.payload.as_ptr()) };
    let c_channels =
        unsafe { c_reference::opus_packet_get_nb_channels(obs.payload.as_ptr()) };
    let c_frames = unsafe {
        c_reference::opus_packet_get_nb_frames(
            obs.payload.as_ptr(),
            obs.payload.len() as c_reference::opus_int32,
        )
    };
    let c_samples_per_frame =
        unsafe { c_reference::opus_packet_get_samples_per_frame(obs.payload.as_ptr(), obs.sample_rate) };
    let c_samples = unsafe {
        c_reference::opus_packet_get_nb_samples(
            obs.payload.as_ptr(),
            obs.payload.len() as c_reference::opus_int32,
            obs.sample_rate,
        )
    };
    assert_eq!(opus_packet_get_bandwidth(&obs.payload), c_bandwidth);
    assert_eq!(opus_packet_get_nb_channels(&obs.payload), c_channels);
    assert_eq!(opus_packet_get_nb_frames(&obs.payload), Ok(c_frames));
    assert_eq!(
        opus_packet_get_samples_per_frame(&obs.payload, obs.sample_rate),
        c_samples_per_frame
    );
    assert_eq!(opus_packet_get_nb_samples(&obs.payload, obs.sample_rate), Ok(c_samples));
    assert_eq!(obs.payload[0] >> 3, 28, "TOC config must be CELT-only");
    assert!(obs.is_celt_only());

    let frame_cap = 5760;
    let (c_ms_ret, c_ms_range) = c_ms_decode_with_range(
        &obs.payload,
        obs.sample_rate,
        obs.channels,
        obs.streams,
        obs.coupled_streams,
        &obs.mapping,
        frame_cap,
    );
    assert_eq!(obs.c_ret, c_ms_ret);

    let mut rust_ms_dec = OpusMSDecoder::new(
        obs.sample_rate,
        obs.channels,
        obs.streams,
        obs.coupled_streams,
        &obs.mapping,
    )
    .unwrap();
    let mut rust_ms_pcm = vec![0i16; frame_cap as usize * obs.channels as usize];
    let rust_ms_ret = rust_ms_dec.decode(
        Some(&obs.payload),
        obs.payload.len() as i32,
        &mut rust_ms_pcm,
        frame_cap,
        false,
    );
    assert_eq!(obs.rust_ret, rust_ms_ret);

    let mut rust_dec = OpusDecoder::new(obs.sample_rate, obs.channels).unwrap();
    let mut rust_pcm = vec![0i16; frame_cap as usize * obs.channels as usize];
    let rust_ret = rust_dec.decode(Some(&obs.payload), &mut rust_pcm, frame_cap, false);
    let rust_range = rust_dec.get_final_range();
    let (c_ret, c_range) = c_decode_with_range(&obs.payload, obs.sample_rate, obs.channels);
    assert_eq!(rust_ms_dec.get_final_range(), c_ms_range.unwrap());
    assert_eq!(rust_range, c_range.unwrap());

    let first_frame_packet = [
        0xe0, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x00, 0x05, 0x35,
        0x04, 0xff, 0x00, 0x00,
    ];
    let second_frame_packet = [0xe0, 0x00];
    let mut rust_first_dec = OpusDecoder::new(obs.sample_rate, obs.channels).unwrap();
    let mut rust_first_pcm = vec![0i16; frame_cap as usize * obs.channels as usize];
    let rust_first =
        rust_first_dec.decode(Some(&first_frame_packet), &mut rust_first_pcm, frame_cap, false);
    let rust_first_range = rust_first_dec.get_final_range();
    let (c_first, c_first_range) =
        c_decode_with_range(&first_frame_packet, obs.sample_rate, obs.channels);
    assert_eq!(rust_first, Ok(30));
    assert_eq!(c_first.as_ref().map(|pcm| pcm.len()), Ok(30));
    assert_eq!(rust_first_range, c_first_range.unwrap());
    assert_eq!(&rust_first_pcm[..30], c_first.as_ref().unwrap());

    let mut rust_seq_dec = OpusDecoder::new(obs.sample_rate, obs.channels).unwrap();
    let mut rust_seq_pcm = vec![0i16; frame_cap as usize * obs.channels as usize];
    assert_eq!(
        rust_seq_dec.decode(Some(&first_frame_packet), &mut rust_seq_pcm, frame_cap, false),
        Ok(30)
    );
    assert_eq!(
        rust_seq_dec.decode(Some(&second_frame_packet), &mut rust_seq_pcm, frame_cap, false),
        Ok(30)
    );

    let c_pcm = c_ret.as_ref().expect("C standalone decode should succeed");
    let rust_samples = rust_ret.expect("Rust standalone decode should succeed") as usize;
    let total = rust_samples * obs.channels as usize;
    assert_eq!(total, c_pcm.len());
    assert_eq!(&rust_pcm[..total], c_pcm);
}

fn assert_family255_asymmetry_fixed(bytes: &[u8]) {
    let obs = decode_observation(bytes);
    assert_eq!(obs.mapping_family, 255);
    assert!(
        matches!((&obs.rust_ret, &obs.c_ret), (Ok(_), Ok(_)) | (Err(_), Err(_))),
        "family=255 decode-result asymmetry should be closed, got Rust={:?}, C={:?}, sr={}, ch={}, streams={}, coupled={}, packet_len={}",
        obs.rust_ret,
        obs.c_ret.as_ref().map(|pcm| pcm.len()),
        obs.sample_rate,
        obs.channels,
        obs.streams,
        obs.coupled_streams,
        obs.payload.len()
    );
    if let Ok(rust_samples) = obs.rust_ret {
        assert_eq!(obs.c_samples_per_channel(), Some(rust_samples as usize));
    }
}

#[test]
fn family255_custom_mapping_error_asymmetries_are_fixed() {
    for bytes in [
        include_bytes!("../seeds/fuzz_multistream/regression-ms-decode-family255-parser-00.bin")
            .as_slice(),
        include_bytes!("../seeds/fuzz_multistream/regression-ms-decode-family255-parser-01.bin")
            .as_slice(),
        include_bytes!("../seeds/fuzz_multistream/regression-ms-decode-family255-parser-02.bin")
            .as_slice(),
        include_bytes!("../seeds/fuzz_multistream/regression-ms-decode-family255-parser-03.bin")
            .as_slice(),
    ] {
        assert_family255_asymmetry_fixed(bytes);
    }
}

#[test]
fn family1_vorbis_surround_over_rejection_is_tracked_precisely() {
    for bytes in [
        include_bytes!("../seeds/fuzz_multistream/regression-ms-decode-family1-parser-00.bin")
            .as_slice(),
        include_bytes!("../seeds/fuzz_multistream/regression-ms-decode-family1-parser-01.bin")
            .as_slice(),
    ] {
        let obs = decode_observation(bytes);
        assert_eq!(obs.sample_rate, 48000);
        assert_eq!(obs.channels, 3);
        assert_eq!(obs.mapping_family, 1);
        assert_eq!(obs.streams, 2);
        assert_eq!(obs.coupled_streams, 1);
        assert_eq!(obs.mapping, [0, 2, 1]);
        let rust_samples = obs.rust_ret.expect("Rust should now decode family=1 fixture");
        assert_eq!(obs.c_samples_per_channel(), Some(5760));
        assert_eq!(rust_samples, 5760);
    }
}

#[test]
fn multistream_recovery_repros_are_silk_hybrid_pcm_only_divergences() {
    for bytes in [
        include_bytes!("../known_failures/multistream-decode-recovery-divergence/crash-00.bin")
            .as_slice(),
        include_bytes!("../known_failures/multistream-decode-recovery-divergence/crash-01.bin")
            .as_slice(),
        include_bytes!("../known_failures/multistream-decode-recovery-divergence/crash-02.bin")
            .as_slice(),
        include_bytes!("../known_failures/multistream-decode-recovery-divergence/crash-03.bin")
            .as_slice(),
    ] {
        let obs = decode_observation(bytes);
        let rust_samples = obs.rust_ret.expect("Rust should decode recovery fixture");
        let c_samples = obs
            .c_samples_per_channel()
            .expect("C should decode recovery fixture");
        assert_eq!(rust_samples as usize, c_samples);
        assert!(
            !obs.is_celt_only(),
            "recovery fixture must stay on SILK/Hybrid oracle path"
        );
        let snr = obs.snr_db().expect("SNR for Ok/Ok fixture");
        assert!(
            snr < oracle::SILK_DECODE_MIN_SNR_DB,
            "fixture should document the recovery PCM divergence, got {snr:.2} dB"
        );
    }
}
