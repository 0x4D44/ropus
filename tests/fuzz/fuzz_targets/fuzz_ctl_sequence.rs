#![no_main]

use libfuzzer_sys::arbitrary::{self, Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use ropus::opus::decoder::{
    opus_packet_get_nb_frames, opus_packet_get_nb_samples, OPUS_BANDWIDTH_FULLBAND,
    OPUS_BANDWIDTH_MEDIUMBAND, OPUS_BANDWIDTH_NARROWBAND, OPUS_BANDWIDTH_SUPERWIDEBAND,
    OPUS_BANDWIDTH_WIDEBAND,
};
use ropus::opus::encoder::{
    OpusEncoder, OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY,
    OPUS_APPLICATION_VOIP, OPUS_AUTO, OPUS_BITRATE_MAX, OPUS_FRAMESIZE_10_MS, OPUS_FRAMESIZE_20_MS,
    OPUS_FRAMESIZE_2_5_MS, OPUS_FRAMESIZE_5_MS, OPUS_FRAMESIZE_ARG, OPUS_SIGNAL_MUSIC,
    OPUS_SIGNAL_VOICE,
};
use std::cell::RefCell;
use std::ptr::NonNull;
use std::sync::Once;

#[path = "c_reference.rs"]
mod c_reference;

thread_local! {
    static CURRENT_INPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn init_panic_capture() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            CURRENT_INPUT.with(|cell| {
                let bytes = cell.borrow();
                if !bytes.is_empty() {
                    use std::collections::hash_map::DefaultHasher;
                    use std::hash::{Hash, Hasher};
                    let mut hasher = DefaultHasher::new();
                    bytes.hash(&mut hasher);
                    let hash = hasher.finish();
                    let dir = std::env::var("FUZZ_PANIC_CAPTURE_DIR")
                        .unwrap_or_else(|_| "fuzz_crashes".to_string());
                    let _ = std::fs::create_dir_all(&dir);
                    let path = std::path::Path::new(&dir).join(format!("crash_{hash:016x}.bin"));
                    if let Err(e) = std::fs::write(&path, bytes.as_slice()) {
                        eprintln!("[PANIC CAPTURE] Failed to write {}: {}", path.display(), e);
                    }
                }
            });
            prev(info);
        }));
    });
}

const SAMPLE_RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];
const APPLICATIONS: [i32; 3] = [
    OPUS_APPLICATION_VOIP,
    OPUS_APPLICATION_AUDIO,
    OPUS_APPLICATION_RESTRICTED_LOWDELAY,
];
const OP_KIND_COUNT: u8 = 17;
const MAX_DATA_BYTES: i32 = 4000;

#[derive(Debug)]
struct CtlSequenceInput {
    sample_rate_idx: u8,
    channels_minus_one: u8,
    application_idx: u8,
    frames: Vec<FrameStep>,
}

#[derive(Debug)]
struct FrameStep {
    ops: Vec<CtlOp>,
    pcm_bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
struct CtlOp {
    kind: u8,
    value_idx: u8,
}

impl<'a> Arbitrary<'a> for CtlSequenceInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let sample_rate_idx = u.int_in_range(0..=4)?;
        let channels_minus_one = u.int_in_range(0..=1)?;
        let application_idx = u.int_in_range(0..=2)?;
        let n_frames = u.int_in_range(1..=8)?;

        let sample_rate = SAMPLE_RATES[sample_rate_idx as usize];
        let channels = channels_minus_one as i32 + 1;
        let pcm_bytes_needed = (sample_rate / 50) as usize * channels as usize * 2;

        let mut frames = Vec::with_capacity(n_frames);
        for _ in 0..n_frames {
            let n_ops = u.int_in_range(1..=6)?;
            let mut ops = Vec::with_capacity(n_ops);
            for _ in 0..n_ops {
                ops.push(CtlOp {
                    kind: u.arbitrary::<u8>()? % OP_KIND_COUNT,
                    value_idx: u.arbitrary()?,
                });
            }
            frames.push(FrameStep {
                ops,
                pcm_bytes: u.bytes(pcm_bytes_needed)?.to_vec(),
            });
        }

        Ok(Self {
            sample_rate_idx,
            channels_minus_one,
            application_idx,
            frames,
        })
    }
}

impl CtlSequenceInput {
    fn fingerprint(&self) -> [u8; 16] {
        let mut fp = [0u8; 16];
        fp[0] = self.sample_rate_idx;
        fp[1] = self.channels_minus_one;
        fp[2] = self.application_idx;
        fp[3] = self.frames.len() as u8;
        let op_count: usize = self.frames.iter().map(|f| f.ops.len()).sum();
        fp[4..8].copy_from_slice(&(op_count as u32).to_le_bytes());
        if let Some(first) = self.frames.first().and_then(|f| f.ops.first()) {
            fp[8] = first.kind;
            fp[9] = first.value_idx;
        }
        fp
    }
}

#[derive(Clone, Copy, Debug)]
enum CtlKind {
    Bitrate,
    Complexity,
    Vbr,
    VbrConstraint,
    InbandFec,
    PacketLossPerc,
    Dtx,
    ForceChannels,
    Bandwidth,
    MaxBandwidth,
    Signal,
    LsbDepth,
    ExpertFrameDuration,
    PredictionDisabled,
    PhaseInversionDisabled,
    Application,
    ResetState,
}

impl CtlOp {
    fn kind(self) -> CtlKind {
        match self.kind % OP_KIND_COUNT {
            0 => CtlKind::Bitrate,
            1 => CtlKind::Complexity,
            2 => CtlKind::Vbr,
            3 => CtlKind::VbrConstraint,
            4 => CtlKind::InbandFec,
            5 => CtlKind::PacketLossPerc,
            6 => CtlKind::Dtx,
            7 => CtlKind::ForceChannels,
            8 => CtlKind::Bandwidth,
            9 => CtlKind::MaxBandwidth,
            10 => CtlKind::Signal,
            11 => CtlKind::LsbDepth,
            12 => CtlKind::ExpertFrameDuration,
            13 => CtlKind::PredictionDisabled,
            14 => CtlKind::PhaseInversionDisabled,
            15 => CtlKind::Application,
            16 => CtlKind::ResetState,
            _ => unreachable!(),
        }
    }

    fn request(self) -> i32 {
        match self.kind() {
            CtlKind::Bitrate => c_reference::OPUS_SET_BITRATE_REQUEST,
            CtlKind::Complexity => c_reference::OPUS_SET_COMPLEXITY_REQUEST,
            CtlKind::Vbr => c_reference::OPUS_SET_VBR_REQUEST,
            CtlKind::VbrConstraint => c_reference::OPUS_SET_VBR_CONSTRAINT_REQUEST,
            CtlKind::InbandFec => c_reference::OPUS_SET_INBAND_FEC_REQUEST,
            CtlKind::PacketLossPerc => c_reference::OPUS_SET_PACKET_LOSS_PERC_REQUEST,
            CtlKind::Dtx => c_reference::OPUS_SET_DTX_REQUEST,
            CtlKind::ForceChannels => c_reference::OPUS_SET_FORCE_CHANNELS_REQUEST,
            CtlKind::Bandwidth => c_reference::OPUS_SET_BANDWIDTH_REQUEST,
            CtlKind::MaxBandwidth => c_reference::OPUS_SET_MAX_BANDWIDTH_REQUEST,
            CtlKind::Signal => c_reference::OPUS_SET_SIGNAL_REQUEST,
            CtlKind::LsbDepth => c_reference::OPUS_SET_LSB_DEPTH_REQUEST,
            CtlKind::ExpertFrameDuration => c_reference::OPUS_SET_EXPERT_FRAME_DURATION_REQUEST,
            CtlKind::PredictionDisabled => c_reference::OPUS_SET_PREDICTION_DISABLED_REQUEST,
            CtlKind::PhaseInversionDisabled => {
                c_reference::OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST
            }
            CtlKind::Application => c_reference::OPUS_SET_APPLICATION_REQUEST,
            CtlKind::ResetState => c_reference::OPUS_RESET_STATE,
        }
    }

    fn value(self) -> i32 {
        select_value(
            self.value_idx,
            match self.kind() {
                CtlKind::Bitrate => &[
                    OPUS_AUTO,
                    OPUS_BITRATE_MAX,
                    1,
                    500,
                    6000,
                    16000,
                    64000,
                    510000,
                    0,
                    -2,
                ],
                CtlKind::Complexity => &[-1, 0, 1, 5, 9, 11],
                CtlKind::Vbr | CtlKind::VbrConstraint | CtlKind::Dtx => &[-1, 0, 1, 2],
                CtlKind::InbandFec => &[-1, 0, 1, 2, 3],
                CtlKind::PacketLossPerc => &[-1, 0, 1, 50, 100, 101],
                CtlKind::ForceChannels => &[OPUS_AUTO, 1, 2, 0, 3],
                CtlKind::Bandwidth => &[
                    OPUS_AUTO,
                    OPUS_BANDWIDTH_NARROWBAND,
                    OPUS_BANDWIDTH_MEDIUMBAND,
                    OPUS_BANDWIDTH_WIDEBAND,
                    OPUS_BANDWIDTH_SUPERWIDEBAND,
                    OPUS_BANDWIDTH_FULLBAND,
                    1100,
                    1106,
                ],
                CtlKind::MaxBandwidth => &[
                    OPUS_BANDWIDTH_NARROWBAND,
                    OPUS_BANDWIDTH_MEDIUMBAND,
                    OPUS_BANDWIDTH_WIDEBAND,
                    OPUS_BANDWIDTH_SUPERWIDEBAND,
                    OPUS_BANDWIDTH_FULLBAND,
                    OPUS_AUTO,
                    1100,
                    1106,
                ],
                CtlKind::Signal => &[
                    OPUS_AUTO,
                    OPUS_SIGNAL_VOICE,
                    OPUS_SIGNAL_MUSIC,
                    0,
                    3000,
                    3003,
                ],
                CtlKind::LsbDepth => &[7, 8, 12, 16, 24, 25],
                CtlKind::ExpertFrameDuration => &[
                    OPUS_FRAMESIZE_ARG,
                    OPUS_FRAMESIZE_2_5_MS,
                    OPUS_FRAMESIZE_5_MS,
                    OPUS_FRAMESIZE_10_MS,
                    OPUS_FRAMESIZE_20_MS,
                    4999,
                    5010,
                ],
                CtlKind::PredictionDisabled => &[-1, 0, 1, 2],
                CtlKind::PhaseInversionDisabled => &[-1, 0, 1, 2],
                CtlKind::Application => &[
                    OPUS_APPLICATION_VOIP,
                    OPUS_APPLICATION_AUDIO,
                    OPUS_APPLICATION_RESTRICTED_LOWDELAY,
                    0,
                    2050,
                ],
                CtlKind::ResetState => &[0],
            },
        )
    }
}

fn select_value(idx: u8, values: &[i32]) -> i32 {
    values[idx as usize % values.len()]
}

fn apply_rust_ctl(enc: &mut OpusEncoder, op: CtlOp) -> i32 {
    let value = op.value();
    match op.kind() {
        CtlKind::Bitrate => enc.set_bitrate(value),
        CtlKind::Complexity => enc.set_complexity(value),
        CtlKind::Vbr => enc.set_vbr(value),
        CtlKind::VbrConstraint => enc.set_vbr_constraint(value),
        CtlKind::InbandFec => enc.set_inband_fec(value),
        CtlKind::PacketLossPerc => enc.set_packet_loss_perc(value),
        CtlKind::Dtx => enc.set_dtx(value),
        CtlKind::ForceChannels => enc.set_force_channels(value),
        CtlKind::Bandwidth => enc.set_bandwidth(value),
        CtlKind::MaxBandwidth => enc.set_max_bandwidth(value),
        CtlKind::Signal => enc.set_signal(value),
        CtlKind::LsbDepth => enc.set_lsb_depth(value),
        CtlKind::ExpertFrameDuration => enc.set_expert_frame_duration(value),
        CtlKind::PredictionDisabled => enc.set_prediction_disabled(value),
        CtlKind::PhaseInversionDisabled => enc.set_phase_inversion_disabled(value),
        CtlKind::Application => enc.set_application(value),
        CtlKind::ResetState => {
            enc.reset();
            c_reference::OPUS_OK
        }
    }
}

struct CEncoder {
    ptr: NonNull<c_reference::OpusEncoder>,
}

impl CEncoder {
    fn new(sample_rate: i32, channels: i32, application: i32) -> Result<Self, i32> {
        unsafe {
            let mut error = 0;
            let ptr =
                c_reference::opus_encoder_create(sample_rate, channels, application, &mut error);
            if ptr.is_null() || error != c_reference::OPUS_OK {
                if !ptr.is_null() {
                    c_reference::opus_encoder_destroy(ptr);
                }
                Err(error)
            } else {
                Ok(Self {
                    ptr: NonNull::new_unchecked(ptr),
                })
            }
        }
    }

    fn ctl(&mut self, op: CtlOp) -> i32 {
        unsafe {
            if matches!(op.kind(), CtlKind::ResetState) {
                c_reference::opus_encoder_ctl(self.ptr.as_ptr(), op.request())
            } else {
                c_reference::opus_encoder_ctl(self.ptr.as_ptr(), op.request(), op.value())
            }
        }
    }

    fn encode(&mut self, pcm: &[i16], frame_size: i32) -> Result<Vec<u8>, i32> {
        let mut out = vec![0u8; MAX_DATA_BYTES as usize];
        let ret = unsafe {
            c_reference::opus_encode(
                self.ptr.as_ptr(),
                pcm.as_ptr(),
                frame_size,
                out.as_mut_ptr(),
                MAX_DATA_BYTES,
            )
        };
        if ret < 0 {
            Err(ret as i32)
        } else {
            out.truncate(ret as usize);
            Ok(out)
        }
    }

    fn final_range(&mut self) -> Result<u32, i32> {
        let mut value = 0u32;
        let ret = unsafe {
            c_reference::opus_encoder_ctl(
                self.ptr.as_ptr(),
                c_reference::OPUS_GET_FINAL_RANGE_REQUEST,
                &mut value as *mut u32,
            )
        };
        if ret == c_reference::OPUS_OK {
            Ok(value)
        } else {
            Err(ret)
        }
    }

    fn get_i32(&mut self, request: i32) -> Result<i32, i32> {
        let mut value = 0i32;
        let ret = unsafe {
            c_reference::opus_encoder_ctl(self.ptr.as_ptr(), request, &mut value as *mut i32)
        };
        if ret == c_reference::OPUS_OK {
            Ok(value)
        } else {
            Err(ret)
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

fn pcm_from_bytes(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn assert_getter_parity(
    rust_enc: &OpusEncoder,
    c_enc: &mut CEncoder,
    frame_idx: usize,
    label: &str,
) {
    let checks = [
        (
            "bitrate",
            rust_enc.get_bitrate(),
            c_enc.get_i32(c_reference::OPUS_GET_BITRATE_REQUEST),
        ),
        (
            "complexity",
            rust_enc.get_complexity(),
            c_enc.get_i32(c_reference::OPUS_GET_COMPLEXITY_REQUEST),
        ),
        (
            "vbr",
            rust_enc.get_vbr(),
            c_enc.get_i32(c_reference::OPUS_GET_VBR_REQUEST),
        ),
        (
            "vbr_constraint",
            rust_enc.get_vbr_constraint(),
            c_enc.get_i32(c_reference::OPUS_GET_VBR_CONSTRAINT_REQUEST),
        ),
        (
            "inband_fec",
            rust_enc.get_inband_fec(),
            c_enc.get_i32(c_reference::OPUS_GET_INBAND_FEC_REQUEST),
        ),
        (
            "packet_loss_perc",
            rust_enc.get_packet_loss_perc(),
            c_enc.get_i32(c_reference::OPUS_GET_PACKET_LOSS_PERC_REQUEST),
        ),
        (
            "dtx",
            rust_enc.get_dtx(),
            c_enc.get_i32(c_reference::OPUS_GET_DTX_REQUEST),
        ),
        (
            "signal",
            rust_enc.get_signal(),
            c_enc.get_i32(c_reference::OPUS_GET_SIGNAL_REQUEST),
        ),
        (
            "application",
            rust_enc.get_application(),
            c_enc.get_i32(c_reference::OPUS_GET_APPLICATION_REQUEST),
        ),
        (
            "force_channels",
            rust_enc.get_force_channels(),
            c_enc.get_i32(c_reference::OPUS_GET_FORCE_CHANNELS_REQUEST),
        ),
        (
            "bandwidth",
            rust_enc.get_bandwidth(),
            c_enc.get_i32(c_reference::OPUS_GET_BANDWIDTH_REQUEST),
        ),
        (
            "max_bandwidth",
            rust_enc.get_max_bandwidth(),
            c_enc.get_i32(c_reference::OPUS_GET_MAX_BANDWIDTH_REQUEST),
        ),
        (
            "lsb_depth",
            rust_enc.get_lsb_depth(),
            c_enc.get_i32(c_reference::OPUS_GET_LSB_DEPTH_REQUEST),
        ),
        (
            "expert_frame_duration",
            rust_enc.get_expert_frame_duration(),
            c_enc.get_i32(c_reference::OPUS_GET_EXPERT_FRAME_DURATION_REQUEST),
        ),
        (
            "prediction_disabled",
            rust_enc.get_prediction_disabled(),
            c_enc.get_i32(c_reference::OPUS_GET_PREDICTION_DISABLED_REQUEST),
        ),
        (
            "phase_inversion_disabled",
            rust_enc.get_phase_inversion_disabled(),
            c_enc.get_i32(c_reference::OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST),
        ),
        (
            "lookahead",
            rust_enc.get_lookahead(),
            c_enc.get_i32(c_reference::OPUS_GET_LOOKAHEAD_REQUEST),
        ),
        (
            "in_dtx",
            rust_enc.get_in_dtx(),
            c_enc.get_i32(c_reference::OPUS_GET_IN_DTX_REQUEST),
        ),
    ];

    for (name, rust_value, c_value) in checks {
        let c_value = c_value.unwrap_or_else(|err| {
            panic!("C getter {name} failed with {err} at frame={frame_idx} after {label}")
        });
        assert_eq!(
            rust_value, c_value,
            "getter {name} mismatch at frame={frame_idx} after {label}"
        );
    }
}

// Stateful CTL-sequence differential target.
//
// This differs from fuzz_encode_multiframe by making the operation stream the
// primary fuzzed object: each frame applies 1-6 arbitrary encoder CTL setters,
// including return-code checks for invalid edge values, before encoding through
// the same Rust and C encoder states. The multiframe target mainly varies a
// fixed prologue subset; this target stresses between-frame setter ordering,
// reset_state, getter parity, application-before-first-frame, bandwidth,
// channel forcing, signal, LSB depth, expert frame duration, prediction,
// phase inversion, and the rate-control knobs.
fuzz_target!(|input: CtlSequenceInput| {
    init_panic_capture();
    CURRENT_INPUT.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.clear();
        buf.extend_from_slice(&input.fingerprint());
    });

    let sample_rate = SAMPLE_RATES[input.sample_rate_idx as usize];
    let channels = input.channels_minus_one as i32 + 1;
    let application = APPLICATIONS[input.application_idx as usize];
    let frame_size = sample_rate / 50;

    let mut rust_enc = match OpusEncoder::new(sample_rate, channels, application) {
        Ok(enc) => enc,
        Err(_) => return,
    };
    let mut c_enc = match CEncoder::new(sample_rate, channels, application) {
        Ok(enc) => enc,
        Err(_) => return,
    };

    for (frame_idx, frame) in input.frames.iter().enumerate() {
        for (op_idx, &op) in frame.ops.iter().enumerate() {
            let rust_ret = apply_rust_ctl(&mut rust_enc, op);
            let c_ret = c_enc.ctl(op);
            assert_eq!(
                rust_ret,
                c_ret,
                "CTL return mismatch at frame={frame_idx}, op={op_idx}, kind={:?}, value={}",
                op.kind(),
                op.value()
            );
            assert_getter_parity(&rust_enc, &mut c_enc, frame_idx, "ctl");
        }

        let pcm = pcm_from_bytes(&frame.pcm_bytes);
        let mut rust_out = vec![0u8; MAX_DATA_BYTES as usize];
        let rust_ret = rust_enc.encode(&pcm, frame_size, &mut rust_out, MAX_DATA_BYTES);
        let c_ret = c_enc.encode(&pcm, frame_size);

        match (rust_ret, c_ret) {
            (Ok(rust_len), Ok(c_packet)) => {
                let rust_len = rust_len as usize;
                rust_out.truncate(rust_len);

                assert_eq!(
                    rust_len,
                    c_packet.len(),
                    "encoded length mismatch at frame={frame_idx}, sr={sample_rate}, ch={channels}, app={application}"
                );
                assert_eq!(
                    rust_out, c_packet,
                    "encoded bytes mismatch at frame={frame_idx}, sr={sample_rate}, ch={channels}, app={application}, len={rust_len}"
                );

                let rust_range = rust_enc.get_final_range();
                let c_range = c_enc
                    .final_range()
                    .expect("C final-range getter should succeed after encode");
                assert_eq!(
                    rust_range, c_range,
                    "final range mismatch at frame={frame_idx}, sr={sample_rate}, ch={channels}, app={application}"
                );
                assert_getter_parity(&rust_enc, &mut c_enc, frame_idx, "encode");

                if !rust_out.is_empty() {
                    let nb_frames = opus_packet_get_nb_frames(&rust_out);
                    assert!(
                        nb_frames.is_ok() && nb_frames.unwrap() > 0,
                        "encoded packet not parseable at frame={frame_idx}, len={rust_len}"
                    );
                    if let Ok(samples) = opus_packet_get_nb_samples(&rust_out, sample_rate) {
                        assert!(
                            samples > 0 && samples <= frame_size,
                            "packet sample count out of bounds at frame={frame_idx}: samples={samples}, frame_size={frame_size}"
                        );
                    }
                }
            }
            (Err(rust_err), Err(c_err)) => {
                assert_eq!(
                    rust_err, c_err,
                    "encode error mismatch at frame={frame_idx}, sr={sample_rate}, ch={channels}, app={application}"
                );
            }
            (Ok(rust_len), Err(c_err)) => panic!(
                "Rust encoded {rust_len} bytes but C errored {c_err} at frame={frame_idx}, sr={sample_rate}, ch={channels}, app={application}"
            ),
            (Err(rust_err), Ok(c_packet)) => panic!(
                "Rust errored {rust_err} but C encoded {} bytes at frame={frame_idx}, sr={sample_rate}, ch={channels}, app={application}",
                c_packet.len()
            ),
        }
    }
});
