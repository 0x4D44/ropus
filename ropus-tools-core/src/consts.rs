//! Constants shared across the CLI.
//!
//! `OGG_STREAM_SERIAL` deliberately lives in `container::ogg`, not here, so it
//! is only reachable from the encode-side writer (which fabricates a serial)
//! and never from the read paths (which must use the input file's actual
//! serial captured from its first OggS page).

/// Opus output sample rate (the codec works natively at 48 kHz).
pub const OPUS_SR: u32 = 48_000;
/// 20 ms frame at 48 kHz = 960 samples per channel.
pub const FRAME_SAMPLES_PER_CH: usize = 960;
/// Maximum bytes for a single Opus frame, per RFC 6716 §3.2.1.
pub const MAX_OPUS_FRAME_BYTES: usize = 1275;
/// Number of Opus frames per packet for our current encoder config (20 ms = 1 frame).
/// If `--frame-duration` becomes configurable (40/60/80/100/120 ms = 2/3/4/5/6
/// frames per packet), update this constant in lockstep with the encoder so
/// `MAX_PACKET_BYTES` remains correct. The `debug_assert_eq!` at the top of
/// `commands::encode` enforces the relationship.
pub const FRAMES_PER_PACKET: usize = 1;
/// Maximum Opus packet bytes for our config: `MAX_OPUS_FRAME_BYTES * FRAMES_PER_PACKET`.
pub const MAX_PACKET_BYTES: usize = MAX_OPUS_FRAME_BYTES * FRAMES_PER_PACKET;
