//! Constants shared across the CLI.
//!
//! `OGG_STREAM_SERIAL` deliberately lives in `container::ogg`, not here, so it
//! is only reachable from the encode-side writer (which fabricates a serial)
//! and never from the read paths (which must use the input file's actual
//! serial captured from its first OggS page).

/// Opus output sample rate (the codec works natively at 48 kHz).
pub const OPUS_SR: u32 = 48_000;
/// Maximum bytes for a single Opus frame, per RFC 6716 §3.2.1.
pub const MAX_OPUS_FRAME_BYTES: usize = 1275;
/// Maximum number of Opus sub-frames libopus packs into a single code-3 packet
/// when `--framesize` ≥ 40 ms. The encoder splits a long requested duration
/// into 20 ms (or shorter) sub-frames and repacks them; 120 ms / 20 ms = 6 is
/// the worst case. At `--framesize 2.5`..20 the encoder emits one sub-frame
/// per packet, but we size for the worst case so the output buffer is always
/// large enough to hold the repacketised result. Cost: ~7.5 KiB of heap for
/// one encoder lifetime. The `debug_assert_eq!` at the top of
/// `commands::encode` enforces the
/// `MAX_PACKET_BYTES = MAX_OPUS_FRAME_BYTES * MAX_SUBFRAMES_PER_PACKET`
/// relationship in debug builds.
pub const MAX_SUBFRAMES_PER_PACKET: usize = 6;
/// Maximum Opus packet bytes for our config:
/// `MAX_OPUS_FRAME_BYTES * MAX_SUBFRAMES_PER_PACKET` = 7650 bytes.
pub const MAX_PACKET_BYTES: usize = MAX_OPUS_FRAME_BYTES * MAX_SUBFRAMES_PER_PACKET;
