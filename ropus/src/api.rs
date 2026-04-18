//! High-level idiomatic Rust API for Opus encoding and decoding.
//!
//! This module is a thin facade over [`crate::opus::encoder::OpusEncoder`] and
//! [`crate::opus::decoder::OpusDecoder`]. It contains no codec logic of its own;
//! every method delegates to the existing implementations and reuses their
//! constants. Use the low-level modules (`ropus::opus::*`, `ropus::celt::*`,
//! etc.) when you need finer control than this surface exposes.
//!
//! # Example
//!
//! ```no_run
//! use ropus::{Application, Bitrate, Channels, Decoder, Encoder};
//!
//! let mut encoder = Encoder::builder(48_000, Channels::Stereo, Application::Audio)
//!     .bitrate(Bitrate::Bits(64_000))
//!     .build()
//!     .unwrap();
//!
//! let pcm_in = vec![0i16; 960 * 2]; // 20 ms stereo @ 48 kHz
//! let mut packet = vec![0u8; 4000];
//! let n = encoder.encode(&pcm_in, &mut packet).unwrap();
//!
//! let mut decoder = Decoder::new(48_000, Channels::Stereo).unwrap();
//! let mut pcm_out = vec![0i16; 960 * 2];
//! let _samples = decoder.decode(&packet[..n], &mut pcm_out, false).unwrap();
//! ```

use core::fmt;

use crate::opus::decoder::{
    OPUS_BAD_ARG, OPUS_BANDWIDTH_FULLBAND, OPUS_BANDWIDTH_MEDIUMBAND, OPUS_BANDWIDTH_NARROWBAND,
    OPUS_BANDWIDTH_SUPERWIDEBAND, OPUS_BANDWIDTH_WIDEBAND, OPUS_BUFFER_TOO_SMALL,
    OPUS_INTERNAL_ERROR, OPUS_INVALID_PACKET, OpusDecoder,
};

/// Map a libopus channel-count `i32` (always 1 or 2 here) back to [`Channels`].
///
/// Inverse of [`Channels::as_c_int`]. Treats any unexpected value as `Mono` —
/// the inner Opus types only ever store 1 or 2, so the fallback exists purely
/// to avoid panicking on a corrupted state.
fn channels_from_c_int(n: i32) -> Channels {
    match n {
        2 => Channels::Stereo,
        _ => Channels::Mono,
    }
}
use crate::opus::encoder::{
    OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_APPLICATION_VOIP, OPUS_AUTO,
    OPUS_BITRATE_MAX, OPUS_FRAMESIZE_2_5_MS, OPUS_FRAMESIZE_5_MS, OPUS_FRAMESIZE_10_MS,
    OPUS_FRAMESIZE_20_MS, OPUS_FRAMESIZE_40_MS, OPUS_FRAMESIZE_60_MS, OPUS_FRAMESIZE_80_MS,
    OPUS_FRAMESIZE_100_MS, OPUS_FRAMESIZE_120_MS, OPUS_FRAMESIZE_ARG, OPUS_SIGNAL_MUSIC,
    OPUS_SIGNAL_VOICE, OpusEncoder,
};

// ===========================================================================
// Typed enums — translate to/from the existing libopus integer constants
// ===========================================================================

/// Channel layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Channels {
    Mono,
    Stereo,
}

impl Channels {
    /// Channel count: 1 for mono, 2 for stereo.
    pub fn count(self) -> usize {
        match self {
            Channels::Mono => 1,
            Channels::Stereo => 2,
        }
    }

    fn as_c_int(self) -> i32 {
        self.count() as i32
    }
}

/// Encoder application hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Application {
    /// Best for VoIP / videoconference; biased toward intelligibility.
    Voip,
    /// Best for general music and high-fidelity content.
    Audio,
    /// Lowest algorithmic delay; restricts the available codec modes.
    RestrictedLowDelay,
}

impl Application {
    fn as_c_int(self) -> i32 {
        match self {
            Application::Voip => OPUS_APPLICATION_VOIP,
            Application::Audio => OPUS_APPLICATION_AUDIO,
            Application::RestrictedLowDelay => OPUS_APPLICATION_RESTRICTED_LOWDELAY,
        }
    }
}

/// Target bitrate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Bitrate {
    /// Let the encoder pick a sensible default for the sample rate / channels.
    Auto,
    /// Use the maximum bitrate the channel/format allow.
    Max,
    /// Explicit bitrate in bits per second.
    ///
    /// Advanced/unchecked: values above `i32::MAX` are silently clamped to
    /// `i32::MAX` when forwarded to libopus. Use [`Bitrate::try_bits`] to get
    /// a validated `Result` instead of the silent clamp.
    Bits(u32),
}

impl Bitrate {
    /// Construct [`Bitrate::Bits`] with overflow validation.
    ///
    /// Returns [`BitrateRangeError`] if `b` exceeds `i32::MAX` (the limit
    /// imposed by libopus's signed `int` API). Prefer this constructor when
    /// the bitrate value originates from untrusted input or a wider integer
    /// type than `u32` you've already narrowed.
    pub fn try_bits(b: u32) -> Result<Self, BitrateRangeError> {
        if b > i32::MAX as u32 {
            Err(BitrateRangeError(b))
        } else {
            Ok(Bitrate::Bits(b))
        }
    }

    fn as_c_int(self) -> i32 {
        match self {
            Bitrate::Auto => OPUS_AUTO,
            Bitrate::Max => OPUS_BITRATE_MAX,
            Bitrate::Bits(b) => i32::try_from(b).unwrap_or(i32::MAX),
        }
    }
}

/// Returned by [`Bitrate::try_bits`] when the requested bitrate exceeds the
/// `i32::MAX` bps limit imposed by libopus's signed integer API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitrateRangeError(pub u32);

impl fmt::Display for BitrateRangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "bitrate {} bps exceeds the libopus i32::MAX limit",
            self.0
        )
    }
}

impl std::error::Error for BitrateRangeError {}

/// Signal-content hint passed to the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Signal {
    Auto,
    Voice,
    Music,
}

impl Signal {
    fn as_c_int(self) -> i32 {
        match self {
            Signal::Auto => OPUS_AUTO,
            Signal::Voice => OPUS_SIGNAL_VOICE,
            Signal::Music => OPUS_SIGNAL_MUSIC,
        }
    }
}

/// Audio bandwidth selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Bandwidth {
    Narrowband,
    Mediumband,
    Wideband,
    Superwideband,
    Fullband,
    Auto,
}

impl Bandwidth {
    fn as_c_int(self) -> i32 {
        match self {
            Bandwidth::Narrowband => OPUS_BANDWIDTH_NARROWBAND,
            Bandwidth::Mediumband => OPUS_BANDWIDTH_MEDIUMBAND,
            Bandwidth::Wideband => OPUS_BANDWIDTH_WIDEBAND,
            Bandwidth::Superwideband => OPUS_BANDWIDTH_SUPERWIDEBAND,
            Bandwidth::Fullband => OPUS_BANDWIDTH_FULLBAND,
            Bandwidth::Auto => OPUS_AUTO,
        }
    }
}

/// Forced channel count for the encoded stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ForceChannels {
    Auto,
    Mono,
    Stereo,
}

impl ForceChannels {
    fn as_c_int(self) -> i32 {
        match self {
            ForceChannels::Auto => OPUS_AUTO,
            ForceChannels::Mono => 1,
            ForceChannels::Stereo => 2,
        }
    }
}

/// Expert frame-duration override.
///
/// `Argument` (the default) keeps the duration determined by the buffer size
/// passed to `encode`; the other variants pin the encoder to that frame length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameDuration {
    Ms2_5,
    Ms5,
    Ms10,
    Ms20,
    Ms40,
    Ms60,
    Ms80,
    Ms100,
    Ms120,
    Argument,
}

impl FrameDuration {
    fn as_c_int(self) -> i32 {
        match self {
            FrameDuration::Ms2_5 => OPUS_FRAMESIZE_2_5_MS,
            FrameDuration::Ms5 => OPUS_FRAMESIZE_5_MS,
            FrameDuration::Ms10 => OPUS_FRAMESIZE_10_MS,
            FrameDuration::Ms20 => OPUS_FRAMESIZE_20_MS,
            FrameDuration::Ms40 => OPUS_FRAMESIZE_40_MS,
            FrameDuration::Ms60 => OPUS_FRAMESIZE_60_MS,
            FrameDuration::Ms80 => OPUS_FRAMESIZE_80_MS,
            FrameDuration::Ms100 => OPUS_FRAMESIZE_100_MS,
            FrameDuration::Ms120 => OPUS_FRAMESIZE_120_MS,
            FrameDuration::Argument => OPUS_FRAMESIZE_ARG,
        }
    }
}

// ===========================================================================
// Error types — generated by `impl_opus_error!` for uniform Display/Error impls
// ===========================================================================

/// Generate a libopus error enum together with `from_code`, `Display`, and the
/// `Error` marker. Each variant maps a known `OPUS_*` integer code to a typed
/// arm; the trailing `Unknown(i32)` arm is the catch-all for unmapped codes.
///
/// `$code` is `pat` rather than `expr` because it appears in `match` arm
/// position; the `OPUS_*` codes are `const i32`, which is valid pattern syntax.
macro_rules! impl_opus_error {
    (
        $(#[$meta:meta])*
        $name:ident {
            $( $variant:ident => $code:pat => $msg:literal ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $name {
            $( $variant, )*
            /// An unrecognized error code from the underlying Opus codec.
            Unknown(i32),
        }

        impl $name {
            fn from_code(code: i32) -> Self {
                // Real Opus error codes are always negative; OPUS_OK == 0 should
                // never reach an *Error::from_code path. Guard against the -1
                // collision between OPUS_BAD_ARG and non-error sentinels such as
                // OPUS_BITRATE_MAX, and against any future positive CTL constant
                // that might otherwise be misclassified as an error variant.
                if code >= 0 {
                    return Self::Unknown(code);
                }
                match code {
                    $( $code => $name::$variant, )*
                    other => $name::Unknown(other),
                }
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    $( $name::$variant => f.write_str($msg), )*
                    $name::Unknown(c) => write!(f, "unknown Opus error code: {c}"),
                }
            }
        }

        impl std::error::Error for $name {}
    };
}

impl_opus_error!(
    /// Errors returned by [`EncoderBuilder::build`].
    ///
    /// Distinct from [`DecoderInitError`] for type-driven matching, even though
    /// variants currently coincide.
    EncoderBuildError {
        BadArg   => OPUS_BAD_ARG        => "invalid argument to Opus encoder builder",
        Internal => OPUS_INTERNAL_ERROR => "Opus encoder internal initialization error",
    }
);

impl_opus_error!(
    /// Errors returned by [`Encoder::encode`] and [`Encoder::encode_float`].
    EncodeError {
        BadArg         => OPUS_BAD_ARG          => "invalid argument to Opus encode",
        BufferTooSmall => OPUS_BUFFER_TOO_SMALL => "output buffer too small for encoded frame",
        Internal       => OPUS_INTERNAL_ERROR   => "Opus encoder internal error",
    }
);

impl_opus_error!(
    /// Errors returned by [`Decoder::new`].
    ///
    /// Distinct from [`EncoderBuildError`] for type-driven matching, even though
    /// variants currently coincide.
    DecoderInitError {
        BadArg   => OPUS_BAD_ARG        => "invalid argument to Opus decoder constructor",
        Internal => OPUS_INTERNAL_ERROR => "Opus decoder internal initialization error",
    }
);

impl_opus_error!(
    /// Errors returned by [`Decoder::decode`] and [`Decoder::decode_float`].
    DecodeError {
        BadArg         => OPUS_BAD_ARG          => "invalid argument to Opus decode",
        BufferTooSmall => OPUS_BUFFER_TOO_SMALL => "output buffer too small for decoded frame",
        Internal       => OPUS_INTERNAL_ERROR   => "Opus decoder internal error",
        InvalidPacket  => OPUS_INVALID_PACKET   => "malformed Opus packet",
    }
);

// ===========================================================================
// Encoder
// ===========================================================================

/// High-level Opus encoder.
pub struct Encoder {
    inner: OpusEncoder,
}

impl Encoder {
    /// Begin building an encoder. Defaults match those of [`OpusEncoder::new`].
    pub fn builder(
        sample_rate: u32,
        channels: Channels,
        application: Application,
    ) -> EncoderBuilder {
        EncoderBuilder {
            sample_rate,
            channels,
            application,
            bitrate: None,
            complexity: None,
            signal: None,
            vbr: None,
            force_channels: None,
            max_bandwidth: None,
            frame_duration: None,
        }
    }

    /// Encode a 16-bit PCM frame.
    ///
    /// `pcm` length must be `frame_size * channels`, where `frame_size` is one
    /// of the supported Opus frame sizes (2.5/5/10/20/40/60 ms at the
    /// configured sample rate).
    pub fn encode(&mut self, pcm: &[i16], output: &mut [u8]) -> Result<usize, EncodeError> {
        let frame_size = self.frame_size_from_pcm_len(pcm.len())?;
        let max = clamp_max_data_bytes(output.len())?;
        self.inner
            .encode(pcm, frame_size, output, max)
            .map(|n| n as usize)
            .map_err(EncodeError::from_code)
    }

    /// Encode a float PCM frame (samples in `[-1.0, 1.0]`).
    ///
    /// `pcm` length must be `frame_size * channels`, where `frame_size` is one
    /// of the supported Opus frame sizes (2.5/5/10/20/40/60 ms at the
    /// configured sample rate).
    pub fn encode_float(&mut self, pcm: &[f32], output: &mut [u8]) -> Result<usize, EncodeError> {
        let frame_size = self.frame_size_from_pcm_len(pcm.len())?;
        let max = clamp_max_data_bytes(output.len())?;
        self.inner
            .encode_float(pcm, frame_size, output, max)
            .map(|n| n as usize)
            .map_err(EncodeError::from_code)
    }

    fn frame_size_from_pcm_len(&self, pcm_len: usize) -> Result<i32, EncodeError> {
        let ch = self.inner.get_channels() as usize;
        if ch == 0 || pcm_len % ch != 0 {
            return Err(EncodeError::BadArg);
        }
        let per_channel = pcm_len / ch;
        // Mirror the decoder's empty-buffer check (Decoder::frame_size_from_pcm_len)
        // so an empty PCM input is reported consistently rather than being shipped
        // down to encode_native to fail later with BadArg.
        if per_channel == 0 {
            return Err(EncodeError::BufferTooSmall);
        }
        i32::try_from(per_channel).map_err(|_| EncodeError::BadArg)
    }

    /// Configured sample rate in Hz (the rate passed to [`Encoder::builder`]).
    pub fn sample_rate(&self) -> u32 {
        // get_sample_rate() returns i32 but is always one of 8000/12000/16000/
        // 24000/48000, so the cast to u32 is lossless.
        self.inner.get_sample_rate() as u32
    }

    /// Configured channel layout.
    pub fn channels(&self) -> Channels {
        channels_from_c_int(self.inner.get_channels())
    }

    /// Encoder lookahead in 48 kHz samples.
    ///
    /// This is the value to write into OpusHead's `pre_skip` field
    /// per RFC 7845. For typical encoders this is 312; in
    /// `OPUS_APPLICATION_RESTRICTED_LOWDELAY` it is 120.
    pub fn lookahead(&self) -> u32 {
        self.inner.get_lookahead().max(0) as u32
    }
}

/// Builder for [`Encoder`].
pub struct EncoderBuilder {
    sample_rate: u32,
    channels: Channels,
    application: Application,
    bitrate: Option<Bitrate>,
    complexity: Option<u8>,
    signal: Option<Signal>,
    vbr: Option<bool>,
    force_channels: Option<ForceChannels>,
    max_bandwidth: Option<Bandwidth>,
    frame_duration: Option<FrameDuration>,
}

impl EncoderBuilder {
    /// Set target bitrate.
    pub fn bitrate(mut self, b: Bitrate) -> Self {
        self.bitrate = Some(b);
        self
    }

    /// Set CPU/quality complexity (0..=10).
    pub fn complexity(mut self, c: u8) -> Self {
        self.complexity = Some(c);
        self
    }

    /// Set the signal-type hint.
    pub fn signal(mut self, s: Signal) -> Self {
        self.signal = Some(s);
        self
    }

    /// Enable or disable variable-bitrate mode.
    pub fn vbr(mut self, on: bool) -> Self {
        self.vbr = Some(on);
        self
    }

    /// Force the encoded stream to a specific channel count.
    pub fn force_channels(mut self, c: ForceChannels) -> Self {
        self.force_channels = Some(c);
        self
    }

    /// Limit the maximum encoded bandwidth.
    pub fn max_bandwidth(mut self, b: Bandwidth) -> Self {
        self.max_bandwidth = Some(b);
        self
    }

    /// Pin the encoder to a specific frame duration.
    pub fn frame_duration(mut self, d: FrameDuration) -> Self {
        self.frame_duration = Some(d);
        self
    }

    /// Construct the encoder, applying the configured settings.
    pub fn build(self) -> Result<Encoder, EncoderBuildError> {
        let fs = i32::try_from(self.sample_rate).map_err(|_| EncoderBuildError::BadArg)?;
        let mut inner = OpusEncoder::new(fs, self.channels.as_c_int(), self.application.as_c_int())
            .map_err(EncoderBuildError::from_code)?;

        if let Some(b) = self.bitrate {
            check_ok(inner.set_bitrate(b.as_c_int()))?;
        }
        if let Some(c) = self.complexity {
            check_ok(inner.set_complexity(c as i32))?;
        }
        if let Some(s) = self.signal {
            check_ok(inner.set_signal(s.as_c_int()))?;
        }
        if let Some(vbr) = self.vbr {
            check_ok(inner.set_vbr(i32::from(vbr)))?;
        }
        if let Some(fc) = self.force_channels {
            check_ok(inner.set_force_channels(fc.as_c_int()))?;
        }
        if let Some(bw) = self.max_bandwidth {
            check_ok(inner.set_max_bandwidth(bw.as_c_int()))?;
        }
        if let Some(d) = self.frame_duration {
            check_ok(inner.set_expert_frame_duration(d.as_c_int()))?;
        }

        Ok(Encoder { inner })
    }
}

fn check_ok(code: i32) -> Result<(), EncoderBuildError> {
    if code == 0 {
        Ok(())
    } else {
        Err(EncoderBuildError::from_code(code))
    }
}

fn clamp_max_data_bytes(len: usize) -> Result<i32, EncodeError> {
    if len == 0 {
        return Err(EncodeError::BufferTooSmall);
    }
    Ok(i32::try_from(len).unwrap_or(i32::MAX))
}

// ===========================================================================
// Decoder
// ===========================================================================

/// High-level Opus decoder.
pub struct Decoder {
    inner: OpusDecoder,
}

impl Decoder {
    /// Create a new decoder.
    pub fn new(sample_rate: u32, channels: Channels) -> Result<Self, DecoderInitError> {
        let fs = i32::try_from(sample_rate).map_err(|_| DecoderInitError::BadArg)?;
        let inner =
            OpusDecoder::new(fs, channels.as_c_int()).map_err(DecoderInitError::from_code)?;
        Ok(Self { inner })
    }

    /// Decode a packet to 16-bit PCM. An empty `packet` triggers PLC
    /// (packet-loss concealment) — pass the next valid packet when available.
    ///
    /// `output` must be sized to hold the maximum frame this packet might
    /// decode to (typically `120 ms * sample_rate / 1000 * channels` samples).
    /// The actual frame size is recovered from the packet.
    pub fn decode(
        &mut self,
        packet: &[u8],
        output: &mut [i16],
        decode_fec: bool,
    ) -> Result<usize, DecodeError> {
        let frame_size = self.frame_size_from_pcm_len(output.len())?;
        let data = packet_arg(packet);
        self.inner
            .decode(data, output, frame_size, decode_fec)
            .map(|n| n as usize)
            .map_err(DecodeError::from_code)
    }

    /// Decode a packet to floating-point PCM.
    ///
    /// `output` must be sized to hold the maximum frame this packet might
    /// decode to (typically `120 ms * sample_rate / 1000 * channels` samples).
    /// The actual frame size is recovered from the packet.
    pub fn decode_float(
        &mut self,
        packet: &[u8],
        output: &mut [f32],
        decode_fec: bool,
    ) -> Result<usize, DecodeError> {
        let frame_size = self.frame_size_from_pcm_len(output.len())?;
        let data = packet_arg(packet);
        self.inner
            .decode_float(data, output, frame_size, decode_fec)
            .map(|n| n as usize)
            .map_err(DecodeError::from_code)
    }

    fn frame_size_from_pcm_len(&self, pcm_len: usize) -> Result<i32, DecodeError> {
        let ch = self.inner.get_channels() as usize;
        if ch == 0 || pcm_len % ch != 0 {
            return Err(DecodeError::BadArg);
        }
        let per_channel = pcm_len / ch;
        if per_channel == 0 {
            return Err(DecodeError::BufferTooSmall);
        }
        i32::try_from(per_channel).map_err(|_| DecodeError::BadArg)
    }

    /// Configured sample rate in Hz (the rate passed to [`Decoder::new`]).
    pub fn sample_rate(&self) -> u32 {
        // get_sample_rate() returns i32 but is always one of 8000/12000/16000/
        // 24000/48000, so the cast to u32 is lossless.
        self.inner.get_sample_rate() as u32
    }

    /// Configured channel layout.
    pub fn channels(&self) -> Channels {
        channels_from_c_int(self.inner.get_channels())
    }
}

fn packet_arg(packet: &[u8]) -> Option<&[u8]> {
    if packet.is_empty() { None } else { Some(packet) }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- as_c_int round-trips against the existing OPUS_* constants ----

    #[test]
    fn typed_enums_as_c_int_match_constants() {
        // Channels: count() defines the contract; as_c_int just casts it.
        assert_eq!(Channels::Mono.as_c_int(), 1);
        assert_eq!(Channels::Stereo.as_c_int(), 2);

        // Application
        assert_eq!(Application::Voip.as_c_int(), OPUS_APPLICATION_VOIP);
        assert_eq!(Application::Audio.as_c_int(), OPUS_APPLICATION_AUDIO);
        assert_eq!(Application::RestrictedLowDelay.as_c_int(), OPUS_APPLICATION_RESTRICTED_LOWDELAY);

        // Bitrate
        assert_eq!(Bitrate::Auto.as_c_int(), OPUS_AUTO);
        assert_eq!(Bitrate::Max.as_c_int(), OPUS_BITRATE_MAX);
        assert_eq!(Bitrate::Bits(64_000).as_c_int(), 64_000);

        // Signal
        assert_eq!(Signal::Auto.as_c_int(), OPUS_AUTO);
        assert_eq!(Signal::Voice.as_c_int(), OPUS_SIGNAL_VOICE);
        assert_eq!(Signal::Music.as_c_int(), OPUS_SIGNAL_MUSIC);

        // Bandwidth
        assert_eq!(Bandwidth::Narrowband.as_c_int(), OPUS_BANDWIDTH_NARROWBAND);
        assert_eq!(Bandwidth::Mediumband.as_c_int(), OPUS_BANDWIDTH_MEDIUMBAND);
        assert_eq!(Bandwidth::Wideband.as_c_int(), OPUS_BANDWIDTH_WIDEBAND);
        assert_eq!(Bandwidth::Superwideband.as_c_int(), OPUS_BANDWIDTH_SUPERWIDEBAND);
        assert_eq!(Bandwidth::Fullband.as_c_int(), OPUS_BANDWIDTH_FULLBAND);
        assert_eq!(Bandwidth::Auto.as_c_int(), OPUS_AUTO);

        // ForceChannels
        assert_eq!(ForceChannels::Auto.as_c_int(), OPUS_AUTO);
        assert_eq!(ForceChannels::Mono.as_c_int(), 1);
        assert_eq!(ForceChannels::Stereo.as_c_int(), 2);

        // FrameDuration
        assert_eq!(FrameDuration::Ms2_5.as_c_int(), OPUS_FRAMESIZE_2_5_MS);
        assert_eq!(FrameDuration::Ms5.as_c_int(), OPUS_FRAMESIZE_5_MS);
        assert_eq!(FrameDuration::Ms10.as_c_int(), OPUS_FRAMESIZE_10_MS);
        assert_eq!(FrameDuration::Ms20.as_c_int(), OPUS_FRAMESIZE_20_MS);
        assert_eq!(FrameDuration::Ms40.as_c_int(), OPUS_FRAMESIZE_40_MS);
        assert_eq!(FrameDuration::Ms60.as_c_int(), OPUS_FRAMESIZE_60_MS);
        assert_eq!(FrameDuration::Ms80.as_c_int(), OPUS_FRAMESIZE_80_MS);
        assert_eq!(FrameDuration::Ms100.as_c_int(), OPUS_FRAMESIZE_100_MS);
        assert_eq!(FrameDuration::Ms120.as_c_int(), OPUS_FRAMESIZE_120_MS);
        assert_eq!(FrameDuration::Argument.as_c_int(), OPUS_FRAMESIZE_ARG);
    }

    // ---- from_code round-trips, plus the positive-code guard from item 1 ----

    #[test]
    fn error_enums_from_code_round_trip() {
        // EncoderBuildError
        assert_eq!(EncoderBuildError::from_code(OPUS_BAD_ARG), EncoderBuildError::BadArg);
        assert_eq!(EncoderBuildError::from_code(OPUS_INTERNAL_ERROR), EncoderBuildError::Internal);
        // Positive guard: 0 (OPUS_OK) and any other >=0 value land in Unknown.
        assert_eq!(EncoderBuildError::from_code(0), EncoderBuildError::Unknown(0));
        assert_eq!(EncoderBuildError::from_code(-99_999), EncoderBuildError::Unknown(-99_999));

        // EncodeError
        assert_eq!(EncodeError::from_code(OPUS_BAD_ARG), EncodeError::BadArg);
        assert_eq!(EncodeError::from_code(OPUS_BUFFER_TOO_SMALL), EncodeError::BufferTooSmall);
        assert_eq!(EncodeError::from_code(OPUS_INTERNAL_ERROR), EncodeError::Internal);
        assert_eq!(EncodeError::from_code(0), EncodeError::Unknown(0));
        assert_eq!(EncodeError::from_code(-99_999), EncodeError::Unknown(-99_999));

        // DecoderInitError
        assert_eq!(DecoderInitError::from_code(OPUS_BAD_ARG), DecoderInitError::BadArg);
        assert_eq!(DecoderInitError::from_code(OPUS_INTERNAL_ERROR), DecoderInitError::Internal);
        assert_eq!(DecoderInitError::from_code(0), DecoderInitError::Unknown(0));
        assert_eq!(DecoderInitError::from_code(-99_999), DecoderInitError::Unknown(-99_999));

        // DecodeError
        assert_eq!(DecodeError::from_code(OPUS_BAD_ARG), DecodeError::BadArg);
        assert_eq!(DecodeError::from_code(OPUS_BUFFER_TOO_SMALL), DecodeError::BufferTooSmall);
        assert_eq!(DecodeError::from_code(OPUS_INTERNAL_ERROR), DecodeError::Internal);
        assert_eq!(DecodeError::from_code(OPUS_INVALID_PACKET), DecodeError::InvalidPacket);
        assert_eq!(DecodeError::from_code(0), DecodeError::Unknown(0));
        assert_eq!(DecodeError::from_code(-99_999), DecodeError::Unknown(-99_999));
    }

    // ---- Encoder/decoder zero-input consistency ----

    #[test]
    fn encoder_and_decoder_reject_zero_length_buffers() {
        let mut encoder = Encoder::builder(48_000, Channels::Mono, Application::Audio)
            .build()
            .expect("encoder builds");
        let mut packet = [0u8; 4000];
        let enc_err = encoder.encode(&[], &mut packet);
        assert!(enc_err.is_err(), "encode(&[], ...) must err, got {enc_err:?}");

        let mut decoder = Decoder::new(48_000, Channels::Mono).expect("decoder builds");
        let dec_err = decoder.decode(&[1u8], &mut [], false);
        assert!(dec_err.is_err(), "decode(.., &mut [], ..) must err, got {dec_err:?}");
    }

    // ---- Accessors round-trip the configuration ----

    #[test]
    fn encoder_decoder_accessors_round_trip() {
        let encoder = Encoder::builder(24_000, Channels::Stereo, Application::Audio)
            .build()
            .expect("encoder builds");
        assert_eq!(encoder.sample_rate(), 24_000);
        assert_eq!(encoder.channels(), Channels::Stereo);

        let decoder = Decoder::new(24_000, Channels::Stereo).expect("decoder builds");
        assert_eq!(decoder.sample_rate(), 24_000);
        assert_eq!(decoder.channels(), Channels::Stereo);
    }

    // ---- Encoder::lookahead matches the RFC 7845 pre_skip default ----

    #[test]
    fn encoder_lookahead_default_is_312() {
        // Audio application at 48 kHz: lookahead = fs/400 (120) + delay_compensation
        // (typically 192) = 312, matching the long-standing OpusHead pre_skip default.
        let encoder = Encoder::builder(48_000, Channels::Mono, Application::Audio)
            .build()
            .expect("encoder builds");
        assert_eq!(encoder.lookahead(), 312);
    }

    // ---- Bitrate::try_bits validates overflow ----

    #[test]
    fn bitrate_try_bits_validates_overflow() {
        // i32::MAX is exactly the boundary — must succeed.
        let ok = Bitrate::try_bits(i32::MAX as u32).expect("i32::MAX bps is accepted");
        assert_eq!(ok, Bitrate::Bits(i32::MAX as u32));

        // One past the boundary must fail with the typed error.
        let too_big = (i32::MAX as u32).checked_add(1).expect("u32 has headroom");
        let err = Bitrate::try_bits(too_big).expect_err("over-i32::MAX bps must error");
        assert_eq!(err, BitrateRangeError(too_big));
        // Display message check (locked-in copy from the doc-comment).
        assert_eq!(
            err.to_string(),
            format!("bitrate {too_big} bps exceeds the libopus i32::MAX limit")
        );
    }

    // ---- Builder -> encode -> decode round-trip (lossy; we only check shape) ----

    #[test]
    fn round_trip_silence_20ms_mono() {
        let mut encoder = Encoder::builder(48_000, Channels::Mono, Application::Audio)
            .bitrate(Bitrate::Bits(64_000))
            .build()
            .expect("encoder builds");
        let pcm_in = vec![0i16; 960]; // 20 ms mono @ 48 kHz
        let mut packet = vec![0u8; 4000];
        let n = encoder.encode(&pcm_in, &mut packet).expect("encode succeeds");
        assert!(n > 0, "encode produced empty packet");

        let mut decoder = Decoder::new(48_000, Channels::Mono).expect("decoder builds");
        let mut pcm_out = vec![0i16; 960];
        let samples = decoder
            .decode(&packet[..n], &mut pcm_out, false)
            .expect("decode succeeds");
        assert_eq!(samples, 960, "decoded sample count mismatch");
    }
}
