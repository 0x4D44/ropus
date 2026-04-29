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
//! use ropus::{Application, Bitrate, Channels, DecodeMode, Decoder, Encoder};
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
//! let _samples = decoder.decode(&packet[..n], &mut pcm_out, DecodeMode::Normal).unwrap();
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

    /// Inverse of [`Bitrate::as_c_int`]. `OPUS_AUTO` (-1000) maps to
    /// [`Bitrate::Auto`], `OPUS_BITRATE_MAX` (-1) maps to [`Bitrate::Max`],
    /// and any positive value maps to [`Bitrate::Bits`]. `0` and stray
    /// negatives are unreachable from the encoder's own stored state, so
    /// the catch-all arm trips a `debug_assert!` and falls back to
    /// [`Bitrate::Auto`].
    #[allow(dead_code)] // Wired up in Stage 2 alongside the runtime getters.
    fn from_c_int(n: i32) -> Bitrate {
        match n {
            OPUS_AUTO => Bitrate::Auto,
            OPUS_BITRATE_MAX => Bitrate::Max,
            n if n > 0 => Bitrate::Bits(n as u32),
            other => {
                debug_assert!(false, "unexpected bitrate value from libopus: {other}");
                Bitrate::Auto
            }
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

    /// Inverse of [`Signal::as_c_int`]. Unknown values are unreachable from
    /// the encoder's own stored state, so the catch-all arm trips a
    /// `debug_assert!` and falls back to [`Signal::Auto`].
    #[allow(dead_code)] // Wired up in Stage 2 alongside the runtime getters.
    fn from_c_int(n: i32) -> Signal {
        match n {
            OPUS_AUTO => Signal::Auto,
            OPUS_SIGNAL_VOICE => Signal::Voice,
            OPUS_SIGNAL_MUSIC => Signal::Music,
            other => {
                debug_assert!(false, "unexpected signal value from libopus: {other}");
                Signal::Auto
            }
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

    /// Inverse of [`Bandwidth::as_c_int`]. Unknown values are unreachable
    /// from the encoder's own stored state, so the catch-all arm trips a
    /// `debug_assert!` and falls back to [`Bandwidth::Fullband`].
    #[allow(dead_code)] // Wired up in Stage 2 alongside the runtime getters.
    fn from_c_int(n: i32) -> Bandwidth {
        match n {
            OPUS_BANDWIDTH_NARROWBAND => Bandwidth::Narrowband,
            OPUS_BANDWIDTH_MEDIUMBAND => Bandwidth::Mediumband,
            OPUS_BANDWIDTH_WIDEBAND => Bandwidth::Wideband,
            OPUS_BANDWIDTH_SUPERWIDEBAND => Bandwidth::Superwideband,
            OPUS_BANDWIDTH_FULLBAND => Bandwidth::Fullband,
            OPUS_AUTO => Bandwidth::Auto,
            other => {
                debug_assert!(false, "unexpected bandwidth value from libopus: {other}");
                Bandwidth::Fullband
            }
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

    /// Inverse of [`ForceChannels::as_c_int`]. Unknown values are
    /// unreachable from the encoder's own stored state, so the catch-all
    /// arm trips a `debug_assert!` and falls back to [`ForceChannels::Auto`].
    #[allow(dead_code)] // Wired up in Stage 2 alongside the runtime getters.
    fn from_c_int(n: i32) -> ForceChannels {
        match n {
            OPUS_AUTO => ForceChannels::Auto,
            1 => ForceChannels::Mono,
            2 => ForceChannels::Stereo,
            other => {
                debug_assert!(
                    false,
                    "unexpected force-channels value from libopus: {other}"
                );
                ForceChannels::Auto
            }
        }
    }
}

/// In-band forward-error-correction mode.
///
/// Wraps libopus' `OPUS_SET_INBAND_FEC` request, which accepts three values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InbandFec {
    /// FEC disabled (CTL value 0). Default.
    Disabled,
    /// FEC enabled (CTL value 1). Encoder may add FEC redundancy when the
    /// packet-loss-percentage hint and bitrate budget permit.
    Enabled,
    /// FEC always enabled (CTL value 2). Forces FEC redundancy regardless
    /// of the bandwidth-adaptation heuristics.
    Forced,
}

impl InbandFec {
    #[allow(dead_code)] // Wired up in Stage 2 alongside the runtime setters.
    fn as_c_int(self) -> i32 {
        match self {
            InbandFec::Disabled => 0,
            InbandFec::Enabled => 1,
            InbandFec::Forced => 2,
        }
    }

    /// Inverse of [`InbandFec::as_c_int`]. Unknown values are unreachable
    /// from the encoder's own stored state, so the catch-all arm trips a
    /// `debug_assert!` and falls back to [`InbandFec::Disabled`].
    #[allow(dead_code)] // Wired up in Stage 2 alongside the runtime getters.
    fn from_c_int(n: i32) -> InbandFec {
        match n {
            0 => InbandFec::Disabled,
            1 => InbandFec::Enabled,
            2 => InbandFec::Forced,
            other => {
                debug_assert!(false, "unexpected inband-fec value from libopus: {other}");
                InbandFec::Disabled
            }
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

/// Whether to attempt forward-error-correction recovery from the previous
/// packet's redundant data. Pass `Normal` for ordinary decoding.
///
/// # Examples
///
/// ```no_run
/// use ropus::{Channels, Decoder, DecodeMode};
/// # fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let mut decoder = Decoder::new(48_000, Channels::Mono)?;
/// let packet: &[u8] = &[];
/// let mut output = vec![0i16; 960];
/// let _n = decoder.decode(packet, &mut output, DecodeMode::Normal)?;
/// # Ok(()) }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecodeMode {
    /// Ordinary decoding (no FEC recovery attempted).
    Normal,
    /// Try to recover the previous packet's content from FEC data
    /// embedded in this packet. Caller must pass an output buffer
    /// sized for the *previous* packet's frame size.
    Fec,
}

impl DecodeMode {
    fn as_decode_fec(self) -> bool {
        match self {
            DecodeMode::Normal => false,
            DecodeMode::Fec => true,
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
            vbr_constraint: None,
            force_channels: None,
            max_bandwidth: None,
            frame_duration: None,
            packet_loss_perc: None,
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
        if ch == 0 || !pcm_len.is_multiple_of(ch) {
            return Err(EncodeError::BadArg);
        }
        let per_channel = pcm_len / ch;
        // Mirror the decoder's empty-buffer check (Decoder::frame_size_from_pcm_len)
        // so an empty PCM input is reported consistently rather than being shipped
        // down to encode_native to fail later with BadArg.
        if per_channel == 0 {
            return Err(EncodeError::BufferTooSmall);
        }
        // Reject anything that isn't one of the nine legal Opus frame sizes
        // for the configured sample rate (2.5/5/10/20/40/60/80/100/120 ms).
        // Without this check, off-by-one buffer lengths get shipped down to
        // encode_native and bounce back as a generic BadArg deep in the stack.
        let fs = self.inner.get_sample_rate() as u32;
        if !is_valid_frame_size_per_channel(fs, per_channel) {
            return Err(EncodeError::BadArg);
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

    /// Per-channel sample counts accepted by [`Encoder::encode`] /
    /// [`Encoder::encode_float`] at the configured sample rate, sorted
    /// ascending (corresponding to 2.5, 5, 10, 20, 40, 60, 80, 100, 120 ms).
    ///
    /// To size an interleaved input buffer, multiply by [`channels`]:
    /// `samples_per_channel * channels.count() as usize`.
    ///
    /// [`channels`]: Self::channels
    pub fn valid_frame_sizes(&self) -> [usize; 9] {
        let fs = self.inner.get_sample_rate() as usize;
        [
            fs / 400,
            fs / 200,
            fs / 100,
            fs / 50,
            fs * 2 / 50,
            fs * 3 / 50,
            fs * 4 / 50,
            fs * 5 / 50,
            fs * 6 / 50,
        ]
    }

    /// Encoder lookahead in 48 kHz samples.
    ///
    /// This is the value to write into OpusHead's `pre_skip` field
    /// per RFC 7845. For typical encoders this is 312; in
    /// `OPUS_APPLICATION_RESTRICTED_LOWDELAY` it is 120.
    pub fn lookahead(&self) -> u32 {
        self.inner.get_lookahead().max(0) as u32
    }

    /// Current constrained-VBR setting, as the 0/1 `i32` libopus stores.
    ///
    /// Mirrors `OPUS_GET_VBR_CONSTRAINT`. Intended for introspection/testing
    /// of settings applied via [`EncoderBuilder::vbr_constraint`].
    pub fn get_vbr_constraint(&self) -> i32 {
        self.inner.get_vbr_constraint()
    }

    /// Current packet-loss-percentage hint, as the 0..=100 `i32` libopus stores.
    ///
    /// Mirrors `OPUS_GET_PACKET_LOSS_PERC`. Intended for introspection/testing
    /// of settings applied via [`EncoderBuilder::packet_loss_perc`].
    pub fn get_packet_loss_perc(&self) -> i32 {
        self.inner.get_packet_loss_perc()
    }
}

/// Whether `per_channel` is one of the nine legal Opus per-channel sample
/// counts for sample rate `fs` (2.5/5/10/20/40/60/80/100/120 ms).
fn is_valid_frame_size_per_channel(fs: u32, per_channel: usize) -> bool {
    let fs = fs as usize;
    matches!(
        per_channel,
        n if n == fs / 400
            || n == fs / 200
            || n == fs / 100
            || n == fs / 50
            || n == fs * 2 / 50
            || n == fs * 3 / 50
            || n == fs * 4 / 50
            || n == fs * 5 / 50
            || n == fs * 6 / 50
    )
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
    vbr_constraint: Option<bool>,
    force_channels: Option<ForceChannels>,
    max_bandwidth: Option<Bandwidth>,
    frame_duration: Option<FrameDuration>,
    packet_loss_perc: Option<u8>,
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

    /// Constrain VBR to meet an averaged bitrate target (CVBR mode).
    ///
    /// Only meaningful when [`vbr`](Self::vbr) is also `true`; has no effect
    /// in CBR mode. Wraps libopus' `OPUS_SET_VBR_CONSTRAINT`.
    pub fn vbr_constraint(mut self, on: bool) -> Self {
        self.vbr_constraint = Some(on);
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

    /// Hint the expected packet-loss percentage (`0..=100`) so the encoder can
    /// bias toward loss-tolerant modes. Wraps libopus' `OPUS_SET_PACKET_LOSS_PERC`;
    /// values outside the range surface as [`EncoderBuildError::BadArg`] from
    /// [`build`](Self::build).
    pub fn packet_loss_perc(mut self, pct: u8) -> Self {
        self.packet_loss_perc = Some(pct);
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
        if let Some(vbrc) = self.vbr_constraint {
            check_ok(inner.set_vbr_constraint(i32::from(vbrc)))?;
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
        if let Some(pct) = self.packet_loss_perc {
            check_ok(inner.set_packet_loss_perc(i32::from(pct)))?;
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

/// Runtime-setter counterpart to [`check_ok`]: maps a libopus `OPUS_*`
/// return code into the [`EncodeError`] surface used by `Encoder`'s
/// `&mut self` setters and `encode` methods. Generic-over-error-type
/// would require `From<i32>` impls on both error types for no readability
/// gain, so the two helpers are kept as parallel three-line copies.
#[allow(dead_code)] // Wired up in Stage 2 alongside the runtime setters.
fn check_ok_runtime(code: i32) -> Result<(), EncodeError> {
    if code == 0 {
        Ok(())
    } else {
        Err(EncodeError::from_code(code))
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

    /// Load DNN weights for neural packet-loss concealment (LPCNet + FARGAN).
    ///
    /// By default the crate embeds weights from the xiph reference sources at
    /// build time if they are on disk (typically via
    /// `cargo run -p fetch-assets -- weights` in-tree development). When
    /// installed from crates.io the embedded blob is empty, and neural PLC
    /// stays disabled unless a caller supplies a blob here; lost frames then
    /// take the classical pitch/noise PLC branch.
    ///
    /// The blob format is xiph's `write_lpcnet_weights` output. Unknown
    /// records (for example DRED or OSCE tables packed into the same tarball)
    /// are silently ignored, matching the C reference's `find_array_check`
    /// semantics.
    pub fn set_dnn_blob(&mut self, data: &[u8]) -> Result<(), DecoderInitError> {
        self.inner
            .set_dnn_blob(data)
            .map_err(DecoderInitError::from_code)
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
        mode: DecodeMode,
    ) -> Result<usize, DecodeError> {
        let frame_size = self.frame_size_from_pcm_len(output.len())?;
        let data = packet_arg(packet);
        self.inner
            .decode(data, output, frame_size, mode.as_decode_fec())
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
        mode: DecodeMode,
    ) -> Result<usize, DecodeError> {
        let frame_size = self.frame_size_from_pcm_len(output.len())?;
        let data = packet_arg(packet);
        self.inner
            .decode_float(data, output, frame_size, mode.as_decode_fec())
            .map(|n| n as usize)
            .map_err(DecodeError::from_code)
    }

    fn frame_size_from_pcm_len(&self, pcm_len: usize) -> Result<i32, DecodeError> {
        let ch = self.inner.get_channels() as usize;
        if ch == 0 || !pcm_len.is_multiple_of(ch) {
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

    /// Maximum per-channel sample count any decoded frame can produce.
    ///
    /// Equals 120 ms at the configured sample rate (e.g. 5760 at 48 kHz,
    /// 960 at 8 kHz). Use this to size the output buffer passed to
    /// [`Decoder::decode`] / [`Decoder::decode_float`].
    pub fn max_frame_samples_per_channel(&self) -> usize {
        let fs = self.inner.get_sample_rate() as usize;
        fs * 6 / 50
    }

    /// Set the decode gain in Q8 dB (range `[-32768, 32767]`).
    ///
    /// Applied inside the decoder in fixed-point before the i16 saturating
    /// clamp — matches `OPUS_SET_GAIN` in libopus. Used by callers honouring
    /// `OpusHead.output_gain` from RFC 7845 §5.1.
    pub fn set_gain(&mut self, gain: i32) -> Result<(), DecodeError> {
        self.inner.set_gain(gain).map_err(DecodeError::from_code)
    }
}

fn packet_arg(packet: &[u8]) -> Option<&[u8]> {
    if packet.is_empty() {
        None
    } else {
        Some(packet)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Thread-safety contract: Encoder and Decoder are Send + Sync ----
    //
    // This is a compile-time assertion — if the inner codec state ever grows
    // a !Send / !Sync field (e.g., Rc, RefCell, raw pointer), the README's
    // thread-safety claim would silently drift and this test would fail to
    // compile. Keep it here so the claim stays honest.

    #[test]
    fn encoder_decoder_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Encoder>();
        assert_send_sync::<Decoder>();
    }

    #[test]
    fn decoder_set_dnn_blob_rejects_garbage() {
        let mut decoder = Decoder::new(48_000, Channels::Mono).unwrap();
        // A short garbage blob can't parse as xiph's weight-record format;
        // the wrapped OpusDecoder returns OPUS_BAD_ARG, which the facade
        // lifts to DecoderInitError::BadArg.
        assert_eq!(
            decoder.set_dnn_blob(&[0u8; 4]),
            Err(DecoderInitError::BadArg),
        );
    }

    #[test]
    fn decoder_set_dnn_blob_accepts_empty_blob() {
        let mut decoder = Decoder::new(48_000, Channels::Mono).unwrap();
        // An empty blob is the build-without-weights default; it should
        // either succeed (no-op) or fail cleanly with BadArg — never panic.
        let _ = decoder.set_dnn_blob(&[]);
    }

    // ---- as_c_int round-trips against the existing OPUS_* constants ----

    #[test]
    fn typed_enums_as_c_int_match_constants() {
        // Channels: count() defines the contract; as_c_int just casts it.
        assert_eq!(Channels::Mono.as_c_int(), 1);
        assert_eq!(Channels::Stereo.as_c_int(), 2);

        // Application
        assert_eq!(Application::Voip.as_c_int(), OPUS_APPLICATION_VOIP);
        assert_eq!(Application::Audio.as_c_int(), OPUS_APPLICATION_AUDIO);
        assert_eq!(
            Application::RestrictedLowDelay.as_c_int(),
            OPUS_APPLICATION_RESTRICTED_LOWDELAY
        );

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
        assert_eq!(
            Bandwidth::Superwideband.as_c_int(),
            OPUS_BANDWIDTH_SUPERWIDEBAND
        );
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
        assert_eq!(
            EncoderBuildError::from_code(OPUS_BAD_ARG),
            EncoderBuildError::BadArg
        );
        assert_eq!(
            EncoderBuildError::from_code(OPUS_INTERNAL_ERROR),
            EncoderBuildError::Internal
        );
        // Positive guard: 0 (OPUS_OK) and any other >=0 value land in Unknown.
        assert_eq!(
            EncoderBuildError::from_code(0),
            EncoderBuildError::Unknown(0)
        );
        assert_eq!(
            EncoderBuildError::from_code(-99_999),
            EncoderBuildError::Unknown(-99_999)
        );

        // EncodeError
        assert_eq!(EncodeError::from_code(OPUS_BAD_ARG), EncodeError::BadArg);
        assert_eq!(
            EncodeError::from_code(OPUS_BUFFER_TOO_SMALL),
            EncodeError::BufferTooSmall
        );
        assert_eq!(
            EncodeError::from_code(OPUS_INTERNAL_ERROR),
            EncodeError::Internal
        );
        assert_eq!(EncodeError::from_code(0), EncodeError::Unknown(0));
        assert_eq!(
            EncodeError::from_code(-99_999),
            EncodeError::Unknown(-99_999)
        );

        // DecoderInitError
        assert_eq!(
            DecoderInitError::from_code(OPUS_BAD_ARG),
            DecoderInitError::BadArg
        );
        assert_eq!(
            DecoderInitError::from_code(OPUS_INTERNAL_ERROR),
            DecoderInitError::Internal
        );
        assert_eq!(DecoderInitError::from_code(0), DecoderInitError::Unknown(0));
        assert_eq!(
            DecoderInitError::from_code(-99_999),
            DecoderInitError::Unknown(-99_999)
        );

        // DecodeError
        assert_eq!(DecodeError::from_code(OPUS_BAD_ARG), DecodeError::BadArg);
        assert_eq!(
            DecodeError::from_code(OPUS_BUFFER_TOO_SMALL),
            DecodeError::BufferTooSmall
        );
        assert_eq!(
            DecodeError::from_code(OPUS_INTERNAL_ERROR),
            DecodeError::Internal
        );
        assert_eq!(
            DecodeError::from_code(OPUS_INVALID_PACKET),
            DecodeError::InvalidPacket
        );
        assert_eq!(DecodeError::from_code(0), DecodeError::Unknown(0));
        assert_eq!(
            DecodeError::from_code(-99_999),
            DecodeError::Unknown(-99_999)
        );
    }

    // ---- Encoder/decoder zero-input consistency ----

    #[test]
    fn encoder_and_decoder_reject_zero_length_buffers() {
        let mut encoder = Encoder::builder(48_000, Channels::Mono, Application::Audio)
            .build()
            .expect("encoder builds");
        let mut packet = [0u8; 4000];
        let enc_err = encoder.encode(&[], &mut packet);
        assert!(
            enc_err.is_err(),
            "encode(&[], ...) must err, got {enc_err:?}"
        );

        let mut decoder = Decoder::new(48_000, Channels::Mono).expect("decoder builds");
        let dec_err = decoder.decode(&[1u8], &mut [], DecodeMode::Normal);
        assert!(
            dec_err.is_err(),
            "decode(.., &mut [], ..) must err, got {dec_err:?}"
        );
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

    // ---- Decoder::max_frame_samples_per_channel matches 120 ms at fs ----

    #[test]
    fn decoder_max_frame_samples_per_channel_matches_120ms() {
        let dec = Decoder::new(48_000, Channels::Mono).unwrap();
        assert_eq!(dec.max_frame_samples_per_channel(), 5760);

        let dec8 = Decoder::new(8_000, Channels::Mono).unwrap();
        assert_eq!(dec8.max_frame_samples_per_channel(), 960);
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
        let n = encoder
            .encode(&pcm_in, &mut packet)
            .expect("encode succeeds");
        assert!(n > 0, "encode produced empty packet");

        let mut decoder = Decoder::new(48_000, Channels::Mono).expect("decoder builds");
        let mut pcm_out = vec![0i16; 960];
        let samples = decoder
            .decode(&packet[..n], &mut pcm_out, DecodeMode::Normal)
            .expect("decode succeeds");
        assert_eq!(samples, 960, "decoded sample count mismatch");
    }

    // ---- DecodeMode helper -> libopus decode_fec int ----

    #[test]
    fn decode_mode_as_decode_fec() {
        assert!(!DecodeMode::Normal.as_decode_fec());
        assert!(DecodeMode::Fec.as_decode_fec());
    }

    // ---- Encoder rejects off-by-one PCM lengths instead of leaking BadArg
    //      from deep inside encode_native. ----

    #[test]
    fn encode_rejects_off_by_one_frame_size() {
        let mut encoder = Encoder::builder(48_000, Channels::Mono, Application::Audio)
            .build()
            .expect("encoder builds");
        let mut packet = vec![0u8; 4000];

        // 481 / 959 / 961 / 1 are all one off from a legal Opus frame size at
        // 48 kHz mono and must surface as a typed BadArg.
        for bad in [1usize, 481, 959, 961] {
            let pcm = vec![0i16; bad];
            let err = encoder
                .encode(&pcm, &mut packet)
                .expect_err("off-by-one frame size must err");
            assert_eq!(
                err,
                EncodeError::BadArg,
                "encode({bad} samples) wrong error variant: {err:?}"
            );
        }
    }

    // ---- All nine valid frame sizes round-trip without error. ----

    #[test]
    fn encode_accepts_all_nine_frame_sizes() {
        let mut encoder = Encoder::builder(48_000, Channels::Mono, Application::Audio)
            .build()
            .expect("encoder builds");
        let mut packet = vec![0u8; 4000];
        for &n in &encoder.valid_frame_sizes() {
            let pcm = vec![0i16; n];
            encoder
                .encode(&pcm, &mut packet)
                .unwrap_or_else(|e| panic!("encode {n} samples failed: {e:?}"));
        }
    }

    // ---- valid_frame_sizes exact values at 48 kHz mono. ----

    #[test]
    fn valid_frame_sizes_at_48k_mono() {
        let encoder = Encoder::builder(48_000, Channels::Mono, Application::Audio)
            .build()
            .expect("encoder builds");
        assert_eq!(
            encoder.valid_frame_sizes(),
            [120, 240, 480, 960, 1920, 2880, 3840, 4800, 5760]
        );
    }

    // ---- vbr_constraint plumbs through without panicking; one-frame encode ----

    #[test]
    fn builder_vbr_constraint_round_trip() {
        for on in [false, true] {
            let mut encoder = Encoder::builder(48_000, Channels::Mono, Application::Audio)
                .bitrate(Bitrate::Bits(64_000))
                .vbr(true)
                .vbr_constraint(on)
                .build()
                .expect("encoder builds with vbr_constraint");
            assert_eq!(
                encoder.get_vbr_constraint(),
                i32::from(on),
                "vbr_constraint({on}) was not stored on the encoder"
            );
            let pcm_in = vec![0i16; 960]; // 20 ms mono @ 48 kHz
            let mut packet = vec![0u8; 4000];
            let n = encoder
                .encode(&pcm_in, &mut packet)
                .expect("encode succeeds");
            assert!(n > 0, "encode produced empty packet");
        }
    }

    // ---- packet_loss_perc plumbs through without panicking; one-frame encode ----

    #[test]
    fn builder_packet_loss_perc_round_trip() {
        for pct in [0u8, 10, 100] {
            let mut encoder = Encoder::builder(48_000, Channels::Mono, Application::Audio)
                .bitrate(Bitrate::Bits(64_000))
                .packet_loss_perc(pct)
                .build()
                .expect("encoder builds with packet_loss_perc");
            assert_eq!(
                encoder.get_packet_loss_perc(),
                pct as i32,
                "packet_loss_perc({pct}) was not stored on the encoder"
            );
            let pcm_in = vec![0i16; 960]; // 20 ms mono @ 48 kHz
            let mut packet = vec![0u8; 4000];
            let n = encoder
                .encode(&pcm_in, &mut packet)
                .expect("encode succeeds");
            assert!(n > 0, "encode produced empty packet");
        }
    }

    // ---- packet_loss_perc > 100 is rejected by the inner setter at build ----

    #[test]
    fn builder_packet_loss_perc_rejects_out_of_range() {
        let result = Encoder::builder(48_000, Channels::Mono, Application::Audio)
            .packet_loss_perc(101)
            .build();
        match result {
            Ok(_) => panic!("packet_loss_perc > 100 must err"),
            Err(e) => assert_eq!(e, EncoderBuildError::BadArg),
        }
    }

    // ---- from_c_int round-trips for every typed enum that has one ----

    #[test]
    fn bitrate_from_c_int_round_trip() {
        for value in [Bitrate::Auto, Bitrate::Max, Bitrate::Bits(64_000)] {
            assert_eq!(
                Bitrate::from_c_int(value.as_c_int()),
                value,
                "Bitrate::from_c_int(as_c_int()) round-trip failed for {value:?}",
            );
        }
    }

    #[test]
    fn signal_from_c_int_round_trip() {
        for value in [Signal::Auto, Signal::Voice, Signal::Music] {
            assert_eq!(
                Signal::from_c_int(value.as_c_int()),
                value,
                "Signal::from_c_int(as_c_int()) round-trip failed for {value:?}",
            );
        }
    }

    #[test]
    fn force_channels_from_c_int_round_trip() {
        for value in [
            ForceChannels::Auto,
            ForceChannels::Mono,
            ForceChannels::Stereo,
        ] {
            assert_eq!(
                ForceChannels::from_c_int(value.as_c_int()),
                value,
                "ForceChannels::from_c_int(as_c_int()) round-trip failed for {value:?}",
            );
        }
    }

    #[test]
    fn bandwidth_from_c_int_round_trip() {
        for value in [
            Bandwidth::Narrowband,
            Bandwidth::Mediumband,
            Bandwidth::Wideband,
            Bandwidth::Superwideband,
            Bandwidth::Fullband,
            Bandwidth::Auto,
        ] {
            assert_eq!(
                Bandwidth::from_c_int(value.as_c_int()),
                value,
                "Bandwidth::from_c_int(as_c_int()) round-trip failed for {value:?}",
            );
        }
    }

    #[test]
    fn inband_fec_from_c_int_round_trip() {
        for value in [InbandFec::Disabled, InbandFec::Enabled, InbandFec::Forced] {
            assert_eq!(
                InbandFec::from_c_int(value.as_c_int()),
                value,
                "InbandFec::from_c_int(as_c_int()) round-trip failed for {value:?}",
            );
        }
    }

    // ---- Release-build fallback for unknown bitrate values ----
    //
    // The catch-all arm in `Bitrate::from_c_int` trips a `debug_assert!`
    // and falls back to `Auto`. The assertion fires in debug builds; in
    // release builds the silent-default behaviour must hold so that a
    // hypothetical future libopus state change cannot panic the facade.
    // Skipping the debug-build symmetric check keeps the test suite clean
    // (the round-trip tests above are the load-bearing coverage).
    #[cfg(not(debug_assertions))]
    #[test]
    fn bitrate_from_c_int_unknown_falls_back_to_auto_in_release() {
        assert_eq!(Bitrate::from_c_int(0), Bitrate::Auto);
        assert_eq!(Bitrate::from_c_int(-42), Bitrate::Auto);
    }
}
