/// Return the largest supported Opus frame size for a sample rate.
///
/// Opus permits frames up to 120 ms. Keep this arithmetic checked because
/// callers at the low-level API boundary supply the rate and frame size as
/// signed integers.
pub(crate) fn max_frame_size(fs: i32) -> Option<i32> {
    fs.checked_mul(6)?.checked_div(50)
}

/// Compute an interleaved sample count without allowing signed-to-unsigned
/// conversion or multiplication to wrap.
pub(crate) fn checked_sample_count(frame_size: i32, channels: i32) -> Option<usize> {
    let frame_size = usize::try_from(frame_size).ok()?;
    let channels = usize::try_from(channels).ok()?;
    frame_size.checked_mul(channels)
}

#[doc(hidden)]
pub mod analysis;
pub mod decoder;
pub mod dred;
pub mod encoder;
pub mod extensions;
pub(crate) mod mlp;
pub(crate) mod mlp_data;
pub mod multistream;
pub mod repacketizer;
pub mod soft_clip;
