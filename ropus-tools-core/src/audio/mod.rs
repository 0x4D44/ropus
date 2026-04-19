//! Audio I/O: decode any input to interleaved f32, downmix, resample, write WAV.

pub mod decode;
pub mod dither;
pub mod downmix;
pub mod resample;
pub mod wav;
