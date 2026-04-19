//! Command implementations: `encode`, `decode`, `info`, `play`. Each takes its
//! options struct from `crate::options` and returns `anyhow::Result<()>`.

mod decode;
mod encode;
mod info;
mod play;

pub use decode::decode;
pub use encode::encode;
pub use info::info;
pub use play::play;
