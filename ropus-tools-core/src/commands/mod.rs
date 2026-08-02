//! Command implementations: `encode`, `decode`, `info`, `play`. Each takes its
//! options struct from `crate::options` and returns `anyhow::Result<()>`.

mod decode;
mod encode;
mod info;
mod play;

pub use decode::{decode, decode_with_policy};
pub use encode::{encode, encode_with_policy};
pub use info::{info, validate_query_key};
pub use play::play;
