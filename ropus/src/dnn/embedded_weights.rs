//! Compile-time default DNN weight blob.
//!
//! Populated by `build.rs` from the xiph weight tables
//! (`reference/dnn/{pitchdnn,fargan,plc}_data.c`). When the reference
//! sources aren't on disk at build time, the blob is empty — callers
//! must call `OpusDecoder::set_dnn_blob` explicitly with a
//! user-provided blob to activate neural PLC.
//!
//! The blob format matches the runtime `parse_weights` input: a
//! concatenation of `WeightHead` (64 bytes) + payload + zero-padding
//! per record, for PitchDNN, then FARGAN, then PLC-predictor arrays.

/// Embedded weight blob. `&[]` when the build couldn't locate the
/// reference DNN sources (e.g. crates.io consumers without a local
/// `cargo run -p fetch-assets -- weights`).
pub const WEIGHTS_BLOB: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/weights_blob.bin"));

/// Convenience: `true` iff `WEIGHTS_BLOB` has at least one record.
/// Used by `OpusDecoder::new` to decide whether to auto-load weights.
#[inline]
pub const fn has_embedded_weights() -> bool {
    !WEIGHTS_BLOB.is_empty()
}
