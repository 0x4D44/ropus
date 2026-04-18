//! CELT static tables.
//!
//! Precomputed lookup tables shared across CELT modules.

/// Mean energy per band in Q4, with BITRES=3 baked in.
/// Used by `denormalise_bands` to add per-band mean before exp2.
/// Matches C `eMeans[25]` from `quant_bands.c`.
pub static E_MEANS: [i8; 25] = [
    9, 10, 10, 11, 11, 12, 12, 12, 11, 11, 11, 10, 10, 8, 6, 5, 3, 1, 4, 4, 3, 2, 3, 4, 3,
];
