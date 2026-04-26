#![cfg(not(no_reference))]
//! DRED-scoped differential tests against the C reference.
//!
//! Keeps Stage 8 tests isolated from `c_ref_differential.rs` (which is
//! already 3000+ lines of pre-Stage-8 coverage). Each sub-stage of the
//! DRED port contributes a targeted test here.

#![allow(clippy::too_many_arguments)]

use std::os::raw::{c_int, c_uchar};

// ---------------------------------------------------------------------------
// Range-coder FFI (subset needed for laplace_p0 tests).
//
// Matches the layout used by `harness/src/bindings.rs` — inlined here to
// keep this integration-test crate self-contained.
// ---------------------------------------------------------------------------

type OpusUint32 = u32;

#[repr(C)]
struct EcCtx {
    buf: *mut c_uchar,
    storage: OpusUint32,
    end_offs: OpusUint32,
    end_window: OpusUint32,
    nend_bits: c_int,
    nbits_total: c_int,
    offs: OpusUint32,
    rng: OpusUint32,
    val: OpusUint32,
    ext: OpusUint32,
    rem: c_int,
    error: c_int,
}

unsafe extern "C" {
    fn ec_enc_init(this: *mut EcCtx, buf: *mut c_uchar, size: OpusUint32);
    fn ec_enc_done(this: *mut EcCtx);
    fn ec_dec_init(this: *mut EcCtx, buf: *mut c_uchar, storage: OpusUint32);

    fn ec_laplace_encode_p0(this: *mut EcCtx, value: c_int, p0: u16, decay: u16);
    fn ec_laplace_decode_p0(this: *mut EcCtx, p0: u16, decay: u16) -> c_int;
}

// ---------------------------------------------------------------------------
// DRED stats-data FFI (Stage 8.2).
//
// Each `extern` is the C `const opus_uint8 dred_*_q8[N]` array linked in from
// `reference/dnn/dred_rdovae_stats_data.c`. `harness/build.rs` compiles that
// TU alongside the rest of the reference library.
// ---------------------------------------------------------------------------

unsafe extern "C" {
    static dred_latent_quant_scales_q8: [u8; 400];
    static dred_latent_dead_zone_q8: [u8; 400];
    static dred_latent_r_q8: [u8; 400];
    static dred_latent_p0_q8: [u8; 400];

    static dred_state_quant_scales_q8: [u8; 800];
    static dred_state_dead_zone_q8: [u8; 800];
    static dred_state_r_q8: [u8; 800];
    static dred_state_p0_q8: [u8; 800];
}

/// `ec_range_bytes` is a static inline in `entcode.h` returning `ctx->offs`.
/// Read the field directly to avoid needing a non-exported symbol.
fn range_bytes(ctx: &EcCtx) -> usize {
    ctx.offs as usize
}

#[allow(dead_code)]
fn new_ctx() -> EcCtx {
    EcCtx {
        buf: std::ptr::null_mut(),
        storage: 0,
        end_offs: 0,
        end_window: 0,
        nend_bits: 0,
        nbits_total: 0,
        offs: 0,
        rng: 0,
        val: 0,
        ext: 0,
        rem: 0,
        error: 0,
    }
}

/// Encode a vector of values with the C reference; return the written bytes
/// (trimmed to `ec_range_bytes`).
fn c_encode_laplace_p0(values: &[i32], p0: u16, decay: u16) -> Vec<u8> {
    let mut buf = vec![0u8; 4096];
    let mut ec = new_ctx();
    unsafe {
        ec_enc_init(&mut ec, buf.as_mut_ptr(), buf.len() as OpusUint32);
        for &v in values {
            ec_laplace_encode_p0(&mut ec, v, p0, decay);
        }
        ec_enc_done(&mut ec);
        let n = range_bytes(&ec);
        buf.truncate(n);
    }
    buf
}

/// Decode `n` values from `bytes` using the C reference.
fn c_decode_laplace_p0(bytes: &[u8], n: usize, p0: u16, decay: u16) -> Vec<i32> {
    let mut buf = bytes.to_vec();
    // Pad in case the range decoder reads past `ec_range_bytes` (it may read
    // the next byte even if it was coded as zero).
    buf.resize(buf.len().max(32), 0);
    let mut ec = new_ctx();
    let mut out = Vec::with_capacity(n);
    unsafe {
        ec_dec_init(&mut ec, buf.as_mut_ptr(), buf.len() as OpusUint32);
        for _ in 0..n {
            out.push(ec_laplace_decode_p0(&mut ec, p0, decay) as i32);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Rust-encoded bytes match C-encoded bytes for identical inputs.
///
/// This is the Stage 8.1 tier-1 anchor: once DRED payloads depend on
/// `ec_laplace_encode_p0`, any drift here will ripple into every DRED
/// redundancy frame. Covering the full expected input range up front
/// rules out that contribution.
#[test]
fn laplace_p0_encode_matches_c() {
    use ropus::celt::range_coder::RangeEncoder;

    // DRED-realistic (p0, decay) pairs. C `dred_encode_latents` skips the
    // call when `p0 == 255` or `r == 0`, so those degenerate cases don't
    // need coverage. We test the actual called range: p0, decay both in
    // roughly [128, 32000] (i.e. u8 << 7 with neither saturated).
    let params: &[(u16, u16)] = &[
        (10000, 14000),
        (1 << 7, 1 << 7),
        (64 << 7, 100 << 7),
        (127 << 7, 200 << 7),
        (200 << 7, 250 << 7),
        (16384, 16384),
    ];
    // Values span the DRED quantiser range plus zero-magnetism.
    let values: &[i32] = &[
        0, 0, 0, 1, -1, 2, -2, 7, -7, 8, -8, 14, -14, 15, -15, 30, -30, 3, -3, 0, 5, -5,
    ];

    for &(p0, decay) in params {
        let c_bytes = c_encode_laplace_p0(values, p0, decay);

        let mut rust_buf = vec![0u8; 4096];
        let rust_bytes = {
            let mut enc = RangeEncoder::new(&mut rust_buf);
            for &v in values {
                enc.encode_laplace_p0(v, p0, decay);
            }
            enc.done();
            assert!(!enc.error(), "rust enc error p0={p0} decay={decay}");
            let n = enc.range_bytes() as usize;
            rust_buf[..n].to_vec()
        };

        assert_eq!(
            rust_bytes, c_bytes,
            "byte divergence p0={p0} decay={decay}: rust={rust_bytes:?} c={c_bytes:?}"
        );
    }
}

/// Rust decodes C-encoded bytes back to the original values.
///
/// Catches the reverse-direction bug: an asymmetric encoder/decoder pair
/// could produce identical encoded bytes by accident (encoder matches C,
/// decoder deviates) and round-trip-test cleanly against itself while
/// silently diverging from the reference stream.
#[test]
fn laplace_p0_decode_matches_c() {
    use ropus::celt::range_coder::RangeDecoder;

    let params: &[(u16, u16)] = &[(10000, 14000), (127 << 7, 200 << 7), (255 << 7, 255 << 7)];
    let values: &[i32] = &[0, 3, -5, 12, -9, 0, 1, -1, 20, -20, 0];

    for &(p0, decay) in params {
        let c_bytes = c_encode_laplace_p0(values, p0, decay);

        // Decode with Rust
        let mut padded = c_bytes.clone();
        padded.resize(padded.len().max(32), 0);
        let mut dec = RangeDecoder::new(&padded);
        let rust_decoded: Vec<i32> = (0..values.len())
            .map(|_| dec.decode_laplace_p0(p0, decay))
            .collect();

        // Decode with C (as a sanity cross-check that c_bytes round-trip)
        let c_decoded = c_decode_laplace_p0(&c_bytes, values.len(), p0, decay);

        assert_eq!(
            rust_decoded, values,
            "rust decode of c bytes diverges p0={p0} decay={decay}"
        );
        assert_eq!(
            c_decoded, values,
            "c decode of c bytes diverges p0={p0} decay={decay}"
        );
    }
}

/// Stage 8.2: every DRED static quantisation table matches the C reference
/// byte-for-byte. Any drift here silently corrupts every redundancy frame the
/// encoder emits (and every one the decoder parses), so the diff is anchored
/// at stage boundary rather than relying on downstream integration to surface
/// it.
#[test]
fn dred_stats_arrays_match_c() {
    use ropus::dnn::dred_stats;

    // SAFETY: `extern static` arrays from `dred_rdovae_stats_data.c` are
    // compile-time constants; reading them is always defined.
    let pairs: &[(&str, &[u8], &[u8])] = unsafe {
        &[
            (
                "dred_latent_quant_scales_q8",
                &dred_stats::dred_latent_quant_scales_q8,
                &dred_latent_quant_scales_q8,
            ),
            (
                "dred_latent_dead_zone_q8",
                &dred_stats::dred_latent_dead_zone_q8,
                &dred_latent_dead_zone_q8,
            ),
            (
                "dred_latent_r_q8",
                &dred_stats::dred_latent_r_q8,
                &dred_latent_r_q8,
            ),
            (
                "dred_latent_p0_q8",
                &dred_stats::dred_latent_p0_q8,
                &dred_latent_p0_q8,
            ),
            (
                "dred_state_quant_scales_q8",
                &dred_stats::dred_state_quant_scales_q8,
                &dred_state_quant_scales_q8,
            ),
            (
                "dred_state_dead_zone_q8",
                &dred_stats::dred_state_dead_zone_q8,
                &dred_state_dead_zone_q8,
            ),
            (
                "dred_state_r_q8",
                &dred_stats::dred_state_r_q8,
                &dred_state_r_q8,
            ),
            (
                "dred_state_p0_q8",
                &dred_stats::dred_state_p0_q8,
                &dred_state_p0_q8,
            ),
        ]
    };

    for (name, rust, c) in pairs {
        assert_eq!(
            rust.len(),
            c.len(),
            "{name}: length mismatch rust={} c={}",
            rust.len(),
            c.len()
        );
        if rust != c {
            // Surface the first divergent index to make failures debuggable.
            let idx = rust
                .iter()
                .zip(c.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(usize::MAX);
            panic!(
                "{name}: byte divergence at index {idx} (rust={} c={})",
                rust[idx], c[idx],
            );
        }
    }
}
