//! Unified entropy coding trait for band quantization.
//!
//! The C reference uses a single `ec_ctx` type for both encoding and decoding.
//! In Rust, `RangeEncoder` and `RangeDecoder` are separate types.
//! This trait provides a unified interface so the same band quantization code
//! can handle both encode and decode paths via monomorphization.

use super::range_coder::{RangeDecoder, RangeEncoder};

/// Unified entropy coder interface.
///
/// Methods with default `unreachable!()` bodies are encode-only or decode-only;
/// the appropriate type overrides just the methods it supports. Calling an
/// unsupported method (e.g., `ec_encode` on a decoder) panics immediately,
/// matching the C behaviour of calling the wrong function on the wrong context.
pub trait EcCoder {
    /// Fractional bit position (1/8-bit units).
    fn ec_tell_frac(&self) -> u32;
    /// Integer bit position.
    fn ec_tell(&self) -> i32;
    /// Total buffer size in bytes.
    fn ec_range_bytes(&self) -> u32;

    // -- Encode-only operations (override in RangeEncoder) --

    fn ec_encode(&mut self, _fl: u32, _fh: u32, _ft: u32) {
        unreachable!("ec_encode called on decoder")
    }
    fn ec_enc_uint(&mut self, _fl: u32, _ft: u32) {
        unreachable!("ec_enc_uint called on decoder")
    }
    fn ec_enc_bits(&mut self, _fl: u32, _bits: u32) {
        unreachable!("ec_enc_bits called on decoder")
    }
    fn ec_enc_bit_logp(&mut self, _val: bool, _logp: u32) {
        unreachable!("ec_enc_bit_logp called on decoder")
    }

    // -- Decode-only operations (override in RangeDecoder) --

    fn ec_decode(&mut self, _ft: u32) -> u32 {
        unreachable!("ec_decode called on encoder")
    }
    fn ec_dec_update(&mut self, _fl: u32, _fh: u32, _ft: u32) {
        unreachable!("ec_dec_update called on encoder")
    }
    fn ec_dec_uint(&mut self, _ft: u32) -> u32 {
        unreachable!("ec_dec_uint called on encoder")
    }
    fn ec_dec_bits(&mut self, _bits: u32) -> u32 {
        unreachable!("ec_dec_bits called on encoder")
    }
    fn ec_dec_bit_logp(&mut self, _logp: u32) -> bool {
        unreachable!("ec_dec_bit_logp called on encoder")
    }

    // -- Encoder snapshot/restore for theta_rdo (override in RangeEncoder) --

    fn ec_snapshot(&self) -> super::range_coder::EncoderSnapshot {
        unreachable!("encoder-only")
    }
    fn ec_restore(&mut self, _snap: &super::range_coder::EncoderSnapshot) {
        unreachable!("encoder-only")
    }
    fn ec_range_bytes_usize(&self) -> usize {
        unreachable!("encoder-only")
    }
    fn ec_storage_usize(&self) -> usize {
        unreachable!("encoder-only")
    }
    fn ec_buffer(&self) -> &[u8] {
        unreachable!("encoder-only")
    }
    fn ec_buffer_mut(&mut self) -> &mut [u8] {
        unreachable!("encoder-only")
    }
    /// Debug: return (offs, end_offs, storage, rem) for tracing.
    fn ec_debug_state(&self) -> (u32, u32, u32, i32) {
        (0, 0, 0, 0) // default for decoder
    }
}

impl<'a> EcCoder for RangeEncoder<'a> {
    fn ec_tell_frac(&self) -> u32 {
        self.tell_frac()
    }
    fn ec_tell(&self) -> i32 {
        self.tell()
    }
    fn ec_range_bytes(&self) -> u32 {
        self.range_bytes()
    }
    fn ec_encode(&mut self, fl: u32, fh: u32, ft: u32) {
        self.encode(fl, fh, ft);
    }
    fn ec_enc_uint(&mut self, fl: u32, ft: u32) {
        self.encode_uint(fl, ft);
    }
    fn ec_enc_bits(&mut self, fl: u32, bits: u32) {
        self.encode_bits(fl, bits);
    }
    fn ec_enc_bit_logp(&mut self, val: bool, logp: u32) {
        self.encode_bit_logp(val, logp);
    }
    fn ec_snapshot(&self) -> super::range_coder::EncoderSnapshot {
        self.snapshot()
    }
    fn ec_restore(&mut self, snap: &super::range_coder::EncoderSnapshot) {
        self.restore(snap)
    }
    fn ec_range_bytes_usize(&self) -> usize {
        self.range_bytes() as usize
    }
    fn ec_storage_usize(&self) -> usize {
        self.storage() as usize
    }
    fn ec_buffer(&self) -> &[u8] {
        self.buffer()
    }
    fn ec_buffer_mut(&mut self) -> &mut [u8] {
        self.buffer_mut()
    }
    fn ec_debug_state(&self) -> (u32, u32, u32, i32) {
        let (offs, eoffs, storage, _, _, rem, _) = self.enc_debug_state();
        (offs, eoffs, storage, rem)
    }
}

impl<'a> EcCoder for RangeDecoder<'a> {
    fn ec_tell_frac(&self) -> u32 {
        self.tell_frac()
    }
    fn ec_tell(&self) -> i32 {
        self.tell()
    }
    fn ec_range_bytes(&self) -> u32 {
        self.range_bytes()
    }
    fn ec_decode(&mut self, ft: u32) -> u32 {
        self.decode(ft)
    }
    fn ec_dec_update(&mut self, fl: u32, fh: u32, ft: u32) {
        self.update(fl, fh, ft);
    }
    fn ec_dec_uint(&mut self, ft: u32) -> u32 {
        self.decode_uint(ft)
    }
    fn ec_dec_bits(&mut self, bits: u32) -> u32 {
        self.decode_bits(bits)
    }
    fn ec_dec_bit_logp(&mut self, logp: u32) -> bool {
        self.decode_bit_logp(logp)
    }
}
