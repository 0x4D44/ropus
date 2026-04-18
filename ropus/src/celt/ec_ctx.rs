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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trait_dispatch_roundtrip_for_encoder_and_decoder() {
        let mut buffer = [0u8; 64];
        let buffer_len = buffer.len();
        {
            let mut enc = RangeEncoder::new(&mut buffer);

            assert_eq!(EcCoder::ec_tell(&enc), enc.tell());
            assert_eq!(EcCoder::ec_tell_frac(&enc), enc.tell_frac());
            assert_eq!(EcCoder::ec_range_bytes(&enc), enc.range_bytes());
            assert_eq!(EcCoder::ec_storage_usize(&enc), buffer_len);
            assert_eq!(EcCoder::ec_buffer(&enc).len(), buffer_len);
            assert_eq!(EcCoder::ec_buffer_mut(&mut enc).len(), buffer_len);

            EcCoder::ec_encode(&mut enc, 1, 2, 4);
            EcCoder::ec_enc_uint(&mut enc, 257, 1000);
            EcCoder::ec_enc_bits(&mut enc, 0b10101, 5);
            EcCoder::ec_enc_bit_logp(&mut enc, true, 1);

            let snapshot = EcCoder::ec_snapshot(&enc);

            EcCoder::ec_enc_uint(&mut enc, 7, 9);
            EcCoder::ec_restore(&mut enc, &snapshot);
            EcCoder::ec_enc_uint(&mut enc, 3, 9);

            enc.done();
        }

        let mut dec = RangeDecoder::new(&buffer);
        assert_eq!(EcCoder::ec_tell(&dec), dec.tell());
        assert_eq!(EcCoder::ec_tell_frac(&dec), dec.tell_frac());
        assert_eq!(EcCoder::ec_range_bytes(&dec), dec.range_bytes());

        let symbol = EcCoder::ec_decode(&mut dec, 4);
        assert_eq!(symbol, 1);
        EcCoder::ec_dec_update(&mut dec, 1, 2, 4);
        assert_eq!(EcCoder::ec_dec_uint(&mut dec, 1000), 257);
        assert_eq!(EcCoder::ec_dec_bits(&mut dec, 5), 0b10101);
        assert!(EcCoder::ec_dec_bit_logp(&mut dec, 1));
        assert_eq!(EcCoder::ec_dec_uint(&mut dec, 9), 3);
        assert!(!dec.error());
    }

    #[test]
    #[should_panic(expected = "ec_decode called on encoder")]
    fn test_encoder_panics_on_decode_only_method() {
        let mut buffer = [0u8; 8];
        let mut enc = RangeEncoder::new(&mut buffer);
        let _ = enc.ec_decode(2);
    }

    #[test]
    #[should_panic(expected = "ec_encode called on decoder")]
    fn test_decoder_panics_on_encode_only_method() {
        let mut dec = RangeDecoder::new(&[0x80u8]);
        dec.ec_encode(0, 1, 2);
    }

    #[test]
    #[should_panic(expected = "encoder-only")]
    fn test_decoder_panics_on_encoder_snapshot_helpers() {
        let dec = RangeDecoder::new(&[0x80u8]);
        let _ = dec.ec_snapshot();
    }

    // --- Coverage additions: more panic paths, restore on decoder ---

    #[test]
    #[should_panic(expected = "ec_dec_update called on encoder")]
    fn test_encoder_panics_on_dec_update() {
        let mut buffer = [0u8; 8];
        let mut enc = RangeEncoder::new(&mut buffer);
        enc.ec_dec_update(0, 1, 2);
    }

    #[test]
    #[should_panic(expected = "ec_dec_uint called on encoder")]
    fn test_encoder_panics_on_dec_uint() {
        let mut buffer = [0u8; 8];
        let mut enc = RangeEncoder::new(&mut buffer);
        let _ = enc.ec_dec_uint(10);
    }

    #[test]
    #[should_panic(expected = "ec_dec_bits called on encoder")]
    fn test_encoder_panics_on_dec_bits() {
        let mut buffer = [0u8; 8];
        let mut enc = RangeEncoder::new(&mut buffer);
        let _ = enc.ec_dec_bits(4);
    }

    #[test]
    #[should_panic(expected = "ec_dec_bit_logp called on encoder")]
    fn test_encoder_panics_on_dec_bit_logp() {
        let mut buffer = [0u8; 8];
        let mut enc = RangeEncoder::new(&mut buffer);
        let _ = enc.ec_dec_bit_logp(2);
    }

    #[test]
    #[should_panic(expected = "ec_enc_uint called on decoder")]
    fn test_decoder_panics_on_enc_uint() {
        let mut dec = RangeDecoder::new(&[0x80u8]);
        dec.ec_enc_uint(5, 10);
    }

    #[test]
    #[should_panic(expected = "ec_enc_bits called on decoder")]
    fn test_decoder_panics_on_enc_bits() {
        let mut dec = RangeDecoder::new(&[0x80u8]);
        dec.ec_enc_bits(0, 4);
    }

    #[test]
    #[should_panic(expected = "ec_enc_bit_logp called on decoder")]
    fn test_decoder_panics_on_enc_bit_logp() {
        let mut dec = RangeDecoder::new(&[0x80u8]);
        dec.ec_enc_bit_logp(true, 2);
    }

    #[test]
    #[should_panic(expected = "encoder-only")]
    fn test_decoder_panics_on_restore() {
        let mut dec = RangeDecoder::new(&[0x80u8]);
        // Get a valid snapshot from an encoder to pass to the decoder
        let mut buf = [0u8; 8];
        let enc = RangeEncoder::new(&mut buf);
        let snap = enc.snapshot();
        dec.ec_restore(&snap);
    }

    #[test]
    fn test_encoder_snapshot_and_restore_via_trait() {
        let mut buffer = [0u8; 64];
        let mut enc = RangeEncoder::new(&mut buffer);
        EcCoder::ec_encode(&mut enc, 0, 1, 4);
        let snap = EcCoder::ec_snapshot(&enc);
        let tell_before = EcCoder::ec_tell(&enc);
        EcCoder::ec_encode(&mut enc, 0, 1, 4);
        EcCoder::ec_restore(&mut enc, &snap);
        assert_eq!(EcCoder::ec_tell(&enc), tell_before);
    }

    #[test]
    fn test_encoder_ec_buffer_and_storage_via_trait() {
        let mut data = [0x55u8; 32];
        let enc = RangeEncoder::new(&mut data);
        assert_eq!(EcCoder::ec_storage_usize(&enc), 32);
        let buf = EcCoder::ec_buffer(&enc);
        assert_eq!(buf.len(), 32);
        assert_eq!(buf[0], 0x55);
    }
}
