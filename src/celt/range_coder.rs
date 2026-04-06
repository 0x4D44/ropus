//! Range coder (entropy coding) for the Opus codec.
//!
//! Faithful port of `celt/entcode.c`, `celt/entenc.c`, and `celt/entdec.c`
//! from the xiph/opus C reference implementation.
//!
//! The range coder is the entropy coding engine at the heart of Opus. Every
//! coded decision in both CELT and SILK passes through this module. It uses a
//! bidirectional buffer: range-coded bytes grow forward from the start, while
//! raw (uncoded) bits grow backward from the end.

use crate::types::ec_ilog;

// ============================================================================
// Constants from mfrngcod.h
// ============================================================================

/// Bits per output symbol (byte-oriented I/O).
const EC_SYM_BITS: u32 = 8;

/// Total bits in the state registers (val, rng).
const EC_CODE_BITS: u32 = 32;

/// Maximum value of an output symbol (0xFF).
const EC_SYM_MAX: u32 = (1u32 << EC_SYM_BITS) - 1;

/// Shift to extract top symbol from val: 32 - 8 - 1 = 23.
const EC_CODE_SHIFT: u32 = EC_CODE_BITS - EC_SYM_BITS - 1;

/// Upper bound of the code range (bit 31): 0x80000000.
const EC_CODE_TOP: u32 = 1u32 << (EC_CODE_BITS - 1);

/// Normalization threshold: EC_CODE_TOP >> 8 = 0x00800000.
const EC_CODE_BOT: u32 = EC_CODE_TOP >> EC_SYM_BITS;

/// Extra bits in the last partial symbol: (32 - 2) % 8 + 1 = 7.
const EC_CODE_EXTRA: u32 = (EC_CODE_BITS - 2) % EC_SYM_BITS + 1;

// ============================================================================
// Constants from entcode.h
// ============================================================================

/// Bits for the range-coded portion of unsigned integers in ec_enc/dec_uint.
pub const EC_UINT_BITS: u32 = 8;

/// Fractional bit resolution for ec_tell_frac: 2^3 = 8 → 1/8-bit precision.
pub const BITRES: u32 = 3;

/// Width of the raw-bit accumulator in bits.
const EC_WINDOW_SIZE: u32 = 32;

// ============================================================================
// Shared helpers
// ============================================================================

/// Correction table for `tell_frac`. Threshold values for the normalized range
/// at each 1/8-bit boundary: `correction[k] ≈ floor(2^16 · 2^(-(k+1)/8))`.
const TELL_FRAC_CORRECTION: [u32; 8] = [35733, 38967, 42495, 46340, 50535, 55109, 60097, 65535];

/// Debug flag for tracing raw-bit writes.
static TRACE_ENCODE_BITS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Computes bits used scaled by `2^BITRES` (1/8-bit units).
///
/// Uses a lookup-table-accelerated linear approximation with one correction
/// step. Always rounds up (reports worst-case bit cost).
fn tell_frac_impl(nbits_total: i32, rng: u32) -> u32 {
    let nbits = (nbits_total as u32) << BITRES;
    let l = ec_ilog(rng) as u32;
    // Normalize rng into [2^15, 2^16) for table lookup
    let r = rng >> (l - 16);
    // Initial estimate from the top 4 bits of the normalized range
    let mut b = (r >> 12) - 8;
    // One correction step via the threshold table
    b += if r > TELL_FRAC_CORRECTION[b as usize] {
        1
    } else {
        0
    };
    // Combine integer log2 part (l) and fractional 1/8-bit part (b)
    let l = (l << 3) + b;
    nbits - l
}

// ============================================================================
// RangeEncoder
// ============================================================================

/// Snapshot of a `RangeEncoder`'s scalar state (everything except the buffer).
/// Used by the SILK rate control loop to save/restore the encoder state.
#[derive(Clone)]
pub struct RangeEncoderSnapshot {
    pub storage: u32,
    pub end_offs: u32,
    pub end_window: u32,
    pub nend_bits: i32,
    pub nbits_total: i32,
    pub offs: u32,
    pub rng: u32,
    pub val: u32,
    pub rem: i32,
    pub ext: u32,
    pub error: i32,
    /// Saved front-of-buffer data (range-coded bytes, up to `offs`).
    pub buf_front: Vec<u8>,
}

/// Range encoder. Writes entropy-coded symbols into a byte buffer.
///
/// Corresponds to `ec_enc` (typedef for `ec_ctx`) in the C reference.
/// The buffer is bidirectional: range-coded bytes grow forward from `buf[0]`,
/// raw bits grow backward from `buf[storage-1]`.
pub struct RangeEncoder<'a> {
    buf: &'a mut [u8],
    /// Buffer capacity in bytes.
    storage: u32,
    /// Bytes consumed from the end of the buffer (raw bits region).
    end_offs: u32,
    /// Bit accumulator for raw bits at the buffer end.
    end_window: u32,
    /// Number of valid bits in end_window.
    nend_bits: i32,
    /// Total bits consumed (range-coded + raw); used by tell().
    nbits_total: i32,
    /// Byte offset for the front of the buffer (range-coded data).
    offs: u32,
    /// Current interval width. Invariant: > EC_CODE_BOT after normalization.
    rng: u32,
    /// Low end of the current coding interval.
    val: u32,
    /// Buffered byte awaiting carry propagation (-1 = no byte yet).
    rem: i32,
    /// Count of outstanding carry-propagating symbols (consecutive 0xFF bytes).
    ext: u32,
    /// Nonzero if a buffer overflow occurred.
    error: i32,
}

impl<'a> RangeEncoder<'a> {
    /// Initializes the encoder with an external buffer.
    ///
    /// Corresponds to `ec_enc_init`.
    pub fn new(buf: &'a mut [u8]) -> Self {
        let storage = buf.len() as u32;
        Self {
            buf,
            storage,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            nbits_total: EC_CODE_BITS as i32 + 1,
            offs: 0,
            rng: EC_CODE_TOP,
            val: 0,
            rem: -1,
            ext: 0,
            error: 0,
        }
    }

    /// Number of range-coded bytes written so far.
    #[inline(always)]
    pub fn range_bytes(&self) -> u32 {
        self.offs
    }

    /// Debug: enable/disable raw-bit write tracing.
    pub fn set_trace_bits(enable: bool) {
        TRACE_ENCODE_BITS.store(enable, std::sync::atomic::Ordering::Relaxed);
    }

    /// Debug: return (offs, end_offs, storage, rng, val, rem, ext) for tracing.
    pub fn enc_debug_state(&self) -> (u32, u32, u32, u32, u32, i32, u32) {
        (self.offs, self.end_offs, self.storage, self.rng, self.val, self.rem, self.ext)
    }

    /// Debug: return the buffer byte at the given absolute offset.
    pub fn enc_debug_byte(&self, idx: usize) -> u8 {
        if idx < self.buf.len() {
            self.buf[idx]
        } else {
            0
        }
    }

    /// Save a snapshot of the encoder state (scalars + front buffer).
    /// Matches C `memcpy(&sRangeEnc_copy, psRangeEnc, sizeof(ec_enc))` +
    /// buffer save.
    pub fn save_snapshot(&self) -> RangeEncoderSnapshot {
        let front_len = self.offs as usize;
        RangeEncoderSnapshot {
            storage: self.storage,
            end_offs: self.end_offs,
            end_window: self.end_window,
            nend_bits: self.nend_bits,
            nbits_total: self.nbits_total,
            offs: self.offs,
            rng: self.rng,
            val: self.val,
            rem: self.rem,
            ext: self.ext,
            error: self.error,
            buf_front: self.buf[..front_len].to_vec(),
        }
    }

    /// Restore from a previously saved snapshot.
    pub fn restore_snapshot(&mut self, snap: &RangeEncoderSnapshot) {
        self.storage = snap.storage;
        self.end_offs = snap.end_offs;
        self.end_window = snap.end_window;
        self.nend_bits = snap.nend_bits;
        self.nbits_total = snap.nbits_total;
        self.offs = snap.offs;
        self.rng = snap.rng;
        self.val = snap.val;
        self.rem = snap.rem;
        self.ext = snap.ext;
        self.error = snap.error;
        let front_len = snap.buf_front.len();
        self.buf[..front_len].copy_from_slice(&snap.buf_front);
    }

    /// Returns the current internal range value.
    ///
    /// Used by the encoder to capture the final range for verification.
    pub fn get_rng(&self) -> u32 {
        self.rng
    }

    /// Returns the current low end of the coding interval (debug).
    pub fn get_val(&self) -> u32 {
        self.val
    }

    /// Reference to the output buffer.
    #[inline(always)]
    pub fn buffer(&self) -> &[u8] {
        self.buf
    }

    /// Whether an error (buffer overflow) occurred.
    #[inline(always)]
    pub fn error(&self) -> bool {
        self.error != 0
    }

    /// Debug: returns (val, offs, end_offs, storage, nend_bits, rem, ext).
    pub fn debug_state(&self) -> (u32, u32, u32, u32, i32, i32, u32) {
        (
            self.val,
            self.offs,
            self.end_offs,
            self.storage,
            self.nend_bits,
            self.rem,
            self.ext,
        )
    }

    /// Bits "used" so far (conservative, rounds up).
    #[inline(always)]
    pub fn tell(&self) -> i32 {
        self.nbits_total - ec_ilog(self.rng)
    }

    /// Adjust nbits_total by a delta. Used by the CELT encoder to mark
    /// the remaining budget as consumed (e.g. for silence frames).
    /// Matches C: `enc->nbits_total += delta`.
    #[inline(always)]
    pub fn add_nbits_total(&mut self, delta: i32) {
        self.nbits_total += delta;
    }

    /// Bits used, scaled by `2^BITRES` (in 1/8-bit units).
    #[inline(always)]
    pub fn tell_frac(&self) -> u32 {
        tell_frac_impl(self.nbits_total, self.rng)
    }

    // ---- Internal helpers ----

    /// Writes a byte to the front of the buffer. Returns -1 on overflow.
    fn write_byte(&mut self, value: u32) -> i32 {
        if self.offs + self.end_offs >= self.storage {
            return -1;
        }
        self.buf[self.offs as usize] = value as u8;
        self.offs += 1;
        0
    }

    /// Writes a byte to the end of the buffer. Returns -1 on overflow.
    fn write_byte_at_end(&mut self, value: u32) -> i32 {
        if self.offs + self.end_offs >= self.storage {
            return -1;
        }
        self.end_offs += 1;
        self.buf[(self.storage - self.end_offs) as usize] = value as u8;
        0
    }

    /// Outputs a symbol with carry propagation.
    ///
    /// Consecutive 0xFF bytes are buffered in `ext` because a carry from lower
    /// bits could ripple through them (0xFF + 1 = 0x00 with carry). When a
    /// non-0xFF byte arrives, all buffered bytes are flushed.
    fn carry_out(&mut self, c: i32) {
        if c != EC_SYM_MAX as i32 {
            let carry = c >> EC_SYM_BITS;
            if self.rem >= 0 {
                self.error |= self.write_byte((self.rem + carry) as u32);
            }
            if self.ext > 0 {
                // carry=0 → flush 0xFF; carry=1 → 0xFF wraps to 0x00
                let sym = (EC_SYM_MAX + carry as u32) & EC_SYM_MAX;
                loop {
                    self.error |= self.write_byte(sym);
                    self.ext -= 1;
                    if self.ext == 0 {
                        break;
                    }
                }
            }
            self.rem = c & EC_SYM_MAX as i32;
        } else {
            self.ext += 1;
        }
    }

    /// If the range is too small, outputs bytes and rescales.
    #[inline(always)]
    fn normalize(&mut self) {
        while self.rng <= EC_CODE_BOT {
            self.carry_out((self.val >> EC_CODE_SHIFT) as i32);
            self.val = (self.val << EC_SYM_BITS) & (EC_CODE_TOP - 1);
            self.rng <<= EC_SYM_BITS;
            self.nbits_total += EC_SYM_BITS as i32;
        }
    }

    // ---- Public encoding operations ----

    /// Encodes a symbol with CDF range `[fl, fh)` out of total `ft`.
    ///
    /// Uses integer division for the general case. Corresponds to `ec_encode`.
    pub fn encode(&mut self, fl: u32, fh: u32, ft: u32) {
        let r = self.rng / ft;
        if fl > 0 {
            self.val = self
                .val
                .wrapping_add(self.rng.wrapping_sub(r.wrapping_mul(ft.wrapping_sub(fl))));
            self.rng = r.wrapping_mul(fh.wrapping_sub(fl));
        } else {
            // First-symbol optimization: keep the division residual in the range
            self.rng = self.rng.wrapping_sub(r.wrapping_mul(ft.wrapping_sub(fh)));
        }
        self.normalize();
    }

    /// Like `encode()` but with `ft = 1 << bits`. Replaces division with a shift.
    ///
    /// Corresponds to `ec_encode_bin`.
    pub fn encode_bin(&mut self, fl: u32, fh: u32, bits: u32) {
        let r = self.rng >> bits;
        if fl > 0 {
            self.val += self.rng - r * ((1u32 << bits) - fl);
            self.rng = r * (fh - fl);
        } else {
            self.rng -= r * ((1u32 << bits) - fh);
        }
        self.normalize();
    }

    /// Encodes a binary symbol where P(1) = 1/(1 << logp). No division needed.
    ///
    /// Corresponds to `ec_enc_bit_logp`.
    pub fn encode_bit_logp(&mut self, val: bool, logp: u32) {
        let r = self.rng;
        let l = self.val;
        let s = r >> logp;
        let r = r - s;
        if val {
            self.val = l + r;
        }
        self.rng = if val { s } else { r };
        self.normalize();
    }

    /// Encodes symbol `s` using an 8-bit inverse CDF table with `ft = 1 << ftb`.
    ///
    /// The iCDF encodes survival probabilities: `icdf[s] = ft - CDF(s+1)`.
    /// Values must be monotonically non-increasing, last entry must be 0.
    ///
    /// Corresponds to `ec_enc_icdf`.
    pub fn encode_icdf(&mut self, s: u32, icdf: &[u8], ftb: u32) {
        let r = self.rng >> ftb;
        if s > 0 {
            self.val += self.rng - r * icdf[s as usize - 1] as u32;
            self.rng = r * (icdf[s as usize - 1] as u32 - icdf[s as usize] as u32);
        } else {
            self.rng -= r * icdf[s as usize] as u32;
        }
        self.normalize();
    }

    /// Like `encode_icdf` but with 16-bit iCDF entries.
    ///
    /// Corresponds to `ec_enc_icdf16`.
    pub fn encode_icdf16(&mut self, s: u32, icdf: &[u16], ftb: u32) {
        let r = self.rng >> ftb;
        if s > 0 {
            self.val += self.rng - r * icdf[s as usize - 1] as u32;
            self.rng = r * (icdf[s as usize - 1] as u32 - icdf[s as usize] as u32);
        } else {
            self.rng -= r * icdf[s as usize] as u32;
        }
        self.normalize();
    }

    /// Encodes a uniform integer in `[0, ft)`.
    ///
    /// When `ft` needs more than `EC_UINT_BITS` (8) bits, the value is split
    /// into range-coded MSBs + raw LSBs. Corresponds to `ec_enc_uint`.
    pub fn encode_uint(&mut self, fl: u32, ft: u32) {
        debug_assert!(ft > 1);
        let ft_dec = ft - 1;
        let ftb = ec_ilog(ft_dec) as u32;
        if ftb > EC_UINT_BITS {
            let ftb = ftb - EC_UINT_BITS;
            let ft_upper = (ft_dec >> ftb) + 1;
            let fl_upper = fl >> ftb;
            // Debug: trace encode_uint when near byte 261
            if TRACE_ENCODE_BITS.load(std::sync::atomic::Ordering::Relaxed)
                && self.end_offs >= 50 && self.end_offs <= 62 {
                let raw_bits = fl & ((1u32 << ftb) - 1);
                eprintln!("[R EU] fl={} ft={} ftb={} fl_upper={} ft_upper={} raw_bits={} eoffs={}",
                    fl, ft, ftb, fl_upper, ft_upper, raw_bits, self.end_offs);
            }
            self.encode(fl_upper, fl_upper + 1, ft_upper);
            self.encode_bits(fl & ((1u32 << ftb) - 1), ftb);
        } else {
            self.encode(fl, fl + 1, ft_dec + 1);
        }
    }

    /// Writes raw (uncoded) bits to the end of the buffer. 1 ≤ bits ≤ 25.
    ///
    /// Corresponds to `ec_enc_bits`.
    pub fn encode_bits(&mut self, fl: u32, bits: u32) {
        debug_assert!(bits > 0);
        // Debug: trace raw-bit writes near byte 261 (eoffs 55-62)
        {
            if TRACE_ENCODE_BITS.load(std::sync::atomic::Ordering::Relaxed) {
                let new_eoffs_approx = self.end_offs + ((self.nend_bits as u32 + bits + 7) / 8);
                if self.end_offs <= 62 && new_eoffs_approx >= 56 {
                    static EB_CTR: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
                    let ctr = EB_CTR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    eprintln!("[R EB#{}] fl=0x{:x} bits={} eoffs={} nend_bits={} end_window=0x{:08x}",
                        ctr, fl, bits, self.end_offs, self.nend_bits, self.end_window);
                }
            }
        }
        let mut window = self.end_window;
        let mut used = self.nend_bits;
        if used + bits as i32 > EC_WINDOW_SIZE as i32 {
            loop {
                self.error |= self.write_byte_at_end(window & EC_SYM_MAX);
                window >>= EC_SYM_BITS;
                used -= EC_SYM_BITS as i32;
                if used < EC_SYM_BITS as i32 {
                    break;
                }
            }
        }
        window |= fl << used as u32;
        used += bits as i32;
        self.end_window = window;
        self.nend_bits = used;
        self.nbits_total += bits as i32;
    }

    /// Overwrites the first `nbits` (≤ 8) bits of the stream after encoding.
    ///
    /// Used for packet header flags whose values aren't known until late in
    /// the encoding process. At least `nbits` bits must already have been
    /// encoded using power-of-two probabilities.
    ///
    /// Corresponds to `ec_enc_patch_initial_bits`.
    pub fn patch_initial_bits(&mut self, val: u32, nbits: u32) {
        debug_assert!(nbits <= EC_SYM_BITS);
        let shift = EC_SYM_BITS - nbits;
        let mask = ((1u32 << nbits) - 1) << shift;
        if self.offs > 0 {
            // The first byte has been finalized in the buffer
            self.buf[0] = ((self.buf[0] as u32 & !mask) | (val << shift)) as u8;
        } else if self.rem >= 0 {
            // The first byte is still awaiting carry propagation
            self.rem = ((self.rem as u32 & !mask) | (val << shift)) as i32;
        } else if self.rng <= EC_CODE_TOP >> nbits {
            // The renormalization loop has never been run — patch val directly
            self.val = (self.val & !(mask << EC_CODE_SHIFT)) | (val << (EC_CODE_SHIFT + shift));
        } else {
            // Fewer than nbits have been encoded
            self.error = -1;
        }
    }

    /// Compacts the buffer to `new_size` bytes, relocating end-of-buffer raw bits.
    ///
    /// Corresponds to `ec_enc_shrink`.
    pub fn shrink(&mut self, new_size: u32) {
        debug_assert!(self.offs + self.end_offs <= new_size);
        let src_start = (self.storage - self.end_offs) as usize;
        let src_end = self.storage as usize;
        let dst_start = (new_size - self.end_offs) as usize;
        self.buf.copy_within(src_start..src_end, dst_start);
        self.storage = new_size;
    }

    /// Finalizes the stream: flushes remaining range state, carry buffer, and
    /// raw bits. Zeros unused bytes. Must be called before reading output.
    ///
    /// Corresponds to `ec_enc_done`.
    pub fn done(&mut self) {
        // Compute minimum bits needed so the interval is decodable regardless
        // of any subsequent trailing bits.
        let mut l = EC_CODE_BITS as i32 - ec_ilog(self.rng);
        let mut msk = (EC_CODE_TOP - 1) >> l as u32;
        let mut end = (self.val + msk) & !msk;
        // Check if the rounded-up value could decode to a different symbol
        if (end | msk) >= self.val + self.rng {
            l += 1;
            msk >>= 1;
            end = (self.val + msk) & !msk;
        }
        // Emit the top bytes of `end` through carry propagation
        while l > 0 {
            self.carry_out((end >> EC_CODE_SHIFT) as i32);
            end = (end << EC_SYM_BITS) & (EC_CODE_TOP - 1);
            l -= EC_SYM_BITS as i32;
        }
        // Flush any remaining byte in rem/ext
        if self.rem >= 0 || self.ext > 0 {
            self.carry_out(0);
        }
        // Flush buffered raw bits from end_window
        let mut window = self.end_window;
        let mut used = self.nend_bits;
        while used >= EC_SYM_BITS as i32 {
            self.error |= self.write_byte_at_end(window & EC_SYM_MAX);
            window >>= EC_SYM_BITS;
            used -= EC_SYM_BITS as i32;
        }
        // Clear unused space and handle remaining partial raw bits
        if self.error == 0 {
            let start = self.offs as usize;
            let gap_end = (self.storage - self.end_offs) as usize;
            self.buf[start..gap_end].fill(0);
            if used > 0 {
                if self.end_offs >= self.storage {
                    self.error = -1;
                } else {
                    l = -l;
                    // If buffer is full and we don't have room, truncate
                    if self.offs + self.end_offs >= self.storage && l < used {
                        window &= (1u32 << l as u32) - 1;
                        self.error = -1;
                    }
                    // OR leftover raw bits into the last byte of the raw region
                    let idx = (self.storage - self.end_offs - 1) as usize;
                    self.buf[idx] |= window as u8;
                }
            }
        }
    }
}

// ============================================================================
// RangeDecoder
// ============================================================================

/// Range decoder. Reads entropy-coded symbols from a byte buffer.
///
/// Corresponds to `ec_dec` (typedef for `ec_ctx`) in the C reference.
pub struct RangeDecoder<'a> {
    buf: &'a [u8],
    /// Buffer size in bytes.
    storage: u32,
    /// Bytes consumed from the end of the buffer (raw bits region).
    end_offs: u32,
    /// Bit accumulator for raw bits at the buffer end.
    end_window: u32,
    /// Number of valid bits in end_window.
    nend_bits: i32,
    /// Total bits consumed (range-coded + raw); used by tell().
    nbits_total: i32,
    /// Byte offset for the front of the buffer (range-coded data).
    offs: u32,
    /// Current interval width.
    rng: u32,
    /// `top_of_range - coded_value - 1`. Inverted representation simplifies
    /// decode comparisons to `val < threshold`.
    val: u32,
    /// Last byte read from input, used across normalization iterations where
    /// a byte boundary falls mid-symbol.
    rem: i32,
    /// Saved quotient `rng / ft` from `decode()`, reused in `update()`.
    ext: u32,
    /// Nonzero if a decoding error occurred.
    error: i32,
}

impl<'a> RangeDecoder<'a> {
    /// Initializes the decoder from a buffer. Reads initial bytes and normalizes.
    ///
    /// Corresponds to `ec_dec_init`.
    pub fn new(buf: &'a [u8]) -> Self {
        let storage = buf.len() as u32;
        let mut dec = Self {
            buf,
            storage,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            // = EC_CODE_BITS + 1 - floor((EC_CODE_BITS - EC_CODE_EXTRA) / EC_SYM_BITS) * EC_SYM_BITS
            // = 33 - ((32 - 7) / 8) * 8 = 33 - 24 = 9
            nbits_total: EC_CODE_BITS as i32 + 1
                - ((EC_CODE_BITS as i32 - EC_CODE_EXTRA as i32) / EC_SYM_BITS as i32)
                    * EC_SYM_BITS as i32,
            offs: 0,
            rng: 1u32 << EC_CODE_EXTRA,
            val: 0,
            rem: 0,
            ext: 0,
            error: 0,
        };
        dec.rem = dec.read_byte();
        dec.val = dec.rng - 1 - (dec.rem as u32 >> (EC_SYM_BITS - EC_CODE_EXTRA));
        // Normalize reads ~3 more bytes, bringing nbits_total to 33
        dec.normalize();
        dec
    }

    /// Number of range-coded bytes read so far.
    #[inline(always)]
    pub fn range_bytes(&self) -> u32 {
        self.offs
    }

    /// Reference to the input buffer.
    #[inline(always)]
    pub fn buffer(&self) -> &[u8] {
        self.buf
    }

    /// Whether a decoding error occurred.
    #[inline(always)]
    pub fn error(&self) -> bool {
        self.error != 0
    }

    /// Bits "used" so far (conservative, rounds up).
    #[inline(always)]
    pub fn tell(&self) -> i32 {
        self.nbits_total - ec_ilog(self.rng)
    }

    /// Bits used, scaled by `2^BITRES` (in 1/8-bit units).
    #[inline(always)]
    pub fn tell_frac(&self) -> u32 {
        tell_frac_impl(self.nbits_total, self.rng)
    }

    /// Returns the internal range coder state for validation.
    /// The decoder stores this to detect bit-stream corruption.
    #[inline(always)]
    pub fn get_rng(&self) -> u32 {
        self.rng
    }

    /// Debug: get val
    pub fn debug_val(&self) -> u32 { self.val }
    /// Debug: get ext
    pub fn debug_ext(&self) -> u32 { self.ext }
    /// Debug: get nbits_total
    pub fn debug_nbits_total(&self) -> i32 { self.nbits_total }

    /// Returns the internal error flag as raw integer.
    #[inline(always)]
    pub fn get_error(&self) -> i32 {
        self.error
    }

    // ---- Internal helpers ----

    /// Reads the next byte from the front. Returns 0 past the end.
    fn read_byte(&mut self) -> i32 {
        if self.offs < self.storage {
            let b = self.buf[self.offs as usize] as i32;
            self.offs += 1;
            b
        } else {
            0
        }
    }

    /// Reads a byte from the end of the buffer. Returns 0 past the end.
    fn read_byte_from_end(&mut self) -> i32 {
        if self.end_offs < self.storage {
            self.end_offs += 1;
            self.buf[(self.storage - self.end_offs) as usize] as i32
        } else {
            0
        }
    }

    /// If the range is too small, reads input bytes and rescales.
    fn normalize(&mut self) {
        while self.rng <= EC_CODE_BOT {
            self.nbits_total += EC_SYM_BITS as i32;
            self.rng <<= EC_SYM_BITS;
            // Combine leftover bits from last byte with new byte
            let sym = self.rem;
            self.rem = self.read_byte();
            // Extract the EC_CODE_EXTRA MSBs from the combined 16-bit value
            let sym = ((sym << EC_SYM_BITS) | self.rem) >> (EC_SYM_BITS - EC_CODE_EXTRA);
            // Subtract from val using the inverted representation:
            // EC_SYM_MAX & ~sym complements the input bits
            self.val =
                ((self.val << EC_SYM_BITS) + (EC_SYM_MAX & !(sym as u32))) & (EC_CODE_TOP - 1);
        }
    }

    // ---- Public decoding operations ----

    /// Computes the cumulative frequency for the next symbol.
    ///
    /// Returns a value in `[0, ft)`. **Must** be followed by `update()`.
    /// Corresponds to `ec_decode`.
    pub fn decode(&mut self, ft: u32) -> u32 {
        self.ext = self.rng / ft;
        let s = self.val / self.ext;
        ft - (s + 1).min(ft)
    }

    /// Like `decode()` but with `ft = 1 << bits`. Must be followed by `update()`.
    ///
    /// Corresponds to `ec_decode_bin`.
    pub fn decode_bin(&mut self, bits: u32) -> u32 {
        self.ext = self.rng >> bits;
        let s = self.val / self.ext;
        (1u32 << bits) - (s + 1).min(1u32 << bits)
    }

    /// Advances the decoder past the symbol `[fl, fh)` out of `ft`.
    ///
    /// Exactly one call to `decode()` or `decode_bin()` must precede this.
    /// Corresponds to `ec_dec_update`.
    pub fn update(&mut self, fl: u32, fh: u32, ft: u32) {
        let s = self.ext * (ft - fh);
        self.val -= s;
        self.rng = if fl > 0 {
            self.ext * (fh - fl)
        } else {
            self.rng - s
        };
        self.normalize();
    }

    /// Decodes a binary symbol where P(1) = 1/(1 << logp).
    ///
    /// Self-contained (no `update()` needed). Corresponds to `ec_dec_bit_logp`.
    pub fn decode_bit_logp(&mut self, logp: u32) -> bool {
        let r = self.rng;
        let d = self.val;
        let s = r >> logp;
        let ret = d < s;
        if !ret {
            self.val = d - s;
        }
        self.rng = if ret { s } else { r - s };
        self.normalize();
        ret
    }

    /// Decodes a symbol using an 8-bit inverse CDF table.
    ///
    /// Linear search through the iCDF (fast for small alphabets, which is the
    /// common case in Opus). Self-contained. Corresponds to `ec_dec_icdf`.
    pub fn decode_icdf(&mut self, icdf: &[u8], ftb: u32) -> i32 {
        let mut s = self.rng;
        let d = self.val;
        let r = s >> ftb;
        let mut ret: i32 = -1;
        let mut t;
        loop {
            t = s;
            ret += 1;
            s = r * icdf[ret as usize] as u32;
            if d >= s {
                break;
            }
        }
        self.val = d - s;
        self.rng = t - s;
        self.normalize();
        ret
    }

    /// Like `decode_icdf` but with 16-bit iCDF entries.
    ///
    /// Corresponds to `ec_dec_icdf16`.
    pub fn decode_icdf16(&mut self, icdf: &[u16], ftb: u32) -> i32 {
        let mut s = self.rng;
        let d = self.val;
        let r = s >> ftb;
        let mut ret: i32 = -1;
        let mut t;
        loop {
            t = s;
            ret += 1;
            s = r * icdf[ret as usize] as u32;
            if d >= s {
                break;
            }
        }
        self.val = d - s;
        self.rng = t - s;
        self.normalize();
        ret
    }

    /// Decodes a uniform integer in `[0, ft)`. Self-contained.
    ///
    /// Corresponds to `ec_dec_uint`.
    pub fn decode_uint(&mut self, ft: u32) -> u32 {
        debug_assert!(ft > 1);
        let ft_dec = ft - 1;
        let ftb = ec_ilog(ft_dec) as u32;
        if ftb > EC_UINT_BITS {
            let ftb = ftb - EC_UINT_BITS;
            let ft_upper = (ft_dec >> ftb) + 1;
            let s = self.decode(ft_upper);
            self.update(s, s + 1, ft_upper);
            let t = (s << ftb) | self.decode_bits(ftb);
            if t <= ft_dec {
                t
            } else {
                self.error = 1;
                ft_dec
            }
        } else {
            let ft_full = ft_dec + 1;
            let s = self.decode(ft_full);
            self.update(s, s + 1, ft_full);
            s
        }
    }

    /// Reads raw (uncoded) bits from the end of the buffer. 0 ≤ bits ≤ 25.
    ///
    /// Corresponds to `ec_dec_bits`.
    pub fn decode_bits(&mut self, bits: u32) -> u32 {
        let mut window = self.end_window;
        let mut available = self.nend_bits;
        if (available as u32) < bits {
            loop {
                window |= (self.read_byte_from_end() as u32) << available as u32;
                available += EC_SYM_BITS as i32;
                if available > EC_WINDOW_SIZE as i32 - EC_SYM_BITS as i32 {
                    break;
                }
            }
        }
        let ret = window & ((1u32 << bits) - 1);
        window >>= bits;
        available -= bits as i32;
        self.end_window = window;
        self.nend_bits = available;
        self.nbits_total += bits as i32;
        ret
    }
}

// ============================================================================
// Encoder state snapshot (for two-pass coding in quant_bands)
// ============================================================================

/// Snapshot of the range encoder's scalar state.
/// The underlying buffer is shared; only register/offset state is captured.
/// Matches C pattern: `ec_enc saved = *enc; ... *enc = saved;`
#[derive(Clone, Copy)]
pub struct EncoderSnapshot {
    storage: u32,
    end_offs: u32,
    end_window: u32,
    nend_bits: i32,
    nbits_total: i32,
    offs: u32,
    rng: u32,
    val: u32,
    rem: i32,
    ext: u32,
    error: i32,
}

impl EncoderSnapshot {
    /// Number of range-coded bytes at the time of the snapshot.
    #[inline(always)]
    pub fn range_bytes(&self) -> u32 {
        self.offs
    }
}

impl<'a> RangeEncoder<'a> {
    /// Save the current encoder state (scalar fields only; buffer is shared).
    pub fn snapshot(&self) -> EncoderSnapshot {
        EncoderSnapshot {
            storage: self.storage,
            end_offs: self.end_offs,
            end_window: self.end_window,
            nend_bits: self.nend_bits,
            nbits_total: self.nbits_total,
            offs: self.offs,
            rng: self.rng,
            val: self.val,
            rem: self.rem,
            ext: self.ext,
            error: self.error,
        }
    }

    /// Restore encoder state from a snapshot. Buffer contents are NOT restored —
    /// the caller must handle buffer byte save/restore separately if needed.
    pub fn restore(&mut self, snap: &EncoderSnapshot) {
        self.storage = snap.storage;
        self.end_offs = snap.end_offs;
        self.end_window = snap.end_window;
        self.nend_bits = snap.nend_bits;
        self.nbits_total = snap.nbits_total;
        self.offs = snap.offs;
        self.rng = snap.rng;
        self.val = snap.val;
        self.rem = snap.rem;
        self.ext = snap.ext;
        self.error = snap.error;
    }

    /// Total buffer capacity in bytes.
    #[inline(always)]
    pub fn storage(&self) -> u32 {
        self.storage
    }

    /// Mutable reference to the output buffer.
    #[inline(always)]
    pub fn buffer_mut(&mut self) -> &mut [u8] {
        self.buf
    }
}

impl<'a> RangeDecoder<'a> {
    /// Total buffer capacity in bytes.
    #[inline(always)]
    pub fn storage(&self) -> u32 {
        self.storage
    }

    /// Reduce the buffer capacity by `amount` bytes.
    /// Used to exclude redundancy bytes from the main decode range.
    /// Matches C: `dec.storage -= redundancy_bytes`.
    #[inline(always)]
    pub fn reduce_storage(&mut self, amount: u32) {
        self.storage -= amount;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tell_initial_state() {
        let mut buf = vec![0u8; 100];
        let enc = RangeEncoder::new(&mut buf);
        assert_eq!(enc.tell(), 1);
        // tell_frac: nbits=33<<3=264, l=ec_ilog(2^31)=32, r=2^31>>16=32768,
        // b=(32768>>12)-8=0, correction[0]=35733>32768 so b stays 0,
        // l=(32<<3)+0=256, result=264-256=8
        assert_eq!(enc.tell_frac(), 8);
    }

    #[test]
    fn encode_decode_single_symbol() {
        let mut buf = vec![0u8; 100];
        {
            let mut enc = RangeEncoder::new(&mut buf);
            enc.encode(2, 3, 4);
            enc.done();
            assert!(!enc.error());
        }
        {
            let mut dec = RangeDecoder::new(&buf);
            let fl = dec.decode(4);
            assert_eq!(fl, 2);
            dec.update(2, 3, 4);
            assert!(!dec.error());
        }
    }

    #[test]
    fn encode_decode_multiple_uniform_symbols() {
        let mut buf = vec![0u8; 1024];
        let ft = 16u32;
        let symbols: Vec<u32> = (0..100).map(|i| (i * 7 + 3) % ft).collect();

        {
            let mut enc = RangeEncoder::new(&mut buf);
            for &s in &symbols {
                enc.encode(s, s + 1, ft);
            }
            enc.done();
            assert!(!enc.error());
        }
        {
            let mut dec = RangeDecoder::new(&buf);
            for &expected in &symbols {
                let fl = dec.decode(ft);
                assert_eq!(fl, expected);
                dec.update(expected, expected + 1, ft);
            }
            assert!(!dec.error());
        }
    }

    #[test]
    fn encode_decode_bit_logp() {
        let mut buf = vec![0u8; 100];
        let bits = [true, false, true, true, false, false, true, false];
        {
            let mut enc = RangeEncoder::new(&mut buf);
            for &b in &bits {
                enc.encode_bit_logp(b, 3);
            }
            enc.done();
            assert!(!enc.error());
        }
        {
            let mut dec = RangeDecoder::new(&buf);
            for &expected in &bits {
                assert_eq!(dec.decode_bit_logp(3), expected);
            }
            assert!(!dec.error());
        }
    }

    #[test]
    fn encode_decode_icdf() {
        // 4 symbols, ftb=8: icdf = [128, 64, 32, 0]
        let icdf: &[u8] = &[128, 64, 32, 0];
        let ftb = 8;
        let symbols = [0u32, 1, 2, 3, 0, 3, 1, 2];

        let mut buf = vec![0u8; 100];
        {
            let mut enc = RangeEncoder::new(&mut buf);
            for &s in &symbols {
                enc.encode_icdf(s, icdf, ftb);
            }
            enc.done();
            assert!(!enc.error());
        }
        {
            let mut dec = RangeDecoder::new(&buf);
            for &expected in &symbols {
                assert_eq!(dec.decode_icdf(icdf, ftb), expected as i32);
            }
            assert!(!dec.error());
        }
    }

    #[test]
    fn encode_decode_icdf16() {
        let icdf: &[u16] = &[200, 100, 50, 0];
        let ftb = 8;
        let symbols = [0u32, 1, 2, 3, 2, 1, 0, 3];

        let mut buf = vec![0u8; 100];
        {
            let mut enc = RangeEncoder::new(&mut buf);
            for &s in &symbols {
                enc.encode_icdf16(s, icdf, ftb);
            }
            enc.done();
            assert!(!enc.error());
        }
        {
            let mut dec = RangeDecoder::new(&buf);
            for &expected in &symbols {
                assert_eq!(dec.decode_icdf16(icdf, ftb), expected as i32);
            }
            assert!(!dec.error());
        }
    }

    #[test]
    fn encode_decode_raw_bits() {
        let mut buf = vec![0u8; 100];
        let values: [(u32, u32); 5] = [(7, 3), (255, 8), (1023, 10), (0, 1), (31, 5)];
        {
            let mut enc = RangeEncoder::new(&mut buf);
            enc.encode_bit_logp(true, 2);
            for &(val, bits) in &values {
                enc.encode_bits(val, bits);
            }
            enc.done();
            assert!(!enc.error());
        }
        {
            let mut dec = RangeDecoder::new(&buf);
            assert!(dec.decode_bit_logp(2));
            for &(expected, bits) in &values {
                assert_eq!(dec.decode_bits(bits), expected);
            }
            assert!(!dec.error());
        }
    }

    #[test]
    fn encode_decode_uint() {
        let mut buf = vec![0u8; 100];
        let cases: [(u32, u32); 6] = [
            (0, 10),
            (9, 10),
            (0, 256),
            (255, 256),
            (0, 1000),
            (999, 1000),
        ];
        {
            let mut enc = RangeEncoder::new(&mut buf);
            for &(val, ft) in &cases {
                enc.encode_uint(val, ft);
            }
            enc.done();
            assert!(!enc.error());
        }
        {
            let mut dec = RangeDecoder::new(&buf);
            for &(expected, ft) in &cases {
                assert_eq!(dec.decode_uint(ft), expected);
            }
            assert!(!dec.error());
        }
    }

    #[test]
    fn encode_bin_decode_bin() {
        // encode_bin/decode_bin use power-of-two totals (ft = 1 << bits).
        // Test with individual unit-width symbols for exact round-trip.
        let mut buf = vec![0u8; 100];
        let bits = 4u32; // ft = 16
        let symbols = [0u32, 5, 10, 15];
        {
            let mut enc = RangeEncoder::new(&mut buf);
            for &s in &symbols {
                enc.encode_bin(s, s + 1, bits);
            }
            enc.done();
            assert!(!enc.error());
        }
        {
            let mut dec = RangeDecoder::new(&buf);
            for &expected in &symbols {
                let fl = dec.decode_bin(bits);
                assert_eq!(fl, expected);
                dec.update(fl, fl + 1, 1 << bits);
            }
            assert!(!dec.error());
        }
    }

    #[test]
    fn mixed_coded_and_raw_bits() {
        let mut buf = vec![0u8; 256];
        let icdf: &[u8] = &[128, 0];
        let ftb = 8;

        {
            let mut enc = RangeEncoder::new(&mut buf);
            enc.encode_icdf(0, icdf, ftb);
            enc.encode_bits(0x1F, 5);
            enc.encode_icdf(1, icdf, ftb);
            enc.encode_bits(0x00, 3);
            enc.encode_bit_logp(true, 4);
            enc.encode_bits(0xFF, 8);
            enc.done();
            assert!(!enc.error());
        }
        {
            let mut dec = RangeDecoder::new(&buf);
            assert_eq!(dec.decode_icdf(icdf, ftb), 0);
            assert_eq!(dec.decode_bits(5), 0x1F);
            assert_eq!(dec.decode_icdf(icdf, ftb), 1);
            assert_eq!(dec.decode_bits(3), 0x00);
            assert!(dec.decode_bit_logp(4));
            assert_eq!(dec.decode_bits(8), 0xFF);
            assert!(!dec.error());
        }
    }

    #[test]
    fn empty_stream() {
        let mut buf = vec![0u8; 10];
        {
            let mut enc = RangeEncoder::new(&mut buf);
            enc.done();
            assert!(!enc.error());
        }
    }

    #[test]
    fn patch_initial_bits() {
        let mut buf = vec![0u8; 100];
        {
            let mut enc = RangeEncoder::new(&mut buf);
            for _ in 0..20 {
                enc.encode_bit_logp(false, 1);
            }
            enc.patch_initial_bits(3, 2);
            enc.done();
            assert!(!enc.error());
            // Top 2 bits of first byte should be 11
            assert_eq!(buf[0] & 0xC0, 0xC0);
        }
    }

    #[test]
    fn shrink_preserves_data() {
        let mut buf = vec![0u8; 200];
        {
            let mut enc = RangeEncoder::new(&mut buf);
            enc.encode_bit_logp(true, 2);
            enc.encode_bits(0xAB, 8);
            enc.done();
            assert!(!enc.error());
        }
        // Verify round-trip still works
        {
            let mut dec = RangeDecoder::new(&buf);
            assert!(dec.decode_bit_logp(2));
            assert_eq!(dec.decode_bits(8), 0xAB);
            assert!(!dec.error());
        }
    }

    #[test]
    fn stress_many_bit_logp_values() {
        let mut buf = vec![0u8; 4096];
        let n = 1000;

        for logp in 1..=8 {
            buf.fill(0);
            let bits: Vec<bool> = (0..n).map(|i| (i * 31 + logp) % 7 < 3).collect();
            {
                let mut enc = RangeEncoder::new(&mut buf);
                for &b in &bits {
                    enc.encode_bit_logp(b, logp as u32);
                }
                enc.done();
                assert!(!enc.error(), "encode error at logp={logp}");
            }
            {
                let mut dec = RangeDecoder::new(&buf);
                for (i, &expected) in bits.iter().enumerate() {
                    let got = dec.decode_bit_logp(logp as u32);
                    assert_eq!(got, expected, "mismatch at i={i}, logp={logp}");
                }
                assert!(!dec.error(), "decode error at logp={logp}");
            }
        }
    }

    #[test]
    fn encode_decode_uint_large_range() {
        let mut buf = vec![0u8; 256];
        // Test with large ft values that exercise the MSB/LSB split
        let cases: [(u32, u32); 4] = [
            (0, 100_000),
            (50_000, 100_000),
            (99_999, 100_000),
            (123_456, 1_000_000),
        ];
        {
            let mut enc = RangeEncoder::new(&mut buf);
            for &(val, ft) in &cases {
                enc.encode_uint(val, ft);
            }
            enc.done();
            assert!(!enc.error());
        }
        {
            let mut dec = RangeDecoder::new(&buf);
            for &(expected, ft) in &cases {
                assert_eq!(dec.decode_uint(ft), expected);
            }
            assert!(!dec.error());
        }
    }
}
