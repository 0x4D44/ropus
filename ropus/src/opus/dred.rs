//! Public DRED decoder API — mirrors `OpusDREDDecoder` in
//! `reference/src/opus_decoder.c:1369-1600`.
//!
//! C allocates the decoder behind `opus_dred_decoder_get_size` +
//! `opus_dred_decoder_init` (reuse-in-place pattern), and the payload state
//! behind `opus_dred_alloc` + `opus_dred_free`. Rust maps both to plain
//! owned structs: `OpusDREDDecoder::new()` for the model-holding handle and
//! `OpusDred::default()` (already defined in `dnn::dred`) for payload state.
//! No `_get_size`/`_init` split is needed because Rust's ownership covers
//! the lifetime story the C API encoded into two functions.
//!
//! Scope for Stage 8.7: `parse` (finds the DRED extension in a packet and
//! runs `dred_ec_decode`) + `process` (seeds the RDOVAE decoder state from
//! the parsed payload and emits `fec_features`). The audio-reconstruction
//! last mile (features → PCM via FARGAN) is Stage 8.9 territory and is NOT
//! exposed here.

use crate::dnn::core::parse_weights;
use crate::dnn::dred::{
    DRED_EXPERIMENTAL_BYTES, DRED_EXPERIMENTAL_VERSION, DRED_EXTENSION_ID, DRED_LATENT_DIM,
    DRED_NUM_FEATURES, DRED_NUM_REDUNDANCY_FRAMES, OpusDred, RDOVAEDec, RDOVAEDecState,
    init_rdovaedec,
};
use crate::dnn::embedded_weights::WEIGHTS_BLOB;
use crate::opus::decoder::{
    MAX_FRAMES, OPUS_BAD_ARG, OPUS_OK, OPUS_UNIMPLEMENTED, PaddingInfo,
    opus_packet_get_samples_per_frame, opus_packet_parse_impl_with_padding,
};
use crate::opus::repacketizer::OpusExtensionIterator;

/// Public DRED decoder handle. Holds the RDOVAE decoder model populated
/// from a weight blob. Mirrors `struct OpusDREDDecoder` in
/// `reference/src/opus_decoder.c:1339`.
pub struct OpusDREDDecoder {
    pub(crate) model: RDOVAEDec,
    pub(crate) loaded: bool,
}

impl Default for OpusDREDDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl OpusDREDDecoder {
    /// Construct a fresh DRED decoder. If the crate-embedded weight blob
    /// carries the RDOVAE decoder layers, the model is populated and
    /// `loaded()` returns `true`. Otherwise the decoder is unloaded and
    /// `parse` / `process` will return `OPUS_UNIMPLEMENTED` until a blob
    /// is supplied via `load_model`.
    ///
    /// Maps the C pair `opus_dred_decoder_get_size` +
    /// `opus_dred_decoder_init` / `_create` into a single Rust
    /// constructor, following the same pattern stage 7b.1.5 adopted for
    /// `OpusDecoder`.
    pub fn new() -> Self {
        let mut dec = Self {
            model: RDOVAEDec::default(),
            loaded: false,
        };
        if !WEIGHTS_BLOB.is_empty() {
            let _ = dec.load_model(WEIGHTS_BLOB);
        }
        dec
    }

    /// Whether an RDOVAE model is populated and ready to decode.
    pub fn loaded(&self) -> bool {
        self.loaded
    }

    /// Parse a weight blob into the RDOVAE decoder model. Mirrors C
    /// `dred_decoder_load_model` in `opus_decoder.c:1369`. Returns
    /// `OPUS_OK` on success or `OPUS_BAD_ARG` on blob errors.
    pub fn load_model(&mut self, data: &[u8]) -> i32 {
        match parse_weights(data) {
            Ok(arrays) => match init_rdovaedec(&arrays) {
                Ok(model) => {
                    self.model = model;
                    self.loaded = true;
                    OPUS_OK
                }
                Err(()) => OPUS_BAD_ARG,
            },
            Err(_) => OPUS_BAD_ARG,
        }
    }

    /// Reset the decoder to an unloaded state. No direct C counterpart —
    /// C relies on the caller to `destroy` + `create` instead. Here we
    /// expose a cheap handle re-use primitive for test / CLI loops.
    pub fn reset(&mut self) {
        self.model = RDOVAEDec::default();
        self.loaded = false;
    }

    /// Locate the DRED extension payload inside an Opus packet and run
    /// `dred_ec_decode`. Mirrors C `opus_dred_parse` in
    /// `opus_decoder.c:1548` (sans the `defer_processing` knob — we always
    /// return the parsed struct and let the caller decide whether to
    /// process, matching the task brief for Stage 8.7).
    ///
    /// Rust `parse` always defers processing; call `process(...)` explicitly.
    /// C's `defer_processing=0` case is not provided.
    ///
    /// Returns a populated `OpusDred` (with `process_stage = 1`) or a
    /// negative `OPUS_*` error code. `OPUS_UNIMPLEMENTED` when the
    /// decoder is unloaded; `0` dred-sample-count ⇒ caller sees empty
    /// `nb_latents`. Matches C semantics for packets lacking a DRED
    /// extension. Callers needing the C return value (usable-samples
    /// count) should compute it via [`OpusDred::usable_samples`].
    ///
    /// `max_dred_samples` and `sampling_rate` are forwarded from
    /// `opus_dred_parse` and gate how many feature frames we're willing
    /// to decode (`min_feature_frames`).
    pub fn parse(
        &self,
        data: &[u8],
        max_dred_samples: i32,
        sampling_rate: i32,
    ) -> Result<OpusDred, i32> {
        if !self.loaded {
            return Err(OPUS_UNIMPLEMENTED);
        }
        // `process_stage = -1` mirrors C `dred->process_stage = -1` at the
        // top of `opus_dred_parse` — marks the struct as "parse attempted
        // but not yet decoded" so a process() call on an untouched OpusDred
        // returns OPUS_BAD_ARG rather than silently running the RDOVAE on
        // zeroed state.
        let mut dred = OpusDred {
            process_stage: -1,
            ..OpusDred::default()
        };

        match dred_find_payload(data) {
            FindPayload::Error(e) => Err(e),
            FindPayload::Missing => Ok(dred),
            FindPayload::Found {
                payload,
                dred_frame_offset,
            } => {
                let offset = 100 * max_dred_samples / sampling_rate;
                let min_feature_frames = (2 + offset).min(2 * DRED_NUM_REDUNDANCY_FRAMES as i32);
                dred.ec_decode(payload, min_feature_frames, dred_frame_offset);
                Ok(dred)
            }
        }
    }

    /// Run the RDOVAE decoder forward pass to populate
    /// `dred.fec_features` from `dred.state` + `dred.latents`. Mirrors C
    /// `opus_dred_process` in `opus_decoder.c:1587`.
    ///
    /// Returns `OPUS_OK` on success, `OPUS_BAD_ARG` if the payload isn't
    /// in a valid processing state, or `OPUS_UNIMPLEMENTED` when the
    /// decoder is unloaded. If the payload is already at
    /// `process_stage == 2` (processed once already), the call is a
    /// no-op that returns `OPUS_OK`.
    pub fn process(&self, dred: &mut OpusDred) -> i32 {
        if !self.loaded {
            return OPUS_UNIMPLEMENTED;
        }
        if dred.process_stage != 1 && dred.process_stage != 2 {
            return OPUS_BAD_ARG;
        }
        if dred.process_stage == 2 {
            return OPUS_OK;
        }
        // Mirrors C `DRED_rdovae_decode_all`: zero a fresh RDOVAEDecState,
        // seed GRUs from `dred.state`, then walk the latents newest→oldest
        // writing into `fec_features`. Each qframe is 4×DRED_NUM_FEATURES
        // floats (80 total); chunk `i` writes to offset `2*i*DRED_NUM_FEATURES`
        // where the outer loop walks `i = 0, 2, 4, ..., 2*nb_latents-2`.
        let mut state = RDOVAEDecState::default();
        state.init_states(&self.model, &dred.state);
        let nb = dred.nb_latents as usize;
        for k in 0..nb {
            let out_offset = 4 * k * DRED_NUM_FEATURES;
            let in_base = k * (DRED_LATENT_DIM + 1);
            let latent = {
                let mut tmp = [0.0f32; DRED_LATENT_DIM + 1];
                tmp.copy_from_slice(&dred.latents[in_base..in_base + DRED_LATENT_DIM + 1]);
                tmp
            };
            state.decode_qframe(
                &self.model,
                &mut dred.fec_features[out_offset..out_offset + 4 * DRED_NUM_FEATURES],
                &latent,
            );
        }
        dred.process_stage = 2;
        OPUS_OK
    }

    // =======================================================================
    // Stage 8.9 — FARGAN reconstruction seam
    // =======================================================================
    //
    // The three `decode*` methods below mirror C's
    // `opus_decoder_dred_decode{,24,_float}` in
    // `reference/src/opus_decoder.c:1609-1691`. C routes through
    // `opus_decode_native(..., dred, dred_offset)`, which then calls
    // `fargan_synthesise_frame` on the DRED-derived features to produce
    // PCM. Until Stage 7 ports FARGAN (see
    // `wrk_docs/2026.04.19 - HLD - dred-port.md` §Stage 7 coordination
    // and §Staging row 8.9), the audio-reconstruction last mile is
    // unavailable. The joint Stage 7 follow-up commit replaces the
    // `Err(OPUS_UNIMPLEMENTED)` return in each body with the actual
    // FARGAN-backed decode path.
    //
    // C places these methods on `OpusDecoder` (they need the classical
    // decoder's channel count, preemph state, resampler state, etc. for
    // the final render). Pending Stage 7, they live here on
    // `OpusDREDDecoder` so the seam stays scoped to `opus/dred.rs`; the
    // follow-up commit is free to relocate them onto `OpusDecoder` when
    // the real body lands.

    /// Reconstruct 16-bit PCM from a parsed DRED payload. Stage 8.9 stub.
    ///
    /// Will eventually drive FARGAN synthesis from `dred.fec_features`
    /// and render `frame_size` samples per channel into `pcm`. Currently
    /// returns [`OPUS_UNIMPLEMENTED`] because FARGAN is Stage 7's
    /// deliverable; see `wrk_docs/2026.04.19 - HLD - dred-port.md`
    /// §Stage 7 coordination.
    ///
    /// Mirrors C `opus_decoder_dred_decode` at
    /// `reference/src/opus_decoder.c:1609`.
    pub fn decode(
        &self,
        _dred: &OpusDred,
        _dred_offset: i32,
        _pcm: &mut [i16],
        _frame_size: i32,
    ) -> Result<i32, i32> {
        Err(OPUS_UNIMPLEMENTED)
    }

    /// Reconstruct 24-bit PCM (packed into `i32`) from a parsed DRED
    /// payload. Stage 8.9 stub.
    ///
    /// Will eventually drive FARGAN synthesis from `dred.fec_features`
    /// and render `frame_size` samples per channel into `pcm`. Currently
    /// returns [`OPUS_UNIMPLEMENTED`] because FARGAN is Stage 7's
    /// deliverable; see `wrk_docs/2026.04.19 - HLD - dred-port.md`
    /// §Stage 7 coordination.
    ///
    /// Mirrors C `opus_decoder_dred_decode24` at
    /// `reference/src/opus_decoder.c:1643`.
    pub fn decode24(
        &self,
        _dred: &OpusDred,
        _dred_offset: i32,
        _pcm: &mut [i32],
        _frame_size: i32,
    ) -> Result<i32, i32> {
        Err(OPUS_UNIMPLEMENTED)
    }

    /// Reconstruct floating-point PCM from a parsed DRED payload. Stage
    /// 8.9 stub.
    ///
    /// Will eventually drive FARGAN synthesis from `dred.fec_features`
    /// and render `frame_size` samples per channel into `pcm`. Currently
    /// returns [`OPUS_UNIMPLEMENTED`] because FARGAN is Stage 7's
    /// deliverable; see `wrk_docs/2026.04.19 - HLD - dred-port.md`
    /// §Stage 7 coordination.
    ///
    /// Mirrors C `opus_decoder_dred_decode_float` at
    /// `reference/src/opus_decoder.c:1677`.
    pub fn decode_float(
        &self,
        _dred: &OpusDred,
        _dred_offset: i32,
        _pcm: &mut [f32],
        _frame_size: i32,
    ) -> Result<i32, i32> {
        Err(OPUS_UNIMPLEMENTED)
    }
}

// ===========================================================================
// DRED extension locator
// ===========================================================================

enum FindPayload<'a> {
    /// Valid packet, no DRED extension present (caller gets an empty OpusDred).
    Missing,
    /// Valid DRED payload located inside the packet.
    Found {
        payload: &'a [u8],
        dred_frame_offset: i32,
    },
    /// Packet parse error — propagate the negative `OPUS_*` code.
    Error(i32),
}

/// Port of `dred_find_payload` in `opus_decoder.c:1466`. Walks the packet's
/// padding region looking for an `DRED_EXTENSION_ID` extension whose
/// 2-byte experimental prefix matches (`'D', DRED_EXPERIMENTAL_VERSION`).
/// Returns the payload bytes with the prefix stripped, plus the DRED
/// offset within the packet (in 2.5 ms units).
fn dred_find_payload(data: &[u8]) -> FindPayload<'_> {
    let mut toc: u8 = 0;
    let mut sizes = [0i16; MAX_FRAMES];
    let mut payload_offset: i32 = 0;
    let mut padding = PaddingInfo { offset: 0, len: 0 };
    let ret = opus_packet_parse_impl_with_padding(
        data,
        data.len() as i32,
        false,
        &mut toc,
        &mut sizes,
        &mut payload_offset,
        None,
        Some(&mut padding),
    );
    if ret < 0 {
        return FindPayload::Error(ret);
    }
    let nb_frames = ret;
    let frame_size = opus_packet_get_samples_per_frame(data, 48000);
    if padding.len == 0 {
        return FindPayload::Missing;
    }
    let pad_start = padding.offset;
    let pad_end = pad_start + padding.len as usize;
    if pad_end > data.len() {
        return FindPayload::Missing;
    }
    let mut iter = OpusExtensionIterator::new(&data[pad_start..pad_end], padding.len, nb_frames);
    loop {
        let (r, ext) = iter.find(DRED_EXTENSION_ID as i32);
        if r < 0 {
            return FindPayload::Error(r);
        }
        if r == 0 {
            return FindPayload::Missing;
        }
        // DRED position in the packet, in units of 2.5 ms like for the
        // signalled DRED offset. Matches C.
        let dred_frame_offset = ext.frame * frame_size / 120;
        if ext.len as usize > DRED_EXPERIMENTAL_BYTES
            && ext.data[0] == b'D'
            && ext.data[1] as u32 == DRED_EXPERIMENTAL_VERSION
        {
            let payload = &ext.data[DRED_EXPERIMENTAL_BYTES..];
            return FindPayload::Found {
                payload,
                dred_frame_offset,
            };
        }
        // Not our version — keep looking.
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dnn::dred::{
        DRED_MAX_DATA_SIZE, DRED_MAX_FRAMES, DRED_STATE_DIM, DREDEnc, compute_quantizer,
    };
    use crate::dnn::dred_stats::{
        dred_state_dead_zone_q8, dred_state_p0_q8, dred_state_quant_scales_q8, dred_state_r_q8,
    };

    /// Stage 8.7 Rust round-trip at the `OpusDRED` level — same oracle as
    /// the in-`dnn::dred` round-trip but routed through the public
    /// `OpusDREDDecoder` surface (via direct `ec_decode` entry, since we
    /// feed the payload bytes directly without an Opus packet wrapper).
    /// Exists so future refactors of `parse`/`process` can't silently
    /// break the decoder's core contract.
    #[test]
    fn opus_dred_decoder_parses_rust_emitted_payload() {
        // Skip if weights aren't available: the test builds a payload with
        // the Rust encoder (which doesn't need weights — we set buffers
        // directly), decodes, and runs `process` (which does need the
        // RDOVAE decoder weights).
        if WEIGHTS_BLOB.is_empty() {
            eprintln!("opus::dred::tests: WEIGHTS_BLOB empty — skipping decoder round-trip test.");
            return;
        }

        const NUM_CHUNKS: i32 = 4;
        let q0: i32 = 6;
        let d_q: i32 = 0;
        let qmax: i32 = 15;

        // Build a DREDEnc without running RDOVAE — we seed buffers by hand.
        let mut enc = DREDEnc::new_unloaded(48000, 1);
        enc.loaded = true;
        enc.latent_offset = 0;
        enc.latents_buffer_fill = 2 * NUM_CHUNKS + 2;
        enc.dred_offset = 0;
        enc.last_extra_dred_offset = 0;

        let mut seed = 0xBADF00D_u32;
        let xsf = |s: &mut u32| {
            *s ^= *s << 13;
            *s ^= *s >> 17;
            *s ^= *s << 5;
            let v = (*s as i32) as f64 / (i32::MAX as f64);
            (v * 2.5) as f32
        };

        for i in 0..DRED_STATE_DIM {
            enc.state_buffer[i] = xsf(&mut seed);
        }
        for chunk in 0..(NUM_CHUNKS as usize) {
            for i in 0..DRED_LATENT_DIM {
                let x = xsf(&mut seed);
                enc.latents_buffer[2 * chunk * DRED_LATENT_DIM + i] = x;
            }
        }
        let activity_mem = vec![1u8; 4 * DRED_MAX_FRAMES];

        // Snapshot what the encoder *would* write per-dim, using the
        // same deadzone+tanh+round path as `dred_encode_latents`.
        fn enc_q(x: &[f32], scale: &[u8], dzone: &[u8], r: &[u8], p0: &[u8]) -> Vec<i32> {
            let dim = x.len();
            let eps = 0.1f32;
            let mut out = vec![0i32; dim];
            for i in 0..dim {
                let delta = dzone[i] as f32 / 256.0;
                let xq = x[i] * scale[i] as f32 / 256.0;
                let dz = (xq / (delta + eps)).tanh();
                let adjusted = xq - delta * dz;
                let qi = (0.5 + adjusted).floor() as i32;
                out[i] = if r[i] == 0 || p0[i] == 255 { 0 } else { qi };
            }
            out
        }
        fn deq(q: i32, scale: u8) -> f32 {
            let s = if scale == 0 { 1.0 } else { scale as f32 };
            (q as f32) * 256.0 / s
        }

        let state_qoffset = (q0 as usize) * DRED_STATE_DIM;
        let mut seed_state = [0.0f32; DRED_STATE_DIM];
        seed_state.copy_from_slice(&enc.state_buffer[..DRED_STATE_DIM]);
        let sq = enc_q(
            &seed_state,
            &dred_state_quant_scales_q8[state_qoffset..state_qoffset + DRED_STATE_DIM],
            &dred_state_dead_zone_q8[state_qoffset..state_qoffset + DRED_STATE_DIM],
            &dred_state_r_q8[state_qoffset..state_qoffset + DRED_STATE_DIM],
            &dred_state_p0_q8[state_qoffset..state_qoffset + DRED_STATE_DIM],
        );
        let expected_state: Vec<f32> = (0..DRED_STATE_DIM)
            .map(|i| deq(sq[i], dred_state_quant_scales_q8[state_qoffset + i]))
            .collect();

        // Encode to bytes.
        let mut payload = vec![0u8; DRED_MAX_DATA_SIZE];
        let nbytes = enc.encode_silk_frame(
            &mut payload,
            NUM_CHUNKS,
            DRED_MAX_DATA_SIZE,
            q0,
            d_q,
            qmax,
            &activity_mem,
        );
        assert!(nbytes > 0);

        // Decode via OpusDred::ec_decode directly.
        let mut dred = OpusDred::default();
        let nb = dred.ec_decode(&payload[..nbytes as usize], 2 * NUM_CHUNKS, 0);
        assert!(nb >= 1);
        assert_eq!(dred.process_stage, 1);

        for i in 0..DRED_STATE_DIM {
            assert_eq!(
                dred.state[i].to_bits(),
                expected_state[i].to_bits(),
                "state[{i}] mismatch via OpusDREDDecoder path",
            );
        }

        // Run `process` through the public decoder handle.
        let decoder = OpusDREDDecoder::new();
        assert!(
            decoder.loaded(),
            "embedded DRED decoder weights should be present when WEIGHTS_BLOB is nonempty"
        );
        let ret = decoder.process(&mut dred);
        assert_eq!(ret, OPUS_OK);
        assert_eq!(dred.process_stage, 2);
        // Rerun should be a no-op.
        assert_eq!(decoder.process(&mut dred), OPUS_OK);
        assert_eq!(dred.process_stage, 2);
    }

    #[test]
    fn opus_dred_decoder_reset_clears_loaded() {
        let mut dec = OpusDREDDecoder::new();
        let was_loaded = dec.loaded();
        dec.reset();
        assert!(!dec.loaded());
        // Re-load returns back to the original state (matches construction).
        if was_loaded {
            assert_eq!(dec.load_model(WEIGHTS_BLOB), OPUS_OK);
            assert!(dec.loaded());
        }
    }

    #[test]
    fn opus_dred_decoder_rejects_empty_blob() {
        let mut dec = OpusDREDDecoder::new();
        dec.reset();
        assert_eq!(dec.load_model(&[]), OPUS_BAD_ARG);
        assert!(!dec.loaded());
    }

    #[test]
    fn parse_unloaded_returns_unimplemented() {
        let mut dec = OpusDREDDecoder::new();
        dec.reset();
        // A dummy 1-byte packet — we won't get past the loaded check.
        let packet = [0x00u8; 1];
        let err = dec.parse(&packet, 1920, 48000).unwrap_err();
        assert_eq!(err, OPUS_UNIMPLEMENTED);
    }

    // Test covering `compute_quantizer` unchanged (just to keep the
    // imports live when re-exported via `opus::dred` — caller doesn't
    // need it directly).
    #[test]
    fn compute_quantizer_reachable_from_dred_scope() {
        assert_eq!(compute_quantizer(0, 0, 15, 0), 0);
    }

    // Stage 8.9 — FARGAN reconstruction seam stubs. These three tests
    // pin the returned error so Stage 7's follow-up commit sees a
    // visible failure when it replaces the stubs with real bodies
    // (the new `Ok(...)` return will trip every assertion below, and
    // whoever wires FARGAN then has to update these alongside the
    // implementation).
    #[test]
    fn decode_returns_unimplemented() {
        let dec = OpusDREDDecoder::new();
        let dred = OpusDred::default();
        let mut pcm = [0i16; 960];
        let err = dec.decode(&dred, 0, &mut pcm, 960).unwrap_err();
        assert_eq!(err, OPUS_UNIMPLEMENTED);
    }

    #[test]
    fn decode24_returns_unimplemented() {
        let dec = OpusDREDDecoder::new();
        let dred = OpusDred::default();
        let mut pcm = [0i32; 960];
        let err = dec.decode24(&dred, 0, &mut pcm, 960).unwrap_err();
        assert_eq!(err, OPUS_UNIMPLEMENTED);
    }

    #[test]
    fn decode_float_returns_unimplemented() {
        let dec = OpusDREDDecoder::new();
        let dred = OpusDred::default();
        let mut pcm = [0.0f32; 960];
        let err = dec.decode_float(&dred, 0, &mut pcm, 960).unwrap_err();
        assert_eq!(err, OPUS_UNIMPLEMENTED);
    }
}
