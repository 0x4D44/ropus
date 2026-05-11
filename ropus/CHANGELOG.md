# Changelog

All notable changes to the `ropus` crate are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
crate aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.12.18] - 2026-05-11

First crates.io publish since `0.12.6` (2026-05-04). This entry consolidates
the work of the unpublished `0.12.7` – `0.12.17` patch bumps so the public
record matches the source. Theme: integration round-trip hardening — every
fix below traces to a divergence the harness or fuzz cluster surfaced after
0.12.0.

### Fixed

- Multistream encoder: `ms_*` setter family was writing only `fec_config`,
  leaving `silk_mode.use_in_band_fec` stale. Collapsed the entire `ms_*`
  setter+getter family into the public `OpusEncoder::set_*` fan-out (mirrors
  C `opus_multistream_encoder_ctl`'s SET path). Closes 24h-fuzz Cluster A
  H1, plus latent H2 (`ms_set_vbr` skipping `silk_mode.use_cbr`). (`739f9c6`)
- SILK prefill branch was structurally diverged from the non-prefill body.
  Unified the two so prefill no longer drifts from the C reference. Closes
  Cluster A B1. (`71b37dd`)
- Multistream wrapper had a complexity-modulus asymmetry vs. C; aligned the
  per-stream complexity computation. Closes Cluster A NEW class. (`05702b5`)
- Multistream packet validation: removed Rust-side error tolerances that
  were masking real Ok/Err divergences against the C reference, and tightened
  the parity oracle. (`e127a4e`, `89c4989`)
- Multiframe encoder VBR/FEC/DTX parity (`0fc1261`), CBR rate-switch
  resampler parity (`96ec3d1`), hybrid CELT↔SILK info handoff parity
  (`27be4ff`), and gain-fade fixed-point parity (`8847c20`).
- CELT decoder PLC default state did not match C on the first PLC frame;
  fixed initial state. (`4f52410`)
- CELT encoder: two silence-path state divergences and a CELT→SILK
  redundancy-frame prediction sequencing bug. (`612fe78`, `9d9d9cc`)
- SILK LBRR `ec_prev` state mutation parity — the LBRR pass was leaving
  `ec_prev` in a different state than the C reference. (`55e224c`)
- PLC short-frame `delay_buffer` contamination: state from the previous
  frame was bleeding through. (`9b8c552`)
- DRED bitrate plumbing: five setter-sequence divergences (F33, F48, F53,
  stale-quantizer, F33b) where the encoder's response to CTL ordering
  differed from C. `compute_dred_bitrate` / `estimate_dred_bitrate` ported
  byte-exact in f32, validated via 15-vector FFI fixture across FEC,
  loss 0–30 %, 5–120 kbps, frame sizes 160/320/480/960, and 8/16/24/48 kHz.
  (`755eb3b`, `8f342bc`, `9a420c4`, `c0c145d`)
- DRED LPCNet feature drift: radix-5 FFT grouping and a `pow` precision
  fix bring DRED feature output substantially closer to C. (`b1bfa6f`)
- DNN/SILK numerical parity: `lpc_from_bands` lag-windowing precision +
  noise-floor constant (`eb4c310`); `if_features` `norm_1` routed through
  f64 to match C semantics (`6b1b8d9`); `compute_burg_cepstrum` f32/f64
  parity via `powi` + `log10` (`6dc7f2e`); `celt_fir_f32` summation order
  (`da62d7a`).
- LFE band-0 boost added to match the C reference. (`f354712`)
- C-ABI shim (`capi/`) float-PCM ingest fix (Cluster B Phase 1). (`3f76dd6`)

### Tooling / Build

- Build cleanliness: `-msse4.1` applied to xiph/opus C reference SIMD
  sources (`368f336`); silenced the upstream "opus will be very slow"
  note (`b8e82a5`); FFI extern blocks annotated with `#[link]` for
  self-contained linking (`2f21d43`); cleared all default-lint warnings
  workspace-wide (`bce8490`); final fmt + clippy sweep (`6cbff17`).

### Testing

These changes don't affect the published API but materially change what
"green" means for the crate.

- Two large fuzz campaigns (24 h cluster + 8 h overnight), 11+ crash
  classes triaged and fixed; all repros migrated to permanent regression
  seeds. New targets: `fuzz_encode_multiframe` (rebuilt + 5 new CTL
  dimensions), `fuzz_multistream`, `fuzz_repacketizer_seq`, `fuzz_dnn_blob`,
  `fuzz_decode_plc_seq`, mode-transition (SILK/HYBRID/CELT), stereo-biased
  multiframe.
- Bounded PLC recovery-drift oracle on `decode24` (exact prefix + bounded
  drift; full-frame exact remains documented debt).
- Release-preflight gate now distinguishes exact byte parity, exact PCM
  parity, bounded drift, SNR-gated parity, recovery-only,
  error-agreement-only, smoke-only, ignored, and asset-skipped lanes.
  Adds release-blocking gates for DRED/DNN assets, generated Ogg
  family-0 corpus, performance threshold (decode 1.26×), and platform/
  sanitizer breadth (x86_64 smoke + cargo-fuzz ASan).
- Mutation-testing audit runner revived against the current
  `ropus/src/...` crate layout.
- Parameterized round-trip differential grid added to the harness.

## [0.12.0] - 2026-04-29

### Added

- Runtime setters on the typed `Encoder` facade, returning
  `Result<(), EncodeError>`: `set_bitrate`, `set_complexity`, `set_signal`,
  `set_vbr`, `set_vbr_constraint`, `set_force_channels`, `set_max_bandwidth`,
  `set_packet_loss_perc`, `set_inband_fec`, `set_dtx`. Each thinly delegates
  to the matching libopus CTL on the underlying `OpusEncoder`. Lets callers
  retune an encoder mid-stream — bitrate, FEC, DTX, channel forcing, etc. —
  without rebuilding it.
- Runtime getters paired with the setters: `bitrate`, `effective_bitrate_bps`
  (the libopus `OPUS_GET_BITRATE` value, which differs from the user-supplied
  request after clamping), `complexity`, `signal`, `vbr`, `force_channels`,
  `max_bandwidth`, `inband_fec`, `dtx`. The getters report the libopus-mandated
  default state on a freshly-built encoder (pinned by a regression test).
- `EncoderBuilder::inband_fec(InbandFec)` and `EncoderBuilder::dtx(bool)`
  builder methods, mirroring the existing `vbr_constraint` / `packet_loss_perc`
  builder shape.
- New `InbandFec { Disabled, Enabled, Forced }` enum re-exported from the
  crate root, replacing the previous bare `i32` for FEC-mode configuration.

### Changed (BREAKING)

- `Encoder::get_vbr_constraint() -> i32` renamed and retyped to
  `Encoder::vbr_constraint() -> bool`. Migration:
  `enc.get_vbr_constraint() != 0` → `enc.vbr_constraint()`.
- `Encoder::get_packet_loss_perc() -> i32` renamed and retyped to
  `Encoder::packet_loss_perc() -> u8`. Migration:
  `enc.get_packet_loss_perc() as u8` → `enc.packet_loss_perc()`.
- The low-level `OpusEncoder::get_vbr_constraint` /
  `get_packet_loss_perc` methods are unchanged; only the high-level typed
  facade was renamed. C-ABI shim (`capi/`) callers are unaffected.

## [0.11.1] - 2026-04-19

### Added

- `Decoder::set_dnn_blob` on the typed facade, delegating to the existing
  low-level `OpusDecoder::set_dnn_blob`. Errors lift to
  `DecoderInitError`. Lets callers using the idiomatic `Decoder` API
  install neural-PLC weights without dropping down to the raw libopus
  surface.
- Compile-time thread-safety assertion
  (`api::tests::encoder_decoder_are_send_sync`) that verifies `Encoder`
  and `Decoder` are `Send + Sync` — catches regressions where a `!Send` /
  `!Sync` field (e.g., an accidentally introduced `Rc` or `RefCell`) would
  otherwise silently drift the README claim.

### Fixed

- SILK PLC state-carryover for the DEEP_PLC recovery path — preserves
  the subframe state across the transition from classical to neural PLC
  so the decoder reconverges cleanly after a burst loss. (`9bde2c1`)

## [0.11.0] - 2026-04-19

### Added

- Builder methods `EncoderBuilder::vbr_constraint(bool)` and
  `EncoderBuilder::packet_loss_perc(u8)` on the typed facade, exposing the
  matching libopus CTLs. (`d5d2d2e`)

## [0.10.0] - 2026-04-19

### Fixed

- `ropus-cli` decode path now applies `OpusHead.output_gain` per RFC 7845
  §5.1. Previously the Q8 gain field was parsed but ignored, so files
  encoded with non-zero header gain decoded at the wrong level. (`37b82a3`)

## [0.9.0] - 2026-04-19

### Added

- Real DNN weight population — the build script embeds LPCNet / FARGAN /
  classical-PLC weights from the xiph reference sources at compile time
  when they are present on disk, and callers can override at runtime via
  `OpusDecoder::set_dnn_blob`. (`93f0274`)

## [0.8.0] - 2026-04-19

### Added

- DNN modules (LPCNet, FARGAN) wired into `OpusDecoder`'s PLC path —
  plumbing-only at this stage, gated behind a tier-2 50 dB SNR floor
  against the C-float reference. (`77d2a3e`)

## [0.7.2] - 2026-04-19

### Fixed

- SILK `silk_plc_update` now saves subframe geometry and applies the LTP
  bias for unvoiced frames, matching the C reference and eliminating a
  drift seen on voiced/unvoiced boundaries. (`60c6a97`)

## [0.7.1] - 2026-04-19

### Fixed

- SILK side-channel resampler is now cloned after `set_fs` during
  mono→stereo transitions, preventing resampler-state corruption when the
  encoder switches channel layout mid-stream. (`a215d9b`)

## [0.7.0] - 2026-04-19

### Added

- Analysis / tonality detection wired into `OpusEncoder` (Stage 6.4).
  Integrated encode reaches a tier-1 95% byte-exact floor against the C
  reference on music vectors, with 0.000 dB decoded-PCM SNR delta on
  music + speech under the SNR-equivalence gate. (`141471a`)

## [0.6.0] - 2026-04-19

### Added

- Five additional encoder CTLs routed through the C-ABI shim (`capi/`),
  bringing the ropus configuration surface closer to full libopus parity.
  (`469a100`)

## [0.5.3] - 2026-04-19

### Fixed

- CELT hybrid-mode `min_allowed` floor for the redundancy signal bit,
  correcting a rate-allocation drift on low-bitrate hybrid frames.
  (`e83f185`)

## [0.5.2] - 2026-04-19

### Added

- Ambisonics mapping matrix data ported for projection / surround encode
  (`channel_mapping == 3`). Enables first- through fifth-order ambisonic
  round-trips through the harness. (`1f43890`)

## [0.5.1] - 2026-04-19

### Fixed

- Removed an `encode_multiframe` `range_final` overwrite that could
  corrupt the entropy-coder range when stitching multi-frame packets.
  (`3009a6b`)

### Changed

- Crate package now includes `examples/**/*` so the end-to-end samples
  ship on crates.io. (`3d5ea2a`)

## [0.5.0] - 2026-04-18

### Changed

- **BREAKING:** `Decoder::decode` and `Decoder::decode_float` now take a
  `DecodeMode` enum in place of the previous `decode_fec: bool` parameter.
  All in-tree call sites were migrated in lockstep. (`f50767e`)
- `Encoder::encode` / `Encoder::encode_float` now reject off-by-one frame-size
  buffers at the high-level facade with `EncodeError::BadArg`, instead of
  surfacing the same error from deep inside mode selection. (`f50767e`)

### Added

- `DecodeMode { Normal, Fec }` enum, re-exported from the crate root.
  (`f50767e`)
- `Encoder::valid_frame_sizes()` returning the legal nine-set
  `[fs/400, fs/200, fs/100, fs/50, fs*2/50, ..., fs*6/50]` for
  discoverability. (`f50767e`)
- `Decoder::max_frame_samples_per_channel()` returning `fs * 6 / 50`.
  (`f50767e`)
- `Encoder::lookahead()` accessor forwarding to `OpusEncoder::get_lookahead()`,
  so callers can populate `OpusHead.pre_skip` with the actual encoder delay
  rather than a hardcoded constant. (`6d8b28f`)

### Fixed

- Restored `uc!` / `uc_set!` / `uc_mut!` macros to unchecked indexing in
  `celt/fft.rs` and `celt/mdct.rs` (95 fuzzed call sites), recovering the
  ~9% decode regression introduced in 0.3.0 by the safe-indexing
  simplification. (`02cd5ff`)
- Examples / packaging: `WAVE_FORMAT_EXTENSIBLE` (0xFFFE) parsing in
  `examples/common/wav.rs`; `roundtrip` example `--bitrate` made optional;
  hardened `ropus-cli` git-SHA capture; declared `BSD-3-Clause` license on
  sibling crates to match `ropus`. (`3b3e28a`)

## [0.4.0] - 2026-04-18

### Added

- High-level `Encoder` / `EncoderBuilder` / `Decoder` facade exposed at the
  crate root, mirroring the libopus C surface in idiomatic Rust. Wraps the
  existing `OpusEncoder` / `OpusDecoder`; no codec logic added. (`3fcc10d`)
- Typed enums: `Channels`, `Application`, `Bitrate`, `Signal`, `Bandwidth`,
  `ForceChannels`, `FrameDuration`. Each has a private `as_c_int()` that
  returns the existing libopus constant; no value duplication. (`3fcc10d`)
- Error types: `EncoderBuildError`, `EncodeError`, `DecoderInitError`,
  `DecodeError`, generated via the `impl_opus_error!` macro with a
  positive-code guard so unrelated CTL value constants (e.g.
  `OPUS_BITRATE_MAX = -1`) cannot be silently misclassified as error
  codes. (`3fcc10d`)
- `Encoder::sample_rate()` / `Encoder::channels()` accessors. (`ff1ffe7`)
- `Decoder::sample_rate()` / `Decoder::channels()` accessors. (`ff1ffe7`)
- `Bitrate::try_bits(u32) -> Result<Self, BitrateRangeError>` validated
  constructor; `Bitrate::Bits(u32)` retained with documented clamp.
  (`ff1ffe7`)
- `ogg = "0.9"` added under `[dev-dependencies]` for the new examples.
  (`ff1ffe7`)

## [0.3.0] - 2026-04-18

### Added

- Crates.io publishing metadata in `Cargo.toml`: `description`, `license`
  (BSD-3-Clause), `repository`, `readme`, `keywords`, `categories`,
  `authors`, `rust-version = "1.85"`, and an `include` whitelist. (`c1ecc6b`)
- `LICENSE` (root + `ropus/LICENSE`) reproducing `reference/COPYING`
  verbatim, preserving upstream Xiph.Org copyright per BSD-3-Clause
  clause 1. (`c1ecc6b`)
- `README.md` library-user oriented quickstart, status, and
  acknowledgements. (`c1ecc6b`)
- Crate-level `//!` docs and stable public-API re-exports at the crate
  root (`OpusEncoder`, `OpusDecoder`, `OpusMSEncoder`, `OpusMSDecoder`,
  `OpusRepacketizer`, and the `OPUS_*` constants). Items reachable via
  `ropus::opus::*` / `celt::*` / `silk::*` / `dnn::*` remain accessible
  but are not subject to semver guarantees pre-1.0. (`7fe3adc`)

### Changed

- Removed `[features]` (`simd`, `unchecked-indexing`, `dnn`); `wide`
  is now an unconditional dependency, and the previously feature-gated
  default branches are the only paths compiled. (`5abd1a9`)
- `uc!` / `uc_set!` / `uc_mut!` macros simplified to safe indexing, so
  the published crate ships unsafe-free. (Restored to unchecked indexing
  in 0.5.0 after a measured regression.) (`5abd1a9`)
- Migrated 6 FFI-dependent differential tests + helper + `extern "C"`
  block out of the library crate into `harness/tests/c_ref_differential.rs`,
  so the library crate no longer carries any FFI surface. (`5abd1a9`)
- Renamed harness binaries `mdopus-compare` -> `ropus-compare`,
  `mdopus-interframe` -> `ropus-interframe`; harness, fuzz, and tooling
  switched from `use mdopus::...` to `use ropus::...`. (`f2f7340`)

### Removed

- `MDOPUS_SKIP_REFERENCE` env-var handling in the harness `build.rs`.
  (`5abd1a9`)
- `ropus/src/main.rs` (Hello-world stub). (`5abd1a9`)

## [0.2.1] - 2026-04-18

### Changed

- Initial public structure: split the single-crate repo into a Cargo
  workspace, with the library crate carved out as `ropus/` (formerly
  `src/` at the repo root). Resolver = 3; `harness/` becomes a separate
  `publish = false` crate hosting the C reference FFI build, integration
  binaries, and tooling. (`fa031ab`)

[0.12.18]: https://github.com/0x4D44/ropus/releases/tag/v0.12.18
[0.12.0]: https://github.com/0x4D44/ropus/releases/tag/v0.12.0
[0.11.1]: https://github.com/0x4D44/ropus/releases/tag/v0.11.1
[0.11.0]: https://github.com/0x4D44/ropus/releases/tag/v0.11.0
[0.10.0]: https://github.com/0x4D44/ropus/releases/tag/v0.10.0
[0.9.0]: https://github.com/0x4D44/ropus/releases/tag/v0.9.0
[0.8.0]: https://github.com/0x4D44/ropus/releases/tag/v0.8.0
[0.7.2]: https://github.com/0x4D44/ropus/releases/tag/v0.7.2
[0.7.1]: https://github.com/0x4D44/ropus/releases/tag/v0.7.1
[0.7.0]: https://github.com/0x4D44/ropus/releases/tag/v0.7.0
[0.6.0]: https://github.com/0x4D44/ropus/releases/tag/v0.6.0
[0.5.3]: https://github.com/0x4D44/ropus/releases/tag/v0.5.3
[0.5.2]: https://github.com/0x4D44/ropus/releases/tag/v0.5.2
[0.5.1]: https://github.com/0x4D44/ropus/releases/tag/v0.5.1
[0.5.0]: https://github.com/0x4D44/ropus/releases/tag/v0.5.0
[0.4.0]: https://github.com/0x4D44/ropus/releases/tag/v0.4.0
[0.3.0]: https://github.com/0x4D44/ropus/releases/tag/v0.3.0
[0.2.1]: https://github.com/0x4D44/ropus/releases/tag/v0.2.1
