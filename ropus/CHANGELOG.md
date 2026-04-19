# Changelog

All notable changes to the `ropus` crate are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
crate aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The latest version published to crates.io is `0.5.0`. Versions `0.5.1`
through `0.11.1` are in-tree milestones identified by the
`ropus/Cargo.toml` version-bump commits in this repository; they will
land on crates.io as a single jump at the next publish.

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
