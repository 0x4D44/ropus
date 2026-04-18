# Changelog

All notable changes to the `ropus` crate are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
crate aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The crate has not yet been published to crates.io; versions below correspond
to in-tree milestones identified by the `ropus/Cargo.toml` version-bump
commits in this repository.

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

[0.5.0]: https://github.com/0x4D44/ropus/releases/tag/v0.5.0
[0.4.0]: https://github.com/0x4D44/ropus/releases/tag/v0.4.0
[0.3.0]: https://github.com/0x4D44/ropus/releases/tag/v0.3.0
[0.2.1]: https://github.com/0x4D44/ropus/releases/tag/v0.2.1
