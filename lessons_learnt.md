- C ABI constructors must stage fallible handle/state allocations before publishing outputs (`capi/src/alloc.rs:try_box`).

  `handle_alloc_error`, `Box::new`, and `Vec::with_capacity` can abort instead of
  returning `OPUS_ALLOC_FAIL`. Allocate through the fallible helpers, release any
  staged sub-handles on failure, and only write C output parameters after the
  complete handle is ready.

- Multistream decoder CTLs need matching C varargs cases and typed Rust arms (`capi/src/ctl_shim.c:opus_multistream_decoder_ctl`).

  A request can be implemented by `OpusMSDecoder` yet still return `OPUS_UNIMPLEMENTED` when the C switch omits it; test the complete C-to-Rust round trip and reject invalid values before fan-out.

- Extension parsers must validate slice capacities and checked frame-count prefixes before indexing (`ropus/src/opus/repacketizer.rs:opus_packet_extensions_parse`).

  Safe slices do not make caller-provided counts safe: reject lengths beyond the input, capacities beyond output slices, negative counts, and overflowing frame prefixes before iterating.

- Multistream decoder resets must clear neural PLC history via `lpcnet.reset()` (`ropus/src/opus/decoder.rs:OpusDecoder::ms_reset`).

  Keep `ms_reset` aligned with the canonical decoder reset: clear FEC/GRU and analysis state while preserving loaded neural model weights.

- Validate expert frame-duration controls before storing them (`ropus/src/opus/multistream.rs:OpusMSEncoder::set_expert_frame_duration`).

  Multistream setters should mirror the single-stream CTL's accepted sentinel and duration range, return `OPUS_BAD_ARG`, and leave the prior duration unchanged; C callers must receive that status instead of an unconditional success.

- Route multistream decoder CTLs through validated OpusDecoder setters (`ropus/src/opus/multistream.rs:OpusMSDecoder::set_complexity`).

  Internal fan-out helpers must preserve public range checks and side effects. Calling raw field assignments can accept invalid values and leave the CELT decoder's complexity out of sync.

- Guard projection frame sizes before buffer allocation (`ropus/src/opus/multistream.rs:OpusProjectionEncoder::encode`).

  Wrapper methods must reject non-positive frame sizes before converting them to `usize`; delegating validation to the underlying multistream codec is too late when the wrapper sizes a temporary buffer first.

- Validate safe-slice lengths before fixed-size constructor copies (`ropus/src/opus/multistream.rs:OpusMSEncoder::new_impl`).

  Safe slices prevent memory unsafety, but direct indexing can still panic when a caller passes a short mapping or projection matrix. Return `OPUS_BAD_ARG` before copying, and use checked arithmetic for projection dimensions.

- Provision ignored C-reference assets before fuzz checks (`.deltic-integrate.toml:harness/fuzz`).

  A clean task worktree has no tracked `reference/` tree, so the fuzz build
  script cannot assume a prior manual fetch. Reuse the pinned, idempotent
  `fetch-assets reference` command in every gate that enters the fuzz crate.

- Check excluded Cargo workspaces with their own manifest (`.deltic-integrate.toml:fuzz`).

  Root `cargo fmt --all` can miss a standalone workspace, while shared source
  included by another package may use a different edition's import ordering.
  Gate the nested manifest explicitly and document any unavoidable shared-file
  formatter conflict narrowly.

- Route CLI banners from typed paths, never raw argv scans (`ropus-tools-core/src/prelude.rs:output_is_stdout`).

  A raw scan cannot distinguish positional inputs from values consumed by
  earlier options. Parse once with Clap, then derive stdout and query routing
  from the resulting fields so text never contaminates binary or scalar output.

- Drain Opus lookahead, but set EOS to source + pre-skip (`ropus-tools-core/src/commands/encode.rs:encode`).

  Packet input must cover the source plus encoder delay and round up to a full
  frame. The EOS granule deliberately excludes packet padding, so subtracting
  `pre_skip` recovers the exact source duration.

- Benchmark guards must inspect dependency features after Cargo unification (`ropus/src/lib.rs:SILK_ENCODE_TRACE_ENABLED`).

  A harness-local feature flag is not enough: another selected workspace package
  can enable instrumentation on the shared dependency. Export the dependency's
  compiled state and make the benchmark reject any instrumented build.

- Cache FFI capacities before range calls; slice types alone cannot prove C writes fit (`harness-deep-plc/src/lib.rs`).

  A safe wrapper must validate signed dimensions, checked products, integer
  conversions, and runtime buffer capacities before crossing the FFI boundary.
  Null-pointer sentinel tests can prove rejection happens before the C call.

- ARM DNN parity needs `DISABLE_NEON`: header kernels bypass source filters (`harness-deep-plc/build.rs`).

  Excluding `dnn/arm/*.c` does not select scalar inference because
  `reference/dnn/vec.h` dispatches to inline NEON code. Differential harnesses
  must pin the intended arithmetic path explicitly.
