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
