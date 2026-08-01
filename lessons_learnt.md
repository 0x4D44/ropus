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
