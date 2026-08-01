- ARM DNN parity needs `DISABLE_NEON`: header kernels bypass source filters (`harness-deep-plc/build.rs`).

  Excluding `dnn/arm/*.c` does not select scalar inference because
  `reference/dnn/vec.h` dispatches to inline NEON code. Differential harnesses
  must pin the intended arithmetic path explicitly.
