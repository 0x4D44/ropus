//! Stage 7b.3 control harness: orchestrates the C-fixed-classical vs
//! C-float-classical comparison for classical SILK PLC.
//!
//! This crate is empty on purpose — all the logic lives in
//! `tests/control_snr.rs`. The crate exists so cargo can register the
//! integration test target; we needed *something* here for `cargo`'s lib
//! vs bin detection to succeed without ambiguity.
