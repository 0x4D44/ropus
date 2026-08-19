# ROP-REQ-KIL-00032 — Tier-2 PLC gates must prove DEEP_PLC engagement, not infer it from SNR

- **State:** Draft
- **Priority:** Should
- **Area:** harness-deep-plc/tests
- **Raised:** 2026-08-19T10:46:33Z
- **Discovery source:** Agent
- **Implemented-by:** —
- **Satisfied-by:** —
- **Violated-by:** —
- **Depends-on:** —
- **Design:** —
- **Flow:** heavy
- **Claimed-by:** —
- **State history:** Draft (2026-08-19T10:46:33Z, raised via `deltic reqs new`)

## Statement

The tier-2 DEEP_PLC gates must carry positive evidence that the neural path actually ran, instead of inferring it from the SNR value. Today harness-deep-plc/tests/tier2_snr.rs sets complexity 10 on both decoders and asserts SNR alone; the 50 dB release threshold sits exactly at the top of the measured classical fixed-vs-float band (harness-control/tests/control_snr.rs:57-58 asserts [35, 50] dB), and the 45 dB calibration floor sits inside it — so if both sides silently fell back to classical PLC (e.g. a complexity or weights-wiring regression affecting both), the "neural" gates could stay green while gating the wrong path. No engagement accessor is reachable from this crate (`celt_last_frame_type` is `#[cfg(test)]`-gated inside ropus, ropus/src/opus/decoder.rs:1632). Needs design: expose a debug/telemetry accessor (Rust side) and a peek (C side, via the guarded shim path once ROP-BUG-KIL-00025 lands) reporting whether concealed frames used the neural branch, and assert it in the tier-2 gates. Related mechanical holes are tracked separately as ROP-BUG-KIL-00022.

## Notes
