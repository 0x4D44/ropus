# Testing Oracle Matrix

Last reviewed: 2026-05-06

Purpose: make the strongest oracle behind each validation lane explicit. A green
test run is only meaningful when read alongside the oracle strength and required
assets below.

## Oracle Classes

| Class | Meaning |
|---|---|
| `exact-byte-parity` | Encoded bytes or payload bytes must match the reference exactly. |
| `exact-pcm-parity` | Decoded PCM samples must match the reference exactly. |
| `snr-gated-parity` | Output is compared by a numeric SNR/quality threshold, not exact samples. |
| `sample-count-only` | Decode result symmetry and sample-count/shape parity only; Hybrid high-band PCM values are not a correctness oracle for attacker-controlled decode inputs. |
| `recovery-or-dtx-only` | Decode result symmetry and sample-count/shape parity only; recovery/DTX PCM values are not a correctness oracle. |
| `error-agreement-only` | Decode success/error symmetry only; sample count is checked only when both sides decode. |
| `smoke-only` | The command exercises a path without proving detailed codec equivalence. |
| `ignored` | The test exists but is not part of normal validation. |
| `asset-skipped` | The strongest path is skipped when external assets are unavailable. |

## Matrix

| Subsystem | Command or target | Strongest oracle | Required assets | Asserts | Does not assert | Release gate | Preflight key | Owner or follow-up | Last reviewed |
|---|---|---|---|---|---|---|---|---|---|
| Conformance | `cargo test -p conformance -- --test-threads=1` | `exact-byte-parity` | `fixed_reference`, `conformance_sources`, `ietf_vectors` for vector binary | Upstream xiph conformance programs pass against the ropus C ABI shim. | DNN/DRED extension parity beyond compiled upstream tests; parallel libtest safety without `--test-threads=1`. | `required-core` | `fixed_reference`, `conformance_sources`, `ietf_vectors` | `/home/md/language/ropus/tests/conformance/README.md` | 2026-05-06 |
| Harness | `cargo run -p ropus-harness --bin ropus-compare -- unit range_coder` | `exact-byte-parity` | `fixed_reference` | Range coder behavior matches the fixed-point C reference for the scoped unit lane. | Full encode/decode parity outside the selected unit. | `required-core` | `fixed_reference` | `/home/md/language/ropus/wrk_docs/2026.05.05 - PLN - testing improvement plan.md` | 2026-05-06 |
| Harness | `cargo run -p ropus-harness --bin ropus-compare -- encode <wav>` | `exact-byte-parity` | `fixed_reference`, fixture WAV | Rust encoded bytes are compared to C-reference encoded bytes with automation-safe failure status after Workstream A. | All parameter combinations. | `required-core` | `fixed_reference` | `/home/md/language/ropus/wrk_docs/2026.05.06 - HLD - harness gate integrity.md` | 2026-05-06 |
| Harness | `cargo run -p ropus-harness --bin ropus-compare -- decode <opus>` | `exact-pcm-parity` | `fixed_reference`, fixture Opus stream | Rust decoded samples are compared to C-reference decoded samples with length-aware comparison after Workstream A. | Container behavior, FEC/PLC not present in the selected fixture. | `required-core` | `fixed_reference` | `/home/md/language/ropus/wrk_docs/2026.05.06 - HLD - harness gate integrity.md` | 2026-05-06 |
| Harness | `cargo run -p ropus-harness --bin ropus-compare -- roundtrip <wav>` | `exact-pcm-parity` | `fixed_reference`, fixture WAV | End-to-end Rust and C-reference encode/decode outputs are compared for the fixture. | Exhaustive mode, bitrate, loss, and state-transition coverage. | `required-core` | `fixed_reference` | `/home/md/language/ropus/wrk_docs/2026.05.06 - HLD - harness gate integrity.md` | 2026-05-06 |
| IETF vectors | `cargo test -p conformance --test ietf_vectors -- --test-threads=1` | `exact-pcm-parity` | `fixed_reference`, `conformance_sources`, `ietf_vectors` | 12 RFC 6716 / RFC 8251 bitstreams are decoded through upstream `opus_demo`/`opus_compare` expectations. | Non-RFC real-world streams; encoder behavior. | `required-core` | `fixed_reference`, `conformance_sources`, `ietf_vectors` | `/home/md/language/ropus/wrk_docs/2026.05.04 - HLD - ietf-vector-provisioning-gate.md` | 2026-05-06 |
| DNN PLC | `harness-deep-plc` tier-2 SNR tests | `snr-gated-parity` | `fixed_reference`, `dnn_base_weights`, `float_deep_plc_assets` | Neural PLC output clears a quality/SNR threshold against the float C reference. | Exact PCM parity; DRED reconstruction; behavior when weights are missing. | `optional-report` in C1; candidate C2 neural profile | `dnn_base_weights`, `float_deep_plc_assets` | `/home/md/language/ropus/wrk_docs/2026.05.05 - PLN - testing improvement plan.md` | 2026-05-06 |
| DRED | DRED payload-byte differential tests | `ignored` | `fixed_reference`, `dnn_base_weights`, `dred_rdovae_weights` | Existing ignored tests document intended payload-byte parity direction. | A passing release gate for DRED encode payloads or final DRED PCM reconstruction. | `debt` | `dnn_base_weights`, `dred_rdovae_weights` | `/home/md/language/ropus/wrk_docs/2026.05.05 - PLN - testing improvement plan.md` | 2026-05-06 |
| Fuzz | Shared fuzz oracle `celt-coded-comparable` class | `exact-pcm-parity` | Target-specific corpus/crash input | Exact PCM/float-bit parity where the target path supports it. | Coverage of SILK/Hybrid SNR-gated and weakened oracle classes. | `warn` | `none` | `/home/md/language/ropus/wrk_docs/2026.05.06 - HLD - fuzz oracle debt reporting.md` | 2026-05-06 |
| Fuzz | Shared fuzz oracle `silk-hybrid-snr-comparable` class | `snr-gated-parity` | Target-specific corpus/crash input | SNR-gated parity when the reference energy floor is met. | Exact PCM parity; coverage of weakened oracle classes. | `warn` | `none` | `/home/md/language/ropus/wrk_docs/2026.05.06 - HLD - fuzz oracle debt reporting.md` | 2026-05-06 |
| Fuzz | Shared fuzz oracle `sample-count-only` class | `sample-count-only` | Target-specific corpus/crash input | Decode result symmetry and sample-count/shape parity only. | PCM sample values, exact PCM parity, and SNR threshold are not asserted for attacker-controlled high-band Hybrid decode inputs. | `debt` | `none` | `/home/md/language/ropus/wrk_docs/2026.05.06 - HLD - fuzz oracle debt reporting.md` | 2026-05-06 |
| Fuzz | Shared fuzz oracle `recovery-or-dtx-only` class | `recovery-or-dtx-only` | Target-specific corpus/crash input | Decode result symmetry and sample-count/shape parity only. | PCM sample values, exact PCM parity, and SNR threshold are not asserted when at least one SILK/Hybrid sub-frame routes through PLC/DTX-style recovery. | `debt` | `none` | `/home/md/language/ropus/wrk_docs/2026.05.06 - HLD - fuzz oracle debt reporting.md` | 2026-05-06 |
| Fuzz | Shared fuzz oracle `error-agreement-only` class | `error-agreement-only` | Target-specific corpus/crash input | Decode success/error symmetry only; sample count only when both sides decode. | PCM sample values, exact PCM parity, SNR threshold, and in some cases sample-count expectations are not asserted because packet structure is not usable for PCM comparison. | `debt` | `none` | `/home/md/language/ropus/wrk_docs/2026.05.06 - HLD - fuzz oracle debt reporting.md` | 2026-05-06 |
| Performance | `full-test` benchmark stage | `smoke-only` | `fixed_reference`, benchmark WAV fixtures | Benchmark commands run and ratios are reported/warned. | Codec correctness beyond command survival; release-blocking performance thresholds. | `warn` | `fixed_reference` | `/home/md/language/ropus/full-test/src/bench.rs` | 2026-05-06 |

## C1 Preflight Keys

| Key | C1 status |
|---|---|
| `fixed_reference` | Required core asset; banner-failing under `--release-preflight` when missing. |
| `conformance_sources` | Required core asset; banner-failing under `--release-preflight` when missing. |
| `ietf_vectors` | Required core asset; existing Stage 2 synthetic failure remains the failure path when unavailable. |
| `dnn_base_weights` | Optional/report-only in C1. |
| `dred_rdovae_weights` | Optional/report-only in C1. |
| `float_deep_plc_assets` | Optional/report-only in C1. |
