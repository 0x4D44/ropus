# Conformance suite

This crate runs the xiph/opus reference test programs (`test_opus_*.c`)
and the IETF RFC 6716 / RFC 8251 bitstream vectors against the ropus
codec via its C ABI shim (`capi/`).

## Running

```
cargo test -p conformance -- --test-threads=1
```

**`--test-threads=1` is mandatory.** The reference C tests have
file-scope state — global RNG seeds (`Rz`/`Rw`/`iseed` in
`test_opus_common.h`), FILE* handles kept across function boundaries in
`opus_demo.c`, static decoder-state comparisons in `test_opus_api.c` —
that cannot safely be driven from more than one thread per process.
Invoking `cargo test` without the flag runs tests in parallel by default
and will produce spurious failures and (worse) silently wrong passes.

## What each test binary covers

| Test binary     | Source                                           | Notes                                                                                                                       |
|-----------------|--------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| `padding`       | `reference/tests/test_opus_padding.c`            | Packet padding round-trip.                                                                                                  |
| `decode`        | `reference/tests/test_opus_decode.c`             | PLC, zero-length, CTL (`OPUS_RESET_STATE`, `GET_FINAL_RANGE`, `LAST_PACKET_DURATION`).                                      |
| `api`           | `reference/tests/test_opus_api.c`                | Every encoder/decoder/multistream/repacketizer CTL surface; state-clone / memcpy semantics.                                 |
| `encode`        | `reference/tests/test_opus_encode.c`             | Encoder modes + `regression_test()` (11 historical crash repros from `opus_encode_regressions.c`; 5 QEXT/DRED-gated compile out). |
| `extensions`    | `reference/tests/test_opus_extensions.c`         | `opus_packet_extensions_{count,parse,generate}`.                                                                            |
| `ietf_vectors`  | `reference/src/opus_{demo,compare}.c` (verbatim) | 12 RFC 6716 / 8251 bitstreams × {mono, stereo} = 24 subtests.                                                               |
| `projection`    | `reference/tests/test_opus_projection.c`         | Ambisonics matrix math + public `opus_projection_*` encode/decode.                                                          |

All 7 binaries pass with 0 failures under `cargo test -p conformance -- --test-threads=1`.

## IETF vectors — provisioned gate, manual direct run

```
tools/fetch_ietf_vectors.sh      # Linux/macOS/Git Bash
tools/fetch_ietf_vectors.ps1     # Windows PowerShell
```

The script downloads the canonical archive from `opus-codec.org`,
verifies its SHA-256, and extracts the 12 vectors (~71 MB) into
`tests/vectors/ietf/` (gitignored). If the SHA doesn't match, the script
aborts without overwriting anything.

`full-test` invokes this provisioning path before it treats conformance as a
gate. If vectors are missing and cannot be provisioned, `full-test` keeps the
non-IETF Stage 2 signal where possible but reports a FAIL-level IETF
provisioning failure; it no longer produces a green gate by silently skipping
the RFC vectors.

If the directory is missing, `cargo test -p conformance --test
ietf_vectors` panics with a message telling you to run the fetch script.
Direct conformance runs stay local-only and do not perform network I/O inside
libtest.

## Notes on the build

- All conformance C sources are compiled with `-DFIXED_POINT=1` so the
  reference's `opus_res` typedef resolves to `opus_int32` and matches our
  fixed-point capi. Mismatching this silently corrupts 24-bit decode
  output because C does not type-check across function-pointer calls.
- `opus_demo.c` and `opus_compare.c` are the upstream `.c` files compiled
  verbatim as static libs with their `main` renamed (via `-Dmain=…`), the
  same pattern already used for the test binaries. No upstream
  modifications.
- `opus_demo.c` calls a handful of DRED symbols that live in the full
  DNN tree we intentionally don't link. Local no-op stubs at
  `src/dred_stub.c` resolve those references. They are inert on the
  decode-only code path `run_vectors.sh` and `ietf_vectors` exercise on
  valid RFC 8251 vectors; see the stub's header comment for the exact
  preconditions.
- All lossgen call sites in `opus_demo.c` are gated by
  `#ifdef ENABLE_LOSSGEN`, which is left undefined for the conformance
  build — so no lossgen symbols are ever referenced and no stub is
  needed.
- The `projection` binary compiles `reference/src/mapping_matrix.c`
  verbatim alongside `test_opus_projection.c` with `-DDISABLE_FLOAT_API=1`
  so only the fixed-point matrix branch is pulled in. Header stubs
  (`arch.h`, `float_cast.h`, `mathops.h`, `os_support.h`) in
  `tests/conformance/include/` keep the reference tree off the include
  path.
