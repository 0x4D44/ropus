# ropus fuzz targets

cargo-fuzz subcrate that exercises the ropus encoder, decoder,
repacketizer, and packet-introspection APIs. Two kinds of targets live
here:

- **Differential targets** — run ropus and the vendored xiph/opus C
  reference on the same input and assert byte-equal (or sample-equal)
  output. These require the C reference source under `reference/` so the
  harness's `build.rs` can compile it.
- **Safety targets** (`*_safety`, `fuzz_packet_parse`) — Rust-only sweeps
  that assert the codec never panics, never indexes out of bounds, and
  obeys documented contracts (sample counts, CBR determinism, etc.).

This subcrate is excluded from the top-level `Cargo` workspace. It builds
via `cargo fuzz`, which re-enters it using its own `Cargo.toml`.

## Targets

Listed in the same order as `Cargo.toml` `[[bin]]` entries. All nine are
defined under `fuzz_targets/`; the tenth file (`c_reference.rs`) is a
shared FFI module, not a target.

| target                    | kind         | what it fuzzes                                                                                      |
| ------------------------- | ------------ | --------------------------------------------------------------------------------------------------- |
| `fuzz_decode`             | differential | Decode a single packet with ropus and the C reference; assert sample-equal output (mode-gated).     |
| `fuzz_encode`             | differential | Encode one PCM frame with ropus and C; assert byte-equal compressed bytes.                          |
| `fuzz_roundtrip`          | differential | Encode then decode; compare compressed bytes and decoded PCM against the C reference.               |
| `fuzz_repacketizer`       | differential | Drive `opus_repacketizer_cat`/`_out` against the C reference; assert identical packet boundaries.   |
| `fuzz_packet_parse`       | differential | Packet introspection (`get_bandwidth`, `get_nb_frames`, `get_nb_channels`, `get_samples_per_frame`, `get_nb_samples`) vs C. |
| `fuzz_decode_safety`      | safety       | Rust-only: arbitrary packet bytes must never panic or OOB in the decoder.                           |
| `fuzz_encode_safety`      | safety       | Rust-only: encoder never panics; packet is parseable; CBR is deterministic; CELT-only is correct.   |
| `fuzz_roundtrip_safety`   | safety       | Rust-only: encode+decode never panics; decoded sample count matches; decode is deterministic; LOWDELAY diff is bounded. |
| `fuzz_encode_multiframe`  | differential | 5-10 sequential frames through a single encoder state; catches state-accumulation bugs.             |

Each target starts with a small panic-capture hook: on Windows, Rust
panics inside `libfuzzer-sys` trigger `__fastfail`, which bypasses
libFuzzer's crash-artifact writer. The hook writes the crashing input to
`$FUZZ_PANIC_CAPTURE_DIR` (or `fuzz_crashes/` by default) so crashes stay
reproducible across platforms.

## Prerequisites

- Rust **nightly** toolchain (cargo-fuzz requires `-Zsanitizer=address`):
  `rustup toolchain install nightly`.
- `cargo fuzz`: `cargo install cargo-fuzz`.
- The xiph/opus C reference checked out under `reference/` in the repo
  root (see the top-level `README`).
- On Windows: Visual Studio with the "C++ AddressSanitizer" workload so
  `clang_rt.asan_dynamic-x86_64.dll` is on PATH. The heavier runner
  script (`tools/fuzz_run.sh`) sets this up automatically; `fuzz_cycle.sh`
  assumes it is already in place.

## Running

### Single target, short burst

```bash
cargo +nightly fuzz run --fuzz-dir tests/fuzz fuzz_decode -- -max_total_time=600
```

The `-- -max_total_time=N` split is required: everything after `--` is
forwarded to libFuzzer, everything before is `cargo fuzz`'s own.

### Full cycle (all nine targets)

```bash
# 10 minutes per target (default, ~1.5h total for all nine)
./tools/fuzz_cycle.sh

# 1 hour per target (overnight, ~9h total)
./tools/fuzz_cycle.sh 3600

# Dry run: list targets without executing
./tools/fuzz_cycle.sh 0
```

Pass any positive integer as the duration in seconds. The default (600s)
matches `fuzz_run.sh`'s default and keeps the weekly cadence practical;
bump it up for longer overnight runs.

Per-target output goes to `logs/fuzz_cycle_<timestamp>/<target>.log`. The
script wraps each `cargo fuzz run` with `mdtimeout` (Windows) or
`timeout` (Linux/macOS) for a hard ceiling, falling back to unwrapped
execution with a warning if neither is on PATH. A final summary table
prints the per-target exit code and the count of new crash files.

For the richer workflow (parallel jobs, Windows ASan bootstrap, crash
regression replay), use `./tools/fuzz_run.sh` instead — `fuzz_cycle.sh`
is deliberately a thin cadence wrapper.

## Corpus

Seed inputs live under `corpus/<target>/`. Each target has its own
directory (e.g. `corpus/fuzz_decode/`, plus a few historical variants
such as `corpus/fuzz_decode_filtered/` retained for comparison runs).
libFuzzer reads the target's own `corpus/<target>/` directory on start
and adds interesting inputs back to it as the session progresses.

### Seeding new inputs

- From deterministic WAV fixtures: `tests/vectors/` holds a small set of
  synthetic signals (sine / noise / silence at each sample rate). Run
  `cargo run --bin ropus-compare -- encode <vector.wav>` to generate the
  corresponding `.opus` bytes, then drop them under
  `corpus/fuzz_decode/`.
- From real-world bitstreams: the Stage 3 real-world corpus at
  `tests/vectors/real_world/` (populated by `tools/fetch_corpus.sh`) is
  a good source of libopus-encoded `.opus` / `.ogg` / `.webm` inputs.
  Extract raw Opus payloads from the container before adding to a
  `fuzz_decode*` corpus — the fuzz targets take bare packet bytes, not
  Ogg pages.
- From a seed generator: `tools/gen_fuzz_seeds.py` and
  `tools/generate_fuzz_seeds.py` emit structured seeds for the encoder
  and repacketizer targets respectively. See the headers of those
  scripts for usage.

Avoid committing multi-megabyte corpora into git. If a new corpus grows
large, add a `.gitignore` entry under the target's subdirectory and
publish the corpus out-of-tree.

## Triage

When a libFuzzer session finds a crash, two things happen:

1. libFuzzer writes `crash-<hash>` (or `leak-*`, `timeout-*`, `oom-*`)
   into the current directory (or the `-artifact_prefix=` path).
2. The project's panic-capture hook writes `crash_<hash>.bin` into
   `$FUZZ_PANIC_CAPTURE_DIR`, falling back to `fuzz_crashes/` in the
   working directory if the env var is unset.

`tools/fuzz_run.sh` promotes both into `tests/fuzz/crashes/<target>/`
with a timestamp prefix. `tools/fuzz_cycle.sh` counts any file newer
than the cycle start under that directory as a "new crash" in its final
summary.

### Reproduce

```bash
cargo +nightly fuzz run --fuzz-dir tests/fuzz <target> \
    tests/fuzz/crashes/<target>/<timestamp>_crash-<hash>.bin
```

Single-shot replay: libFuzzer exits after one iteration over the given
file. Combine with `RUST_BACKTRACE=1` for a panic backtrace.

### Minimize

```bash
cargo +nightly fuzz tmin --fuzz-dir tests/fuzz <target> \
    tests/fuzz/crashes/<target>/<timestamp>_crash-<hash>.bin
```

`cargo fuzz tmin` shrinks the input while keeping the crash reproducible.
The minimized file usually belongs in the corpus for that target, not in
the crash directory — once the underlying bug is fixed, it acts as
regression coverage.

### Fix workflow

1. Open a `wrk_journals/<date> - JRN - <short-slug>.md` entry: attach
   the minimized input, the stack trace, and a one-paragraph hypothesis.
2. Land a unit or integration test under `ropus/src/**` (or the
   appropriate module's test file) that encodes the failing input by
   value — don't make tests read from `tests/fuzz/crashes/`.
3. Land the fix. Rerun the single crash input to confirm it passes, then
   run `./tools/fuzz_run.sh --check-crashes` to re-run every historical
   crash input as a regression gate.
4. Promote the minimized input into `corpus/<target>/` so the next fuzz
   session starts from richer coverage.

## Cadence

**Pre-1.0 (today): manual, human-run.** Target is one full
`./tools/fuzz_cycle.sh` cycle per week, with single-target bursts on any
code change that touches a fuzzed module. There is intentionally no CI
job wired up — continuous fuzzing takes dedicated machine time and
crash-triage discipline we don't have automation for yet.

**Post-1.0 (not yet decided): nightly CI.** Candidate setup is a
scheduled GitHub Actions job that runs each target for 30 minutes, fails
the build on any new crash artifact under `tests/fuzz/crashes/`, and
opens an issue with the minimized reproducer attached. Tracked as
post-1.0 work — do not add CI config for this before then.
