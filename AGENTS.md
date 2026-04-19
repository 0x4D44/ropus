# Repository Guidelines

## Project Structure & Module Organization
This project is a Rust port of the Opus codec. Core code lives in `src/`.
- `src/lib.rs`: crate exports and module wiring.
- `src/celt/`, `src/silk/`, `src/opus/`, `src/dnn/`: codec implementations.
- `src/types.rs`: fixed-point numeric types and helper utilities.
- `tests/harness/`: `mdopus-compare` binary, C FFI build script, and bindings used for bit-exact validation.
- `tests/vectors/`: deterministic WAV fixtures for comparison runs.
- `wrk_docs/design_docs/`: architecture/design notes and generated documentation.
- `tools/`: coordinator and workflow scripts.
- `logs/`, `wrk_docs/`, `wrk_journals/`: process notes and artifacts.

## Build, Test, and Development Commands
- `cargo build`  
  Build the crate and compile the comparison harness binary (`mdopus-compare`).
- `cargo run --bin mdopus-compare -- encode assets/test.wav`  
  Compare encode output from Rust vs C reference.
- `cargo run --bin mdopus-compare -- decode tests/vectors/48k_...opus`  
  Compare decode output sample-by-sample.
- `cargo run --bin mdopus-compare -- roundtrip tests/vectors/48k_sine1k_loud.wav`  
  End-to-end encode/decode parity check.
- `cargo run --bin mdopus-compare -- unit range_coder`  
  Run module-level unit comparison.
- `cargo run --bin mdopus-compare -- bench tests/vectors/48k_sine1k_loud.wav --iters 20`  
  Run codec performance comparison.
- `python tools/coordinator.py status`  
  Show phase/status of the multi-agent workflow.
- `cargo fmt`, `cargo llvm-cov`  
  Formatting and coverage instrumentation.

## Coding Style & Naming Conventions
- Use Rust 2024 style and keep API/logic changes minimal.
- Naming: `snake_case` for functions/modules/files, `PascalCase` for types, `SCREAMING_SNAKE_CASE` for constants.
- Preserve fixed-point and parity-sensitive logic; avoid “cleanup” refactors in codec hot paths unless behavior is identical.
- Prefer Rust idioms, but bit-exactness takes priority over aesthetic refactors.
- `unsafe` should be confined to harness/FFI boundaries; keep codec modules safe-by-default.

## Testing Guidelines
- Treat `mdopus-compare` as the truth source for regression validation.
- Every codec logic change should include at least one of: `encode`, `decode`, or `roundtrip`.
- Unit tests are defined inline with `#[cfg(test)]` in `src/`.
- For touched comparison paths, prefer one deterministic fixture and a command example above before opening PR.

## Commit & Pull Request Guidelines
- Recent commit messages are mostly imperative and action-led: `Fix ...`, `Add ...`, `Enable ...`, `Update ...`.
- PRs should include:
  - short summary + affected modules,
  - files changed,
  - exact commands run,
  - expected vs actual outcomes for harness checks.

## Security & Configuration Notes
- `reference/` must contain the xiph/opus source tree; harness build fails if missing.
- Requires a working C compiler toolchain (`cc` crate build path) for comparison tests.
- Do not commit generated reference outputs, logs, or large binary fixtures unless explicitly requested.
