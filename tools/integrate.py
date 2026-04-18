#!/usr/bin/env python3
"""
mdopus integration tester - generates test corpus, runs bit-exact comparisons
between C reference and Rust implementation, iterates fixing issues.

Usage:
    python tools/integrate.py run          # Full: generate + test + fix loop
    python tools/integrate.py generate     # Just generate test corpus
    python tools/integrate.py test         # Just run tests (corpus must exist)
    python tools/integrate.py status       # Show test results summary

Phases:
    1. Wire up Rust encoder/decoder stubs in the comparison harness
    2. Generate test WAV corpus (various signals, rates, channels)
    3. Run ropus-compare for each test case
    4. Collect failures, feed to Claude for fixing
    5. Repeat until all pass or max iterations reached
"""

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import textwrap
import time
import logging
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "tests" / "vectors"
RESULTS_FILE = ROOT / "tools" / "integration_results.json"
LOGS = ROOT / "logs"

# ---------------------------------------------------------------------------
# WAV generation
# ---------------------------------------------------------------------------

def write_wav(path: Path, samples: list[int], sample_rate: int = 48000,
              channels: int = 1, bits: int = 16):
    """Write a 16-bit PCM WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = len(samples)
    byte_rate = sample_rate * channels * (bits // 8)
    block_align = channels * (bits // 8)
    data_size = num_samples * (bits // 8)

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # chunk size
        f.write(struct.pack("<H", 1))   # PCM
        f.write(struct.pack("<H", channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits))
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        for s in samples:
            f.write(struct.pack("<h", max(-32768, min(32767, int(s)))))


def gen_silence(duration_s: float, sample_rate: int, channels: int) -> list[int]:
    return [0] * int(duration_s * sample_rate * channels)


def gen_sine(freq: float, duration_s: float, sample_rate: int, channels: int,
             amplitude: float = 0.8) -> list[int]:
    n = int(duration_s * sample_rate)
    samples = []
    for i in range(n):
        val = int(amplitude * 32767 * math.sin(2.0 * math.pi * freq * i / sample_rate))
        for _ in range(channels):
            samples.append(val)
    return samples


def gen_white_noise(duration_s: float, sample_rate: int, channels: int,
                    seed: int = 42) -> list[int]:
    """Simple LCG pseudo-random noise for reproducibility."""
    n = int(duration_s * sample_rate * channels)
    state = seed
    samples = []
    for _ in range(n):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        val = ((state >> 16) & 0xFFFF) - 32768
        samples.append(max(-32768, min(32767, val)))
    return samples


def gen_impulse(duration_s: float, sample_rate: int, channels: int) -> list[int]:
    n = int(duration_s * sample_rate)
    samples = [0] * (n * channels)
    # Place impulse at sample 100
    if n > 100:
        for c in range(channels):
            samples[100 * channels + c] = 32767
    return samples


def gen_square_wave(freq: float, duration_s: float, sample_rate: int,
                    channels: int, amplitude: float = 0.8) -> list[int]:
    n = int(duration_s * sample_rate)
    period = sample_rate / freq
    samples = []
    for i in range(n):
        val = int(amplitude * 32767) if (i % int(period)) < int(period / 2) else int(-amplitude * 32767)
        for _ in range(channels):
            samples.append(val)
    return samples


def gen_sweep(f_start: float, f_end: float, duration_s: float,
              sample_rate: int, channels: int, amplitude: float = 0.8) -> list[int]:
    """Logarithmic frequency sweep."""
    n = int(duration_s * sample_rate)
    samples = []
    for i in range(n):
        t = i / sample_rate
        # Log sweep: f(t) = f_start * (f_end/f_start)^(t/T)
        f = f_start * (f_end / f_start) ** (t / duration_s)
        phase = 2.0 * math.pi * f_start * duration_s / math.log(f_end / f_start) * \
                ((f_end / f_start) ** (t / duration_s) - 1.0)
        val = int(amplitude * 32767 * math.sin(phase))
        for _ in range(channels):
            samples.append(val)
    return samples


# ---------------------------------------------------------------------------
# Test corpus definition
# ---------------------------------------------------------------------------

TEST_CASES = []

# For each sample rate Opus supports
for sr in [8000, 12000, 16000, 24000, 48000]:
    for ch in [1, 2]:
        ch_label = "mono" if ch == 1 else "stereo"
        prefix = f"{sr}hz_{ch_label}"
        duration = 1.0  # 1 second per test

        TEST_CASES.append({
            "name": f"{prefix}_silence",
            "gen": lambda d=duration, s=sr, c=ch: gen_silence(d, s, c),
            "sample_rate": sr, "channels": ch,
        })
        TEST_CASES.append({
            "name": f"{prefix}_sine440",
            "gen": lambda d=duration, s=sr, c=ch: gen_sine(440.0, d, s, c),
            "sample_rate": sr, "channels": ch,
        })
        TEST_CASES.append({
            "name": f"{prefix}_noise",
            "gen": lambda d=duration, s=sr, c=ch: gen_white_noise(d, s, c),
            "sample_rate": sr, "channels": ch,
        })

# Extended tests at 48kHz mono only (the most common case)
for name, gen_fn in [
    ("48k_impulse", lambda: gen_impulse(1.0, 48000, 1)),
    ("48k_square1k", lambda: gen_square_wave(1000.0, 1.0, 48000, 1)),
    ("48k_sweep", lambda: gen_sweep(100.0, 20000.0, 2.0, 48000, 1)),
    ("48k_sine1k_loud", lambda: gen_sine(1000.0, 1.0, 48000, 1, amplitude=1.0)),
    ("48k_sine1k_quiet", lambda: gen_sine(1000.0, 1.0, 48000, 1, amplitude=0.01)),
]:
    TEST_CASES.append({
        "name": name, "gen": gen_fn,
        "sample_rate": 48000, "channels": 1,
    })


def generate_corpus(log: logging.Logger) -> int:
    """Generate all test WAV files. Returns count."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for tc in TEST_CASES:
        path = CORPUS_DIR / f"{tc['name']}.wav"
        if path.exists():
            continue
        samples = tc["gen"]()
        write_wav(path, samples, tc["sample_rate"], tc["channels"])
        log.info(f"  Generated {tc['name']}.wav ({len(samples)} samples)")
        count += 1
    log.info(f"  Corpus: {len(TEST_CASES)} test files in {CORPUS_DIR}")
    return len(TEST_CASES)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

BITRATES = [6000, 8000, 10000, 16000, 24000, 32000, 48000, 64000, 96000, 128000, 256000, 320000, 510000]
COMPARE_BIN = ROOT / "target" / "debug" / "ropus-compare"
if sys.platform == "win32":
    COMPARE_BIN = COMPARE_BIN.with_suffix(".exe")


def build_harness(log: logging.Logger) -> bool:
    """Build the comparison harness."""
    log.info("Building comparison harness...")
    result = subprocess.run(
        ["cargo", "build"], cwd=str(ROOT),
        capture_output=True, text=True, timeout=300,
        encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        log.error(f"Build failed:\n{result.stderr[:2000]}")
        return False
    log.info("  Build OK")
    return True


def run_comparison(wav_path: Path, mode: str, bitrate: int = 64000,
                   complexity: int = 10) -> dict:
    """
    Run ropus-compare and parse results.
    Returns dict with keys: mode, wav, bitrate, passed, output, error
    """
    cmd = [str(COMPARE_BIN)]
    if mode == "encode":
        cmd += ["encode", str(wav_path), "--bitrate", str(bitrate), "--complexity", str(complexity)]
    elif mode == "decode":
        # For decode tests we'd need pre-encoded opus files; skip for now
        return {"mode": mode, "wav": str(wav_path), "passed": None, "output": "skipped"}
    elif mode == "roundtrip":
        cmd += ["roundtrip", str(wav_path), "--bitrate", str(bitrate)]
    else:
        return {"mode": mode, "wav": str(wav_path), "passed": None, "output": f"unknown mode {mode}"}

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60,
            encoding="utf-8", errors="replace",
        )
        output = result.stdout + result.stderr
        passed = result.returncode == 0 and "FAIL" not in output
        return {
            "mode": mode, "wav": str(wav_path), "bitrate": bitrate,
            "passed": passed, "output": output.strip(),
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "mode": mode, "wav": str(wav_path), "bitrate": bitrate,
            "passed": False, "output": "TIMEOUT", "returncode": -1,
        }
    except Exception as e:
        return {
            "mode": mode, "wav": str(wav_path), "bitrate": bitrate,
            "passed": False, "output": str(e), "returncode": -1,
        }


def run_all_tests(log: logging.Logger) -> list[dict]:
    """Run encode + roundtrip comparisons for all test cases at various bitrates."""
    results = []
    total = len(TEST_CASES) * len(BITRATES) * 2  # encode + roundtrip
    done = 0

    for tc in TEST_CASES:
        wav_path = CORPUS_DIR / f"{tc['name']}.wav"
        if not wav_path.exists():
            log.warning(f"  Missing: {wav_path}")
            continue

        for bitrate in BITRATES:
            for mode in ["encode", "roundtrip"]:
                r = run_comparison(wav_path, mode, bitrate=bitrate)
                results.append(r)
                done += 1
                status = "PASS" if r["passed"] else ("SKIP" if r["passed"] is None else "FAIL")
                if status == "FAIL":
                    log.warning(f"  [{done}/{total}] {status}: {tc['name']} {mode} @{bitrate} - {r['output'][:200]}")
                elif done % 20 == 0:
                    log.info(f"  [{done}/{total}] Progress...")

    return results


def summarize_results(results: list[dict], log: logging.Logger) -> tuple[int, int, int]:
    """Print summary and return (passed, failed, skipped)."""
    passed = sum(1 for r in results if r["passed"] is True)
    failed = sum(1 for r in results if r["passed"] is False)
    skipped = sum(1 for r in results if r["passed"] is None)

    log.info(f"\n{'='*60}")
    log.info(f"RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
    log.info(f"{'='*60}")

    if failed > 0:
        log.info("\nFailed tests:")
        for r in results:
            if r["passed"] is False:
                log.info(f"  {r.get('mode','?')} {Path(r.get('wav','')).stem} @{r.get('bitrate','?')}")
                for line in r.get("output", "").split("\n")[:5]:
                    log.info(f"    {line}")

    return passed, failed, skipped


# ---------------------------------------------------------------------------
# Fix loop
# ---------------------------------------------------------------------------

MAX_FIX_ITERATIONS = 10

def invoke_claude(prompt: str, timeout: int = 3600) -> tuple[bool, str]:
    """Invoke Claude CLI in headless mode."""
    cmd = [
        "claude", "-p",
        "--model", "claude-opus-4-6",
        "--effort", "max",
        "--permission-mode", "bypassPermissions",
        "--output-format", "text",
    ]
    try:
        result = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True,
            input=prompt, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout}s"
    except Exception as e:
        return False, f"ERROR: {e}"


WIRE_UP_PROMPT = textwrap.dedent("""\
    The mdopus Rust port has all 26 modules compiling. Now the comparison
    harness (tests/harness/main.rs) needs its Rust stubs wired up.

    The functions `rust_encode` and `rust_decode` in tests/harness/main.rs
    are currently stubs that return empty vectors. Wire them up to call
    the actual Rust implementation in src/.

    The Rust implementation is a library crate (src/lib.rs). The harness
    binary already depends on it via `extern crate mdopus` (it's in the
    same package).

    Look at src/lib.rs and src/opus/ to find the Rust encoder/decoder
    API, then update rust_encode and rust_decode in tests/harness/main.rs
    to call them.

    If the Rust API doesn't yet expose encode/decode at the top level,
    add the necessary public API in src/lib.rs or src/opus/mod.rs.

    The C reference functions c_encode/c_decode in the same file show
    the expected calling convention (sample_rate, channels, bitrate, etc.)
    Match that pattern.
""")


FIX_PROMPT = textwrap.dedent("""\
    The mdopus Opus comparison harness is failing. Fix the Rust implementation
    to match the C reference output.

    ## Failure summary
    {failure_summary}

    ## Sample failure output
    ```
    {sample_output}
    ```

    ## Rules
    1. The C reference is always correct — match its behavior exactly
    2. Focus on the FIRST failing test case — fix the root cause
    3. Check for: integer overflow wrapping, signed shift behavior,
       division rounding, cast truncation, off-by-one in buffer indexing
    4. Make the smallest change that fixes the issue
    5. Do NOT modify the C reference code or the harness comparison logic
""")


def fix_loop(results: list[dict], log: logging.Logger) -> bool:
    """Iterate: collect failures, ask Claude to fix, re-test."""
    for iteration in range(MAX_FIX_ITERATIONS):
        failures = [r for r in results if r["passed"] is False]
        if not failures:
            log.info("All tests passing!")
            return True

        log.info(f"\n--- Fix iteration {iteration + 1}/{MAX_FIX_ITERATIONS} ---")
        log.info(f"  {len(failures)} failing tests")

        # Build failure summary
        failure_groups = {}
        for f in failures:
            key = f"{f.get('mode','?')} @{f.get('bitrate','?')}"
            if key not in failure_groups:
                failure_groups[key] = []
            failure_groups[key].append(Path(f.get("wav", "")).stem)

        summary_lines = []
        for key, wavs in failure_groups.items():
            summary_lines.append(f"  {key}: {len(wavs)} failures ({', '.join(wavs[:5])}{'...' if len(wavs) > 5 else ''})")
        failure_summary = "\n".join(summary_lines)

        # Get first few failure outputs for context
        sample_outputs = []
        for f in failures[:3]:
            sample_outputs.append(f"-- {Path(f.get('wav','')).stem} {f.get('mode','')} @{f.get('bitrate','')} --")
            sample_outputs.append(f.get("output", "")[:500])
        sample_output = "\n".join(sample_outputs)

        prompt = FIX_PROMPT.format(
            failure_summary=failure_summary,
            sample_output=sample_output,
        )

        log.info("  Sending failures to Claude for fixing...")
        t0 = time.time()
        ok, output = invoke_claude(prompt)
        elapsed = time.time() - t0
        log.info(f"  Fix returned in {elapsed:.0f}s, ok={ok}")

        # Rebuild
        if not build_harness(log):
            log.error("  Build failed after fix — continuing")
            continue

        # Re-test
        log.info("  Re-running tests...")
        results = run_all_tests(log)
        passed, failed, skipped = summarize_results(results, log)

        if failed == 0:
            return True

    log.error(f"  Exhausted {MAX_FIX_ITERATIONS} fix iterations")
    return False


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    LOGS.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS / f"integrate_{timestamp}.log"

    logger = logging.getLogger("integrate")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def cmd_run(args):
    log = setup_logging()
    log.info("=" * 60)
    log.info("mdopus integration test suite")
    log.info("=" * 60)

    # Step 0: Wire up Rust stubs if needed
    harness_main = ROOT / "tests" / "harness" / "main.rs"
    harness_text = harness_main.read_text(encoding="utf-8", errors="replace")
    if "Rust encoder not yet implemented" in harness_text:
        log.info("\nStep 0: Wiring up Rust encoder/decoder stubs...")
        ok, output = invoke_claude(WIRE_UP_PROMPT)
        if not ok:
            log.warning("  Wire-up returned failure, continuing anyway")

    # Step 1: Build
    if not build_harness(log):
        log.error("Build failed, asking Claude to fix...")
        ok, _ = invoke_claude(
            f"cargo build is failing. Fix the build errors.\n"
            f"Run cargo build and fix any errors you find."
        )
        if not build_harness(log):
            log.error("Build still failing after fix attempt")
            return 1

    # Step 2: Generate corpus
    log.info("\nStep 1: Generating test corpus...")
    generate_corpus(log)

    # Step 3: Run tests
    log.info("\nStep 2: Running comparison tests...")
    results = run_all_tests(log)
    passed, failed, skipped = summarize_results(results, log)

    # Save results
    save_results(results)

    if failed == 0:
        log.info("\nALL TESTS PASSING!")
        return 0

    # Step 4: Fix loop
    log.info(f"\nStep 3: Fix loop ({failed} failures to resolve)...")
    if fix_loop(results, log):
        log.info("\nALL TESTS PASSING after fixes!")
        save_results(results)
        return 0
    else:
        log.error("\nSome tests still failing after fix loop")
        save_results(results)
        return 1


def cmd_generate(args):
    log = setup_logging()
    generate_corpus(log)
    return 0


def cmd_test(args):
    log = setup_logging()
    if not build_harness(log):
        return 1
    results = run_all_tests(log)
    passed, failed, skipped = summarize_results(results, log)
    save_results(results)
    return 0 if failed == 0 else 1


def cmd_status(args):
    if not RESULTS_FILE.exists():
        print("No results yet. Run: python tools/integrate.py run")
        return 0
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    passed = sum(1 for r in results if r.get("passed") is True)
    failed = sum(1 for r in results if r.get("passed") is False)
    skipped = sum(1 for r in results if r.get("passed") is None)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    if failed > 0:
        print("\nFailing:")
        for r in results:
            if r.get("passed") is False:
                print(f"  {r.get('mode','?')} {Path(r.get('wav','')).stem} @{r.get('bitrate','?')}")
    return 0


def save_results(results: list[dict]):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="mdopus integration tester")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run", help="Full: generate + test + fix loop")
    sub.add_parser("generate", help="Generate test corpus only")
    sub.add_parser("test", help="Run tests only")
    sub.add_parser("status", help="Show results summary")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 1

    dispatch = {"run": cmd_run, "generate": cmd_generate, "test": cmd_test, "status": cmd_status}
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
