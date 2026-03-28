#!/usr/bin/env python3
"""
mdopus bisect fixer - surgical bit-exactness debugging.

Instead of throwing 200+ failures at Claude, this script:
1. Picks the simplest failing test case (silence, lowest bitrate)
2. Runs a single encode comparison
3. Captures the exact byte offset of first divergence
4. Asks Claude to trace the specific computation and fix it
5. Rebuilds, re-tests, repeats

Usage:
    python tools/bisect_fix.py run          # Full surgical fix loop
    python tools/bisect_fix.py scan         # Quick scan of all tests
    python tools/bisect_fix.py test <wav>   # Test a specific file
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap
import time
import logging
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "tests" / "vectors"
LOGS = ROOT / "logs"
COMPARE_BIN = ROOT / "target" / "debug" / "mdopus-compare"
if sys.platform == "win32":
    COMPARE_BIN = COMPARE_BIN.with_suffix(".exe")

MAX_FIX_ITERATIONS = 30
BITRATES = [16000, 32000, 64000, 128000]

# Ordered from simplest to most complex
TEST_PRIORITY = [
    ("48000hz_mono_silence", 48000, 1),
    ("48000hz_mono_sine440", 48000, 1),
    ("48000hz_mono_noise", 48000, 1),
    ("48000hz_stereo_silence", 48000, 2),
    ("24000hz_mono_silence", 24000, 1),
    ("16000hz_mono_silence", 16000, 1),
    ("8000hz_mono_silence", 8000, 1),
    ("48k_impulse", 48000, 1),
    ("48k_square1k", 48000, 1),
    ("48k_sweep", 48000, 1),
]


def setup_logging() -> logging.Logger:
    LOGS.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS / f"bisect_{timestamp}.log"

    logger = logging.getLogger("bisect")
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


def build(log: logging.Logger) -> bool:
    result = subprocess.run(
        ["cargo", "build"], cwd=str(ROOT),
        capture_output=True, text=True, timeout=300,
        encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        log.error(f"Build failed:\n{result.stderr[:2000]}")
        return False
    return True


def run_test(wav_name: str, bitrate: int, mode: str = "encode") -> dict:
    """Run a single comparison test, return detailed results."""
    wav_path = CORPUS_DIR / f"{wav_name}.wav"
    if not wav_path.exists():
        return {"passed": None, "output": f"File not found: {wav_path}"}

    cmd = [str(COMPARE_BIN)]
    if mode == "encode":
        cmd += ["encode", str(wav_path), "--bitrate", str(bitrate), "--complexity", "10"]
    elif mode == "roundtrip":
        cmd += ["roundtrip", str(wav_path), "--bitrate", str(bitrate)]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace",
        )
        output = (result.stdout + result.stderr).strip()
        passed = result.returncode == 0 and "FAIL" not in output
        return {"passed": passed, "output": output, "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        return {"passed": False, "output": "TIMEOUT"}
    except Exception as e:
        return {"passed": False, "output": str(e)}


def scan_all(log: logging.Logger) -> list[dict]:
    """Quick scan of all test cases, return sorted by difficulty."""
    results = []
    for name, sr, ch in TEST_PRIORITY:
        for br in BITRATES:
            r = run_test(name, br)
            r.update({"name": name, "bitrate": br, "sample_rate": sr, "channels": ch})
            status = "PASS" if r["passed"] else "FAIL"
            if r["passed"]:
                log.info(f"  PASS: {name} @{br}")
            results.append(r)

    # Also scan remaining test files not in priority list
    for wav in sorted(CORPUS_DIR.glob("*.wav")):
        stem = wav.stem
        if any(stem == t[0] for t in TEST_PRIORITY):
            continue
        for br in [64000]:  # Just one bitrate for non-priority
            r = run_test(stem, br)
            r.update({"name": stem, "bitrate": br})
            results.append(r)

    passed = sum(1 for r in results if r.get("passed"))
    failed = sum(1 for r in results if r.get("passed") is False)
    log.info(f"\nScan: {passed} passed, {failed} failed out of {len(results)}")
    return results


def find_simplest_failure(results: list[dict]) -> dict | None:
    """Find the simplest failing test case to focus on."""
    # Priority: silence first, then simple signals, lowest bitrate first
    for name, sr, ch in TEST_PRIORITY:
        for br in BITRATES:
            for r in results:
                if r.get("name") == name and r.get("bitrate") == br and r.get("passed") is False:
                    return r
    # Fallback: any failure
    for r in results:
        if r.get("passed") is False:
            return r
    return None


def invoke_claude(prompt: str, timeout: int = 3600) -> tuple[bool, str]:
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
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)


SURGICAL_FIX_PROMPT = textwrap.dedent("""\
    You are debugging a C-to-Rust port of the Opus audio codec. The Rust
    implementation compiles and runs but does not produce bit-exact output
    matching the C reference.

    ## Target test case
    Input: {wav_name}.wav ({sample_rate}Hz, {channels}ch)
    Bitrate: {bitrate}
    Mode: encode

    ## Comparison output
    ```
    {comparison_output}
    ```

    ## Your task
    1. The comparison shows the FIRST byte offset where C and Rust diverge
    2. Trace backwards from that divergence point:
       - What function produces those bytes?
       - What intermediate values feed into it?
       - Where does the Rust computation differ from C?
    3. Common root causes in C-to-Rust ports:
       - Integer overflow: C wraps silently, Rust panics. Use wrapping_mul/wrapping_add/wrapping_sub
       - Signed right shift: C is arithmetic (sign-extending), ensure Rust matches
       - Integer division: C truncates toward zero, Rust also truncates toward zero (same)
       - Cast truncation: `(int16_t)x` in C truncates, ensure `as i16` does the same
       - Operator precedence: C and Rust have different precedence for bitwise ops
       - Array indexing: off-by-one between C pointer arithmetic and Rust slice indexing
    4. Fix the ROOT CAUSE, not just the symptom at the divergence point

    ## Rules
    - Read the C reference code in reference/ and compare with src/
    - Make the SMALLEST change that fixes the divergence
    - Do NOT add debug prints, do NOT modify the test harness
    - Do NOT modify the C reference or bindings
    - After fixing, verify with: cargo build
""")


def surgical_fix_loop(log: logging.Logger) -> bool:
    """Main loop: find simplest failure, fix it, repeat."""
    for iteration in range(MAX_FIX_ITERATIONS):
        log.info(f"\n{'='*60}")
        log.info(f"Iteration {iteration + 1}/{MAX_FIX_ITERATIONS}")
        log.info(f"{'='*60}")

        # Scan
        log.info("Scanning tests...")
        results = scan_all(log)

        passed = sum(1 for r in results if r.get("passed"))
        failed = sum(1 for r in results if r.get("passed") is False)
        total = passed + failed

        if failed == 0:
            log.info(f"\nALL {total} TESTS PASSING!")
            return True

        # Find simplest failure
        target = find_simplest_failure(results)
        if not target:
            log.error("No failures found but count says otherwise?")
            return False

        log.info(f"\nFocusing on: {target['name']} @{target['bitrate']}")
        log.info(f"Output: {target['output'][:300]}")

        # Send to Claude
        prompt = SURGICAL_FIX_PROMPT.format(
            wav_name=target["name"],
            sample_rate=target.get("sample_rate", 48000),
            channels=target.get("channels", 1),
            bitrate=target["bitrate"],
            comparison_output=target["output"][:2000],
        )

        log.info("Sending to Claude for surgical fix...")
        t0 = time.time()
        ok, output = invoke_claude(prompt)
        elapsed = time.time() - t0
        log.info(f"Claude returned in {elapsed:.0f}s, ok={ok}")

        # Rebuild
        if not build(log):
            log.warning("Build failed after fix, sending build errors to Claude...")
            build_result = subprocess.run(
                ["cargo", "build"], cwd=str(ROOT),
                capture_output=True, text=True, timeout=300,
                encoding="utf-8", errors="replace",
            )
            fix_prompt = f"cargo build failed. Fix the errors:\n```\n{build_result.stderr[:3000]}\n```"
            invoke_claude(fix_prompt, timeout=600)
            if not build(log):
                log.error("Still broken after build fix attempt")
                continue

        # Re-test the target
        r = run_test(target["name"], target["bitrate"])
        if r["passed"]:
            log.info(f"  FIXED: {target['name']} @{target['bitrate']} now passes!")
        else:
            log.warning(f"  Still failing: {r['output'][:200]}")
            # Check if we made progress (higher match %)
            log.info("  Continuing to next iteration...")

        # Periodic commit
        if (iteration + 1) % 5 == 0:
            subprocess.run(
                ["git", "-c", "user.name=0x4D44", "-c", "user.email=martingdavidson@gmail.com",
                 "add", "-A"], cwd=str(ROOT), capture_output=True
            )
            subprocess.run(
                ["git", "-c", "user.name=0x4D44", "-c", "user.email=martingdavidson@gmail.com",
                 "commit", "-m", f"Bisect fix iteration {iteration + 1}: {passed}/{total} passing"],
                cwd=str(ROOT), capture_output=True
            )
            log.info(f"  Committed checkpoint: {passed}/{total} passing")

    log.error(f"Exhausted {MAX_FIX_ITERATIONS} iterations")
    return False


def cmd_run(args):
    log = setup_logging()
    log.info("mdopus surgical bisect fixer")
    log.info("=" * 60)

    if not build(log):
        log.error("Initial build failed")
        return 1

    return 0 if surgical_fix_loop(log) else 1


def cmd_scan(args):
    log = setup_logging()
    if not build(log):
        return 1
    scan_all(log)
    return 0


def cmd_test(args):
    log = setup_logging()
    if not build(log):
        return 1
    name = Path(args.wav).stem
    for br in BITRATES:
        r = run_test(name, br)
        status = "PASS" if r["passed"] else "FAIL"
        log.info(f"  {status}: {name} @{br}")
        if not r["passed"]:
            log.info(f"    {r['output'][:300]}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="mdopus surgical bisect fixer")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run", help="Full surgical fix loop")
    sub.add_parser("scan", help="Quick scan all tests")
    p_test = sub.add_parser("test", help="Test a specific WAV")
    p_test.add_argument("wav")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 1

    dispatch = {"run": cmd_run, "scan": cmd_scan, "test": cmd_test}
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
