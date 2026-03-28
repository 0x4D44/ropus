#!/usr/bin/env python3
"""
mdopus trace-fix loop - automated surgical bit-exactness debugging.

Cycle:
1. Run comparison harness, capture debug output from both C and Rust
2. Parse paired debug lines [C TAG] / [RS TAG]
3. Find the FIRST tag where C and Rust values diverge
4. If found: send Claude a targeted "fix this specific value" prompt
5. If not found but output still differs: send Claude "add more debug prints"
6. Rebuild, re-test, repeat

Usage:
    python tools/trace_fix.py run         # Full trace-fix loop
    python tools/trace_fix.py trace       # Just run and show divergence
"""

import argparse
import re
import subprocess
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "tests" / "vectors"
LOGS = ROOT / "logs"
COMPARE_BIN = ROOT / "target" / "debug" / "mdopus-compare"
if sys.platform == "win32":
    COMPARE_BIN = COMPARE_BIN.with_suffix(".exe")

MAX_ITERATIONS = 50
TEST_WAV = "48000hz_mono_silence"
TEST_BITRATE = 16000

# Regex to parse debug lines: [C TAG] key=val key=val ...
# or [RS TAG] key=val key=val ...
DEBUG_RE = re.compile(r'^\[(C|RS)\s+([A-Z_ ]+)\]\s*(.+)$')
KV_RE = re.compile(r'(\w+)=([-\d\[\], ]+)')


def setup_logging() -> logging.Logger:
    LOGS.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS / f"tracefix_{timestamp}.log"
    logger = logging.getLogger("tracefix")
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


def build(log: logging.Logger, clean: bool = False) -> bool:
    if clean:
        subprocess.run(["cargo", "clean"], cwd=str(ROOT),
                       capture_output=True, timeout=60)
    result = subprocess.run(
        ["cargo", "build"], cwd=str(ROOT),
        capture_output=True, text=True, timeout=300,
        encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        log.error(f"Build failed:\n{result.stderr[:2000]}")
        return False
    return True


def run_comparison(log: logging.Logger, wav: str = TEST_WAV,
                   bitrate: int = TEST_BITRATE) -> tuple[bool, str]:
    """Run comparison and return (passed, full_output)."""
    wav_path = CORPUS_DIR / f"{wav}.wav"
    cmd = [str(COMPARE_BIN), "encode", str(wav_path),
           "--bitrate", str(bitrate), "--complexity", "10"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace",
        )
        output = result.stderr + result.stdout
        passed = result.returncode == 0 and "FAIL" not in output
        return passed, output
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"


def parse_debug_lines(output: str) -> tuple[list[dict], list[dict]]:
    """Parse debug output into paired C and RS entries."""
    c_lines = []
    rs_lines = []
    for line in output.split('\n'):
        m = DEBUG_RE.match(line.strip())
        if not m:
            continue
        side, tag, rest = m.group(1), m.group(2).strip(), m.group(3)
        kvs = OrderedDict(KV_RE.findall(rest))
        entry = {"tag": tag, "raw": line.strip(), "values": kvs}
        if side == "C":
            c_lines.append(entry)
        else:
            rs_lines.append(entry)
    return c_lines, rs_lines


def find_first_divergence(c_lines: list[dict], rs_lines: list[dict]) -> dict | None:
    """Find the first debug checkpoint where C and Rust values differ."""
    # Match by tag in order
    c_by_tag = {}
    rs_by_tag = {}
    for entry in c_lines:
        tag = entry["tag"]
        if tag not in c_by_tag:
            c_by_tag[tag] = []
        c_by_tag[tag].append(entry)
    for entry in rs_lines:
        tag = entry["tag"]
        if tag not in rs_by_tag:
            rs_by_tag[tag] = []
        rs_by_tag[tag].append(entry)

    # Compare first occurrence of each tag
    for tag in c_by_tag:
        if tag not in rs_by_tag:
            continue
        c_entries = c_by_tag[tag]
        rs_entries = rs_by_tag[tag]
        count = min(len(c_entries), len(rs_entries))
        for i in range(count):
            c_vals = c_entries[i]["values"]
            rs_vals = rs_entries[i]["values"]
            for key in c_vals:
                if key in rs_vals and c_vals[key] != rs_vals[key]:
                    return {
                        "tag": tag,
                        "occurrence": i,
                        "key": key,
                        "c_value": c_vals[key],
                        "rs_value": rs_vals[key],
                        "c_line": c_entries[i]["raw"],
                        "rs_line": rs_entries[i]["raw"],
                        "c_all": c_vals,
                        "rs_all": rs_vals,
                    }
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
        return result.returncode == 0, (result.stdout + result.stderr)
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)


FIX_PROMPT = """\
You are debugging a C-to-Rust port of the Opus audio codec to achieve
bit-exact output. We have debug checkpoints in both C and Rust that
emit intermediate values at the same computation points.

## Divergence found

At checkpoint **[{tag}]** (occurrence #{occurrence}), the variable
**{key}** differs:
- C reference: {key}={c_value}
- Rust port:   {key}={rs_value}

Full C checkpoint:   {c_line}
Full Rust checkpoint: {rs_line}

## Your task

1. Find the code that computes `{key}` in both C and Rust:
   - C: look in reference/silk/ or reference/celt/ for the function
     that emits the [C {tag}] debug line
   - Rust: look in src/silk/ or src/celt/ for the [RS {tag}] debug line
2. Trace backward from {key}: what intermediate values feed into it?
3. Find the EXACT line where C and Rust diverge
4. Common causes: wrapping arithmetic, missing terms, wrong constants,
   different operator precedence, missing code paths
5. Fix the ROOT CAUSE in the Rust code

## Rules
- Make the SMALLEST change that fixes the divergence
- Do NOT remove or modify ANY debug print lines (eprintln!/fprintf)
- Do NOT modify C reference code
- Do NOT modify the test harness
- After fixing, the code must compile: cargo build
"""

INSTRUMENT_PROMPT = """\
You are debugging a C-to-Rust port of the Opus audio codec. The Rust
encoder produces different bytes from the C reference at byte offset
{offset}. All existing debug checkpoints match, so the divergence is
in code that runs AFTER the noise shape analysis / gain tweaking.

## Byte comparison (first divergence)
```
{tail_output}
```

## What we know
- The SILK noise_shape gains, SNR adjustment, and gain_mult all match
- The divergence is in the SILK encoding pipeline AFTER gain computation:
  process_gains, find_pred_coefs, NSQ (noise shaping quantizer),
  encode_indices, encode_pulses, or the range coder output
- Both encoders produce the same packet LENGTH but different content

## Your task
1. Read the SILK encode pipeline in both C (reference/silk/) and Rust (src/silk/encoder.rs)
2. The encode path after noise_shape is roughly:
   - silk_process_gains_FIX (gain smoothing/limiting)
   - silk_find_pred_coefs_FIX (LPC analysis, residual energy)
   - silk_NSQ / silk_NSQ_del_dec (noise shaping quantizer - produces excitation)
   - silk_encode_indices (writes indices to range coder)
   - silk_encode_pulses (writes excitation pulses to range coder)
3. Compare C and Rust implementations of these functions
4. Look for: wrapping arithmetic differences, missing code, wrong constants,
   different control flow, off-by-one errors
5. Fix the ROOT CAUSE in the Rust code

## Rules
- Make the SMALLEST change that fixes the issue
- Do NOT remove debug prints
- Do NOT modify C reference or test harness
- Must compile: cargo build
"""


def trace_fix_loop(log: logging.Logger) -> bool:
    """Main loop: trace → find divergence → fix → repeat."""
    for iteration in range(MAX_ITERATIONS):
        log.info(f"\n{'='*60}")
        log.info(f"Iteration {iteration + 1}/{MAX_ITERATIONS}")
        log.info(f"{'='*60}")

        # Run comparison
        passed, output = run_comparison(log)
        if passed:
            log.info("TEST PASSES! Checking other test cases...")
            # Try a few more
            all_pass = True
            for wav, br in [
                ("48000hz_mono_silence", 64000),
                ("48000hz_mono_sine440", 16000),
                ("48000hz_mono_noise", 16000),
                ("48000hz_stereo_silence", 16000),
            ]:
                p, _ = run_comparison(log, wav, br)
                status = "PASS" if p else "FAIL"
                log.info(f"  {wav} @{br}: {status}")
                if not p:
                    all_pass = False
            if all_pass:
                log.info("\nALL TESTS PASSING!")
                return True
            log.info("Primary test passes, switching to next failing case...")
            # TODO: switch TEST_WAV/BITRATE to a failing case
            continue

        # Parse debug output
        c_lines, rs_lines = parse_debug_lines(output)
        log.info(f"  Parsed {len(c_lines)} C debug lines, {len(rs_lines)} RS debug lines")

        # Find divergence
        div = find_first_divergence(c_lines, rs_lines)

        if div:
            log.info(f"  DIVERGENCE at [{div['tag']}] occurrence #{div['occurrence']}:")
            log.info(f"    {div['key']}: C={div['c_value']}, RS={div['rs_value']}")
            log.info(f"    C:  {div['c_line']}")
            log.info(f"    RS: {div['rs_line']}")

            # Send targeted fix prompt
            prompt = FIX_PROMPT.format(**div)
            log.info("  Sending targeted fix to Claude...")
            t0 = time.time()
            ok, claude_out = invoke_claude(prompt)
            elapsed = time.time() - t0
            log.info(f"  Claude returned in {elapsed:.0f}s, ok={ok}")

        else:
            log.info("  No divergence in debug checkpoints — need finer instrumentation")

            # Extract comparison tail for context
            tail_lines = output.strip().split('\n')
            tail_output = '\n'.join(tail_lines[-20:])

            # Find offset from FAIL line
            offset = "unknown"
            for line in tail_lines:
                if "FAIL at offset" in line:
                    m = re.search(r'offset (\d+)', line)
                    if m:
                        offset = m.group(1)

            last_c = c_lines[-1]["raw"] if c_lines else "(none)"
            last_rs = rs_lines[-1]["raw"] if rs_lines else "(none)"

            prompt = INSTRUMENT_PROMPT.format(
                offset=offset,
                last_c_line=last_c,
                last_rs_line=last_rs,
                tail_output=tail_output,
            )
            log.info("  Sending instrumentation request to Claude...")
            t0 = time.time()
            ok, claude_out = invoke_claude(prompt, timeout=1800)
            elapsed = time.time() - t0
            log.info(f"  Claude returned in {elapsed:.0f}s, ok={ok}")

            # Need clean rebuild for C changes
            if not build(log, clean=True):
                log.warning("  Clean build failed after instrumentation, trying fix...")
                build_err = subprocess.run(
                    ["cargo", "build"], cwd=str(ROOT),
                    capture_output=True, text=True, timeout=300,
                    encoding="utf-8", errors="replace",
                ).stderr[:3000]
                invoke_claude(
                    f"cargo build failed after adding debug prints. Fix errors:\n```\n{build_err}\n```",
                    timeout=600,
                )
                if not build(log, clean=True):
                    log.error("  Still broken, skipping iteration")
                    continue
            continue

        # Rebuild after fix
        if not build(log):
            log.warning("  Build failed after fix, asking Claude to fix build...")
            build_err = subprocess.run(
                ["cargo", "build"], cwd=str(ROOT),
                capture_output=True, text=True, timeout=300,
                encoding="utf-8", errors="replace",
            ).stderr[:3000]
            invoke_claude(
                f"cargo build failed. Fix errors:\n```\n{build_err}\n```",
                timeout=600,
            )
            if not build(log):
                log.error("  Still broken after build fix")
                continue

        # Re-test to see progress
        new_passed, new_output = run_comparison(log)
        if new_passed:
            log.info(f"  TARGET TEST NOW PASSES!")
        else:
            # Check if divergence moved
            new_c, new_rs = parse_debug_lines(new_output)
            new_div = find_first_divergence(new_c, new_rs)
            if new_div:
                if new_div["tag"] != div["tag"] or new_div["key"] != div["key"]:
                    log.info(f"  Progress! Divergence moved: [{new_div['tag']}].{new_div['key']}")
                else:
                    log.info(f"  Same divergence point, values may have changed")
                    log.info(f"    C={new_div['c_value']}, RS={new_div['rs_value']}")
            else:
                log.info("  Debug checkpoints now match — need finer instrumentation next")

        # Periodic commit
        if (iteration + 1) % 3 == 0:
            subprocess.run(
                ["git", "-c", "user.name=0x4D44", "-c", "user.email=martingdavidson@gmail.com",
                 "add", "-A"], cwd=str(ROOT), capture_output=True
            )
            subprocess.run(
                ["git", "-c", "user.name=0x4D44", "-c", "user.email=martingdavidson@gmail.com",
                 "commit", "-m", f"Trace-fix iteration {iteration + 1}\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"],
                cwd=str(ROOT), capture_output=True
            )
            log.info(f"  Committed checkpoint")

    log.error(f"Exhausted {MAX_ITERATIONS} iterations")
    return False


def cmd_run(args):
    log = setup_logging()
    log.info("mdopus trace-fix loop")
    if not build(log):
        log.error("Build failed")
        return 1
    return 0 if trace_fix_loop(log) else 1


def cmd_trace(args):
    log = setup_logging()
    if not build(log):
        return 1
    passed, output = run_comparison(log)
    c_lines, rs_lines = parse_debug_lines(output)
    log.info(f"Parsed {len(c_lines)} C lines, {len(rs_lines)} RS lines")
    div = find_first_divergence(c_lines, rs_lines)
    if div:
        log.info(f"First divergence at [{div['tag']}].{div['key']}:")
        log.info(f"  C:  {div['c_value']}")
        log.info(f"  RS: {div['rs_value']}")
        log.info(f"  C line:  {div['c_line']}")
        log.info(f"  RS line: {div['rs_line']}")
    elif not passed:
        log.info("Debug checkpoints match but output differs — need more checkpoints")
    else:
        log.info("Test passes!")
    return 0


def main():
    parser = argparse.ArgumentParser(description="mdopus trace-fix loop")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run", help="Full trace-fix loop")
    sub.add_parser("trace", help="Just show current divergence")
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 1
    return {"run": cmd_run, "trace": cmd_trace}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
