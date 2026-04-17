#!/usr/bin/env bash
#
# Campaign 8 Phase-2 launcher — extended 36h differential run 2026-04-17/19
#
# Purpose: after the original 24h campaign 8 differential targets exit naturally
# at approximately 2026-04-17 21:06, rebuild them from the current master tip
# and relaunch for 36h in the campaign8_ext artifact tree. Seed corpora stay
# under tests/fuzz/corpus/<target>/ so they carry forward enrichment from the
# phase-1 run.
#
# Prerequisites:
#   - fuzz_decode / fuzz_encode / fuzz_roundtrip processes must already have
#     exited (their .exe files must be unlocked so `cargo +nightly fuzz build`
#     can overwrite them).
#   - Source-level Bug D fix and panic-capture hooks must be in place (applied
#     via the campaign8-extension session).
#
# Usage: ./tools/campaign8_phase2_launcher.sh
#

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CAMPAIGN_DIR="$ROOT/logs/fuzz-findings/campaign8_ext"
CORPUS_DIR="$ROOT/tests/fuzz/corpus"
BIN_DIR="$ROOT/tests/fuzz/target/x86_64-pc-windows-msvc/release"

# Windows ASan setup
ASAN_DIR="/c/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64"
export PATH="$ASAN_DIR:$PATH"
export ASAN_OPTIONS="detect_odr_violation=0:detect_leaks=0"

# Panic capture for Windows (when a Rust panic aborts without libFuzzer saving
# the crashing input)
export FUZZ_PANIC_CAPTURE_DIR="$ROOT/fuzz_crashes"
mkdir -p "$FUZZ_PANIC_CAPTURE_DIR"

# Safety: refuse to run if any of the phase-1 targets are still alive.
for t in fuzz_decode fuzz_encode fuzz_roundtrip; do
    if tasklist 2>/dev/null | grep -qE "^${t}\.exe[[:space:]]"; then
        echo "ERROR: ${t}.exe still running — phase-1 has not completed."
        echo "Wait for the original 24h session to exit (~21:06) before relaunching."
        exit 1
    fi
done

echo "=== Campaign 8 Phase-2: 36-hour Differential Extension ==="
echo "Start: $(date)"
echo ""

# --- Rebuild from current master tip ---
echo "--- Rebuilding differential fuzz binaries ---"
for t in fuzz_decode fuzz_encode fuzz_roundtrip; do
    echo "  Building $t..."
    if ! cargo +nightly fuzz build --fuzz-dir tests/fuzz "$t" 2>&1 | tail -3; then
        echo "ERROR: build failed for $t"
        exit 1
    fi
done
echo ""

run_target() {
    local target="$1"
    local duration="$2"
    local out_dir="$CAMPAIGN_DIR/$target"
    local corpus_dir="$CORPUS_DIR/$target"
    local binary="$BIN_DIR/${target}.exe"

    mkdir -p "$out_dir" "$corpus_dir"

    if [[ ! -f "$binary" ]]; then
        echo "  ERROR: binary not found: $binary"
        return 1
    fi

    local hrs=$((duration/3600))
    local mins=$((duration%3600/60))
    echo "[$(date +%H:%M:%S)] Starting $target (${duration}s / ${hrs}h${mins}m)..."

    nohup "$binary" \
        "$corpus_dir" \
        -max_total_time="$duration" \
        -max_len=16384 \
        -artifact_prefix="$out_dir/" \
        -print_final_stats=1 \
        > "$out_dir/fuzz.log" 2>&1 &

    local wrapper_pid=$!
    echo "  wrapper_pid=$wrapper_pid, log=$out_dir/fuzz.log"
    # Note: real exe PID is captured below after giving processes time to start.
}

# --- Launch all three differential targets for 36h (129600s) ---
echo "--- Launching differential targets (129600s / 36h each) ---"
run_target fuzz_decode 129600
run_target fuzz_encode 129600
run_target fuzz_roundtrip 129600

echo ""
echo "Waiting 15s for processes to fully start..."
sleep 15

# --- Capture actual exe PIDs (wrapper $! is useless for monitoring) ---
for t in fuzz_decode fuzz_encode fuzz_roundtrip; do
    pid=$(tasklist 2>/dev/null | awk -v t="${t}.exe" '$1==t {print $2; exit}')
    if [[ -n "$pid" ]]; then
        echo "$pid" > "$CAMPAIGN_DIR/$t/pid"
        echo "  $t: exe_pid=$pid"
    else
        echo "  WARN: $t exe not found in tasklist — may still be starting"
    fi
done

echo ""
echo "Phase-2 launched."
echo "Campaign dir: $CAMPAIGN_DIR"
echo "Expected completion: $(date -d '+36 hours' 2>/dev/null || date)"
