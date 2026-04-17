#!/usr/bin/env bash
#
# Campaign 8 launcher — overnight 24-hour fuzz session 2026-04-16
# Direct invocation of pre-built libFuzzer binaries.
#

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CAMPAIGN_DIR="$ROOT/logs/fuzz-findings/campaign8"
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

    local pid=$!
    echo "$pid" > "$out_dir/pid"
    echo "  PID=$pid, log=$out_dir/fuzz.log"
}

echo "=== Campaign 8: 24-hour Overnight Fuzz Session ==="
echo "Start: $(date)"
echo "Binary dir: $BIN_DIR"
echo ""

# Phase 1: Differential targets (highest value) — full 24h each
echo "--- Phase 1: Differential targets (86400s / 24h each) ---"
run_target fuzz_decode 86400
run_target fuzz_encode 86400
run_target fuzz_roundtrip 86400

# Phase 2: Safety targets — 20h each
echo ""
echo "--- Phase 2: Safety targets (72000s / 20h each) ---"
run_target fuzz_decode_safety 72000
run_target fuzz_encode_safety 72000
run_target fuzz_roundtrip_safety 72000

# Phase 2b: Lighter differential — 18h each
echo ""
echo "--- Phase 2b: Lighter differential (64800s / 18h each) ---"
run_target fuzz_repacketizer 64800
run_target fuzz_packet_parse 64800
run_target fuzz_encode_multiframe 64800

echo ""
echo "All 9 targets launched."
echo "Campaign dir: $CAMPAIGN_DIR"
