#!/usr/bin/env bash
#
# Campaign 7 launcher — overnight fuzz session 2026-04-14
# 8-hour run. Invokes pre-built libFuzzer binaries directly.
#

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CAMPAIGN_DIR="$ROOT/logs/fuzz-findings/campaign7"
CORPUS_DIR="$ROOT/tests/fuzz/corpus"
BIN_DIR="$ROOT/tests/fuzz/target/x86_64-pc-windows-msvc/release"

# Windows ASan setup
ASAN_DIR="/c/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64"
export PATH="$ASAN_DIR:$PATH"
export ASAN_OPTIONS="detect_odr_violation=0:detect_leaks=0"

# Panic capture for Windows
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

    echo "[$(date +%H:%M:%S)] Starting $target (${duration}s / $((duration/3600))h$((duration%3600/60))m)..."

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

echo "=== Campaign 7: Overnight Fuzz Session (8h) ==="
echo "Start: $(date)"
echo "Binary dir: $BIN_DIR"
echo ""

# Phase 1: Differential targets (highest value) — full 8h each
echo "--- Phase 1: Differential targets (28800s / 8h each) ---"
run_target fuzz_decode 28800
run_target fuzz_encode 28800
run_target fuzz_roundtrip 28800

# Phase 2: Safety targets — 6h each
echo ""
echo "--- Phase 2: Safety targets (21600s / 6h each) ---"
run_target fuzz_decode_safety 21600
run_target fuzz_encode_safety 21600
run_target fuzz_roundtrip_safety 21600

# Phase 2b: Lighter differential — 5h each
echo ""
echo "--- Phase 2b: Lighter differential (18000s / 5h each) ---"
run_target fuzz_repacketizer 18000
run_target fuzz_packet_parse 18000
run_target fuzz_encode_multiframe 18000

echo ""
echo "All 9 targets launched."
echo "Expected completion: ~$(date -d '+8 hours' +%H:%M 2>/dev/null || echo '06:00')"
echo "Campaign dir: $CAMPAIGN_DIR"
