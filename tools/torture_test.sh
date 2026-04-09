#!/usr/bin/env bash
# Torture test suite for mdopus
# Usage: bash tools/torture_test.sh [--duration MINUTES]
#
# Runs a battery of stress tests against the mdopus-compare binary:
#   1. Torture soak (random configs for N minutes)
#   2. Mode transitions (SILK<->CELT switching)
#   3. Full parameter sweep
#   4. Packet loss concealment
#   5. Forward error correction
#   6. Discontinuous transmission
#
# Exits 0 if all pass, 1 if any fail.

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

DURATION_MINS=30

while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --duration requires a value (minutes)" >&2
                exit 1
            fi
            DURATION_MINS="$2"
            shift 2
            ;;
        --duration=*)
            DURATION_MINS="${1#*=}"
            shift
            ;;
        -h|--help)
            echo "Usage: bash tools/torture_test.sh [--duration MINUTES]"
            echo "  --duration MINUTES  Duration for the torture soak (default: 30)"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# Validate duration is a positive integer
if ! [[ "$DURATION_MINS" =~ ^[0-9]+$ ]] || [[ "$DURATION_MINS" -eq 0 ]]; then
    echo "Error: duration must be a positive integer, got '$DURATION_MINS'" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== Building mdopus (release) ==="
if ! cargo build --release 2>&1; then
    echo "FATAL: cargo build --release failed" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Locate binary
# ---------------------------------------------------------------------------

if [[ "$(uname -s)" == MINGW* ]] || [[ "$(uname -s)" == MSYS* ]] || [[ "$(uname -s)" == CYGWIN* ]] || [[ -n "${OS:-}" && "${OS:-}" == "Windows_NT" ]]; then
    COMPARE_BIN="$ROOT/target/release/mdopus-compare.exe"
else
    COMPARE_BIN="$ROOT/target/release/mdopus-compare"
fi

if [[ ! -x "$COMPARE_BIN" ]]; then
    echo "FATAL: binary not found at $COMPARE_BIN" >&2
    exit 1
fi

echo "Using binary: $COMPARE_BIN"
echo ""

# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

TESTS_RUN=0
TESTS_PASSED=0
declare -a TEST_NAMES=()
declare -a TEST_RESULTS=()

run_test() {
    local name="$1"
    shift
    local desc="$1"
    shift

    TESTS_RUN=$((TESTS_RUN + 1))
    TEST_NAMES+=("$desc")

    echo "--- [$TESTS_RUN] $desc ---"
    echo "  Command: $*"

    local rc=0
    "$@" || rc=$?

    if [[ $rc -eq 0 ]]; then
        echo "  Result: PASS"
        TEST_RESULTS+=("PASS")
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo "  Result: FAIL (exit code $rc)"
        TEST_RESULTS+=("FAIL")
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------

DURATION_SECS=$((DURATION_MINS * 60))
VECTORS_DIR="$ROOT/tests/vectors"

echo "=== Torture Test Suite ==="
echo "  Duration: ${DURATION_MINS}m (${DURATION_SECS}s) for soak test"
echo "  Vectors:  $VECTORS_DIR"
echo ""

# 1. Torture soak — mono 48kHz
run_test "torture_mono" \
    "Torture soak mono 48k (${DURATION_MINS} min)" \
    "$COMPARE_BIN" torture --duration "$DURATION_SECS" --seed 42

# 2. Torture soak — stereo 48kHz
run_test "torture_stereo" \
    "Torture soak stereo 48k (${DURATION_MINS} min)" \
    "$COMPARE_BIN" torture --duration "$DURATION_SECS" --seed 43 --channels 2

# 3. Mode transitions
run_test "mode_transitions" \
    "Mode transitions" \
    "$COMPARE_BIN" transitions

# 4. Full sweep
run_test "full_sweep" \
    "Full configuration sweep" \
    "$COMPARE_BIN" sweep

# 5. PLC — 48kHz
run_test "plc_48k" \
    "PLC (48k mono noise)" \
    "$COMPARE_BIN" plc "$VECTORS_DIR/48000hz_mono_noise.wav"

# 6. PLC — 8kHz (telephony)
run_test "plc_8k" \
    "PLC (8k mono noise)" \
    "$COMPARE_BIN" plc "$VECTORS_DIR/8000hz_mono_noise.wav" --bitrate 12000

# 7. FEC test
run_test "fec" \
    "FEC (48k mono noise)" \
    "$COMPARE_BIN" fec "$VECTORS_DIR/48000hz_mono_noise.wav"

# 8. DTX test
run_test "dtx" \
    "DTX (generated signal)" \
    "$COMPARE_BIN" dtx generate

# 9. Longsoak — 48kHz
run_test "longsoak_48k" \
    "Longsoak 48k (60s)" \
    "$COMPARE_BIN" longsoak --duration 60

# 10. Longsoak — 16kHz
run_test "longsoak_16k" \
    "Longsoak 16k (60s)" \
    "$COMPARE_BIN" longsoak --duration 60 --sample-rate 16000

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo "=== Torture Test Summary ==="
for i in "${!TEST_NAMES[@]}"; do
    if [[ "${TEST_RESULTS[$i]}" == "PASS" ]]; then
        echo "  [PASS] ${TEST_NAMES[$i]}"
    else
        echo "  [FAIL] ${TEST_NAMES[$i]}"
    fi
done
echo ""
echo "Result: ${TESTS_PASSED}/${TESTS_RUN} passed"

if [[ "$TESTS_PASSED" -eq "$TESTS_RUN" ]]; then
    echo "ALL TESTS PASSED"
    exit 0
else
    echo "SOME TESTS FAILED"
    exit 1
fi
