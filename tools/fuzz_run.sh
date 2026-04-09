#!/usr/bin/env bash
#
# Fuzz testing runner for mdopus.
#
# Runs all (or selected) libFuzzer targets with seed corpus, collects findings,
# and reports results.
#
# Usage:
#   ./tools/fuzz_run.sh                        # all targets, 10 min each
#   ./tools/fuzz_run.sh --duration 300         # all targets, 5 min each
#   ./tools/fuzz_run.sh --target fuzz_decode   # single target
#   ./tools/fuzz_run.sh --target fuzz_decode --target fuzz_encode  # multiple
#   ./tools/fuzz_run.sh --jobs 4               # parallel libFuzzer jobs
#   ./tools/fuzz_run.sh --no-diff              # skip differential mode (no C ref needed)
#   ./tools/fuzz_run.sh --list                 # list available targets
#   ./tools/fuzz_run.sh --check-crashes        # re-run crash corpus only (fast regression check)
#
# Environment:
#   FUZZ_DURATION   — seconds per target (default: 600)
#   FUZZ_JOBS       — parallel libFuzzer workers (default: 1)
#   FUZZ_MAX_LEN    — max input length in bytes (default: 16384)
#
# Requirements:
#   - Linux or macOS (libFuzzer sanitizers not available on Windows)
#   - Rust nightly toolchain
#   - cargo-fuzz: cargo install cargo-fuzz
#   - C reference source in reference/ (clone https://github.com/xiph/opus)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FUZZ_DIR="$ROOT/tests/fuzz"
CORPUS_DIR="$FUZZ_DIR/corpus"
CRASHES_DIR="$FUZZ_DIR/crashes"
FINDINGS_DIR="$ROOT/logs/fuzz-findings"

ALL_TARGETS=(
    fuzz_decode
    fuzz_encode
    fuzz_roundtrip
    fuzz_repacketizer
    fuzz_packet_parse
)

# Defaults
DURATION="${FUZZ_DURATION:-600}"
JOBS="${FUZZ_JOBS:-1}"
MAX_LEN="${FUZZ_MAX_LEN:-16384}"
TARGETS=()
CHECK_CRASHES_ONLY=false
LIST_ONLY=false
NO_DIFF=false

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration)
            DURATION="$2"; shift 2 ;;
        --target)
            TARGETS+=("$2"); shift 2 ;;
        --jobs)
            JOBS="$2"; shift 2 ;;
        --no-diff)
            NO_DIFF=true; shift ;;
        --list)
            LIST_ONLY=true; shift ;;
        --check-crashes)
            CHECK_CRASHES_ONLY=true; shift ;;
        --max-len)
            MAX_LEN="$2"; shift 2 ;;
        -h|--help)
            head -20 "$0" | tail -17; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Use all targets if none specified
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=("${ALL_TARGETS[@]}")
fi

if $LIST_ONLY; then
    echo "Available fuzz targets:"
    for t in "${ALL_TARGETS[@]}"; do
        corpus_count=0
        if [[ -d "$CORPUS_DIR/$t" ]]; then
            corpus_count=$(find "$CORPUS_DIR/$t" -type f 2>/dev/null | wc -l)
        fi
        crash_count=0
        if [[ -d "$CRASHES_DIR/$t" ]]; then
            crash_count=$(find "$CRASHES_DIR/$t" -type f -name '*.bin' 2>/dev/null | wc -l)
        fi
        printf "  %-25s  corpus: %3d  crashes: %3d\n" "$t" "$corpus_count" "$crash_count"
    done
    exit 0
fi

# --------------------------------------------------------------------------- #
# Pre-flight checks
# --------------------------------------------------------------------------- #
echo "=== mdopus fuzz runner ==="
echo "  Duration per target: ${DURATION}s"
echo "  Parallel jobs:       $JOBS"
echo "  Max input length:    $MAX_LEN bytes"
echo "  Targets:             ${TARGETS[*]}"
echo ""

# Ensure cargo-fuzz is installed
if ! cargo fuzz --version &>/dev/null; then
    echo "ERROR: cargo-fuzz not found. Install with: cargo install cargo-fuzz" >&2
    exit 1
fi

# Create findings directory for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$FINDINGS_DIR/$TIMESTAMP"
mkdir -p "$RUN_DIR"

echo "Findings will be saved to: $RUN_DIR"
echo ""

# --------------------------------------------------------------------------- #
# Build fuzz targets
# --------------------------------------------------------------------------- #
echo "--- Building fuzz targets ---"
cd "$ROOT"

# Common cargo fuzz args — our fuzz dir is non-standard
CARGO_FUZZ=(cargo fuzz --fuzz-dir tests/fuzz)

# Verify targets build before starting long runs
for target in "${TARGETS[@]}"; do
    echo "  Building $target..."
    "${CARGO_FUZZ[@]}" build "$target" 2>&1 | tail -1
done
echo ""

# --------------------------------------------------------------------------- #
# Crash regression check mode
# --------------------------------------------------------------------------- #
if $CHECK_CRASHES_ONLY; then
    echo "--- Crash regression check ---"
    FAILURES=0
    for target in "${TARGETS[@]}"; do
        crash_dir="$CRASHES_DIR/$target"
        if [[ ! -d "$crash_dir" ]] || [[ -z "$(ls -A "$crash_dir" 2>/dev/null)" ]]; then
            echo "  $target: no crashes to check (skip)"
            continue
        fi
        crash_files=$(find "$crash_dir" -type f -name '*.bin' | wc -l)
        echo "  $target: checking $crash_files crash files..."

        fail_count=0
        for crash_file in "$crash_dir"/*.bin; do
            if ! "${CARGO_FUZZ[@]}" run "$target" -- -runs=0 "$crash_file" &>/dev/null; then
                echo "    FAIL: $crash_file still crashes!"
                ((fail_count++))
            fi
        done

        if [[ $fail_count -eq 0 ]]; then
            echo "    OK: all $crash_files crash files handled without crash"
        else
            echo "    FAILED: $fail_count/$crash_files still crash"
            ((FAILURES += fail_count))
        fi
    done

    if [[ $FAILURES -gt 0 ]]; then
        echo ""
        echo "RESULT: $FAILURES crash regressions found!"
        exit 1
    else
        echo ""
        echo "RESULT: All crash regressions pass."
        exit 0
    fi
fi

# --------------------------------------------------------------------------- #
# Main fuzzing loop
# --------------------------------------------------------------------------- #
TOTAL_FINDINGS=0
SUMMARY=""

for target in "${TARGETS[@]}"; do
    echo "--- Fuzzing: $target (${DURATION}s, $JOBS jobs) ---"

    target_corpus="$CORPUS_DIR/$target"
    target_crashes="$CRASHES_DIR/$target"
    target_findings="$RUN_DIR/$target"
    mkdir -p "$target_findings"
    mkdir -p "$target_corpus"
    mkdir -p "$target_crashes"

    seed_count=0
    if [[ -d "$target_corpus" ]]; then
        seed_count=$(find "$target_corpus" -type f 2>/dev/null | wc -l)
    fi
    echo "  Seed corpus: $seed_count files"

    # Build libFuzzer arguments
    FUZZ_ARGS=(
        -max_total_time="$DURATION"
        -max_len="$MAX_LEN"
        -jobs="$JOBS"
        -workers="$JOBS"
        -artifact_prefix="$target_findings/"
        -print_final_stats=1
    )

    # Run the fuzzer, capturing output
    log_file="$target_findings/fuzz.log"

    set +e
    "${CARGO_FUZZ[@]}" run "$target" \
        "$target_corpus" \
        -- "${FUZZ_ARGS[@]}" \
        2>&1 | tee "$log_file"
    fuzz_exit=$?
    set -e

    # Count new findings
    new_findings=$(find "$target_findings" -maxdepth 1 -name 'crash-*' -o -name 'leak-*' -o -name 'timeout-*' -o -name 'oom-*' 2>/dev/null | wc -l)

    # Copy crash artifacts to permanent crash corpus
    if [[ $new_findings -gt 0 ]]; then
        echo "  FOUND $new_findings new findings!"
        for artifact in "$target_findings"/crash-* "$target_findings"/leak-* "$target_findings"/timeout-* "$target_findings"/oom-*; do
            if [[ -f "$artifact" ]]; then
                base=$(basename "$artifact")
                cp "$artifact" "$target_crashes/${TIMESTAMP}_${base}.bin"
            fi
        done
    else
        echo "  No new findings."
    fi

    TOTAL_FINDINGS=$((TOTAL_FINDINGS + new_findings))
    SUMMARY+="  $target: $new_findings findings (exit=$fuzz_exit)\n"
    echo ""
done

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
echo "=== Fuzz run complete ==="
echo "  Run ID: $TIMESTAMP"
echo "  Total findings: $TOTAL_FINDINGS"
echo ""
echo "Per-target results:"
echo -e "$SUMMARY"

if [[ $TOTAL_FINDINGS -gt 0 ]]; then
    echo "Findings saved to: $RUN_DIR"
    echo "Crash regressions copied to: $CRASHES_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Examine findings: ls $RUN_DIR/<target>/"
    echo "  2. Reproduce: cargo fuzz --fuzz-dir tests/fuzz run <target> <finding-file>"
    echo "  3. After fixing, run: $0 --check-crashes"
    exit 1
else
    echo "All clear - no issues found in this run."
    exit 0
fi
