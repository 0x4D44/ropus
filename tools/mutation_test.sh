#!/usr/bin/env bash
#
# Mutation testing runner for mdopus.
#
# Runs cargo-mutants on prioritized source files and collects results.
# Designed for overnight/unattended runs — each file's results are saved
# incrementally so partial runs are still useful.
#
# Usage:
#   ./tools/mutation_test.sh                    # all priority modules
#   ./tools/mutation_test.sh --file silk/common # single module
#   ./tools/mutation_test.sh --list             # list modules & mutant counts
#   ./tools/mutation_test.sh --jobs 8           # parallel threads
#
# Results are saved to logs/mutation-testing/<timestamp>/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$ROOT/logs/mutation-testing"

# Priority order from testing improvement roadmap (Phase 5)
ALL_MODULES=(
    "src/silk/common.rs"
    "src/opus/decoder.rs"
    "src/opus/encoder.rs"
    "src/silk/decoder.rs"
    "src/celt/decoder.rs"
    "src/celt/encoder.rs"
)

# Defaults
JOBS=4
TIMEOUT_MULT=3
MODULES=()
LIST_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --file)
            # Accept short name (silk/common) or full path (src/silk/common.rs)
            f="$2"
            [[ "$f" != src/* ]] && f="src/$f"
            [[ "$f" != *.rs ]] && f="$f.rs"
            MODULES+=("$f")
            shift 2 ;;
        --jobs)
            JOBS="$2"; shift 2 ;;
        --timeout-multiplier)
            TIMEOUT_MULT="$2"; shift 2 ;;
        --list)
            LIST_ONLY=true; shift ;;
        -h|--help)
            head -14 "$0" | tail -11; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

[[ ${#MODULES[@]} -eq 0 ]] && MODULES=("${ALL_MODULES[@]}")

cd "$ROOT"

if $LIST_ONLY; then
    echo "Priority modules for mutation testing:"
    for mod in "${ALL_MODULES[@]}"; do
        count=$(cargo mutants --file "$mod" --list 2>/dev/null | wc -l)
        printf "  %-30s %4d mutants\n" "$mod" "$count"
    done
    exit 0
fi

# Verify cargo-mutants is installed
if ! cargo mutants --version &>/dev/null; then
    echo "ERROR: cargo-mutants not found. Install with: cargo install cargo-mutants" >&2
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$RESULTS_DIR/$TIMESTAMP"
mkdir -p "$RUN_DIR"

echo "=== mdopus mutation testing ==="
echo "  Modules:    ${MODULES[*]}"
echo "  Jobs:       $JOBS"
echo "  Timeout:    ${TIMEOUT_MULT}x"
echo "  Results:    $RUN_DIR"
echo ""

SUMMARY=""

for mod in "${MODULES[@]}"; do
    mod_short="${mod#src/}"
    mod_short="${mod_short%.rs}"
    mod_safe="${mod_short//\//_}"
    log_file="$RUN_DIR/${mod_safe}.log"

    count=$(cargo mutants --file "$mod" --list 2>/dev/null | wc -l)
    echo "--- $mod_short ($count mutants) ---"

    start_time=$(date +%s)

    set +e
    cargo mutants --file "$mod" -j"$JOBS" --timeout-multiplier "$TIMEOUT_MULT" 2>&1 | tee "$log_file"
    exit_code=$?
    set -e

    end_time=$(date +%s)
    elapsed=$((end_time - start_time))

    # Tally results
    caught=$(grep -c '^CAUGHT' "$log_file" 2>/dev/null || true)
    missed=$(grep -c '^MISSED' "$log_file" 2>/dev/null || true)
    timeout=$(grep -c '^TIMEOUT' "$log_file" 2>/dev/null || true)
    unviable=$(grep -c '^UNVIABLE' "$log_file" 2>/dev/null || true)
    total=$((caught + missed + timeout + unviable))

    if [[ $total -gt 0 ]]; then
        kill_rate=$(( (caught + timeout) * 100 / total ))
    else
        kill_rate=0
    fi

    result_line="$mod_short: $caught caught, $missed missed, $timeout timeout, $unviable unviable ($total total, ${kill_rate}% kill rate, ${elapsed}s)"
    echo "  $result_line"
    SUMMARY+="  $result_line\n"

    # Extract missed mutants for triage
    grep '^MISSED' "$log_file" > "$RUN_DIR/${mod_safe}_missed.txt" 2>/dev/null || true

    echo ""
done

# Write summary
{
    echo "=== Mutation Testing Results ==="
    echo "Date: $(date)"
    echo "Modules: ${MODULES[*]}"
    echo ""
    echo "Results:"
    echo -e "$SUMMARY"
} > "$RUN_DIR/summary.txt"

echo "=== Complete ==="
echo -e "$SUMMARY"
echo "Full results: $RUN_DIR"
echo "Missed mutant lists: $RUN_DIR/*_missed.txt"
