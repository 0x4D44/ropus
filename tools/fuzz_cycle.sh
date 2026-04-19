#!/usr/bin/env bash
#
# ropus fuzz cadence runner.
#
# Runs every cargo-fuzz target under tests/fuzz/ for a fixed wall-clock
# duration (default: 10 min per target), wraps each invocation with
# `mdtimeout` (Windows) or `timeout` (Linux/macOS) for a hard ceiling, logs
# stdout+stderr per target to `logs/fuzz_cycle_<timestamp>/<target>.log`,
# and prints a per-target summary at the end.
#
# A "new crash" is any file under `tests/fuzz/crashes/<target>/` with a
# modification time newer than the start of this cycle. Crashes written by
# the project's panic-capture hook land in that directory; the libFuzzer
# artifact_prefix (see tools/fuzz_run.sh) also copies crash-* files there.
#
# Usage:
#   ./tools/fuzz_cycle.sh              # 600s per target (~1.5h total)
#   ./tools/fuzz_cycle.sh 3600         # 1 hour per target (overnight)
#   ./tools/fuzz_cycle.sh 0            # dry run: list targets, exit
#
# This script deliberately stays small and CI-free. The cadence is a
# manual, human-run process pre-1.0 — see tests/fuzz/README.md.

set -euo pipefail

# Track the target currently running so an interrupted cycle can report
# which one was in flight. Cleared once the cycle completes normally.
CURRENT_TARGET=""
on_interrupt() {
    if [[ -n "$CURRENT_TARGET" ]]; then
        echo "" >&2
        echo "⚠ cycle aborted at target $CURRENT_TARGET" >&2
    fi
}
trap on_interrupt INT
trap on_interrupt EXIT

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FUZZ_DIR="$ROOT/tests/fuzz"
CRASHES_DIR="$FUZZ_DIR/crashes"

# ---------------------------------------------------------------------------
# Targets: keep in lock-step with tests/fuzz/Cargo.toml [[bin]] entries.
# ---------------------------------------------------------------------------
TARGETS=(
    fuzz_decode
    fuzz_encode
    fuzz_roundtrip
    fuzz_repacketizer
    fuzz_packet_parse
    fuzz_decode_safety
    fuzz_encode_safety
    fuzz_roundtrip_safety
    fuzz_encode_multiframe
)

# ---------------------------------------------------------------------------
# Duration: positional arg 1, else default 600s.
# Zero means "dry run" (list targets and exit without running cargo-fuzz).
# ---------------------------------------------------------------------------
DURATION="${1:-600}"
if ! [[ "$DURATION" =~ ^[0-9]+$ ]]; then
    echo "ERROR: duration must be a non-negative integer (got: $DURATION)" >&2
    exit 2
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/fuzz_cycle_$TIMESTAMP"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Pick a timeout wrapper. We prefer mdtimeout (Windows helper) if present,
# then GNU coreutils `timeout`. Without either we run unwrapped and warn.
# The wrapper's timeout is padded by 30s so cargo-fuzz gets a chance to
# flush its own stats before we SIGTERM it.
# ---------------------------------------------------------------------------
TIMEOUT_BIN=""
if command -v mdtimeout >/dev/null 2>&1; then
    TIMEOUT_BIN="mdtimeout"
elif command -v timeout >/dev/null 2>&1; then
    TIMEOUT_BIN="timeout"
else
    echo "WARNING: neither mdtimeout nor timeout found on PATH — running cargo-fuzz unwrapped." >&2
    echo "         Hung targets will not be killed automatically." >&2
fi

WRAPPER_SECS=$((DURATION + 30))

echo "=== ropus fuzz cycle ==="
echo "  Timestamp:         $TIMESTAMP"
echo "  Duration / target: ${DURATION}s"
echo "  Wrapper:           ${TIMEOUT_BIN:-none}"
echo "  Log dir:           $LOG_DIR"
echo "  Targets:           ${#TARGETS[@]}"
for t in "${TARGETS[@]}"; do
    echo "    - $t"
done
echo ""

if [[ "$DURATION" == "0" ]]; then
    echo "Duration is 0 — dry run complete, nothing to execute."
    exit 0
fi

# ---------------------------------------------------------------------------
# Require cargo-fuzz up front. The HLD accepts this as a hard prerequisite.
# ---------------------------------------------------------------------------
if ! cargo fuzz --version >/dev/null 2>&1; then
    echo "ERROR: cargo-fuzz is not installed." >&2
    echo "       Install with: cargo install cargo-fuzz" >&2
    exit 3
fi

# Record a cycle-start marker so per-target crash detection can use it.
CYCLE_START_FILE="$LOG_DIR/.cycle_start"
touch "$CYCLE_START_FILE"

# ---------------------------------------------------------------------------
# Run each target. We do NOT `set -e` across a failing target — a crash
# found by libFuzzer returns nonzero, and we want to keep going so the
# cycle covers every target.
# ---------------------------------------------------------------------------
declare -a EXIT_CODES
declare -a NEW_CRASHES
declare -a TARGET_ORDER

for target in "${TARGETS[@]}"; do
    CURRENT_TARGET="$target"
    echo "--- $target ---"
    log_file="$LOG_DIR/$target.log"
    target_crashes_dir="$CRASHES_DIR/$target"
    mkdir -p "$target_crashes_dir"

    # Build the cargo-fuzz command. --fuzz-dir is a per-subcommand flag and
    # must sit after `run`, not before.
    cargo_fuzz_cmd=(
        cargo +nightly fuzz run
            --fuzz-dir "$FUZZ_DIR"
            "$target"
            --
            -max_total_time="$DURATION"
            -print_final_stats=1
    )

    set +e
    if [[ -n "$TIMEOUT_BIN" ]]; then
        "$TIMEOUT_BIN" "${WRAPPER_SECS}s" "${cargo_fuzz_cmd[@]}" >"$log_file" 2>&1
    else
        "${cargo_fuzz_cmd[@]}" >"$log_file" 2>&1
    fi
    exit_code=$?
    set -e

    # Count files in crashes/<target>/ newer than the cycle marker. We use
    # -newer against the marker file because portable `find -newermt` is
    # not available on all BusyBox / msys builds. Caveat: a crash file
    # re-promoted by `fuzz_run.sh` during this cycle will register as
    # "new" even if its hash content is stale.
    new_count=0
    if [[ -d "$target_crashes_dir" ]]; then
        new_count=$(find "$target_crashes_dir" -type f -newer "$CYCLE_START_FILE" 2>/dev/null | wc -l | tr -d ' ')
    fi

    EXIT_CODES+=("$exit_code")
    NEW_CRASHES+=("$new_count")
    TARGET_ORDER+=("$target")

    echo "  exit=$exit_code  new-crashes=$new_count  log=$log_file"
    # Surface failures mid-run so a multi-hour cycle doesn't hide them
    # until the final summary.
    if [[ "$exit_code" -ne 0 || "$new_count" -gt 0 ]]; then
        echo "✗ $target exited $exit_code, $new_count new crash file(s) in $target_crashes_dir/; continuing to next target." >&2
    fi
    echo ""
done

# Cycle ran to completion — clear the in-flight marker so the EXIT trap
# stays silent unless something below re-sets it.
CURRENT_TARGET=""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=== Cycle complete ==="
printf "%-28s %-6s %s\n" "target" "exit" "new-crashes"
printf "%-28s %-6s %s\n" "------" "----" "-----------"
total_crashes=0
total_failed=0
for i in "${!TARGET_ORDER[@]}"; do
    printf "%-28s %-6s %s\n" "${TARGET_ORDER[$i]}" "${EXIT_CODES[$i]}" "${NEW_CRASHES[$i]}"
    total_crashes=$((total_crashes + ${NEW_CRASHES[$i]}))
    if [[ "${EXIT_CODES[$i]}" -ne 0 ]]; then
        total_failed=$((total_failed + 1))
    fi
done
echo ""
echo "  targets with nonzero exit: $total_failed / ${#TARGET_ORDER[@]}"
echo "  total new crash artifacts: $total_crashes"
echo "  logs:                      $LOG_DIR"

# Re-emit the no-timeout warning in the summary footer so an operator
# skimming the per-target table at the end still notices. The original
# warning at script start is easy to miss beneath multi-hour cargo-fuzz
# output.
if [[ -z "$TIMEOUT_BIN" ]]; then
    echo ""
    echo "⚠ Note: no \`timeout\` / \`mdtimeout\` binary found; targets ran without wall-clock enforcement." >&2
fi

# Nonzero exit if we saw anything interesting, so a human running this
# under a pager or wrapper script knows to look.
if [[ "$total_crashes" -gt 0 || "$total_failed" -gt 0 ]]; then
    exit 1
fi
exit 0
