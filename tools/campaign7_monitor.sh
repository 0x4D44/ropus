#!/usr/bin/env bash
#
# Campaign 7 monitor — check status, crashes, and stats for all running fuzzers.
#

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CAMPAIGN_DIR="$ROOT/logs/fuzz-findings/campaign7"

echo "=== Campaign 7 Status @ $(date '+%Y-%m-%d %H:%M:%S') ==="
echo ""

# Process status
running=0
dead=0
for f in "$CAMPAIGN_DIR"/*/pid; do
    [[ -f "$f" ]] || continue
    target=$(basename "$(dirname "$f")")
    pid=$(cat "$f" 2>/dev/null)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        ((running++))
    else
        ((dead++))
        echo "  DONE: $target"
    fi
done
echo "Running: $running  Completed: $dead"
echo ""

# Crash check
echo "--- Crashes ---"
crash_count=0
for d in "$CAMPAIGN_DIR"/*/; do
    target=$(basename "$d")
    crashes=$(find "$d" -maxdepth 1 \( -name 'crash-*' -o -name 'leak-*' -o -name 'timeout-*' -o -name 'oom-*' \) 2>/dev/null)
    if [[ -n "$crashes" ]]; then
        n=$(echo "$crashes" | wc -l)
        crash_count=$((crash_count + n))
        echo "  $target: $n findings"
        echo "$crashes" | while read -r f; do echo "    $(basename "$f")"; done
    fi
done
# Also check fuzz_crashes/ panic capture dir
if [[ -d "$ROOT/fuzz_crashes" ]]; then
    # Only count files newer than campaign start
    if [[ -d "$CAMPAIGN_DIR" ]]; then
        panic_crashes=$(find "$ROOT/fuzz_crashes" -type f -newer "$CAMPAIGN_DIR" 2>/dev/null | wc -l)
        if [[ $panic_crashes -gt 0 ]]; then
            echo "  fuzz_crashes/ (panic capture): $panic_crashes new files"
            crash_count=$((crash_count + panic_crashes))
        fi
    fi
fi
if [[ $crash_count -eq 0 ]]; then
    echo "  None found."
fi
echo ""

# Per-target stats (from last log line with coverage info)
echo "--- Per-target Stats ---"
printf "  %-25s %10s %8s %8s %10s\n" "TARGET" "EXECS" "COV" "CORPUS" "EXEC/S"
for d in "$CAMPAIGN_DIR"/*/; do
    target=$(basename "$d")
    log="$d/fuzz.log"
    [[ -f "$log" ]] || continue
    pid_file="$d/pid"
    status="RUN"
    if [[ -f "$pid_file" ]]; then
        pid=$(cat "$pid_file" 2>/dev/null)
        if ! kill -0 "$pid" 2>/dev/null; then
            status="DONE"
        fi
    fi
    # Parse last stats line
    last_stats=$(grep -E '#[0-9]+' "$log" 2>/dev/null | tail -1)
    if [[ -n "$last_stats" ]]; then
        execs=$(echo "$last_stats" | grep -oP '#\K[0-9]+' | head -1)
        cov=$(echo "$last_stats" | grep -oP 'cov: \K[0-9]+' | head -1)
        corp=$(echo "$last_stats" | grep -oP 'corp: \K[0-9]+' | head -1)
        execps=$(echo "$last_stats" | grep -oP 'exec/s: \K[0-9]+' | head -1)
        printf "  %-25s %10s %8s %8s %10s  [%s]\n" "$target" "${execs:-?}" "${cov:-?}" "${corp:-?}" "${execps:-?}" "$status"
    else
        printf "  %-25s %10s  [%s]\n" "$target" "(no stats yet)" "$status"
    fi
done
echo ""

echo "Total findings: $crash_count"
