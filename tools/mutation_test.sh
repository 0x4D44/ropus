#!/usr/bin/env bash
#
# Mutation testing runner for ropus.
#
# This is a periodic/on-demand unit-test sensitivity audit for the ropus crate.
# It is not a release gate, and it does not run the C-reference differential
# harness. Results are saved to logs/mutation-testing/<timestamp>/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$ROOT/logs/mutation-testing"
PACKAGE="ropus"
SOURCE_PREFIX="ropus/src"

# Priority order from the testing improvement roadmap. These are package-scoped
# paths as reported by `cargo mutants --package ropus --list-files`.
ALL_MODULES=(
    "ropus/src/silk/common.rs"
    "ropus/src/opus/decoder.rs"
    "ropus/src/opus/encoder.rs"
    "ropus/src/silk/decoder.rs"
    "ropus/src/celt/decoder.rs"
    "ropus/src/celt/encoder.rs"
)

JOBS=2
TIMEOUT_MULT=3
MODULES=()
LIST_ONLY=false
BASELINE_MODE="strict"
BASELINE_SKIPS=()
SHARD=""

usage() {
    cat <<'USAGE'
Usage:
  bash tools/mutation_test.sh --list
  bash tools/mutation_test.sh --file celt/math_ops --list
  bash tools/mutation_test.sh --file celt/math_ops --jobs 2 --timeout-multiplier 3

Options:
  --file PATH              Audit one file. Accepts celt/math_ops, src/celt/math_ops,
                           ropus/src/celt/math_ops.rs, or an absolute path under repo root.
  --list                   List selected files and mutant counts, then exit.
  --jobs N                 cargo-mutants jobs. Default: 2.
  --timeout-multiplier N   cargo-mutants test timeout multiplier. Default: 3.
  --shard N/M              Run one cargo-mutants shard for each selected file.
  --skip-test FILTER       Explicitly run in skip-degraded mode with this cargo test skip.
                           May be repeated. Strict mode is the default.
  -h, --help               Show this help.

Policy:
  Unit-test-only audit for package ropus. The C-reference differential harness
  is not in this mutation loop. The lane is periodic/on-demand and
  non-release-blocking; no kill-rate threshold is applied.
USAGE
}

error() {
    echo "ERROR: $*" >&2
}

format_command() {
    printf '%q ' "$@"
}

resolve_module() {
    local input="$1"
    local rel

    if [[ "$input" == /* ]]; then
        case "$input" in
            "$ROOT"/*) input="${input#$ROOT/}" ;;
            *)
                error "path is outside workspace: $input"
                return 1
                ;;
        esac
    fi

    input="${input#./}"
    if [[ "$input" == *".."* ]]; then
        error "path traversal is not allowed: $input"
        return 1
    fi

    case "$input" in
        "$SOURCE_PREFIX"/*) rel="$input" ;;
        src/*) rel="ropus/$input" ;;
        *) rel="$SOURCE_PREFIX/$input" ;;
    esac

    [[ "$rel" == *.rs ]] || rel="$rel.rs"

    if [[ "$rel" != "$SOURCE_PREFIX"/* ]]; then
        error "resolved path is outside $SOURCE_PREFIX: $rel"
        return 1
    fi

    if [[ ! -f "$ROOT/$rel" ]]; then
        error "resolved source file does not exist: $rel"
        return 1
    fi

    printf '%s\n' "$rel"
}

module_safe_name() {
    local mod="$1"
    mod="${mod#$SOURCE_PREFIX/}"
    mod="${mod%.rs}"
    printf '%s\n' "${mod//\//_}"
}

count_mutants() {
    local mod="$1"
    cargo mutants \
        --package "$PACKAGE" \
        --file "$mod" \
        --list \
        --no-times \
        --colors never \
        | awk 'NF { count++ } END { print count + 0 }'
}

write_command_log() {
    local label="$1"
    shift

    {
        echo "## $label"
        format_command "$@"
        echo
        echo
    } >> "$RUN_DIR/commands.txt"
}

append_baseline_skips() {
    local array_name="$1"
    local -n baseline_cmd_ref="$array_name"

    if [[ ${#BASELINE_SKIPS[@]} -eq 0 ]]; then
        return
    fi

    baseline_cmd_ref+=(--)
    for skip in "${BASELINE_SKIPS[@]}"; do
        baseline_cmd_ref+=(--skip "$skip")
    done
}

append_cargo_mutants_test_args() {
    local array_name="$1"
    local -n mutants_cmd_ref="$array_name"

    mutants_cmd_ref+=(-- --lib)
    append_baseline_skips "$array_name"
}

print_inventory() {
    local status=0
    local mod count

    echo "Mutation testing audit inventory"
    echo "Package: $PACKAGE"
    echo "Oracle scope: ropus library cargo tests only. C-reference differential harness is not in this mutation loop."
    echo "Policy: periodic/on-demand, non-release-blocking, no kill-rate threshold."
    echo ""

    for mod in "${MODULES[@]}"; do
        if ! count=$(count_mutants "$mod"); then
            error "failed to list mutants for $mod"
            status=1
            continue
        fi

        printf "  %-36s %5d mutants\n" "$mod" "$count"
        if [[ "$count" -eq 0 ]]; then
            error "$mod resolved successfully but produced zero mutants"
            status=1
        fi
    done

    return "$status"
}

parse_count() {
    local prefix="$1"
    local file="$2"
    grep -Eic "^${prefix}([[:space:]]|$)" "$file" 2>/dev/null || true
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --file)
            if [[ $# -lt 2 ]]; then
                error "--file requires a path"
                exit 1
            fi
            resolved="$(resolve_module "$2")" || exit 1
            MODULES+=("$resolved")
            shift 2
            ;;
        --jobs)
            if [[ $# -lt 2 ]]; then
                error "--jobs requires a value"
                exit 1
            fi
            JOBS="$2"
            shift 2
            ;;
        --timeout-multiplier)
            if [[ $# -lt 2 ]]; then
                error "--timeout-multiplier requires a value"
                exit 1
            fi
            TIMEOUT_MULT="$2"
            shift 2
            ;;
        --shard)
            if [[ $# -lt 2 ]]; then
                error "--shard requires a value like 1/4"
                exit 1
            fi
            SHARD="$2"
            shift 2
            ;;
        --skip-test)
            if [[ $# -lt 2 ]]; then
                error "--skip-test requires a cargo test filter"
                exit 1
            fi
            BASELINE_MODE="skip-degraded"
            BASELINE_SKIPS+=("$2")
            shift 2
            ;;
        --list)
            LIST_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "unknown option: $1"
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ${#MODULES[@]} -eq 0 ]]; then
    MODULES=("${ALL_MODULES[@]}")
fi

cd "$ROOT"

if ! MUTANTS_VERSION="$(cargo mutants --version 2>/dev/null)"; then
    error "cargo-mutants not found. Install with: cargo install cargo-mutants"
    exit 1
fi

if $LIST_ONLY; then
    print_inventory
    exit $?
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$RESULTS_DIR/$TIMESTAMP"
SUMMARY_FILE="$RUN_DIR/summary.md"
mkdir -p "$RUN_DIR"

{
    echo "# Mutation Testing Audit"
    echo
    echo "Date: $(date -Is)"
    echo "Package: $PACKAGE"
    echo "cargo-mutants: $MUTANTS_VERSION"
    echo "Baseline mode: $BASELINE_MODE"
    echo "Jobs: $JOBS"
    echo "Timeout multiplier: $TIMEOUT_MULT"
    if [[ -n "$SHARD" ]]; then
        echo "Shard: $SHARD"
    else
        echo "Shard: none"
    fi
    echo
    echo "Oracle scope: ropus library cargo tests only. C-reference differential harness is not in this mutation loop."
    echo
    echo "Policy: periodic/on-demand audit, non-release-blocking, no kill-rate threshold."
    echo
    if [[ ${#BASELINE_SKIPS[@]} -gt 0 ]]; then
        echo "Explicit skipped tests:"
        for skip in "${BASELINE_SKIPS[@]}"; do
            echo "- $skip"
        done
        echo
    fi
} > "$SUMMARY_FILE"

{
    echo "cargo-mutants: $MUTANTS_VERSION"
    echo "run directory: $RUN_DIR"
    echo
} > "$RUN_DIR/commands.txt"

echo "=== ropus mutation testing audit ==="
echo "  Modules:    ${MODULES[*]}"
echo "  Baseline:   $BASELINE_MODE"
echo "  Jobs:       $JOBS"
echo "  Timeout:    ${TIMEOUT_MULT}x"
echo "  Results:    $RUN_DIR"
echo "  Scope:      unit-test-only; C-reference differential harness not in loop"
echo ""

echo "## Inventory" >> "$SUMMARY_FILE"
inventory_status=0
for mod in "${MODULES[@]}"; do
    safe="$(module_safe_name "$mod")"
    list_file="$RUN_DIR/${safe}_mutants.txt"

    if ! cargo mutants \
        --package "$PACKAGE" \
        --file "$mod" \
        --list \
        --no-times \
        --colors never \
        > "$list_file"; then
        error "failed to list mutants for $mod"
        echo "- ${mod}: inventory_failed" >> "$SUMMARY_FILE"
        inventory_status=1
        continue
    fi

    count="$(awk 'NF { count++ } END { print count + 0 }' "$list_file")"
    echo "- ${mod}: ${count} mutants" >> "$SUMMARY_FILE"
    if [[ "$count" -eq 0 ]]; then
        error "$mod resolved successfully but produced zero mutants"
        inventory_status=1
    fi
done
echo >> "$SUMMARY_FILE"

if [[ "$inventory_status" -ne 0 ]]; then
    echo "Status: inventory_failed" >> "$SUMMARY_FILE"
    exit 1
fi

baseline_cmd=(cargo test --package "$PACKAGE" --lib)
append_baseline_skips baseline_cmd
write_command_log "baseline" "${baseline_cmd[@]}"

echo "=== baseline ==="
set +e
"${baseline_cmd[@]}" 2>&1 | tee "$RUN_DIR/baseline.log"
baseline_status=${PIPESTATUS[0]}
set -e

{
    echo "## Baseline"
    if [[ "$baseline_status" -eq 0 ]]; then
        echo "- Status: pass"
    else
        echo "- Status: baseline_failed"
    fi
    echo -n "- Command: \`"
    format_command "${baseline_cmd[@]}"
    echo "\`"
    echo "- Log: baseline.log"
    echo
} >> "$SUMMARY_FILE"

if [[ "$baseline_status" -ne 0 ]]; then
    error "baseline failed; mutation audit stopped without publishing a score"
    exit 1
fi

echo "## Results" >> "$SUMMARY_FILE"
echo "| File | Listed | Caught | Missed | Timeout | Unviable | cargo-mutants exit | Elapsed | Artifact |" >> "$SUMMARY_FILE"
echo "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |" >> "$SUMMARY_FILE"

final_status=0
for mod in "${MODULES[@]}"; do
    safe="$(module_safe_name "$mod")"
    log_file="$RUN_DIR/${safe}.log"
    missed_file="$RUN_DIR/${safe}_missed.txt"
    output_dir="$RUN_DIR/${safe}_native"
    count="$(awk 'NF { count++ } END { print count + 0 }' "$RUN_DIR/${safe}_mutants.txt")"
    mutants_cmd=(
        cargo mutants
        --package "$PACKAGE"
        --file "$mod"
        --baseline skip
        --timeout-multiplier "$TIMEOUT_MULT"
        --output "$output_dir"
        --colors never
        --caught
        --unviable
        -j "$JOBS"
    )

    if [[ -n "$SHARD" ]]; then
        mutants_cmd+=(--shard "$SHARD")
    fi
    append_cargo_mutants_test_args mutants_cmd
    write_command_log "mutation $mod" "${mutants_cmd[@]}"

    echo "--- $mod ($count mutants listed) ---"
    start_time="$(date +%s)"
    set +e
    "${mutants_cmd[@]}" 2>&1 | tee "$log_file"
    mutants_status=${PIPESTATUS[0]}
    set -e
    end_time="$(date +%s)"
    elapsed="$((end_time - start_time))s"

    caught="$(parse_count CAUGHT "$log_file")"
    missed="$(parse_count MISSED "$log_file")"
    timeout_count="$(parse_count TIMEOUT "$log_file")"
    unviable="$(parse_count UNVIABLE "$log_file")"
    parsed_total="$((caught + missed + timeout_count + unviable))"

    grep -Ei '^MISSED([[:space:]]|$)' "$log_file" > "$missed_file" 2>/dev/null || true

    if [[ "$mutants_status" -ne 0 && "$parsed_total" -eq 0 ]]; then
        final_status=1
    fi

    echo "| $mod | $count | $caught | $missed | $timeout_count | $unviable | $mutants_status | $elapsed | ${safe}_native/mutants.out |" >> "$SUMMARY_FILE"
    echo "  $caught caught, $missed missed, $timeout_count timeout, $unviable unviable; cargo-mutants exit $mutants_status; elapsed $elapsed"
    echo ""
done

{
    echo
    if [[ "$final_status" -eq 0 ]]; then
        echo "Status: audit_completed"
    else
        echo "Status: tool_failed"
        echo
        echo "At least one cargo-mutants command exited non-zero without parseable mutant outcomes."
    fi
    echo
    echo "Notes:"
    echo "- Survivors are triage input, not a release-blocking failure in this lane."
    echo "- Zero-mutant selected files and baseline failures are non-green because they make the audit uninterpretable."
} >> "$SUMMARY_FILE"

echo "=== complete ==="
echo "Summary: $SUMMARY_FILE"
echo "Commands: $RUN_DIR/commands.txt"
echo "Missed mutant lists: $RUN_DIR/*_missed.txt"
exit "$final_status"
