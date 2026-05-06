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
#   ./tools/fuzz_run.sh --sanity               # build targets + replay committed crashes only
#   ./tools/fuzz_run.sh --check-crashes        # alias for --sanity
#
# Environment:
#   FUZZ_DURATION   — seconds per target (default: 600)
#   FUZZ_JOBS       — parallel libFuzzer workers (default: 1)
#   FUZZ_MAX_LEN    — max input length in bytes (default: 16384)
#
# Requirements:
#   - Linux, macOS, or Windows (Git Bash / MSYS2 shell)
#   - Rust nightly toolchain
#   - cargo-fuzz: cargo install cargo-fuzz
#   - C reference source in reference/ (clone https://github.com/xiph/opus)
#   - On Windows: Visual Studio with "C++ AddressSanitizer" component installed
#                 (provides clang_rt.asan_dynamic-x86_64.dll)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FUZZ_DIR="$ROOT/tests/fuzz"
CORPUS_DIR="$FUZZ_DIR/corpus"
CRASHES_DIR="$FUZZ_DIR/crashes"
FINDINGS_DIR="$ROOT/logs/fuzz-findings"
. "$SCRIPT_DIR/fuzz_targets.sh"

# --------------------------------------------------------------------------- #
# Windows ASan runtime setup
# --------------------------------------------------------------------------- #
# Rust nightly on Windows MSVC does not ship librustc-nightly_rt.asan.a, so
# rust-lld fails when cargo-fuzz asks for -Zsanitizer=address. The workaround:
#   1. Force cargo to use MSVC's link.exe (which knows the MSVC lib paths and
#      can find clang_rt.asan_dynamic-x86_64.lib on its own).
#   2. Prepend the VS ASan runtime directory to PATH so the fuzz binaries can
#      load clang_rt.asan_dynamic-x86_64.dll at runtime.
#   3. Suppress ASan LeakSanitizer noise from the vendored C reference.
setup_windows_asan() {
    local os
    os=$(uname -s 2>/dev/null || echo unknown)
    case "$os" in
        MINGW*|MSYS*|CYGWIN*) ;;
        *) return 0 ;;  # not Windows — no setup needed
    esac

    echo "  Windows detected — setting up MSVC ASan runtime"

    # Find the latest VS installation with an x64 ASan DLL
    local vs_parent="/c/Program Files/Microsoft Visual Studio"
    if [[ ! -d "$vs_parent" ]]; then
        echo "  WARNING: $vs_parent not found — Windows fuzz build will likely fail"
        return 0
    fi

    local asan_dir=""
    local candidate
    while IFS= read -r candidate; do
        if [[ -f "$candidate/clang_rt.asan_dynamic-x86_64.dll" ]]; then
            asan_dir="$candidate"
        fi
    done < <(find "$vs_parent" -type d -path '*/bin/Hostx64/x64' 2>/dev/null | sort)

    if [[ -z "$asan_dir" ]]; then
        echo "  WARNING: clang_rt.asan_dynamic-x86_64.dll not found under $vs_parent"
        echo "  Install 'C++ AddressSanitizer' via the Visual Studio Installer"
        return 0
    fi

    echo "  Using ASan runtime at: $asan_dir"

    # Override the linker — rust-lld can't find the ASan runtime, but MSVC
    # link.exe can (it knows about the MSVC lib search paths automatically).
    export CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER="link.exe"

    # Put the ASan DLL directory on PATH so the fuzz binaries can load it.
    # The Windows path format is required here so link.exe / the loader can
    # resolve it; bash $PATH is auto-translated in Git Bash.
    export PATH="$asan_dir:$PATH"

    # Suppress LeakSanitizer — vendored C reference code has no leaks we care
    # about, and LSan false-positives would derail the fuzz session.
    export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_odr_violation=0:detect_leaks=0}"
}

target_text=$(discover_fuzz_targets "$FUZZ_DIR/Cargo.toml") || die "failed to discover fuzz targets from $FUZZ_DIR/Cargo.toml"
[[ -n "$target_text" ]] || die "no fuzz targets declared in $FUZZ_DIR/Cargo.toml"
mapfile -t ALL_TARGETS <<<"$target_text"

# Defaults
DURATION="${FUZZ_DURATION:-600}"
JOBS="${FUZZ_JOBS:-1}"
MAX_LEN="${FUZZ_MAX_LEN:-16384}"
TARGETS=()
CHECK_CRASHES_ONLY=false
SANITY_ONLY=false
LIST_ONLY=false
NO_DIFF=false

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
require_value() {
    local option="$1"
    local value="${2:-}"

    [[ -n "$value" ]] || die "$option requires a value"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration)
            require_value "$1" "${2:-}"
            DURATION="$2"; shift 2 ;;
        --target)
            require_value "$1" "${2:-}"
            TARGETS+=("$2"); shift 2 ;;
        --jobs)
            require_value "$1" "${2:-}"
            JOBS="$2"; shift 2 ;;
        --no-diff)
            NO_DIFF=true; shift ;;
        --list)
            LIST_ONLY=true; shift ;;
        --check-crashes)
            CHECK_CRASHES_ONLY=true; SANITY_ONLY=true; shift ;;
        --sanity)
            CHECK_CRASHES_ONLY=true; SANITY_ONLY=true; shift ;;
        --max-len)
            require_value "$1" "${2:-}"
            MAX_LEN="$2"; shift 2 ;;
        -h|--help)
            head -21 "$0" | tail -18; exit 0 ;;
        *)
            die "unknown option: $1" ;;
    esac
done

target_is_declared() {
    local requested="$1"
    local declared

    for declared in "${ALL_TARGETS[@]}"; do
        if [[ "$declared" == "$requested" ]]; then
            return 0
        fi
    done
    return 1
}

validate_crash_inventory() {
    local failures=0
    local crash_dir target first_bin

    [[ -d "$CRASHES_DIR" ]] || return 0

    while IFS= read -r crash_dir; do
        target=$(basename "$crash_dir")
        if target_is_declared "$target"; then
            continue
        fi
        first_bin=$(find "$crash_dir" -type f -name '*.bin' -print -quit 2>/dev/null)
        if [[ -n "$first_bin" ]]; then
            echo "ERROR: undeclared fuzz crash corpus contains committed .bin files: $crash_dir" >&2
            ((failures += 1))
        fi
    done < <(find "$CRASHES_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)

    [[ $failures -eq 0 ]]
}

for target in "${TARGETS[@]}"; do
    target_is_declared "$target" || die "fuzz target is not declared in $FUZZ_DIR/Cargo.toml: $target"
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
if $SANITY_ONLY; then
    echo "  Mode:                sanity (build + committed crash replay only)"
fi
echo ""

validate_crash_inventory || exit 1

# Ensure cargo-fuzz is installed
if ! cargo fuzz --version &>/dev/null; then
    echo "ERROR: cargo-fuzz not found. Install with: cargo install cargo-fuzz" >&2
    exit 1
fi

if ! cargo +nightly --version &>/dev/null; then
    echo "ERROR: Rust nightly toolchain not found. Install with: rustup toolchain install nightly" >&2
    exit 1
fi

# Windows-specific: wire up MSVC ASan runtime (no-op on Linux/macOS)
setup_windows_asan

if ! $SANITY_ONLY; then
    # Create findings directory for campaign runs only. Sanity mode is
    # deliberately no-campaign and should not create run artefact directories.
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_DIR="$FINDINGS_DIR/$TIMESTAMP"
    mkdir -p "$RUN_DIR"

    echo "Findings will be saved to: $RUN_DIR"
    echo ""
fi

# --------------------------------------------------------------------------- #
# Build fuzz targets
# --------------------------------------------------------------------------- #
echo "--- Building fuzz targets ---"
cd "$ROOT"

# cargo-fuzz requires nightly. --fuzz-dir is a per-subcommand arg (must appear
# after build/run, not before).
CARGO_FUZZ=(cargo +nightly fuzz)
FUZZ_DIR_ARG=(--fuzz-dir tests/fuzz)

print_compact_output() {
    local text="$1"
    if [[ -z "$text" ]]; then
        echo "      <no output>"
    else
        echo "$text" | tail -40 | sed 's/^/      /'
    fi
}

# Verify targets build before starting long runs
BUILD_FAILURES=0
for target in "${TARGETS[@]}"; do
    echo "  Building $target..."
    if $SANITY_ONLY; then
        set +e
        build_output=$("${CARGO_FUZZ[@]}" build "${FUZZ_DIR_ARG[@]}" "$target" 2>&1)
        build_exit=$?
        set -e
        echo "$build_output" | tail -1
        if [[ $build_exit -ne 0 ]]; then
            echo "$target build=fail crashes=0 replay=not_run"
            ((BUILD_FAILURES += 1))
        fi
    else
        "${CARGO_FUZZ[@]}" build "${FUZZ_DIR_ARG[@]}" "$target" 2>&1 | tail -1
    fi
done
echo ""

if $SANITY_ONLY && [[ $BUILD_FAILURES -gt 0 ]]; then
    echo "RESULT: $BUILD_FAILURES fuzz target build failures found!"
    exit 1
fi

# --------------------------------------------------------------------------- #
# Crash regression check mode
# --------------------------------------------------------------------------- #
if $CHECK_CRASHES_ONLY; then
    echo "--- Crash regression check ---"
    FAILURES=0
    for target in "${TARGETS[@]}"; do
        crash_dir="$CRASHES_DIR/$target"
        crash_files=()
        if [[ -d "$crash_dir" ]]; then
            mapfile -t crash_files < <(find "$crash_dir" -type f -name '*.bin' 2>/dev/null | sort)
        fi
        if [[ ${#crash_files[@]} -eq 0 ]]; then
            echo "  $target: no committed crash .bin files to check (skip)"
            echo "$target build=pass crashes=0 replay=skip"
            continue
        fi
        echo "  $target: checking ${#crash_files[@]} crash files..."

        fail_count=0
        for crash_file in "${crash_files[@]}"; do
            set +e
            replay_output=$("${CARGO_FUZZ[@]}" run "${FUZZ_DIR_ARG[@]}" "$target" "$crash_file" -- -runs=0 2>&1)
            replay_exit=$?
            set -e
            if [[ $replay_exit -ne 0 ]]; then
                echo "    FAIL: $crash_file still crashes!"
                echo "    replay command exited with status $replay_exit; last output lines:"
                print_compact_output "$replay_output"
                ((fail_count += 1))
            fi
        done

        if [[ $fail_count -eq 0 ]]; then
            echo "    OK: all ${#crash_files[@]} crash files handled without crash"
            echo "$target build=pass crashes=${#crash_files[@]} replay=pass"
        else
            echo "    FAILED: $fail_count/${#crash_files[@]} still crash"
            echo "$target build=pass crashes=${#crash_files[@]} replay=fail"
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
    "${CARGO_FUZZ[@]}" run "${FUZZ_DIR_ARG[@]}" "$target" \
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
