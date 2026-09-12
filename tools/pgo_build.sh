#!/usr/bin/env bash
#
# PGO (Profile-Guided Optimization) build script for mdopus.
#
# Usage:
#   ./tools/pgo_build.sh              # full PGO build + bench comparison
#   ./tools/pgo_build.sh --train-only # just generate profiles, skip final build
#   ./tools/pgo_build.sh --bench-only # skip training, rebuild from existing profiles
#
# Requires: cargo, llvm-profdata (ships with rustup)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HARNESS_MANIFEST="$ROOT/harness/Cargo.toml"
TARGET_DIR="$ROOT/target"
PROFILE_DIR="$TARGET_DIR/pgo-profiles"
MERGED_PROFILE="$TARGET_DIR/pgo-merged.profdata"
RELEASE_DIR="$TARGET_DIR/release"
VECTORS_DIR="$ROOT/tests/vectors"
BENCH_WAV="$VECTORS_DIR/48k_sine1k_loud.wav"

# Find llvm-profdata from the active rustup toolchain
TOOLCHAIN_DIR="$(rustc --print sysroot)"
LLVM_PROFDATA="$TOOLCHAIN_DIR/lib/rustlib/x86_64-pc-windows-msvc/bin/llvm-profdata.exe"
if [[ ! -f "$LLVM_PROFDATA" ]]; then
    # Try unix path
    LLVM_PROFDATA="$TOOLCHAIN_DIR/lib/rustlib/x86_64-unknown-linux-gnu/bin/llvm-profdata"
fi
if [[ ! -f "$LLVM_PROFDATA" ]]; then
    echo "ERROR: llvm-profdata not found. Install it with: rustup component add llvm-tools"
    exit 1
fi

TRAIN_ONLY=false
BENCH_ONLY=false
BENCH_ITERS=10

for arg in "$@"; do
    case "$arg" in
        --train-only) TRAIN_ONLY=true ;;
        --bench-only) BENCH_ONLY=true ;;
        --iters=*)    BENCH_ITERS="${arg#--iters=}" ;;
        *)            echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

sha256_of() {
    local path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$path" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$path" | awk '{print $1}'
    else
        echo "ERROR: need sha256sum or shasum to identify the built binary" >&2
        return 1
    fi
}

release_binary() {
    local base="$RELEASE_DIR/ropus-compare"
    if [[ -f "${base}.exe" ]]; then
        printf '%s\n' "${base}.exe"
    elif [[ -f "$base" ]]; then
        printf '%s\n' "$base"
    else
        echo "ERROR: ropus-compare binary not found under $RELEASE_DIR" >&2
        return 1
    fi
}

run_benchmark() {
    local label="$1"
    local binary="$2"
    if [[ ! -f "$binary" ]]; then
        echo "ERROR: $label binary not found: $binary" >&2
        return 1
    fi

    local before after
    before="$(sha256_of "$binary")"
    echo "  $label binary: $binary"
    echo "  $label binary identity: $before"
    if ! "$binary" bench "$BENCH_WAV" --iters "$BENCH_ITERS"; then
        echo "ERROR: $label benchmark failed" >&2
        return 1
    fi
    after="$(sha256_of "$binary")"
    if [[ "$before" != "$after" ]]; then
        echo "ERROR: $label binary changed during measurement (before=$before after=$after)" >&2
        return 1
    fi
    echo "  $label binary unchanged after benchmark: $after"
}

# ── Step 0: Baseline bench (non-PGO release) ──────────────────────────
if [[ "$BENCH_ONLY" == false ]]; then
    echo ""
    echo "═══ Step 0: Baseline release build ═══"
    cargo build --release --manifest-path "$HARNESS_MANIFEST" \
        --target-dir "$TARGET_DIR" \
        --package ropus-harness --bin ropus-compare 2>&1 | tail -3

    if [[ -f "$BENCH_WAV" ]]; then
        echo ""
        echo "── Baseline benchmark ──"
        BASELINE_BIN="$(release_binary)"
        run_benchmark "Baseline" "$BASELINE_BIN"
    fi
fi

# ── Step 1: Instrumented build ────────────────────────────────────────
if [[ "$BENCH_ONLY" == false ]]; then
    echo ""
    echo "═══ Step 1: Instrumented build (profile-generate) ═══"
    rm -rf "$PROFILE_DIR"
    mkdir -p "$PROFILE_DIR"

    RUSTFLAGS="-Cprofile-generate=$PROFILE_DIR" \
        cargo build --release --manifest-path "$HARNESS_MANIFEST" \
            --target-dir "$TARGET_DIR" \
            --package ropus-harness --bin ropus-compare 2>&1 | tail -3

    # ── Step 2: Training workload ─────────────────────────────────────
    echo ""
    echo "═══ Step 2: Training workload ═══"
    INSTRUMENTED_BIN="$(release_binary)"
    echo "  Instrumented binary: $INSTRUMENTED_BIN"
    echo "  Instrumented binary identity: $(sha256_of "$INSTRUMENTED_BIN")"

    BITRATES=(16000 32000 64000 128000)
    WAV_COUNT=0

    for wav in "$VECTORS_DIR"/*.wav; do
        [[ -f "$wav" ]] || continue
        for br in "${BITRATES[@]}"; do
            "$INSTRUMENTED_BIN" roundtrip "$wav" --bitrate "$br" > /dev/null 2>&1 || true
        done
        WAV_COUNT=$((WAV_COUNT + 1))
    done

    echo "  Trained on $WAV_COUNT WAV files x ${#BITRATES[@]} bitrates = $((WAV_COUNT * ${#BITRATES[@]})) roundtrips"

    # ── Step 3: Merge profiles ────────────────────────────────────────
    echo ""
    echo "═══ Step 3: Merging profiles ═══"
    "$LLVM_PROFDATA" merge -o "$MERGED_PROFILE" "$PROFILE_DIR"

    PROFILE_SIZE=$(du -h "$MERGED_PROFILE" | cut -f1)
    echo "  Merged profile: $MERGED_PROFILE ($PROFILE_SIZE)"

    if [[ "$TRAIN_ONLY" == true ]]; then
        echo ""
        echo "Done (--train-only). Re-run with --bench-only to build and measure."
        exit 0
    fi
fi

# ── Step 4: Optimized build ───────────────────────────────────────────
echo ""
echo "═══ Step 4: PGO-optimized build (profile-use) ═══"

if [[ ! -f "$MERGED_PROFILE" ]]; then
    echo "ERROR: No merged profile at $MERGED_PROFILE. Run without --bench-only first."
    exit 1
fi

RUSTFLAGS="-Cprofile-use=$MERGED_PROFILE -Cllvm-args=-pgo-warn-missing-function" \
    cargo build --release --manifest-path "$HARNESS_MANIFEST" \
        --target-dir "$TARGET_DIR" \
        --package ropus-harness --bin ropus-compare 2>&1 | tail -3

# ── Step 5: PGO bench ────────────────────────────────────────────────
if [[ -f "$BENCH_WAV" ]]; then
    echo ""
    echo "═══ Step 5: PGO benchmark ═══"
    PGO_BIN="$(release_binary)"
    run_benchmark "PGO" "$PGO_BIN"
fi

echo ""
echo "Done. PGO binary at: $RELEASE_DIR/ropus-compare"
