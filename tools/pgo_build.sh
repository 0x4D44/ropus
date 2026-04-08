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
PROFILE_DIR="$ROOT/target/pgo-profiles"
MERGED_PROFILE="$ROOT/target/pgo-merged.profdata"
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

# ── Step 0: Baseline bench (non-PGO release) ──────────────────────────
if [[ "$BENCH_ONLY" == false ]]; then
    echo ""
    echo "═══ Step 0: Baseline release build ═══"
    cargo build --release --manifest-path "$ROOT/Cargo.toml" 2>&1 | tail -3

    if [[ -f "$BENCH_WAV" ]]; then
        echo ""
        echo "── Baseline benchmark ──"
        cargo run --release --manifest-path "$ROOT/Cargo.toml" \
            --bin mdopus-compare -- bench "$BENCH_WAV" --iters "$BENCH_ITERS"
    fi
fi

# ── Step 1: Instrumented build ────────────────────────────────────────
if [[ "$BENCH_ONLY" == false ]]; then
    echo ""
    echo "═══ Step 1: Instrumented build (profile-generate) ═══"
    rm -rf "$PROFILE_DIR"
    mkdir -p "$PROFILE_DIR"

    RUSTFLAGS="-Cprofile-generate=$PROFILE_DIR" \
        cargo build --release --manifest-path "$ROOT/Cargo.toml" 2>&1 | tail -3

    # ── Step 2: Training workload ─────────────────────────────────────
    echo ""
    echo "═══ Step 2: Training workload ═══"
    INSTRUMENTED_BIN="$ROOT/target/release/mdopus-compare.exe"
    if [[ ! -f "$INSTRUMENTED_BIN" ]]; then
        INSTRUMENTED_BIN="$ROOT/target/release/mdopus-compare"
    fi

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
    cargo build --release --manifest-path "$ROOT/Cargo.toml" 2>&1 | tail -3

# ── Step 5: PGO bench ────────────────────────────────────────────────
if [[ -f "$BENCH_WAV" ]]; then
    echo ""
    echo "═══ Step 5: PGO benchmark ═══"
    cargo run --release --manifest-path "$ROOT/Cargo.toml" \
        --bin mdopus-compare -- bench "$BENCH_WAV" --iters "$BENCH_ITERS"
fi

echo ""
echo "Done. PGO binary at: $ROOT/target/release/mdopus-compare"
