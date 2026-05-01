#!/usr/bin/env bash
# Overnight fuzz launcher — kicks all 10 targets in parallel, each capped
# at $DURATION seconds. Per-target stdout/stderr → $CAMPAIGN_DIR/<target>/run.log.
# libFuzzer artifact_prefix and panic-capture dir scoped per-target so a
# crash in one target doesn't pollute another's triage.
#
# Usage: ./tools/overnight_fuzz_launch.sh <CAMPAIGN_DIR> [DURATION_SECS]

set -u

CAMPAIGN_DIR="${1:?campaign dir required}"
DURATION="${2:-28800}"   # 8h default
MAX_LEN="${MAX_LEN:-16384}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

TARGETS=(
    fuzz_decode
    fuzz_decode_safety
    fuzz_encode
    fuzz_encode_safety
    fuzz_encode_multiframe
    fuzz_roundtrip
    fuzz_roundtrip_safety
    fuzz_repacketizer_seq
    fuzz_multistream
    fuzz_dnn_blob
)

cd "$ROOT"

# Seed-copy preamble (HLD V2 Stream D, gap 9): lift pathological seeds from
# tests/fuzz/seeds/<target>/ into tests/fuzz/corpus/<target>/ at run start.
# Uses cp -n so libFuzzer's evolved corpus from prior runs isn't overwritten.
for t in "${TARGETS[@]}"; do
    SEEDS_DIR="$ROOT/tests/fuzz/seeds/$t"
    CORPUS_DIR="$ROOT/tests/fuzz/corpus/$t"
    if [ -d "$SEEDS_DIR" ]; then
        mkdir -p "$CORPUS_DIR"
        cp -n "$SEEDS_DIR"/* "$CORPUS_DIR/" 2>/dev/null || true
    fi
done

echo "Campaign:  $CAMPAIGN_DIR" | tee "$CAMPAIGN_DIR/launcher.log"
echo "Duration:  ${DURATION}s ($((DURATION/3600))h $((DURATION%3600/60))m)" | tee -a "$CAMPAIGN_DIR/launcher.log"
echo "Started:   $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$CAMPAIGN_DIR/launcher.log"
echo "" | tee -a "$CAMPAIGN_DIR/launcher.log"

# Kick each target. Trailing slash on artifact_prefix is required by libFuzzer.
for t in "${TARGETS[@]}"; do
    LOG="$CAMPAIGN_DIR/$t/run.log"
    ARTIFACTS="$CAMPAIGN_DIR/$t/artifacts/"
    CAPTURE="$CAMPAIGN_DIR/$t/capture"

    # Per-target max_len override. fuzz_encode_multiframe needs ~31 KB to fit
    # 16 frames at 48 kHz stereo (1920 PCM bytes/frame + per-frame config);
    # the default 16 KB caps it well below that.
    case "$t" in
        fuzz_encode_multiframe) MAX_LEN_FOR_TARGET=65536 ;;
        *)                      MAX_LEN_FOR_TARGET="$MAX_LEN" ;;
    esac

    (
        export FUZZ_PANIC_CAPTURE_DIR="$CAPTURE"
        cargo +nightly fuzz run --fuzz-dir tests/fuzz "$t" -- \
            -max_total_time="$DURATION" \
            -max_len="$MAX_LEN_FOR_TARGET" \
            -artifact_prefix="$ARTIFACTS" \
            -print_final_stats=1 \
            > "$LOG" 2>&1
    ) &

    pid=$!
    echo "$pid" > "$CAMPAIGN_DIR/$t/pid"
    echo "[$t] launched pid=$pid → $LOG" | tee -a "$CAMPAIGN_DIR/launcher.log"
done

echo "" | tee -a "$CAMPAIGN_DIR/launcher.log"
echo "All ${#TARGETS[@]} targets launched. Use:" | tee -a "$CAMPAIGN_DIR/launcher.log"
echo "  ls $CAMPAIGN_DIR/<target>/{artifacts,capture}/ to find crashes" | tee -a "$CAMPAIGN_DIR/launcher.log"
echo "  tail -f $CAMPAIGN_DIR/<target>/run.log         to watch a target" | tee -a "$CAMPAIGN_DIR/launcher.log"
echo "  cat $CAMPAIGN_DIR/<target>/pid                  for pid" | tee -a "$CAMPAIGN_DIR/launcher.log"

wait
echo "" | tee -a "$CAMPAIGN_DIR/launcher.log"
echo "Finished:  $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$CAMPAIGN_DIR/launcher.log"
