#!/usr/bin/env bash
# 24h fuzz campaign launcher with asymmetric CPU allocation.
#
# Per-target worker counts bias compute toward new/rebuilt targets and
# saturated targets get a token job for regression value.
#
# Each (target, worker) pair gets its own subdir with run-wN.log,
# artifacts-wN/, capture-wN/ — but they share tests/fuzz/corpus/<target>/
# so libFuzzer's cross-process corpus sync (inotify) lets workers
# co-discover branches.
#
# Usage: ./tools/overnight_fuzz_launch.sh <CAMPAIGN_DIR> [DURATION_SECS]

set -u

CAMPAIGN_DIR="${1:?campaign dir required}"
DURATION="${2:-86400}"   # 24h default
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# target:workers:max_len
#
# Heavy (4): new/rebuilt with deep unexplored surface
#   - fuzz_encode_multiframe (Stream B, 310× cov jump in 30s smoke)
#   - fuzz_multistream (Stream C, P0 OOB just fixed — re-explore)
#
# Medium (2): Stream A targets with new VBR/FEC/DTX/float surface +
# rebuilt repacketizer
#   - fuzz_repacketizer_seq, fuzz_encode, fuzz_encode_safety,
#     fuzz_roundtrip, fuzz_roundtrip_safety, fuzz_decode_safety
#
# Light (1): saturated last campaign, regression value only
#   - fuzz_decode (cov 5501 plateau at minute 78 of last 8h)
#   - fuzz_dnn_blob (saturates ~30s; tiny surface)
#
# Total: 4 + 4 + 2*6 + 1*2 = 22 jobs. Leaves ~6 cores for triage agents.
TARGET_JOBS=(
    "fuzz_encode_multiframe:4:65536"
    "fuzz_multistream:4:16384"
    "fuzz_repacketizer_seq:2:16384"
    "fuzz_encode:2:16384"
    "fuzz_encode_safety:2:16384"
    "fuzz_roundtrip:2:16384"
    "fuzz_roundtrip_safety:2:16384"
    "fuzz_decode_safety:2:16384"
    "fuzz_decode:1:16384"
    "fuzz_dnn_blob:1:4096"
)

cd "$ROOT"

mkdir -p "$CAMPAIGN_DIR"

# Seed-copy preamble: lift pathological seeds from
# tests/fuzz/seeds/<target>/ into tests/fuzz/corpus/<target>/.
# Uses cp -n so libFuzzer's evolved corpus from prior runs isn't overwritten.
for spec in "${TARGET_JOBS[@]}"; do
    t="${spec%%:*}"
    SEEDS_DIR="$ROOT/tests/fuzz/seeds/$t"
    CORPUS_DIR="$ROOT/tests/fuzz/corpus/$t"
    if [ -d "$SEEDS_DIR" ]; then
        mkdir -p "$CORPUS_DIR"
        cp -n "$SEEDS_DIR"/* "$CORPUS_DIR/" 2>/dev/null || true
    fi
done

# Pre-build everything so the parallel cargo invocations below don't all
# fight over the cargo file lock during incremental rebuild.
echo "Pre-build: cargo +nightly fuzz build --fuzz-dir tests/fuzz" \
    | tee "$CAMPAIGN_DIR/launcher.log"
cargo +nightly fuzz build --fuzz-dir tests/fuzz \
    >> "$CAMPAIGN_DIR/launcher.log" 2>&1
if [ $? -ne 0 ]; then
    echo "Pre-build failed; abort." | tee -a "$CAMPAIGN_DIR/launcher.log"
    exit 1
fi

echo "" | tee -a "$CAMPAIGN_DIR/launcher.log"
echo "Campaign:  $CAMPAIGN_DIR" | tee -a "$CAMPAIGN_DIR/launcher.log"
echo "Duration:  ${DURATION}s ($((DURATION/3600))h $((DURATION%3600/60))m)" \
    | tee -a "$CAMPAIGN_DIR/launcher.log"
echo "Started:   $(date '+%Y-%m-%d %H:%M:%S %Z')" \
    | tee -a "$CAMPAIGN_DIR/launcher.log"
echo "" | tee -a "$CAMPAIGN_DIR/launcher.log"

TOTAL_JOBS=0
for spec in "${TARGET_JOBS[@]}"; do
    IFS=: read -r t workers max_len <<< "$spec"
    mkdir -p "$CAMPAIGN_DIR/$t"
    for ((w=0; w<workers; w++)); do
        LOG="$CAMPAIGN_DIR/$t/run-w$w.log"
        ARTIFACTS="$CAMPAIGN_DIR/$t/artifacts-w$w/"
        CAPTURE="$CAMPAIGN_DIR/$t/capture-w$w"
        mkdir -p "$ARTIFACTS" "$CAPTURE"
        (
            export FUZZ_PANIC_CAPTURE_DIR="$CAPTURE"
            cargo +nightly fuzz run --fuzz-dir tests/fuzz "$t" -- \
                -max_total_time="$DURATION" \
                -max_len="$max_len" \
                -artifact_prefix="$ARTIFACTS" \
                -print_final_stats=1 \
                > "$LOG" 2>&1
        ) &
        pid=$!
        echo "$pid" > "$CAMPAIGN_DIR/$t/pid-w$w"
        echo "[$t w$w] launched pid=$pid → $LOG" \
            | tee -a "$CAMPAIGN_DIR/launcher.log"
        TOTAL_JOBS=$((TOTAL_JOBS + 1))
    done
done

echo "" | tee -a "$CAMPAIGN_DIR/launcher.log"
echo "All $TOTAL_JOBS jobs launched across ${#TARGET_JOBS[@]} targets." \
    | tee -a "$CAMPAIGN_DIR/launcher.log"
echo "  ls $CAMPAIGN_DIR/<target>/{artifacts-wN,capture-wN}/ for crashes" \
    | tee -a "$CAMPAIGN_DIR/launcher.log"
echo "  tail -f $CAMPAIGN_DIR/<target>/run-wN.log to watch a worker" \
    | tee -a "$CAMPAIGN_DIR/launcher.log"

wait
echo "" | tee -a "$CAMPAIGN_DIR/launcher.log"
echo "Finished:  $(date '+%Y-%m-%d %H:%M:%S %Z')" \
    | tee -a "$CAMPAIGN_DIR/launcher.log"
