#!/usr/bin/env bash
# 24h fuzz campaign launcher — v2 (fork-mode resilient).
#
# Differences from overnight_fuzz_launch.sh:
#  * Uses libFuzzer `-fork=N` so a panic kills only the child process;
#    the parent records the crash artifact and restarts the child.
#    No more "one bug → one dead worker for 22 h".
#  * One cargo invocation per TARGET (not per worker) — fork mode shares
#    corpus internally so this is structurally cleaner.
#  * Saturated targets dropped (`fuzz_repacketizer_seq`, `fuzz_dnn_blob`):
#    campaign #1 confirmed both at hard saturation (cov 715 / 503).
#  * CPU reallocated onto the differential targets that the per-class
#    skip filters now keep alive.
#  * Adds `fuzz_decode_plc_seq` (Phase 3 new target).
#
# Usage: ./tools/fuzz_24h_launch_v2.sh <CAMPAIGN_DIR> [DURATION_SECS]

set -u

CAMPAIGN_DIR="${1:?campaign dir required}"
DURATION="${2:-54000}"   # 15h default — Phase 4 budget
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# target:forks:max_len
#
# Heavy (6 forks): encode_multiframe — biggest unblocked surface;
#                  per-frame setter shuffle + skip filter for sr=8k VOIP.
# Heavy (4): multistream — encode-bytes class skipped for family=0 ch=1
#            sr=8000; rest of family/ch surface still differential.
# NEW (4): fuzz_decode_plc_seq — multi-packet decode with random drops.
# Med (3): roundtrip — float-12k-AUDIO-vbr class skipped; rest differential.
# Med (2): encode — float-LOWDELAY-8k class skipped; rest differential.
# Light (1): decode + safety variants — token job, panics recorded via fork.
TARGET_JOBS=(
    "fuzz_encode_multiframe:6:65536"
    "fuzz_multistream:4:32768"
    "fuzz_decode_plc_seq:4:32768"
    "fuzz_roundtrip:3:16384"
    "fuzz_encode:2:16384"
    "fuzz_decode:1:16384"
    "fuzz_decode_safety:1:16384"
    "fuzz_encode_safety:1:16384"
    "fuzz_roundtrip_safety:1:16384"
)

cd "$ROOT"

mkdir -p "$CAMPAIGN_DIR"

# Seed-copy preamble: lift pathological seeds from
# tests/fuzz/seeds/<target>/ into tests/fuzz/corpus/<target>/.
for spec in "${TARGET_JOBS[@]}"; do
    t="${spec%%:*}"
    SEEDS_DIR="$ROOT/tests/fuzz/seeds/$t"
    CORPUS_DIR="$ROOT/tests/fuzz/corpus/$t"
    if [ -d "$SEEDS_DIR" ]; then
        mkdir -p "$CORPUS_DIR"
        cp -n "$SEEDS_DIR"/* "$CORPUS_DIR/" 2>/dev/null || true
    fi
done

# Pre-build to dodge cargo file-lock contention at launch.
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

TOTAL_FORKS=0
for spec in "${TARGET_JOBS[@]}"; do
    IFS=: read -r t forks max_len <<< "$spec"
    mkdir -p "$CAMPAIGN_DIR/$t"
    LOG="$CAMPAIGN_DIR/$t/run.log"
    ARTIFACTS="$CAMPAIGN_DIR/$t/artifacts/"
    CAPTURE="$CAMPAIGN_DIR/$t/capture"
    mkdir -p "$ARTIFACTS" "$CAPTURE"
    (
        export FUZZ_PANIC_CAPTURE_DIR="$CAPTURE"
        # `-fork=N` + `-ignore_crashes=1`: with =0 (the default) the
        # parent libFuzzer process EXITS on the first child panic,
        # which campaign-2's 5-min smoke confirmed in the wild — both
        # heaviest targets dead within minutes. With =1 the parent
        # saves the artifact and respawns the child, fuzzing continues
        # for the full -max_total_time. The per-class skip filters
        # inside the targets are belt-and-braces.
        cargo +nightly fuzz run --fuzz-dir tests/fuzz "$t" -- \
            -fork="$forks" \
            -ignore_crashes=1 \
            -ignore_timeouts=1 \
            -ignore_ooms=1 \
            -max_total_time="$DURATION" \
            -max_len="$max_len" \
            -artifact_prefix="$ARTIFACTS" \
            -print_final_stats=1 \
            > "$LOG" 2>&1
    ) &
    pid=$!
    echo "$pid" > "$CAMPAIGN_DIR/$t/pid"
    echo "[$t fork=$forks] launched pid=$pid → $LOG" \
        | tee -a "$CAMPAIGN_DIR/launcher.log"
    TOTAL_FORKS=$((TOTAL_FORKS + forks))
done

echo "" | tee -a "$CAMPAIGN_DIR/launcher.log"
echo "All ${#TARGET_JOBS[@]} targets launched ($TOTAL_FORKS total fork children)." \
    | tee -a "$CAMPAIGN_DIR/launcher.log"
echo "  ls $CAMPAIGN_DIR/<target>/{artifacts,capture}/ for crashes" \
    | tee -a "$CAMPAIGN_DIR/launcher.log"
echo "  tail -f $CAMPAIGN_DIR/<target>/run.log to watch a target" \
    | tee -a "$CAMPAIGN_DIR/launcher.log"

wait
echo "" | tee -a "$CAMPAIGN_DIR/launcher.log"
echo "Finished:  $(date '+%Y-%m-%d %H:%M:%S %Z')" \
    | tee -a "$CAMPAIGN_DIR/launcher.log"
