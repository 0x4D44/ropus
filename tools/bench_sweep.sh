#!/usr/bin/env bash
# Run ropus-compare bench across a representative set of WAV vectors and
# emit a single summary table (C ratio per op) for each input.
#
# Usage: bash tools/bench_sweep.sh [--iters N]
set -euo pipefail

ITERS="${ITERS:-50}"
for arg in "$@"; do
    case "$arg" in
        --iters=*) ITERS="${arg#--iters=}" ;;
    esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/target/release/ropus-compare.exe"
[[ -x "$BIN" ]] || BIN="$ROOT/target/release/ropus-compare"
[[ -x "$BIN" ]] || { echo "Bench binary not found under $ROOT/target/release"; exit 1; }

parse_timing_rows() {
    local output_path="$1"
    awk -F'│' '
        function trim(value) {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
            return value
        }
        function valid_timing(value) {
            return value ~ /^[0-9]+([.][0-9]+)?$/ && (value + 0) > 0
        }
        {
            label = trim($2)
            if (label != "C encode" && label != "Rust encode" &&
                label != "C decode" && label != "Rust decode") {
                next
            }
            count[label]++
            timing = trim($4)
            sub(/ms$/, "", timing)
            gsub(/[[:space:]]/, "", timing)
            if (!valid_timing(timing)) {
                bad = 1
            } else {
                value[label] = timing
            }
        }
        END {
            if (count["C encode"] != 1 || count["Rust encode"] != 1 ||
                count["C decode"] != 1 || count["Rust decode"] != 1 || bad) {
                exit 1
            }
            printf "%s\t%s\t%s\t%s\n", value["C encode"], value["Rust encode"],
                value["C decode"], value["Rust decode"]
        }
    ' "$output_path"
}

# label|wav|bitrate
VECTORS=(
    "SILK NB 8k mono noise|$ROOT/tests/vectors/8000hz_mono_noise.wav|16000"
    "SILK WB 16k mono noise|$ROOT/tests/vectors/16000hz_mono_noise.wav|24000"
    "Hybrid 24k mono noise|$ROOT/tests/vectors/24000hz_mono_noise.wav|32000"
    "CELT FB 48k mono noise|$ROOT/tests/vectors/48000hz_mono_noise.wav|64000"
    "CELT FB 48k stereo noise|$ROOT/tests/vectors/48000hz_stereo_noise.wav|96000"
    "CELT 48k mono sine 1k loud|$ROOT/tests/vectors/48k_sine1k_loud.wav|64000"
    "CELT 48k mono sweep|$ROOT/tests/vectors/48k_sweep.wav|64000"
    "CELT 48k mono square 1k|$ROOT/tests/vectors/48k_square1k.wav|64000"
    "SPEECH 48k mono (SAPI TTS)|$ROOT/tests/vectors/speech_48k_mono.wav|64000"
    "MUSIC 48k stereo|$ROOT/tests/vectors/music_48k_stereo.wav|128000"
)

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
: > "$TMP/summary.txt"
failures=0

for entry in "${VECTORS[@]}"; do
    IFS='|' read -r label wav bitrate <<< "$entry"
    if [[ ! -f "$wav" ]]; then
        echo "SKIP $label — missing $wav"
        continue
    fi
    out="$TMP/$(printf '%s' "$label" | tr ' /' '__').txt"
    "$BIN" bench "$wav" --iters "$ITERS" --repeats 1 --bitrate "$bitrate" > "$out" 2>&1 || {
        echo "FAIL $label"
        cat "$out"
        failures=$((failures + 1))
        continue
    }
    # Save full output for later inspection.
    echo "── $label (bitrate=$bitrate, iters=$ITERS) ──"
    if ! parsed="$(parse_timing_rows "$out")"; then
        echo "FAIL $label — expected exactly one finite timing row for each C/Rust encode/decode operation"
        cat "$out"
        failures=$((failures + 1))
        continue
    fi
    IFS=$'\t' read -r c_enc r_enc c_dec r_dec <<< "$parsed"
    # Keep a compact table row.
    enc_ratio=$(awk -v rust="$r_enc" -v c="$c_enc" 'BEGIN { printf "%.3f", rust / c }')
    dec_ratio=$(awk -v rust="$r_dec" -v c="$c_dec" 'BEGIN { printf "%.3f", rust / c }')
    printf '  %-36s  C-enc=%6sms  R-enc=%6sms  enc_ratio=%s   C-dec=%6sms  R-dec=%6sms  dec_ratio=%s\n' \
        "$label" "$c_enc" "$r_enc" "$enc_ratio" "$c_dec" "$r_dec" "$dec_ratio" \
        >> "$TMP/summary.txt"
done

echo
echo "═══ SWEEP SUMMARY (medians, iters=$ITERS) ═══"
cat "$TMP/summary.txt"

if (( failures > 0 )); then
    echo "SWEEP FAILED: $failures vector(s) failed." >&2
    exit 1
fi
