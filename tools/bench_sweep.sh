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
[[ -x "$BIN" ]] || { echo "Bench binary not found at $BIN"; exit 1; }

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

for entry in "${VECTORS[@]}"; do
    IFS='|' read -r label wav bitrate <<< "$entry"
    if [[ ! -f "$wav" ]]; then
        echo "SKIP $label — missing $wav"
        continue
    fi
    out="$TMP/$(printf '%s' "$label" | tr ' /' '__').txt"
    "$BIN" bench "$wav" --iters "$ITERS" --repeats 1 --bitrate "$bitrate" > "$out" 2>&1 || {
        echo "FAIL $label"; cat "$out"; continue
    }
    # Save full output for later inspection.
    echo "── $label (bitrate=$bitrate, iters=$ITERS) ──"
    grep -E "^  encode *:|^  decode *:" "$out"
    # Also keep a compact table row.
    c_enc=$(awk -F'│' '/C encode/   {gsub(/ms| /,"",$4); print $4}' "$out")
    r_enc=$(awk -F'│' '/Rust encode/{gsub(/ms| /,"",$4); print $4}' "$out")
    c_dec=$(awk -F'│' '/C decode/   {gsub(/ms| /,"",$4); print $4}' "$out")
    r_dec=$(awk -F'│' '/Rust decode/{gsub(/ms| /,"",$4); print $4}' "$out")
    enc_ratio=$(awk "BEGIN{printf \"%.3f\", $r_enc / $c_enc}")
    dec_ratio=$(awk "BEGIN{printf \"%.3f\", $r_dec / $c_dec}")
    printf '  %-36s  C-enc=%6sms  R-enc=%6sms  enc_ratio=%s   C-dec=%6sms  R-dec=%6sms  dec_ratio=%s\n' \
        "$label" "$c_enc" "$r_enc" "$enc_ratio" "$c_dec" "$r_dec" "$dec_ratio" \
        >> "$TMP/summary.txt"
done

echo
echo "═══ SWEEP SUMMARY (medians, iters=$ITERS) ═══"
cat "$TMP/summary.txt"
