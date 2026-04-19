#!/usr/bin/env bash
# fetch_corpus.sh — download a BASELINE reference-encoded Opus corpus.
#
# IMPORTANT — what this script provides (and does not).
# ------------------------------------------------------
# Every URL in SOURCES[] below points at a file that was itself encoded by
# opusenc / libopus (the xiph reference encoder). That makes the corpus this
# script fetches a useful *sanity baseline* — it exercises the Ogg container
# parser and decoder against known-good bitstreams — but it does NOT exercise
# non-reference encoder divergence. Catching bitstream quirks produced by
# encoders OTHER than libopus (FFmpeg's native `opus` encoder, WebRTC, Android
# AOSP, etc.) requires samples encoded by those other producers.
#
# To populate the corpus with genuine non-reference samples, drop files into
# $TARGET_DIR manually. See tests/vectors/real_world/README.md for concrete
# recipes (e.g. `ffmpeg -c:a opus` vs `-c:a libopus`).
#
# What the script does. For each entry in SOURCES[]:
#   1. If destination already exists AND its SHA-256 matches the pinned value,
#      skip.
#   2. Otherwise curl the URL to a temporary file, verify the SHA-256, and
#      move into place.
#   3. If a source is pinned as `TBD` it is REFUSED — the temp file is NOT
#      promoted, the actual digest is printed so the maintainer can pin, and
#      the script moves to the next source. If EVERY source is TBD the script
#      exits nonzero because nothing was usable.
#   4. On checksum mismatch or network failure, the error is counted and the
#      script continues to the next source; the run exits 1 if any failures
#      remain unresolved.
#
# Design notes.
# - Checksums guard against format drift and tampering. Any `TBD` entry is a
#   hard refusal (not trust-on-first-use) — a dev must pin the digest in the
#   script before the file will be admitted.
# - No temp-file leak: `trap` cleans up on any exit path.
# - `bash -n tools/fetch_corpus.sh` passes as a syntax check in environments
#   without curl/network access.

set -euo pipefail

# Allow callers to override the target dir; default matches the HLD.
TARGET_DIR="${TARGET_DIR:-tests/vectors/real_world}"

# ---------------------------------------------------------------------------
# Source list: "url|sha256|dest_filename"
# ---------------------------------------------------------------------------
#
# Selection criteria:
# - Publicly hosted on xiph.org / opus-codec.org / ffmpeg.org (all clearly
#   redistributable under the Opus reference licence or FFmpeg's sample
#   licence).
# - Short samples — big music files add minutes per run without catching
#   anything a short sample wouldn't.
# - Mix of bit-rates and content types (speech, music, mixed).
#
# PROVENANCE CAVEAT: every listed source was produced by opusenc / libopus
# (the xiph reference encoder). Using these alone means ropus is being diffed
# against output that the reference encoder itself produced — useful as a
# sanity baseline, NOT a substitute for samples from non-reference encoders
# (FFmpeg `-c:a opus`, WebRTC, Android AOSP, etc.). See the README for how to
# add those.
#
# CHECKSUMS must be real digests. A `TBD` entry is refused at runtime — the
# maintainer must pin the digest (the script prints the observed hash to make
# this easy) before the file will be admitted to the corpus.
SOURCES=(
    # Xiph Opus reference samples (classical and sample-track showcases used
    # on the opus-codec.org landing page).
    "https://opus-codec.org/static/examples/ehren-paper_lights-96.opus|TBD|ehren-paper_lights-96.opus"
    "https://opus-codec.org/static/examples/samples/speech_orig.opus|TBD|speech_orig.opus"
    "https://opus-codec.org/static/examples/samples/music_orig.opus|TBD|music_orig.opus"

    # FFmpeg FATE sample — short Opus-in-Ogg clip from the upstream FATE
    # test suite. Redistribution is covered by the FATE licence (see
    # https://ffmpeg.org/fate.html).
    "https://samples.ffmpeg.org/A-codecs/opus/intro-v2.opus|TBD|ffmpeg-intro-v2.opus"

    # Xiph test sample (short sine — sanity check the trivial case is still
    # in the corpus alongside complex signals).
    "https://opus-codec.org/static/examples/samples/sine.opus|TBD|sine.opus"
)

# ---------------------------------------------------------------------------
# Non-redistributable sources — NOT fetched. Add your own files locally:
# ---------------------------------------------------------------------------
#
# Sources with unclear / unfree redistribution rights are DELIBERATELY not
# listed above. If you want to exercise files captured from those services
# locally, drop them into $TARGET_DIR and corpus_diff will pick them up on
# its next scan. Example sources local developers tend to keep offline:
#
#   - Spotify-extracted .opus segments (encoder: ffmpeg/libopus)
#   - YouTube `yt-dlp -x --audio-format opus` extracts (encoder: FFmpeg
#     remuxed from WebM)
#   - WhatsApp voice notes (encoder: libopus via WebRTC, 16 kHz mono)
#   - Discord voice captures (encoder: libopus via WebRTC, 48 kHz mono/stereo)
#   - Any WebRTC SFU recording
#
# These matter for coverage because each encoder has its own quirks that the
# synthetic-vector conformance suite cannot reproduce.

# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

missing_tool() {
    local tool="$1"
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: required tool '$tool' is not on PATH" >&2
        return 1
    fi
    return 0
}

# Compute sha256 of a file. Prefers `sha256sum` (Linux/msys2/cygwin/WSL),
# falls back to `shasum -a 256` (macOS). Emits just the hex digest on stdout.
sha256_of() {
    local f="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$f" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$f" | awk '{print $1}'
    else
        echo "ERROR: need sha256sum or shasum on PATH" >&2
        return 1
    fi
}

main() {
    if ! missing_tool curl; then
        echo "fetch_corpus.sh: curl is unavailable; exiting without fetching." >&2
        echo "                 (the corpus is optional — run anyway to smoke-test.)" >&2
        exit 1
    fi

    mkdir -p "$TARGET_DIR"

    local tmp
    tmp="$(mktemp -t ropus_corpus_XXXXXX)" || {
        echo "ERROR: cannot create temp file" >&2
        exit 1
    }
    # shellcheck disable=SC2064
    trap "rm -f '$tmp'" EXIT

    local fetched=0
    local skipped=0
    local failed=0
    local tbd_refused=0
    local total=${#SOURCES[@]}

    for entry in "${SOURCES[@]}"; do
        local url checksum dest_name dest_path
        IFS='|' read -r url checksum dest_name <<<"$entry"
        dest_path="$TARGET_DIR/$dest_name"

        # Hard refusal: a TBD entry has no pinned checksum, so admitting its
        # bytes would be trust-on-first-use. Fetch to a temp path so we can
        # show the maintainer the observed digest (convenient for pinning),
        # but DO NOT promote it into $TARGET_DIR.
        if [[ "$checksum" == "TBD" ]]; then
            echo "  REFUSE $dest_name (checksum pinned as TBD)"
            echo "         $url"
            if curl --fail --silent --show-error --location -o "$tmp" "$url"; then
                local actual
                actual="$(sha256_of "$tmp")"
                echo "         observed sha256: $actual"
                echo "         To admit this file, replace TBD in SOURCES[] with the digest above"
                echo "         after confirming the bytes are the intended test material."
            else
                echo "         (unable to fetch for digest preview; pin manually)"
            fi
            tbd_refused=$((tbd_refused + 1))
            continue
        fi

        if [[ -f "$dest_path" ]]; then
            local actual
            actual="$(sha256_of "$dest_path")"
            if [[ "$actual" == "$checksum" ]]; then
                echo "  SKIP $dest_name (already present, checksum ok)"
                skipped=$((skipped + 1))
                continue
            else
                echo "  STALE $dest_name (checksum drift; refetching)"
            fi
        fi

        echo "  FETCH $dest_name <- $url"
        if ! curl --fail --silent --show-error --location -o "$tmp" "$url"; then
            echo "    ERROR: curl failed for $url" >&2
            failed=$((failed + 1))
            continue
        fi

        local actual
        actual="$(sha256_of "$tmp")"
        if [[ "$actual" != "$checksum" ]]; then
            echo "    ERROR: checksum mismatch for $dest_name" >&2
            echo "           expected $checksum" >&2
            echo "           actual   $actual" >&2
            failed=$((failed + 1))
            continue
        fi

        mv "$tmp" "$dest_path"
        # Re-create the temp so subsequent iterations have one to write to.
        tmp="$(mktemp -t ropus_corpus_XXXXXX)" || {
            echo "ERROR: cannot create temp file" >&2
            exit 1
        }
        fetched=$((fetched + 1))
    done

    echo "---"
    echo "fetch_corpus.sh: $fetched fetched, $skipped already present, $failed failed, $tbd_refused refused (TBD)"

    # Exit nonzero if any pinned fetch failed, OR if every source was TBD
    # (nothing usable was downloaded and nothing was already on disk).
    if [[ "$failed" -gt 0 ]]; then
        exit 1
    fi
    if [[ "$tbd_refused" -eq "$total" && "$skipped" -eq 0 ]]; then
        echo "ERROR: every source is pinned as TBD — pin at least one checksum before re-running." >&2
        exit 1
    fi
    if [[ "$tbd_refused" -gt 0 ]]; then
        echo "WARN: $tbd_refused source(s) refused for missing checksum; corpus is usable but incomplete." >&2
    fi
}

main "$@"
