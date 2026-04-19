# Real-world Opus corpus

This directory holds `.opus` / `.ogg` / `.webm` files used by the
`corpus_diff` harness to diff ropus vs the C reference decoder
sample-for-sample.

## Scope and honest caveat

> **The `fetch_corpus.sh` script produces a BASELINE of reference-encoded
> samples — not a non-reference encoder diversity corpus.**

Every file the fetch script pulls (opus-codec.org examples, FFmpeg FATE)
was itself encoded by `opusenc` / libopus (the xiph reference encoder).
Diffing ropus against the C reference while decoding libopus-encoded input
is useful as a sanity baseline — it exercises the container parser and
decode path against known-good bitstreams — but by construction it cannot
catch encoder-specific bitstream quirks produced by encoders OTHER than
libopus.

Stage 3 of the deferred-work closeout intentionally shipped the baseline +
scaffolding. **Populating the corpus with genuine non-reference samples is
still open work.** This README calls out how to do it.

## How to populate (baseline)

```bash
bash tools/fetch_corpus.sh
```

Fetches the sources in `tools/fetch_corpus.sh` SOURCES[], writing into this
directory. Idempotent: files with a matching SHA-256 are skipped. Any
source pinned as `TBD` is refused — the maintainer must pin a real digest
before the script will admit it. Requires `curl` and either `sha256sum`
(Linux / msys2 / WSL) or `shasum -a 256` (macOS).

### Sources fetched by the script

All of these are redistributable under the Opus reference license or
FFmpeg's sample distribution terms. Exact file list and pinned checksums
live in `tools/fetch_corpus.sh` (SOURCES[]) — that file is the source of
truth; this table is a human-readable summary.

| Filename                     | Source                                                            | Encoder (provenance)    |
|------------------------------|-------------------------------------------------------------------|-------------------------|
| `ehren-paper_lights-96.opus` | https://opus-codec.org/static/examples/                           | opusenc (libopus)       |
| `speech_orig.opus`           | https://opus-codec.org/static/examples/samples/                   | opusenc (libopus)       |
| `music_orig.opus`            | https://opus-codec.org/static/examples/samples/                   | opusenc (libopus)       |
| `ffmpeg-intro-v2.opus`       | https://samples.ffmpeg.org/A-codecs/opus/                         | libopus (FATE sample)   |
| `sine.opus`                  | https://opus-codec.org/static/examples/samples/                   | opusenc (libopus)       |

## How to populate (non-reference — the gap)

To actually exercise the decoder against bitstreams ropus has not already
been tuned to match, drop files from these sources into this directory.
`corpus_diff` picks up anything with a `.opus` / `.ogg` / `.webm`
extension on its next run.

### Concrete recipes

**FFmpeg's native `opus` encoder** — distinct from libopus; exercises a
different encoder code path even though the bitstream format is the same:

```bash
# NON-reference (FFmpeg native encoder, experimental flag required):
ffmpeg -i input.wav -c:a opus -strict -2 ffmpeg-native.opus

# Compare against the libopus-linked build (for reference):
ffmpeg -i input.wav -c:a libopus libopus.opus
```

The `-c:a opus` encoder is the one to drop here. `-c:a libopus` is just
another opusenc under a different wrapper and belongs in the baseline set.

**WebRTC captures** — endpoints (browsers, SFUs, Android `MediaRecorder`)
typically run libopus via WebRTC but with constrained-VBR settings,
20 ms frames, and DTX/PLC patterns uncommon in opusenc. The bitstream
tolerances exercised are different:

- `chrome://webrtc-internals` → dump recorded streams.
- SFU recordings (Janus, LiveKit, Jitsi) — grab `.ogg` or `.webm` files
  straight off the server.
- Android `MediaRecorder` with `AudioEncoder.OPUS` + `.ogg` container.

**Streaming / consumer output** — usually libopus but with platform-specific
remux quirks (granule-pos handling, pre-skip, tag blocks):

- `yt-dlp -x --audio-format opus <url>` (FFmpeg remux from WebM).
- Spotify / Apple Music extracts (if you have rights to the decoded stream).
- WhatsApp / Telegram voice notes (libopus via WebRTC, 16 kHz mono).

Anything encoder-specific you can get your hands on helps. A dozen files
covering three or four distinct encoder implementations catches orders of
magnitude more bugs than fifty reference-encoded files.

## What `corpus_diff` does

```bash
cargo run -p ropus-harness --bin corpus_diff -- tests/vectors/real_world
```

For each file it:

1. Parses the Ogg container (no libopusfile dependency — just OggS pages +
   `OpusHead`).
2. Decodes every audio packet through ropus.
3. Decodes the same packets through the C reference via the existing FFI
   bindings (`harness/src/bindings.rs`).
4. Asserts PCM output matches sample-for-sample.

### Exit codes (stable contract)

| Exit | Meaning                                                           |
|------|-------------------------------------------------------------------|
| `0`  | Directory had files and every valid file matched sample-for-sample. |
| `1`  | One or more mismatches or panics, OR the directory argument was missing / unreadable. |
| `2`  | Directory exists but contains no candidate files (not populated). Distinct from `0` so CI cannot silently pass against an unpopulated corpus. Gate the CI step on `fetch_corpus.sh` or a populate-step having completed first. |

### Limitations (intentional for the P2 scope)

- Channel-mapping family 0 only. Multichannel / ambisonic files are logged
  and skipped — `projection_roundtrip` covers family 3.
- `.webm` files are logged and skipped (Matroska container parsing is not
  in scope for this harness; transcode to `.opus` first if you care).
- No seek, no FEC, no PLC — straight linear decode.
