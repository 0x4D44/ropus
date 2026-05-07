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
scaffolding. The first release gate is now a **generated non-reference smoke**
defined by `corpus_manifest.toml`: `full-test --release-preflight` generates a
temporary FFmpeg-native Opus file from an existing checked-in WAV fixture and
requires exact PCM parity against the C reference. Broader curated
non-reference samples are still open work.

## Release gate manifest

`corpus_manifest.toml` is the source of truth for entries that may satisfy a
release corpus claim. Arbitrary local `.opus` / `.ogg` / `.webm` files are still
useful for exploration through `corpus_diff <dir>`, but they do not satisfy the
release gate unless the manifest marks them required and supported.

The manifest declares three required entries:

- `ffmpeg-native-sine-32k` (unpinned smoke; sine WAV, 32 kbps mono).
- `ffmpeg-native-sine-24k-mono` (pinned; sine WAV, 24 kbps mono).
- `ffmpeg-native-speech-40k-mono` (pinned; speech WAV, 40 kbps mono).

Each entry's `generation_command` documents the exact recipe, e.g.

```bash
ffmpeg -y -hide_banner -loglevel error \
  -i tests/vectors/48k_sine1k_loud.wav \
  -c:a opus -strict -2 -b:a 24k -ac 1 \
  tests/vectors/real_world/ffmpeg-native-sine-24k-mono.opus
```

The generated `.opus` files stay ignored/local by default. Non-quick
`full-test --release-preflight` uses a temporary directory for the generation,
runs `corpus_diff`, and fails the corpus claim if FFmpeg or the native `opus`
encoder is unavailable, if any pinned manifest entry's SHA256 disagrees with
the produced digest, or if `corpus_diff` exits with `2`/`3`/`1`. Quick
release-preflight explicitly makes no real-world/generated corpus claim.

### Pinned digests (locked under)

```
ffmpeg version 7.0.2-static https://johnvansickle.com/ffmpeg/  Copyright (c) 2000-2024 the FFmpeg developers
```

The pin hashes the Opus packet **payload bytes only** — the concatenated
payloads of every Ogg page after `OpusHead` and `OpusTags`. This pins encoder
output without depending on the Ogg muxer's randomised stream serial or on
metadata vendor strings, both of which would otherwise vary across runs and
across FFmpeg versions.

A maintainer running a different FFmpeg version may still see a hard-fail
SHA256 mismatch when the encoder output itself changes (the pin fires on real
encoder drift, not on muxer/metadata variation). In that case either match
the locked FFmpeg version or re-lock both digests in `corpus_manifest.toml`
and update the version line above.

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
| `0`  | Directory had at least one supported file that decoded nonzero audio, and every decoded file matched sample-for-sample. |
| `1`  | One or more mismatches or panics, the directory argument was missing / unreadable, or candidate files existed but none decoded nonzero audio for comparison. |
| `2`  | Directory exists but contains no candidate files (not populated). Distinct from `0` so CI cannot silently pass against an unpopulated corpus. Gate the CI step on `fetch_corpus.sh` or a populate-step having completed first. |
| `3`  | All candidate files were deferred (e.g. only `.webm`). Distinct from `0` so a release-preflight cannot satisfy a corpus claim with only deferred-container entries. |

### Limitations (intentional for the P2 scope)

- Channel-mapping family 0 only. Multichannel / ambisonic files are logged
  and skipped — `projection_roundtrip` covers family 3.
- `.webm` files are loud-deferred with a per-file `DEFER ... reason=webm-matroska-container-deferred`
  line (Matroska container parsing is not in scope for this harness; transcode
  to `.opus` first if you care). Exit code `3` is reserved for the all-deferred case.
- The oracle is exact PCM parity versus the C reference for supported Ogg
  family-0 packet decode only.
- No WebM/player semantics, seek, output gain, pre-skip trimming, granule
  position behavior, FEC, or PLC — straight linear packet decode.
