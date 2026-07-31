# ROP-REQ-FLUX-00058 — Stream audio tools through bounded-memory pipelines

- **State:** Draft
- **Priority:** Should
- **Area:** ropus-tools-core/audio-streaming
- **Raised:** 2026-07-31
- **Implemented-by:** —
- **Satisfied-by:** —
- **Violated-by:** —
- **Flow:** heavy
- **Claimed-by:** —
- **State history:** Draft (2026-07-31, raised via `deltic reqs new` model=gpt-5.6-sol@xhigh)

## Statement

Replace whole-track PCM batching in ropus-tools-core with bounded streaming stages. /Users/md/language/ropus/ropus-tools-core/src/audio/decode.rs:155-235 accumulates every decoded sample and feeds encode/play at commands/encode.rs:112-153 and commands/play.rs:132-177. /Users/md/language/ropus/ropus-tools-core/src/commands/decode.rs:249-370 likewise retains the entire output and can create further full copies while trimming, resampling, and quantizing. Multi-hour stereo inputs can therefore require GiBs despite sequential codecs and sinks. Provide a reusable streaming decode/resample/downmix/sink contract with explicit final trimming, bounded queues, stdin policy, WAV header patch/RF64 or explicit size failure, and resource oracles proving memory does not scale with duration. Preserve current file/stdin behavior and codec output semantics. Raised from static review at origin/main ac7ff8a; no workload or measurement ran.

## Notes

### `ropusdec` acceptance detail (2026-07-31)

Static review at `origin/main` `bfe19ba` adds these bounded-pipeline acceptance
cases:

- `/Users/md/language/ropus/ropus-tools-core/src/commands/decode.rs:188,211,252-348`
  sends `--rate 48000 --no-dither` through whole-track float accumulation,
  cloning, and requantization even though `need_resample` is false. Identity
  rate must take the direct path and remain byte-identical to omitted rate.
- Raw and WAV writers issue one buffered `write_all` per sample at
  `/Users/md/language/ropus/ropus-tools-core/src/commands/decode.rs:424-444`
  and `/Users/md/language/ropus/ropus-tools-core/src/audio/wav.rs:64-69,149-154`.
  Use bounded byte blocks; benchmark before claiming a material speedup.
- Decode retains the complete `OpusTags` packet and owned copies of vendor and
  comments although it reports only vendor and count
  (`/Users/md/language/ropus/ropus-tools-core/src/commands/decode.rs:148-159`;
  `/Users/md/language/ropus/ropus-tools-core/src/container/ogg.rs:42-114`).
  Give metadata an explicit budget and drop unneeded storage before audio
  accumulation.
- Compute WAV/RF64 representability before opening a regular destination.
  Current path writers create the file before the 4 GiB checks at
  `/Users/md/language/ropus/ropus-tools-core/src/audio/wav.rs:18-47,78-108`.

No workload, benchmark, decoder, or test ran. Per-sample write cost,
large-metadata impact, and atomic replacement policy remain unverified, so they
were not raised as separate defects.

### `ropusplay` acceptance detail (2026-07-31)

- `/Users/md/language/ropus/ropus-tools-core/src/audio/decode.rs:155-235` and
  `/Users/md/language/ropus/ropus-tools-core/src/commands/play.rs:110-178`
  retain the complete current track before creating the sink. Bound memory
  independently of duration and begin playback before full-file decode.
- Do not wait for the next full decode at every track transition. Preserve the
  signed non-gapless policy while bounding queue depth and first-sound latency.
- The display path at
  `/Users/md/language/ropus/ropus-tools-core/src/commands/play.rs:488-511,550-608,673-686`
  owns arbitrary tag strings and rescans the full selected label on each
  100 ms UI tick before truncation. Apply a metadata/display budget once and
  make repaint work independent of the original tag length.

Use memory, first-sound, transition-latency, and examined-label resource
oracles. Static review at `origin/main` `e5d7113`; no workload, player,
benchmark, or test ran.
