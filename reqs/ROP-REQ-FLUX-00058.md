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
