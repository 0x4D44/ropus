# ROP-REQ-FLUX-00067 — Extract a pure playback controller from terminal and audio effects

- **State:** Draft
- **Priority:** Should
- **Area:** ropus-tools-core/playback-controller
- **Raised:** 2026-07-31
- **Implemented-by:** —
- **Satisfied-by:** —
- **Violated-by:** —
- **Flow:** heavy
- **Claimed-by:** —
- **State history:** Draft (2026-07-31, raised via `deltic reqs new` model=gpt-5.6-sol@xhigh)

## Statement

Extract the playback transition logic in /Users/md/language/ropus/ropus-tools-core/src/commands/play.rs:60-224,341-441 from the coupled filesystem, decode, rodio, crossterm, and printing adapters. Introduce a pure controller that accepts typed events such as TrackReady, DecodeFailed, Drained, Pause, Resume, Next, Prev, Quit, and elapsed position, and returns explicit state transitions and effects. Keep device, sink, terminal, and file I/O in thin adapters. Hardware-free table tests must prove Off/All/Single end behavior, one and all decode failures, empty decode rejection, Prev at both sides of the two-second threshold, pause/resume, Next, Quit/Ctrl-C, and no empty-playlist indexing. Preserve the signed CLI and UI behavior. This is distinct from ROP-REQ-FLUX-00058, which owns streaming data flow, and should supply stable control-plane oracles while that pipeline changes. Raised from static review at origin/main e5d7113; no player, terminal, build, test, or harness ran.

## Notes
