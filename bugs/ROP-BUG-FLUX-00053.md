# ROP-BUG-FLUX-00053 — Prelude can print banners into implicit stdout binary streams

- **State:** Open
- **Priority:** Must
- **Severity:** High
- **Area:** ropus-tools-core/prelude-routing
- **Raised:** 2026-07-31
- **Owner:** -
- **Owner role:** -
- **Owner run:** -
- **Owner host:** -
- **Owner branch:** -
- **Owner base:** -
- **Owner fingerprint:** -
- **Owner since:** -
- **Owner until:** -
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main ac7ff8a. /Users/md/language/ropus/ropus-tools-core/src/prelude.rs:43-103 re-parses raw argv without knowing option arity. For ropusenc --bitrate 64000 - or ropusdec --rate 44100 -, it treats the option value as the first positional, ignores the later stdin sentinel, and reports output_is_stdout=false. The typed command later correctly maps stdin without -o to stdout, so the binary banner prefixes and corrupts Ogg/WAV/PCM bytes. Tests at prelude.rs:136-179 cover only explicit output spellings. Fix: derive banner routing after authoritative CLI parsing instead of maintaining a second parser, and add every value-taking option before/after stdin plus binary-prefix checks. Static review only; no binary or test ran.

## Fix

<unfixed — raised only>

## Notes
