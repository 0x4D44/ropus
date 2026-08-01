# ROP-BUG-FLUX-00053 — Prelude can print banners into implicit stdout binary streams

- **State:** Open
- **Priority:** Must
- **Severity:** High
- **Area:** ropus-tools-core/prelude-routing
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260801T122743Z-p98923-n509169000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00053-run-fix-20260801T122743Z-p98923-n509169000-c1
- **Owner base:** 37096076d022a92d1f816e89d69d0d568c32d0a0
- **Owner fingerprint:** -
- **Owner since:** 2026-08-01T12:27:43Z
- **Owner until:** 2026-08-01T14:27:43Z
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

### Confirmed in `ropusinfo` strict-query routing (2026-07-31)

`/Users/md/language/ropus/ropusinfo/src/main.rs:44-62` duplicates raw argument
parsing to decide whether the banner is safe. Clap accepts `-q=duration`, but
the scanner misses it and prints the banner before the scalar query result.
Arguments after `--` are also not separated from options. Derive output routing
from the authoritative parsed command and add attached-short-query and
end-of-options cases without injecting `--quiet`. Static review at
`origin/main` `6a312e1`; no binary or test ran.
