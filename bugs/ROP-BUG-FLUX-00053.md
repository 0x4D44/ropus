# ROP-BUG-FLUX-00053 — Prelude can print banners into implicit stdout binary streams

- **State:** Closed
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T122743Z-p98923-n509169000-c1 branch=task/bug-ROP-BUG-FLUX-00053-run-fix-20260801T122743Z-p98923-n509169000-c1 code=7201b5a gate=manual) -> Closed (2026-08-01, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main dc05a88; fixer was a prior automated fix session, verifier is a different actor)

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


### Verification — Closed (2026-08-01, independent two-eyes, host flux)

Covers both the original `ropus-tools-core` prelude finding and the `ropusinfo` note.

- Fix derives banner/stdout routing from the authoritative Clap-parsed command instead of a
  second raw-argv scanner in `ropus-tools-core/src/prelude.rs`; new test
  `typed_paths_select_stdout_without_reparsing_argv` passes.
- `ropusinfo/src/main.rs` likewise now derives routing from the parsed command; new tests
  `typed_query_controls_banner_for_all_value_spellings` and
  `end_of_options_keeps_query_like_input_positional` pass, plus integration test
  `info_query_attached_short_form_returns_bare_number_without_quiet` covers the `-q=duration`
  attached-short-query case called out in the note.
- `cargo clippy -p ropus-tools-core --all-targets --locked -- -D warnings` and `cargo clippy -p
  ropusinfo --all-targets --locked -- -D warnings` both clean; `cargo test -p ropus-tools-core
  --locked` (112 passed) and `cargo test -p ropusinfo --locked` (9 passed) both 0 failed.
