# ROP-BUG-FLUX-00062 — Quiet and no-color flags do not control all CLI output

- **State:** Open
- **Priority:** Should
- **Severity:** Low
- **Area:** ropusenc/output-controls
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

Static review at origin/main b4e2c31. The signed parity goal at /Users/md/language/ropus/wrk_docs/2026.04.19 - HLD - opus-tools-parity.md:10-20 makes opus-tools flag behavior the contract, and https://opus-codec.org/docs/opus-tools/opusenc.html defines --quiet as displaying no messages. /Users/md/language/ropus/ropusenc/src/main.rs:136-140 defines quiet and no-color, but quiet only guards the banner at :244-271; /Users/md/language/ropus/ropus-tools-core/src/commands/encode.rs:79-110,129-157,321-345 still prints headings, paths, progress, bitrate, and completion. main.rs:22 also fixes Clap to ColorChoice::Auto, while :250-253 discards parsed no_color and /Users/md/language/ropus/ropus-tools-core/src/prelude.rs:33-35 disables only the separate colored crate. On a TTY, Clap help and parse errors therefore remain eligible for ANSI despite --no-color. Fix: carry one typed output policy into command reporting and suppress all informational messages under quiet; construct Clap with ColorChoice::Never when --no-color is present. Add quiet success/failure reporting tests and pseudo-terminal help/error ANSI tests. Static review only; no application or test ran.

## Fix

<unfixed — raised only>

## Notes

### Confirmed in `ropusdec` (2026-07-31)

`/Users/md/language/ropus/ropusdec/src/main.rs:67-99` uses quiet only for the
banner and discards typed `no_color`. The shared decoder still emits headings,
paths, tags, active flags, and completion at
`/Users/md/language/ropus/ropus-tools-core/src/commands/decode.rs:87-118,141-159,190-227,460-495`,
while Clap remains `ColorChoice::Auto` at
`/Users/md/language/ropus/ropusdec/src/main.rs:12-18`.

Extend the output-policy fix and oracles to `ropusdec`: successful file and
stdout decodes under quiet must emit no informational text, and pseudo-terminal
help/error output under `--no-color` must contain no ANSI. Static review at
`origin/main` `bfe19ba`; no binary or test ran.

### Confirmed in `ropusinfo` (2026-07-31)

`/Users/md/language/ropus/ropusinfo/src/main.rs:12-18,44-67` keeps Clap at
`ColorChoice::Auto`, uses quiet only for its banner, and relies on an
approximate raw-argument scan to infer query mode. Keep the intentional
banner-only meaning of `--quiet` for this read-only tool, but make strict query
mode itself select no banner and no colour after authoritative parsing.
Pseudo-terminal help/error and strict-query output must contain no ANSI.
Static review at `origin/main` `6a312e1`; no binary or test ran.

### Confirmed in `ropusplay` (2026-07-31)

`/Users/md/language/ropus/ropusplay/src/main.rs:78-90` prints the banner before
authoritative parsing, so plain `--list-devices` violates its one-device-name
per-line stdout contract. It also fixes Clap to `ColorChoice::Auto` and
discards typed `no_color` at :32-37,78-104.

The noninteractive path is structurally malformed:
`/Users/md/language/ropus/ropus-tools-core/src/commands/play.rs:113-139`
unconditionally writes a carriage-return progress prefix and, on failure,
ANSI erase-to-line-end before the plain summary at :444-470. Redirected and
quiet playback can therefore emit joined `decoding …playing …` text and raw
escapes. Quiet intentionally means noninteractive rather than silent for this
tool. Render ephemeral/ANSI UI only on the interactive TTY path, select
list-device and color policy after typed parsing, and assert exact redirected,
quiet, list, and pseudo-terminal output. Static review at `origin/main`
`e5d7113`; no binary, terminal, or test ran.
