# ROP-BUG-FLUX-00055 — Encode and decode can overwrite their own input

- **State:** Open
- **Priority:** Must
- **Severity:** High
- **Area:** ropus-tools-core/path-safety
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

Static review at origin/main ac7ff8a. /Users/md/language/ropus/ropus-tools-core/src/commands/encode.rs:66-77 derives output by replacing the extension with opus, so encoding x.opus without -o targets the source itself. Explicit output can also be the input or a symlink/hardlink alias. After decoding into memory, encode.rs:193-203 unconditionally File::create truncates that path. Decode accepts the same explicit alias at commands/decode.rs:74-85 and truncates through :398-413 or audio/wav.rs:20-27,78-88. Fix: reject lexical and existing-file identity aliases before mutation, choose a noncolliding default for .opus input, and cover direct/symlink/hardlink cases. Static review only; no file was written and no command ran.

## Fix

<unfixed — raised only>

## Notes

### Additional default-output aliases (2026-07-31)

Static review at `origin/main` `bfe19ba` confirmed two implicit decode aliases
in addition to explicit direct, symlink, and hard-link output aliases.
`/Users/md/language/ropus/ropus-tools-core/src/commands/decode.rs:78-84`
derives the destination by replacing the extension, but input format detection
is content-based. A valid Ogg Opus stream named `x.wav` therefore defaults to
itself in WAV mode, and `x.pcm --raw` defaults to itself in raw mode. Output
creation at `:395-457` or
`/Users/md/language/ropus/ropus-tools-core/src/audio/wav.rs:18-28,78-88`
truncates the source after decoding.

Add both unchanged-extension cases to the non-colliding default policy and
source-preservation tests. This was a static review; no file or decoder ran.
