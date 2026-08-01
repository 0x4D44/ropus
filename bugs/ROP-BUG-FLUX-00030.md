# ROP-BUG-FLUX-00030 — WAV readers trust chunk extents and panic on malformed input

- **State:** Open
- **Priority:** Could
- **Severity:** Low
- **Area:** harness/wav-parser
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260801T200923Z-p96292-n603336000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00030-run-fix-20260801T200923Z-p96292-n603336000-c1
- **Owner base:** 316856551feb0723f8e2ac5299ee2f9538889f39
- **Owner fingerprint:** -
- **Owner since:** 2026-08-01T20:09:23Z
- **Owner until:** 2026-08-01T22:09:23Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at `origin/main` `d0ab87e`. Three duplicated readers at `/Users/md/language/ropus/harness/src/cli.rs:33`, `/Users/md/language/ropus/harness/src/bin_inner/projection_roundtrip.rs:46`, and `/Users/md/language/ropus/harness/src/bin_inner/pcm_drift.rs:34` inspect only the chunk header before reading fixed `fmt` fields or slicing the declared `data` extent. Oversized/truncated chunks cause bounds panics; policies also differ, and zero channels can reach division at `/Users/md/language/ropus/harness/src/bin_inner/projection_roundtrip.rs:273`. Expected: malformed user-supplied media produces a controlled nonzero error. Actual: it can panic, and each copy accepts a different malformed subset. Fix: replace the copies with one shared fallible PCM16 RIFF parser using checked extent/alignment arithmetic, validating `fmt` size/format, supported nonzero rate/channels, sample alignment, and chunk bounds. Add truncated `fmt`, oversized `data`, zero-channel, and odd-alignment cases. Static review only; no WAV was parsed.

## Fix

<unfixed — raised only>

## Notes
