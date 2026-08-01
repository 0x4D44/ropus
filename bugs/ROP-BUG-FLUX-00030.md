# ROP-BUG-FLUX-00030 — WAV readers trust chunk extents and panic on malformed input

- **State:** Fixed
- **Priority:** Could
- **Severity:** Low
- **Area:** harness/wav-parser
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T200923Z-p96292-n603336000-c1 branch=task/bug-ROP-BUG-FLUX-00030-run-fix-20260801T200923Z-p96292-n603336000-c1 code=bac27b1 gate=manual)

## Observation

Static review at `origin/main` `d0ab87e`. Three duplicated readers at `/Users/md/language/ropus/harness/src/cli.rs:33`, `/Users/md/language/ropus/harness/src/bin_inner/projection_roundtrip.rs:46`, and `/Users/md/language/ropus/harness/src/bin_inner/pcm_drift.rs:34` inspect only the chunk header before reading fixed `fmt` fields or slicing the declared `data` extent. Oversized/truncated chunks cause bounds panics; policies also differ, and zero channels can reach division at `/Users/md/language/ropus/harness/src/bin_inner/projection_roundtrip.rs:273`. Expected: malformed user-supplied media produces a controlled nonzero error. Actual: it can panic, and each copy accepts a different malformed subset. Fix: replace the copies with one shared fallible PCM16 RIFF parser using checked extent/alignment arithmetic, validating `fmt` size/format, supported nonzero rate/channels, sample alignment, and chunk bounds. Add truncated `fmt`, oversized `data`, zero-channel, and odd-alignment cases. Static review only; no WAV was parsed.

## Fix

<unfixed — raised only>

## Notes
