# ROP-BUG-KIL-00051 — Extension generation trusts negative frame counts and short payload spans

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** ropus/opus-extensions
- **Raised:** 2026-08-22T08:28:39Z
- **Discovery source:** Agent
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
- **State history:** Open (2026-08-22T08:28:39Z, raised via `deltic bugs new` model=gpt-5.6-sol@max)

## Observation

Static review at HEAD 3972b03. opus_packet_extensions_generate at ropus/src/opus/repacketizer.rs:794-844 rejects nb_frames above MAX_FRAMES but not negative values; casting a negative count to usize drives fixed-array indexing. write_extension_payload at :706-755 reads ext.data[0] or ext.data[..ext.len] without checking the descriptor length against the payload slice, and generation writes according to len rather than the actual output slice capacity. Safe callers can therefore panic and may leave partial output when counts, descriptors, or buffer lengths disagree. Expected: the generator returns OPUS_BAD_ARG or OPUS_BUFFER_TOO_SMALL without mutating output for invalid arguments. Fix: preflight nb_frames in 0..=MAX_FRAMES, checked lengths/arithmetic, every descriptor payload span, and output capacity before writing; add negative-count, short-payload, and claimed-length tests. This is distinct from closed ROP-BUG-FLUX-00013, which fixed extension parsing rather than generation. Static inspection only; no code, app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
