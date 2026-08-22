# ROP-BUG-KIL-00050 — Low-level codec methods trust PCM slice capacities and unbounded frame sizes

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** ropus/opus-codec-api
- **Raised:** 2026-08-22T08:28:29Z
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
- **State history:** Open (2026-08-22T08:28:29Z, raised via `deltic bugs new` model=gpt-5.6-sol@max)

## Observation

Static review at HEAD 3972b03. OpusEncoder::encode and encode_float do not prove pcm.len() >= frame_size * channels before indexing at ropus/src/opus/encoder.rs:634-640 and :1695-1712; the native path also treats max_data_bytes as authoritative rather than data.len(). OpusDecoder::decode, decode24, and decode_float accept only a positive frame_size at ropus/src/opus/decoder.rs:1337-1410, then slice or index caller output at :1204-1223 and :1276-1315. The 24-bit and float wrappers allocate from the caller size before the inner decoder bounds work, while frame_size_select uses unchecked i32 products at encoder.rs:441-451. A safe direct caller can cause a panic, overflow-dependent acceptance, or infallible oversized allocation. Expected: invalid dimensions and undersized slices return an Opus error before indexing or allocation. Fix: use checked frame arithmetic, enforce codec maximums, validate input/output and declared byte capacities, and make scratch allocation bounded/fallible; add short-buffer and oversized-frame tests. This is residual core behavior distinct from projection-wrapper allocation bug ROP-BUG-CRUCIBLE-00005. Static inspection only; no code, app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
