# ROP-BUG-FLUX-00029 — Control decoder trusts unbounded packet-file allocation sizes

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness/control-decoder
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

Static review at `origin/main` `d0ab87e`. `/Users/md/language/ropus/harness/src/bin_inner/ctrl_decode_fixed.rs:89` through `:92` accepts unbounded packet-file header values, multiplies signed values cast to `usize` into the scratch allocation at `:116`, and trusts the low 31 bits of `flags_len` for `vec![0; len]` at `:118` through `:127`. A tiny malformed file can request about 2 GiB before `read_exact` proves no payload exists; frame count and output size are also unbounded. Fix: validate Opus sample rate/channels/frame size, require packet length at most 1275 and zero for lost frames, use checked arithmetic, cap frames/output, and validate before creating the output file or allocating. Static review only; no control decoder or malformed file ran.

## Fix

<unfixed — raised only>

## Notes

### Float control decoder repeats the unbounded-input path (2026-07-31)

`/Users/md/language/ropus/harness-deep-plc/src/bin_inner/ctrl_decode_float.rs:73-76,92-109`
trusts unsigned packet-file dimensions, frame count, and a 31-bit payload
length. Signed-to-`usize` casts and unchecked multiplication can request nearly
the address space, while a tiny record can request about 2 GiB before
`read_exact` discovers EOF. Apply the same supported-rate/channel/frame,
checked-arithmetic, packet-length, frame-count, and output-budget validation to
both control decoders before creating output. Static review at `origin/main`
`b65f812`; no decoder or malformed input ran.
