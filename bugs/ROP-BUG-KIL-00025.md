# ROP-BUG-KIL-00025 — peek.c layout mirror compiles without config.h and has no drift guard

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness-deep-plc/shims
- **Raised:** 2026-08-19T10:46:15Z
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
- **State history:** Open (2026-08-19T10:46:15Z, raised via `deltic bugs new`)

## Observation

Static review. harness-deep-plc/c/peek.c hand-mirrors the private `OpusCustomDecoder` prefix (peek.c:36-75), including an `#ifdef ENABLE_DEEP_PLC` block (:66-72) — but the file never includes config.h (its includes at :14-18 are opus_types.h/opus_defines.h/arch.h/modes.h/structs.h) and build.rs passes `ENABLE_DEEP_PLC` only via config.h (harness-deep-plc/config.h:32), never as a `-D` flag. Xiph convention puts the `#ifdef HAVE_CONFIG_H #include "config.h"` prologue in each .c file, and the sibling fixed-point harness shims follow it (harness/silk_encode_frame_FIX_traced.c:60), but none of harness-deep-plc's four local C files do. If no reference header transitively includes config.h (needs the git-ignored reference/ checkout to prove — absent on this host), peek.c's TU omits the 1288-byte plc block that celt_decoder.c's live struct has, and every CELT trailing-array peek reads shifted memory. No guard can catch it: there is no `_Static_assert` on sizeof/offsetof anywhere in the file despite its own ":33 must match celt_decoder.c byte-for-byte" note, `PLC_UPDATE_SAMPLES` is a bare `640` literal (:69), and `validate_peek_capacities` (src/lib.rs:521-532) reads only fields BEFORE the divergence point. Corroboration: tests/state_divergence.rs:11-15 records C-side CELT arrays as "garbage/uninitialised" where an `OPUS_CLEAR`ed decoder should read zeros. Also unresolved without reference/: whether `peek_celt_trailing_base` (peek.c:123-127) must skip the `lpc[channels*CELT_LPC_ORDER]` block preceding `oldBandE` in upstream layout (reviewers disagreed; harness/debug_helper.c:730 shares the same formula). Fix: add the config.h prologue to all four shim files, add `_Static_assert`s on the mirrored layout (derive 640 as `4 * FRAME_SIZE`), delete the dead `peek_celt_old_log_e2` (:155-164, zero references workspace-wide), and add one positive-path test that decodes a CELT frame and checks a peeked value against a known-good property — that test also settles the trailing-base question. Static inspection only; no build or test ran.

## Fix

<unfixed — raised only>

## Notes
