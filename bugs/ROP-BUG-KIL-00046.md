# ROP-BUG-KIL-00046 — Strict info queries accept malformed nonempty Opus packets

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-tools-core/info-validation
- **Raised:** 2026-08-22T07:33:50Z
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
- **State history:** Open (2026-08-22T07:33:50Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at HEAD f9a3871. ropus-tools-core/src/container/ogg.rs:216-226 names validate_opus_audio_packet as packet validation but rejects only an empty slice. ropus-tools-core/src/container/toc.rs:146-150 accepts code-3 frame count zero, and a one-byte code-3 packet yields an unknown count. With a usable EOS granule, commands/info.rs:481-498 uses the shallow validator and strict duration/bitrate queries never decode the packet, so malformed nonempty packets can return a plausible scalar with exit zero. Extended output at info.rs:361-382 also prints frames=? while inventing a one-frame duration through unwrap_or(1). Fix by validating the full Opus packet layout before trusting granule-derived strict output, including code-3 count, lengths, padding, and the 120 ms limit; extended output must reject malformed packets or keep duration unknown. Add zero-frame and truncated-code-3 fixtures. This is residual after closed ROP-BUG-FLUX-00063. Static inspection only; no app, build, test, or harness ran.

## Fix

<unfixed — raised only>

## Notes
