# ROP-REQ-KILN-00015 — Derive every fuzz campaign target set from the Cargo manifest

- **State:** Draft
- **Priority:** Should
- **Area:** tools/fuzz-campaign-discovery
- **Raised:** 2026-08-13T17:17:49Z
- **Discovery source:** Agent
- **Implemented-by:** —
- **Satisfied-by:** —
- **Violated-by:** —
- **Depends-on:** —
- **Design:** —
- **Flow:** light
- **Claimed-by:** —
- **State history:** Draft (2026-08-13T17:17:49Z, raised via `deltic reqs new` model=gpt-5.6-sol@high)

## Statement

The system must derive the runnable fuzz-target set from tests/fuzz/Cargo.toml for every general or overnight campaign. Per-target worker counts, maximum lengths, and deliberate exclusions may remain checked-in policy, but each policy entry must resolve to a declared target and every newly declared target must be surfaced as unassigned or explicitly excluded rather than silently omitted. The shared discovery helper should be the single parser so campaign launchers cannot drift independently.

## Notes
