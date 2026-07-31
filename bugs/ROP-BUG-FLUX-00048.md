# ROP-BUG-FLUX-00048 — Public command options permit panics and invalid loss semantics

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** ropus-tools-core/options-validation
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

Static review at origin/main ac7ff8a. Public options are intended for GUI/plugin callers at /Users/md/language/ropus/ropus-tools-core/src/options.rs:1-6, but command code relies on CLI-only assumptions. /Users/md/language/ropus/ropus-tools-core/src/commands/encode.rs:31-49 panics on the constructible FrameDuration::Argument variant. /Users/md/language/ropus/ropus-tools-core/src/commands/decode.rs:50-56,167-175 accepts huge finite gain, then a saturated cast plus header gain can overflow in checked builds or wrap in release. The documented 0..=100 packet_loss_pct at options.rs:78-81 is not checked, and decode.rs:255-281 makes 101..255 mean 100 percent. Fix: validate every public command boundary, return typed errors for the sentinel and out-of-domain values, and use checked gain conversion/addition. Static review only; no command or test ran.

## Fix

<unfixed — raised only>

## Notes
