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

### Additional `ropusenc` boundary cases (2026-07-31)

Static review at `origin/main` `b4e2c31` found two more triggers covered by this
bug's command-boundary validation fix:

- `/Users/md/language/ropus/ropusenc/src/main.rs:44-47` accepts complexity
  `11..=255`, then
  `/Users/md/language/ropus/ropus-tools-core/src/commands/encode.rs:112-177`
  decodes and resamples the complete input before the encoder rejects it.
- `/Users/md/language/ropus/ropusenc/src/main.rs:36-42` accepts every `u32`
  bitrate, and
  `/Users/md/language/ropus/ropus-tools-core/src/commands/encode.rs:162-164`
  uses unchecked `Bitrate::Bits`. `/Users/md/language/ropus/ropus/src/api.rs:105-136`
  says untrusted values should use `Bitrate::try_bits`, because values above
  `i32::MAX` silently clamp. A huge requested bitrate can therefore succeed at
  a different effective value.

Add early Clap validation for complexity and bitrate, repeat the validation at
the public core boundary, and prove invalid options consume no input and create
no output. This was a static review; no command or test ran.
