# ROP-BUG-FLUX-00048 — Public command options permit panics and invalid loss semantics

- **State:** Closed
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T221503Z-p96641-n943371000-c1 branch=task/bug-ROP-BUG-FLUX-00048-run-fix-20260801T221503Z-p96641-n943371000-c1 code=39f955467fc65a7af104eebf0bd345dade7a8152 gate=manual) -> Closed (2026-08-02, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main 3528b9e; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main ac7ff8a. Public options are intended for GUI/plugin callers at /Users/md/language/ropus/ropus-tools-core/src/options.rs:1-6, but command code relies on CLI-only assumptions. /Users/md/language/ropus/ropus-tools-core/src/commands/encode.rs:31-49 panics on the constructible FrameDuration::Argument variant. /Users/md/language/ropus/ropus-tools-core/src/commands/decode.rs:50-56,167-175 accepts huge finite gain, then a saturated cast plus header gain can overflow in checked builds or wrap in release. The documented 0..=100 packet_loss_pct at options.rs:78-81 is not checked, and decode.rs:255-281 makes 101..255 mean 100 percent. Fix: validate every public command boundary, return typed errors for the sentinel and out-of-domain values, and use checked gain conversion/addition. Static review only; no command or test ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-02, independent two-eyes, host flux)

Covers the main observation plus the `ropusenc`, `ropusdec`, and `ropusplay` boundary notes.

- Fix commit `39f9554` adds boundary validation to the public core (`commands/decode.rs`,
  `commands/encode.rs`, `commands/play.rs`, `options.rs`) plus new CLI parsers
  (`parse_bitrate`, `parse_complexity`, `parse_gain_db`, `parse_volume`) in
  `ropusdec`/`ropusenc`/`ropusplay`.
- Regression re-verified by construction across the affected areas at `39f95546~1`:
  - `commands::decode::tests::public_decode_options_reject_invalid_loss_before_io` and
    `..._reject_unrepresentable_gain_before_io` failed pre-fix — the assertions on the typed
    error message failed because the code instead tried (and failed) to open a missing
    input file, proving loss/gain validation ran after I/O, not before.
  - `commands::play::tests::play_rejects_nonfinite_volume_before_device_or_input` failed
    pre-fix — it enumerated real audio devices instead of rejecting NaN volume first.
  - `parse_bitrate`/`parse_complexity` (and the encode-options equivalent
    `validate_encode_options`) did not exist pre-fix at all (compile error), confirming the
    CLI accepted unranged values with no boundary check.
  - All corresponding tests pass at the current tree.
- `cargo clippy -p ropus-tools-core --all-targets --locked -- -D warnings` clean; `cargo test
  -p ropus-tools-core -p ropusdec -p ropusenc -p ropusplay`: 129+22+2+4+2+8+2 passed, 0
  failed.

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

### Additional `ropusdec` boundary cases (2026-07-31)

Static review at `origin/main` `bfe19ba` confirmed split validation ownership in
the decoder. `/Users/md/language/ropus/ropusdec/src/main.rs:99-106` rejects
packet loss above 100 only for CLI callers, while public
`DecodeOptions.packet_loss_pct` values `101..=255` reach
`/Users/md/language/ropus/ropus-tools-core/src/commands/decode.rs:255-281` and
make every packet lost. The core rejects non-finite gain at `:50-56`, but a huge
finite gain is saturated during float-to-`i32` conversion and then added to the
header gain at `:167-175`, which can panic in checked builds or wrap in release.

Centralize validation in the public core boundary, use checked Q8 conversion
and addition, and keep Clap range parsers only for early diagnostics. Cover
direct library construction as well as wrapper parsing. This was a static
review; no decoder or test ran.

### Additional `ropusplay` boundary cases (2026-07-31)

`/Users/md/language/ropus/ropusplay/src/main.rs:43-45` accepts every `f32`
volume. `/Users/md/language/ropus/ropus-tools-core/src/commands/play.rs:171-173`
passes NaN through `clamp` to the audio sink. Reject non-finite volume at the
public command boundary and constrain the CLI to the documented `0.0..=1.0`.
Finite out-of-range values may retain the signed design's clamping policy.

The same command accepts `+128 dB` at
`/Users/md/language/ropus/ropus-tools-core/src/commands/play.rs:227-243`, but
the promised Q8 decoder gain tops out at 32767, just below +128 dB. Validate
the representable combined header-plus-user gain before decoding. Cover CLI and
direct `PlayOptions` construction without opening a device or input. Static
review at `origin/main` `e5d7113`; no player, device, or test ran.
