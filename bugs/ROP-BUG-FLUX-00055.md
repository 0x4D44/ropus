# ROP-BUG-FLUX-00055 — Encode and decode can overwrite their own input

- **State:** Closed
- **Priority:** Must
- **Severity:** High
- **Area:** ropus-tools-core/path-safety
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-02, deltic:auto role=fix run=fix-20260801T230202Z-p61873-n904081000-c1 branch=task/bug-ROP-BUG-FLUX-00055-run-fix-20260801T230202Z-p61873-n904081000-c1 code=5b63404714043ac391bcb58a459374fadac79861 gate=manual) -> Closed (2026-08-02, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main 3528b9e; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main ac7ff8a. /Users/md/language/ropus/ropus-tools-core/src/commands/encode.rs:66-77 derives output by replacing the extension with opus, so encoding x.opus without -o targets the source itself. Explicit output can also be the input or a symlink/hardlink alias. After decoding into memory, encode.rs:193-203 unconditionally File::create truncates that path. Decode accepts the same explicit alias at commands/decode.rs:74-85 and truncates through :398-413 or audio/wav.rs:20-27,78-88. Fix: reject lexical and existing-file identity aliases before mutation, choose a noncolliding default for .opus input, and cover direct/symlink/hardlink cases. Static review only; no file was written and no command ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-02, independent two-eyes, host flux)

Covers the main observation plus the additional default-output-alias note. The float
control-decoder note's own alias check (`ctrl_decode_float.rs`) could not be independently
re-executed in this environment — see below.

- Fix commit `5b63404` adds `reject_input_output_alias`, `paths_refer_to_same_file`,
  `metadata_identity`, and `noncolliding_default_output` to `ropus-tools-core/src/util.rs`
  (entirely new; the module previously had no identity-aliasing checks at all) and wires
  them into `commands/decode.rs` and `commands/encode.rs` before any output is created.
- Confirmed by reading the pre-fix tree at `5b63404~1`: `commands/encode.rs` and
  `commands/decode.rs` called `File::create` unconditionally with no alias check
  (`grep -n 'File::create\|reject_input_output_alias'` found the create calls and no guard).
  The fix's new tests (`direct_and_lexical_aliases_are_rejected`,
  `symlink_and_hard_link_aliases_are_rejected`, `default_extension_collision_gets_a_safe_suffix`,
  `decode_rejects_direct_input_output_alias_before_reading`,
  `encode_rejects_direct_input_output_alias_before_decode`) all fail to compile against the
  pre-fix tree (`reject_input_output_alias`/`noncolliding_default_output` not found),
  confirming the capability was entirely absent. All 5 tests pass at the current tree.
- The float control-decoder's own alias guard (`harness-deep-plc/src/bin_inner/ctrl_decode_float.rs`)
  is present in the current tree (`reject_input_output_alias` called before `File::create`,
  confirmed by reading the code) but its test (`output_aliases_are_rejected_before_decode`)
  is compiled only under `#[cfg(not(no_reference))]`, gated on the xiph/opus C reference
  sources that are git-ignored and not fetched in this environment
  (`cargo run -p fetch-assets -- all`) — I did not independently re-execute that one test.
  This does not affect the `ropus-tools-core` fix, which is fully verified above.
- `cargo clippy -p ropus-tools-core --all-targets --locked -- -D warnings` clean; `cargo test
  -p ropus-tools-core --locked`: 129 lib + 22 integration passed, 0 failed.

### Additional default-output aliases (2026-07-31)

Static review at `origin/main` `bfe19ba` confirmed two implicit decode aliases
in addition to explicit direct, symlink, and hard-link output aliases.
`/Users/md/language/ropus/ropus-tools-core/src/commands/decode.rs:78-84`
derives the destination by replacing the extension, but input format detection
is content-based. A valid Ogg Opus stream named `x.wav` therefore defaults to
itself in WAV mode, and `x.pcm --raw` defaults to itself in raw mode. Output
creation at `:395-457` or
`/Users/md/language/ropus/ropus-tools-core/src/audio/wav.rs:18-28,78-88`
truncates the source after decoding.

Add both unchanged-extension cases to the non-colliding default policy and
source-preservation tests. This was a static review; no file or decoder ran.

### Float control decoder explicit alias (2026-07-31)

`/Users/md/language/ropus/harness-deep-plc/src/bin_inner/ctrl_decode_float.rs:68-87`
opens the packet input, reads only its header, then creates the caller-selected
PCM output without rejecting direct, symlink, or hard-link aliases. The create
truncates the still-needed packet stream before frame decoding begins. Include
this binary in the shared file-identity rejection and source-preservation
oracles. Static review at `origin/main` `b65f812`; no file was written.
