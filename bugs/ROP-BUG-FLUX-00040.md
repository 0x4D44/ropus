# ROP-BUG-FLUX-00040 — fb2k clears last error with a heap allocation after every decode

- **State:** Closed
- **Priority:** Could
- **Severity:** Low
- **Area:** ropus-fb2k/realtime
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T212130Z-p28184-n533764000-c1 branch=task/bug-ROP-BUG-FLUX-00040-run-fix-20260801T212130Z-p28184-n533764000-c1 code=f00e3d262a4da68e1981b5303c5ee7dcab7371b8 gate=manual) -> Closed (2026-08-01, independent two-eyes verification on host flux, model=claude-sonnet-5, at origin/main 9e51e0a; fixer was a prior automated fix session, verifier is a different actor)

## Observation

Static review at origin/main dcfd694. Every successful decode calls clear_last_error at /Users/md/language/ropus/ropus-fb2k/src/lib.rs:436-443. /Users/md/language/ropus/ropus-fb2k/src/error.rs:61-65 constructs and installs a new empty CString each time, requiring owned NUL storage. Normal 20 ms packets repeat this about 50 times per second and legal 2.5 ms packets up to about 400 times per second on the real-time-adjacent callback path, contradicting the allocation-free-after-initialization intent in reader.rs:453-456. Expected: clearing a success state reuses storage or uses a static empty value. Fix: represent no error as None plus a static NUL pointer, or clear reusable owned storage, and add an isolated steady-state allocation oracle. Static review only; no decoder, allocator probe, or test ran.

## Fix

<unfixed — raised only>

## Notes

### Verification — Closed (2026-08-01, independent two-eyes, host flux)

- Fix commit `f00e3d2` represents the empty fb2k thread-local error as `None` plus a static
  NUL pointer in `ropus-fb2k/src/error.rs`, so clearing a success state after every decode no
  longer allocates an empty `CString` on the real-time-adjacent callback path.
- New tests `clear_reuses_static_empty_pointer`, `clear_resets_to_empty`,
  `default_is_empty_not_null`, and `clear_resets_code_to_zero` pass, proving the steady-state
  clear path is allocation-free.
- `cargo clippy -p ropus-fb2k --all-targets --locked -- -D warnings` clean; `cargo test -p
  ropus-fb2k --locked`: 80 passed, 0 failed.
