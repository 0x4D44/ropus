# ROP-BUG-FLUX-00040 — fb2k clears last error with a heap allocation after every decode

- **State:** Open
- **Priority:** Could
- **Severity:** Low
- **Area:** ropus-fb2k/realtime
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260801T212130Z-p28184-n533764000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00040-run-fix-20260801T212130Z-p28184-n533764000-c1
- **Owner base:** d04b92346d4ec8e5e959b38803ceccbe54b6fd04
- **Owner fingerprint:** -
- **Owner since:** 2026-08-01T21:21:30Z
- **Owner until:** 2026-08-01T23:21:30Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

Static review at origin/main dcfd694. Every successful decode calls clear_last_error at /Users/md/language/ropus/ropus-fb2k/src/lib.rs:436-443. /Users/md/language/ropus/ropus-fb2k/src/error.rs:61-65 constructs and installs a new empty CString each time, requiring owned NUL storage. Normal 20 ms packets repeat this about 50 times per second and legal 2.5 ms packets up to about 400 times per second on the real-time-adjacent callback path, contradicting the allocation-free-after-initialization intent in reader.rs:453-456. Expected: clearing a success state reuses storage or uses a static empty value. Fix: represent no error as None plus a static NUL pointer, or clear reusable owned storage, and add an isolated steady-state allocation oracle. Static review only; no decoder, allocator probe, or test ran.

## Fix

<unfixed — raised only>

## Notes
