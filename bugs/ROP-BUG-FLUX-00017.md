# ROP-BUG-FLUX-00017 — C ABI constructors abort instead of returning allocation failure

- **State:** Fixed
- **Priority:** Could
- **Severity:** Medium
- **Area:** capi/allocation-errors
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
- **State history:** Open (2026-07-31, raised via `deltic bugs new`) -> Fixed (2026-08-01, deltic:auto role=fix run=fix-20260801T181555Z-p60679-n523856000-c1 branch=task/bug-ROP-BUG-FLUX-00017-run-fix-20260801T181555Z-p60679-n523856000-c1 code=552461ad79ee928ba7f625c376cd261ae4131d97 gate=manual)

## Observation

Valid *_create calls do not honor the exported OPUS_ALLOC_FAIL contract under memory pressure. capi/src/encoder.rs:49-58, decoder.rs:78-87, ms_encoder.rs:39-46, ms_decoder.rs:34-41, projection.rs:59-78, and repacketizer.rs:72-81 call std::alloc::handle_alloc_error when handle allocation returns null; Box and Vec construction is also infallible and aborting. ffi_guard catches unwinds only, so it cannot translate allocator aborts. Expected: return NULL (or an error status) and write OPUS_ALLOC_FAIL. Actual: the process aborts. Fix: make every allocation in create/init fallible, clean up partial state, commit output parameters only after full initialization, and return the libopus allocation error without unwinding or aborting.

## Fix

<unfixed — raised only>

## Notes
