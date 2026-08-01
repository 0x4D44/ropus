# ROP-BUG-FLUX-00017 — C ABI constructors abort instead of returning allocation failure

- **State:** Open
- **Priority:** Could
- **Severity:** Medium
- **Area:** capi/allocation-errors
- **Raised:** 2026-07-31
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260801T181555Z-p60679-n523856000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-ROP-BUG-FLUX-00017-run-fix-20260801T181555Z-p60679-n523856000-c1
- **Owner base:** 12c0973b5ecf1c3b762d583b29f11ed30369d48c
- **Owner fingerprint:** -
- **Owner since:** 2026-08-01T18:15:55Z
- **Owner until:** 2026-08-01T20:15:55Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31, raised via `deltic bugs new`)

## Observation

Valid *_create calls do not honor the exported OPUS_ALLOC_FAIL contract under memory pressure. capi/src/encoder.rs:49-58, decoder.rs:78-87, ms_encoder.rs:39-46, ms_decoder.rs:34-41, projection.rs:59-78, and repacketizer.rs:72-81 call std::alloc::handle_alloc_error when handle allocation returns null; Box and Vec construction is also infallible and aborting. ffi_guard catches unwinds only, so it cannot translate allocator aborts. Expected: return NULL (or an error status) and write OPUS_ALLOC_FAIL. Actual: the process aborts. Fix: make every allocation in create/init fallible, clean up partial state, commit output parameters only after full initialization, and return the libopus allocation error without unwinding or aborting.

## Fix

<unfixed — raised only>

## Notes
