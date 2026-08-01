//! Fallible allocations used by the C ABI boundary.
//!
//! Rust's infallible `Box::new`, `Vec::with_capacity`, and
//! `std::alloc::handle_alloc_error` paths abort on allocator exhaustion.  A C
//! API must instead translate those failures into `OPUS_ALLOC_FAIL`, so every
//! constructor stages its heap state through these helpers before publishing
//! a handle or output parameter.

use std::alloc::{self, Layout};
use std::ptr::{self, NonNull};

#[cfg(test)]
use std::cell::Cell;

#[cfg(test)]
const FAILPOINT_DISABLED: usize = usize::MAX;

#[cfg(test)]
thread_local! {
    static FAIL_AFTER: Cell<usize> = const { Cell::new(FAILPOINT_DISABLED) };
}

#[cfg(test)]
pub(crate) fn fail_after(allowed_allocations: usize) {
    FAIL_AFTER.with(|remaining| remaining.set(allowed_allocations));
}

#[cfg(test)]
pub(crate) fn clear_failpoint() {
    FAIL_AFTER.with(|remaining| remaining.set(FAILPOINT_DISABLED));
}

#[cfg(test)]
#[inline]
fn failpoint_trips() -> bool {
    FAIL_AFTER.with(|remaining| {
        let current = remaining.get();
        if current == FAILPOINT_DISABLED {
            return false;
        }
        if current == 0 {
            remaining.set(FAILPOINT_DISABLED);
            return true;
        }
        remaining.set(current - 1);
        false
    })
}

#[cfg(not(test))]
#[inline]
fn failpoint_trips() -> bool {
    false
}

pub(crate) fn try_alloc_zeroed_layout(layout: Layout) -> Option<NonNull<u8>> {
    if failpoint_trips() {
        return None;
    }
    // SAFETY: `layout` was built by the caller from a valid Rust type or a
    // checked size/alignment pair.  A null result is propagated to the C ABI.
    NonNull::new(unsafe { alloc::alloc_zeroed(layout) })
}

pub(crate) fn try_alloc_zeroed<T>() -> Option<NonNull<T>> {
    try_alloc_zeroed_layout(Layout::new::<T>()).map(|ptr| ptr.cast())
}

pub(crate) fn try_box<T>(value: T) -> Result<Box<T>, ()> {
    let layout = Layout::new::<T>();
    let Some(raw) = try_alloc_zeroed_layout(layout) else {
        // Dropping the staged value releases any allocations it owns.  The
        // caller has not published a handle yet, so no C-visible state exists.
        drop(value);
        return Err(());
    };
    let ptr = raw.as_ptr() as *mut T;
    // SAFETY: `ptr` is an exclusive allocation with the exact layout for `T`.
    // Writing the value makes it valid for `Box::from_raw`.
    unsafe {
        ptr::write(ptr, value);
        Ok(Box::from_raw(ptr))
    }
}

pub(crate) unsafe fn dealloc_layout(ptr: *mut u8, layout: Layout) {
    if !ptr.is_null() {
        // SAFETY: `ptr` was allocated with this exact layout by a helper in
        // this module, and the caller has exclusive ownership of it.
        unsafe { alloc::dealloc(ptr, layout) };
    }
}

pub(crate) fn try_vec_with_capacity<T>(capacity: usize) -> Result<Vec<T>, ()> {
    if failpoint_trips() {
        return Err(());
    }
    let mut values = Vec::new();
    values.try_reserve_exact(capacity).map_err(|_| ())?;
    Ok(values)
}
