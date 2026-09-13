//! Fallible heap allocation primitives shared by the codec and C ABI.
//!
//! Constructors in the C ABI must report allocator exhaustion instead of
//! reaching Rust's aborting allocation paths.  This module keeps the actual
//! allocation step explicit so nested codec construction can propagate the
//! same error as the outer handle allocation.

use std::alloc::{self, Layout};
use std::cell::Cell;
use std::ptr::{self, NonNull};

const FAILPOINT_DISABLED: usize = usize::MAX;

thread_local! {
    static FAIL_AFTER: Cell<usize> = const { Cell::new(FAILPOINT_DISABLED) };
}

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

/// Test-only controls exposed through a hidden module so the C API's unit
/// tests can inject failures into the `ropus` dependency as well as its own
/// wrapper allocations.  The failpoint is disabled by default and has no
/// observable effect on normal callers.
#[doc(hidden)]
pub mod test_support {
    use super::{FAIL_AFTER, FAILPOINT_DISABLED};

    /// Allow `allowed_allocations` fallible allocation attempts, then fail the
    /// next attempt on this thread.
    pub fn fail_after(allowed_allocations: usize) {
        FAIL_AFTER.with(|remaining| remaining.set(allowed_allocations));
    }

    /// Disable allocation-failure injection on this thread.
    pub fn clear_failpoint() {
        FAIL_AFTER.with(|remaining| remaining.set(FAILPOINT_DISABLED));
    }
}

/// Allocate zeroed storage without invoking Rust's aborting OOM handler.
pub fn try_alloc_zeroed_layout(layout: Layout) -> Option<NonNull<u8>> {
    if failpoint_trips() {
        return None;
    }
    // SAFETY: `layout` comes from a valid Rust type or a checked layout at the
    // C ABI boundary. A null result is propagated to the caller.
    NonNull::new(unsafe { alloc::alloc_zeroed(layout) })
}

/// Allocate zeroed storage for `T` without constructing a stack temporary.
pub fn try_alloc_zeroed<T>() -> Option<NonNull<T>> {
    try_alloc_zeroed_layout(Layout::new::<T>()).map(|ptr| ptr.cast())
}

/// Move a staged value into fallibly allocated heap storage.
pub fn try_box<T>(value: T) -> Result<Box<T>, ()> {
    let layout = Layout::new::<T>();
    let Some(raw) = try_alloc_zeroed_layout(layout) else {
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

/// Allocate a zero-filled `Vec` with an exact capacity.
pub fn try_vec_with_capacity<T>(capacity: usize) -> Result<Vec<T>, ()> {
    if failpoint_trips() {
        return Err(());
    }
    let mut values = Vec::new();
    values.try_reserve_exact(capacity).map_err(|_| ())?;
    Ok(values)
}

/// Allocate and fill a vector without a second infallible allocation.
pub fn try_vec_with_len<T: Clone>(len: usize, value: T) -> Result<Vec<T>, ()> {
    let mut values = try_vec_with_capacity(len)?;
    values.resize(len, value);
    Ok(values)
}

/// Allocate and copy a slice without a second infallible allocation.
pub fn try_vec_from_slice<T: Clone>(source: &[T]) -> Result<Vec<T>, ()> {
    let mut values = try_vec_with_capacity(source.len())?;
    values.extend_from_slice(source);
    Ok(values)
}

/// Convert UTF-8 bytes with replacement semantics using fallible storage.
pub fn try_string_from_utf8_lossy(source: &[u8]) -> Result<String, ()> {
    // A replacement character takes at most three UTF-8 bytes. Reserving this
    // upper bound lets the subsequent pushes run without another allocation.
    let capacity = source.len().checked_mul(3).ok_or(())?;
    let mut bytes = try_vec_with_capacity(capacity)?;
    for character in String::from_utf8_lossy(source).chars() {
        let mut encoded = [0u8; 4];
        bytes.extend_from_slice(character.encode_utf8(&mut encoded).as_bytes());
    }
    String::from_utf8(bytes).map_err(|_| ())
}

/// Allocate a zeroed `Box<T>` for types whose all-zero byte pattern is valid.
///
/// The caller must only use this for plain-data types where zero is a valid
/// value for every field.
pub fn try_box_zeroed<T>() -> Result<Box<T>, ()> {
    let Some(raw) = try_alloc_zeroed::<T>() else {
        return Err(());
    };
    // SAFETY: the allocation has exactly `T`'s layout and was zeroed. The
    // function's contract requires that zero is a valid value for `T`.
    unsafe { Ok(Box::from_raw(raw.as_ptr())) }
}
