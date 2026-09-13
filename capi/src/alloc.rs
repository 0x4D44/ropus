//! Fallible allocations used by the C ABI boundary.
//!
//! Rust's infallible `Box::new`, `Vec::with_capacity`, and
//! `std::alloc::handle_alloc_error` paths abort on allocator exhaustion.  A C
//! API must instead translate those failures into `OPUS_ALLOC_FAIL`, so every
//! constructor stages its heap state through these helpers before publishing
//! a handle or output parameter.

use std::alloc::{self, Layout};
use std::ptr::NonNull;

use ropus::allocation;

#[cfg(test)]
pub(crate) fn fail_after(allowed_allocations: usize) {
    allocation::test_support::fail_after(allowed_allocations);
}

#[cfg(test)]
pub(crate) fn clear_failpoint() {
    allocation::test_support::clear_failpoint();
}

pub(crate) fn try_alloc_zeroed_layout(layout: Layout) -> Option<NonNull<u8>> {
    allocation::try_alloc_zeroed_layout(layout)
}

pub(crate) fn try_alloc_zeroed<T>() -> Option<NonNull<T>> {
    allocation::try_alloc_zeroed::<T>()
}

pub(crate) fn try_box<T>(value: T) -> Result<Box<T>, ()> {
    allocation::try_box(value)
}

pub(crate) unsafe fn dealloc_layout(ptr: *mut u8, layout: Layout) {
    if !ptr.is_null() {
        // SAFETY: `ptr` was allocated with this exact layout by a helper in
        // this module, and the caller has exclusive ownership of it.
        unsafe { alloc::dealloc(ptr, layout) };
    }
}

pub(crate) fn try_vec_with_capacity<T>(capacity: usize) -> Result<Vec<T>, ()> {
    allocation::try_vec_with_capacity(capacity)
}

pub(crate) fn try_vec_with_len<T: Clone>(len: usize, value: T) -> Result<Vec<T>, ()> {
    allocation::try_vec_with_len(len, value)
}
