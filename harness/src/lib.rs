//! ropus-harness lib: hosts the FFI bindings so each bin can `use` them
//! instead of each pulling in a fresh copy via `#[path]`.
//!
//! The bindings module links against the xiph/opus C reference compiled
//! by `build.rs`. When that build step can't find `reference/`, it sets
//! `cfg(no_reference)` and this crate compiles to nothing — each
//! FFI-using binary stubs its `main` behind the same cfg. This keeps
//! `cargo build` at the workspace root succeeding on a fresh clone.

#[cfg(not(no_reference))]
pub mod bindings;
