// Clippy allows for C-to-Rust codec port patterns.
// This is a bit-exact port of xiph/opus; these patterns are intentional.
#![allow(
    // C reference uses exact float literals (approximation coefficients, fixed-point
    // conversion inputs) that must be preserved for bit-exactness.
    clippy::excessive_precision,
    clippy::approx_constant,
    // C-style explicit casts kept for clarity and 1:1 correspondence with reference.
    clippy::unnecessary_cast,
    // Unrolled loops and table indexing use `+ 0`, `<< 0`, `0 * stride` for pattern
    // clarity with subsequent iterations.
    clippy::identity_op,
    clippy::erasing_op,
    // C codec functions have many parameters; matching the reference API exactly.
    clippy::too_many_arguments,
    // C-style operator precedence preserved for readability against reference.
    clippy::precedence,
    // C-style range checks and control flow preserved for 1:1 correspondence.
    clippy::manual_range_contains,
    clippy::collapsible_if,
    clippy::collapsible_else_if,
    // C-style manual slice copies with computed indices.
    clippy::manual_memcpy,
    // Port uses Result<_, ()> for C-style error returns.
    clippy::result_unit_err,
    // C-style indexed loops preserved for 1:1 reference correspondence.
    clippy::needless_range_loop,
    // C-style min/max chains preserved for bit-exactness verification.
    clippy::manual_clamp,
    // C-style variable declarations (declare then assign in branches).
    clippy::needless_late_init,
    // C-style assign patterns (x = x + y) preserved for reference readability.
    clippy::assign_op_pattern,
    // C-style boolean and comparison patterns.
    clippy::int_plus_one,
    clippy::nonminimal_bool,
    // C-style let-then-return for clarity in complex expressions.
    clippy::let_and_return,
    // Codec types have complex initialization; new() is preferred over Default.
    clippy::new_without_default,
    // C-style manual loop counters preserved for reference correspondence.
    clippy::explicit_counter_loop,
)]

pub mod celt;
pub mod dnn;
pub mod opus;
pub mod silk;
pub mod types;

// Safe-indexing hot-path macros (historical shorthand; plain `[]` / `&mut []`).
macro_rules! uc {
    ($slice:expr, $idx:expr) => {
        $slice[$idx]
    };
}

macro_rules! uc_set {
    ($slice:expr, $idx:expr, $val:expr) => {
        $slice[$idx] = $val
    };
}

macro_rules! uc_mut {
    ($slice:expr, $idx:expr) => {
        &mut $slice[$idx]
    };
}

pub(crate) use uc;
pub(crate) use uc_mut;
pub(crate) use uc_set;

#[cfg(test)]
mod coverage_tests;
#[cfg(test)]
mod property_tests_packet;
#[cfg(test)]
mod property_tests_codec;
