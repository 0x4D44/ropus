//! ropus-compare: thin entry point.
//!
//! The real CLI lives in `cli.rs` and links against the xiph/opus C
//! reference via `bindings.rs`. When `build.rs` can't find `reference/`
//! it sets `cfg(no_reference)` and this binary stubs out — invoking it
//! prints a fetch-assets hint and exits. This keeps `cargo build` at
//! the workspace root working on a fresh clone.

#[cfg(not(no_reference))]
mod cli;

#[cfg(no_reference)]
fn main() {
    eprintln!(
        "ropus-compare requires the xiph/opus C reference source.\n\
         Run: cargo run -p fetch-assets -- reference\n\
         (or `-- all` to also fetch DNN model weights)"
    );
    std::process::exit(2);
}

#[cfg(not(no_reference))]
fn main() {
    cli::run();
}
