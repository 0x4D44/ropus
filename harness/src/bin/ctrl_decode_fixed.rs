//! Thin shell for `ctrl_decode_fixed` — dispatches to the real binary in
//! `bin_inner/` when the C reference is available, otherwise prints the
//! fetch-assets hint. See `harness/build.rs` for the cfg source.

#[cfg(no_reference)]
fn main() {
    eprintln!(
        "ctrl_decode_fixed: requires the xiph/opus C reference source.\n\
         Run: cargo run -p fetch-assets -- reference"
    );
    std::process::exit(2);
}

#[cfg(not(no_reference))]
#[path = "../bin_inner/ctrl_decode_fixed.rs"]
mod inner;

#[cfg(not(no_reference))]
fn main() {
    inner::main();
}
