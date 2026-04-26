#[cfg(no_reference)]
fn main() {
    eprintln!(
        "projection_roundtrip: requires the xiph/opus C reference source.\n\
         Run: cargo run -p fetch-assets -- reference"
    );
    std::process::exit(2);
}

#[cfg(not(no_reference))]
#[path = "../bin_inner/projection_roundtrip.rs"]
mod inner;

#[cfg(not(no_reference))]
fn main() {
    inner::main();
}
