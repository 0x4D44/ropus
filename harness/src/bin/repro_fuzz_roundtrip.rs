#[cfg(no_reference)]
fn main() {
    eprintln!(
        "repro-fuzz-roundtrip: requires the xiph/opus C reference source.\n\
         Run: cargo run -p fetch-assets -- reference"
    );
    std::process::exit(2);
}

#[cfg(not(no_reference))]
#[path = "../bin_inner/repro_fuzz_roundtrip.rs"]
mod inner;

#[cfg(not(no_reference))]
fn main() {
    inner::main();
}
