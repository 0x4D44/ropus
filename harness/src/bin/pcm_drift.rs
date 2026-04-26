#[cfg(no_reference)]
fn main() {
    eprintln!(
        "pcm_drift: requires the xiph/opus C reference source.\n\
         Run: cargo run -p fetch-assets -- reference"
    );
    std::process::exit(2);
}

#[cfg(not(no_reference))]
#[path = "../bin_inner/pcm_drift.rs"]
mod inner;

#[cfg(not(no_reference))]
fn main() {
    inner::main();
}
