#[cfg(no_reference)]
fn main() {
    eprintln!(
        "ropus-interframe: requires the xiph/opus C reference source.\n\
         Run: cargo run -p fetch-assets -- reference"
    );
    std::process::exit(2);
}

#[cfg(not(no_reference))]
#[path = "../bin_inner/interframe.rs"]
mod inner;

#[cfg(not(no_reference))]
fn main() {
    inner::main();
}
