#[cfg(no_reference)]
fn main() {
    eprintln!(
        "replay_fuzz_decode: requires the xiph/opus C reference source.\n\
         Run: cargo run -p fetch-assets -- reference"
    );
    std::process::exit(2);
}

#[cfg(not(no_reference))]
#[path = "../bin_inner/replay_fuzz_decode.rs"]
mod inner;

#[cfg(not(no_reference))]
fn main() {
    inner::main();
}
