#[cfg(no_reference)]
fn main() {
    eprintln!(
        "ctrl_decode_float: requires the xiph/opus C reference + DNN weights.\n\
         Run: cargo run -p fetch-assets -- all"
    );
    std::process::exit(2);
}

#[cfg(not(no_reference))]
#[path = "../bin_inner/ctrl_decode_float.rs"]
mod inner;

#[cfg(not(no_reference))]
fn main() {
    inner::main();
}
