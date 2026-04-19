#[cfg(no_reference)]
fn main() {
    eprintln!(
        "trace_silk_wb: requires the xiph/opus C reference source.\n\
         Run: cargo run -p fetch-assets -- reference"
    );
    std::process::exit(2);
}

#[cfg(not(no_reference))]
#[path = "../bin_inner/trace_silk_wb.rs"]
mod inner;

#[cfg(not(no_reference))]
fn main() {
    inner::main();
}
