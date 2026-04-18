//! Compile the C CTL varargs shim that dispatches into typed Rust entry points.
//!
//! The Opus CTL interface is C variadic. Rather than enable the unstable
//! `c_variadic` feature, a small C shim uses `va_arg` to unpack the argument
//! and calls a typed Rust wrapper per request kind.

fn main() {
    println!("cargo:rerun-if-changed=src/ctl_shim.c");
    cc::Build::new()
        .file("src/ctl_shim.c")
        .warnings(true)
        .compile("mdopus_ctl_shim");
}
