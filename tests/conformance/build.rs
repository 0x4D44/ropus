//! Compile each conformance `.c` test as a static lib with `main` renamed.
//!
//! Approach: we use cc's `-Dmain=<test>_main` trick to rename each test
//! binary's `main` so it can be called from Rust. Each renamed entry point
//! becomes one `#[test]` in `tests/run.rs`. This keeps everything cargo-native
//! — per-test pass/fail in the normal test output, no spawned processes.
//!
//! Rename note: the reference `.c` sources remain strictly unmodified. The
//! `-Dmain=<name>` define substitutes the `main` token at the preprocessor
//! level, satisfying the HLD's "zero modifications" constraint.

use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .expect("tests/conformance has a parent")
        .parent()
        .expect("tests has a parent (repo root)");
    let ref_tests = repo_root.join("reference").join("tests");

    let include_dir = manifest_dir.join("include");

    // (source file, library name / main symbol)
    //
    // phase 1: only test_opus_padding.c is wired in. Adding more tests means
    // (a) extending this list and (b) adding a matching `#[test]` in run.rs.
    let tests: &[(&str, &str)] = &[("test_opus_padding.c", "test_opus_padding")];

    for (src_name, stem) in tests {
        let src_path = ref_tests.join(src_name);
        if !src_path.exists() {
            panic!(
                "Reference test source not found: {}. \
                 Clone https://github.com/xiph/opus into reference/.",
                src_path.display()
            );
        }

        let mut build = cc::Build::new();
        build
            .file(&src_path)
            .include(&include_dir)
            .include(&ref_tests) // for test_opus_common.h
            .define("main", &*format!("{stem}_main"))
            .warnings(false);

        build.compile(stem);
        println!("cargo:rerun-if-changed={}", src_path.display());
    }

    // Rebuild if any vendored header changes.
    println!("cargo:rerun-if-changed=include/opus.h");
    println!("cargo:rerun-if-changed=include/opus_defines.h");
    println!("cargo:rerun-if-changed=include/opus_types.h");
    println!("cargo:rerun-if-changed=include/opus_multistream.h");
    println!("cargo:rerun-if-changed=build.rs");
}
