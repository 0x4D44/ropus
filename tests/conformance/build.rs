//! Compile each conformance `.c` test as a static lib with `main` renamed.
//!
//! Approach: we use cc's `-Dmain=<test>_main` trick to rename each test
//! binary's `main` so it can be called from Rust. Each renamed entry point
//! becomes one `#[test]` in its own cargo-integration-test file in `tests/`
//! (cargo compiles each such file into a separate test binary). That
//! one-binary-per-test-file isolation is *load-bearing*: `test_opus_common.h`
//! defines a common `regression_test(void)` declaration that every conformance
//! `.c` file then provides its own definition for — packing more than one into
//! the same linker image would surface as LNK2005 (duplicate symbol).
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

    // (source file, library name / main symbol).
    //
    // One entry per reference test. Each produces its own static lib; the
    // matching Rust file in `tests/` links exactly that one lib into its
    // test binary. Adding a new reference test is three edits:
    //   1. a row here,
    //   2. a new `tests/<name>.rs` that declares the renamed `main` and
    //      wraps it in a `#[test]`,
    //   3. any needed header vendored into `include/`.
    let tests: &[(&str, &str)] = &[
        ("test_opus_padding.c", "test_opus_padding"),
        ("test_opus_decode.c", "test_opus_decode"),
    ];

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
