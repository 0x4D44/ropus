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
//!
//! For `test_opus_encode.c`, the reference source does
//! `#include "../src/opus_private.h"` — which, resolved relative to the file
//! in `reference/tests/`, would pull in the real upstream `opus_private.h`
//! and its transitive closure (arch.h, celt.h, ...). We intercept that by
//! force-including our stubbed `opus_private.h` (`/FI` on MSVC, `-include`
//! elsewhere) before any other header, claiming the header guard so the
//! real include becomes a no-op.
//!
//! `test_opus_encode.c` also calls `regression_test()`, defined upstream in
//! `opus_encode_regressions.c`. That file exercises surround and projection
//! encoding — both out of scope (see the `publish-as-crate` HLD's non-goals).
//! We substitute our own no-op stub (`src/regression_test_stub.c`) so the
//! link resolves without dragging in surround/ambisonics paths. This is a
//! deviation: we lose coverage of the specific historical regressions, but
//! the main test body still runs.

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
    let private_h = include_dir.join("opus_private.h");

    // (source file, library name / main symbol, extra_c_files).
    //
    // One entry per reference test. Each produces its own static lib; the
    // matching Rust file in `tests/` links exactly that one lib into its
    // test binary. Adding a new reference test is three edits:
    //   1. a row here,
    //   2. a new `tests/<name>.rs` that declares the renamed `main` and
    //      wraps it in a `#[test]`,
    //   3. any needed header vendored into `include/`.
    //
    // `extra` files are compiled into the same static lib as the reference
    // test — used for `test_opus_encode.c` to bring in our
    // `regression_test()` stub.
    let tests: &[(&str, &str, &[&str])] = &[
        ("test_opus_padding.c", "test_opus_padding", &[]),
        ("test_opus_decode.c", "test_opus_decode", &[]),
        ("test_opus_api.c", "test_opus_api", &[]),
        (
            "test_opus_encode.c",
            "test_opus_encode",
            &["src/regression_test_stub.c"],
        ),
        ("test_opus_extensions.c", "test_opus_extensions", &[]),
    ];

    for (src_name, stem, extras) in tests {
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

        // test_opus_encode.c and test_opus_extensions.c both do
        // `#include "../src/opus_private.h"` which resolves to the real
        // upstream header; force-include our stub first so its header
        // guard wins.
        if *stem == "test_opus_encode" || *stem == "test_opus_extensions" {
            let compiler = build.get_compiler();
            let flag = if compiler.is_like_msvc() {
                format!("/FI{}", private_h.display())
            } else {
                format!("-include{}", private_h.display())
            };
            build.flag(&flag);
        }

        for extra in *extras {
            build.file(manifest_dir.join(extra));
        }

        build.compile(stem);
        println!("cargo:rerun-if-changed={}", src_path.display());
        for extra in *extras {
            println!(
                "cargo:rerun-if-changed={}",
                manifest_dir.join(extra).display()
            );
        }
    }

    // Rebuild if any vendored header changes.
    println!("cargo:rerun-if-changed=include/opus.h");
    println!("cargo:rerun-if-changed=include/opus_defines.h");
    println!("cargo:rerun-if-changed=include/opus_types.h");
    println!("cargo:rerun-if-changed=include/opus_multistream.h");
    println!("cargo:rerun-if-changed=include/opus_private.h");
    println!("cargo:rerun-if-changed=include/arch.h");
    println!("cargo:rerun-if-changed=build.rs");
}
