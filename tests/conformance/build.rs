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
//! `opus_encode_regressions.c`. With Pieces A+B's capi surface now wired,
//! we compile that file verbatim into the encode test's static lib so the
//! 11 historical crash repros (7 unconditional + 3 `!DISABLE_FLOAT_API` +
//! `projection_overflow`) run against our codec. The 5 `ENABLE_QEXT` /
//! `ENABLE_DRED`-guarded repros compile out — both macros stay undefined
//! for the whole conformance build, matching the rest of the suite.
//! `opus_encode_regressions.c` also does `#include "../src/opus_private.h"`,
//! so the same `/FI` / `-include` force-include trick used for
//! `test_opus_encode.c` applies when it compiles as an extra.

use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .expect("tests/conformance has a parent")
        .parent()
        .expect("tests has a parent (repo root)");
    let ref_tests = repo_root.join("reference").join("tests");
    let ref_src = repo_root.join("reference").join("src");

    let include_dir = manifest_dir.join("include");
    let private_h = include_dir.join("opus_private.h");

    // (source dir, source file, library name / main symbol, extra_c_files).
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
    // `regression_test()` stub, and for `opus_demo` to bring in our
    // lossgen + DRED stubs.
    //
    // `source_dir` selects between `reference/tests/` (the default) and
    // `reference/src/` (for opus_demo.c and opus_compare.c, which are
    // binaries rather than tests but built the same way).
    enum SrcDir {
        Tests,
        Src,
    }
    // Extras are resolved relative to `manifest_dir` if they start with
    // "src/" (our own stubs) and relative to `repo_root` otherwise (for
    // reference upstream sources like `reference/src/mapping_matrix.c`).
    let tests: &[(SrcDir, &str, &str, &[&str])] = &[
        (
            SrcDir::Tests,
            "test_opus_padding.c",
            "test_opus_padding",
            &[],
        ),
        (SrcDir::Tests, "test_opus_decode.c", "test_opus_decode", &[]),
        (SrcDir::Tests, "test_opus_api.c", "test_opus_api", &[]),
        (
            SrcDir::Tests,
            "test_opus_encode.c",
            "test_opus_encode",
            &["reference/tests/opus_encode_regressions.c"],
        ),
        (
            SrcDir::Tests,
            "test_opus_extensions.c",
            "test_opus_extensions",
            &[],
        ),
        // Piece A — IETF vectors via opus_demo + opus_compare.
        //
        // `opus_demo.c` pulls in DRED symbols unconditionally; we replace
        // them with no-op stubs (see `src/dred_stub.c`) since our decode-only
        // driver on valid RFC 8251 vectors never enters those code paths.
        // All lossgen references in opus_demo.c are gated by
        // `#ifdef ENABLE_LOSSGEN`, which we don't define — so no lossgen
        // symbols are ever link-referenced and no lossgen stub is needed.
        // `opus_compare.c` is a pure PCM comparator with no libopus
        // dependency — it links clean.
        (
            SrcDir::Src,
            "opus_demo.c",
            "opus_demo",
            &["src/dred_stub.c"],
        ),
        (SrcDir::Src, "opus_compare.c", "opus_compare", &[]),
        // Piece B — projection conformance.
        //
        // `test_opus_projection.c` exercises the ambisonics matrix math
        // against the reference C implementation (compiled in via
        // `reference/src/mapping_matrix.c` as an extra) plus the public
        // `opus_projection_*` encode/decode path against our capi.
        //
        // `mapping_matrix.c` is self-contained: depends only on
        // `opus_types.h`, `opus_projection.h`, `float_cast.h`, and a few
        // macros from `arch.h` (`align`, `SAT16`, `PSHR32`, `INT24TORES`).
        // Our vendored `include/arch.h` supplies those in fixed-point,
        // non-ENABLE_RES24 mode.
        //
        // The test also does `#include "../src/mapping_matrix.h"` — which
        // resolves relative to the source file at `reference/tests/`, i.e.
        // directly to `reference/src/mapping_matrix.h`. No include-path
        // plumbing needed for that.
        (
            SrcDir::Tests,
            "test_opus_projection.c",
            "test_opus_projection",
            &["reference/src/mapping_matrix.c"],
        ),
    ];

    for (src_dir, src_name, stem, extras) in tests {
        let src_root = match src_dir {
            SrcDir::Tests => &ref_tests,
            SrcDir::Src => &ref_src,
        };
        let src_path = src_root.join(src_name);
        if !src_path.exists() {
            panic!(
                "Reference source not found: {}. \
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
            // `FIXED_POINT` is load-bearing: with it off, the reference
            // headers typedef `opus_res` to `float`, which would mismatch
            // our capi's `opus_int32*` 24-bit pcm pointer type silently at
            // the C call site (C doesn't type-check through the function
            // pointer). Apply uniformly for hygiene even to tests that
            // don't call `opus_res*` functions today.
            .define("FIXED_POINT", "1")
            .warnings(false);

        // opus_demo.c does `#include "debug.h"` (the SILK timer/debug
        // header, which is a no-op when SILK_DEBUG and SILK_TIC_TOC are
        // 0). Add `reference/silk/` to the include path so that resolves.
        // This is the HLD's preferred "include-path-first" strategy.
        if *stem == "opus_demo" {
            build.include(repo_root.join("reference").join("silk"));
        }

        // test_opus_projection.c does `#include "float_cast.h"` purely out
        // of convention — neither the test itself nor `mapping_matrix.c`'s
        // fixed-point paths call into it. Rather than add `reference/celt/`
        // to the include path (which pulls in SSE intrinsics via
        // `<intrin.h>` on MSVC and blows up with `cc`'s default command
        // line), we vendor a minimal `float_cast.h` stub in `include/`.
        //
        // We also `DISABLE_FLOAT_API` so `mapping_matrix.c` compiles out
        // the float API branch (which uses `RES2FLOAT` — not needed for the
        // i16 test paths we exercise). The test itself guards its float
        // assertions with the same macro plus `!FIXED_POINT`.
        //
        // We do NOT include `reference/src/` for the projection test:
        // `mapping_matrix.c` does `#include "opus_private.h"` which we want
        // to resolve to our stub (already on the include path), not to the
        // real `reference/src/opus_private.h` that drags in `celt.h`.
        if *stem == "test_opus_projection" {
            build.define("DISABLE_FLOAT_API", "1");
        }

        // test_opus_encode.c and test_opus_extensions.c both do
        // `#include "../src/opus_private.h"` which resolves to the real
        // upstream header; force-include our stub first so its header
        // guard wins. opus_demo.c does `#include "opus_private.h"` via
        // the standard include path — our vendored stub in `include/`
        // already wins there by virtue of being on the include path
        // before `reference/src/`, but we still force-include for belt-
        // and-braces parity.
        //
        // `test_opus_projection` is also in the list: the extras-compiled
        // `reference/src/mapping_matrix.c` does `#include "opus_private.h"`
        // which the compiler's current-directory rule would otherwise
        // resolve to the real `reference/src/opus_private.h` (dragging in
        // `celt.h`). Force-including our stub first claims the guard and
        // defuses the transitive pull.
        let needs_private_h = matches!(
            *stem,
            "test_opus_encode" | "test_opus_extensions" | "opus_demo" | "test_opus_projection"
        );
        if needs_private_h {
            let compiler = build.get_compiler();
            let flag = if compiler.is_like_msvc() {
                format!("/FI{}", private_h.display())
            } else {
                format!("-include{}", private_h.display())
            };
            build.flag(&flag);
        }

        // Resolve extras: paths starting with "src/" live in the conformance
        // crate (our own stubs); everything else is relative to the repo
        // root (reference upstream sources like `reference/src/mapping_matrix.c`).
        let resolve_extra = |e: &str| -> PathBuf {
            if e.starts_with("src/") {
                manifest_dir.join(e)
            } else {
                repo_root.join(e)
            }
        };
        for extra in *extras {
            build.file(resolve_extra(extra));
        }

        build.compile(stem);
        println!("cargo:rerun-if-changed={}", src_path.display());
        for extra in *extras {
            println!("cargo:rerun-if-changed={}", resolve_extra(extra).display());
        }
    }

    // Rebuild if any vendored header changes.
    println!("cargo:rerun-if-changed=include/opus.h");
    println!("cargo:rerun-if-changed=include/opus_defines.h");
    println!("cargo:rerun-if-changed=include/opus_types.h");
    println!("cargo:rerun-if-changed=include/opus_multistream.h");
    println!("cargo:rerun-if-changed=include/opus_private.h");
    println!("cargo:rerun-if-changed=include/opus_projection.h");
    println!("cargo:rerun-if-changed=include/mapping_matrix.h");
    println!("cargo:rerun-if-changed=include/os_support.h");
    println!("cargo:rerun-if-changed=include/float_cast.h");
    println!("cargo:rerun-if-changed=include/mathops.h");
    println!("cargo:rerun-if-changed=include/arch.h");
    println!("cargo:rerun-if-changed=build.rs");
}
