//! Build script for ropus.
//!
//! Produces `$OUT_DIR/weights_blob.bin` — a runtime weight blob ready to
//! feed into `OpusDecoder::set_dnn_blob` — by compiling and running a
//! tiny C driver (`build/gen_weights_blob.c`) that pulls in the
//! compile-time `*_data.c` arrays from xiph's weights tarball.
//!
//! If the xiph reference sources aren't on disk (e.g. a fresh clone
//! without `cargo run -p fetch-assets -- weights`), the blob is written
//! as a zero-byte file; the Rust side treats this as "no embedded
//! weights" and callers must provide a blob via `set_dnn_blob`
//! themselves. This keeps `cargo build -p ropus` working from a bare
//! checkout without forcing a weights download.

use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let out_dir: PathBuf = std::env::var_os("OUT_DIR")
        .expect("OUT_DIR must be set by cargo")
        .into();
    let blob_path = out_dir.join("weights_blob.bin");

    let manifest_dir: PathBuf = std::env::var_os("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR must be set by cargo")
        .into();
    let repo_root = manifest_dir
        .parent()
        .expect("ropus crate must sit one level under the repo root")
        .to_path_buf();
    let ref_dir = repo_root.join("reference");
    let dnn_dir = ref_dir.join("dnn");

    // Declare the data files + the custom driver as rerun triggers,
    // so a weights refetch re-embeds without needing `cargo clean`.
    for rel in [
        "dnn/pitchdnn_data.c",
        "dnn/fargan_data.c",
        "dnn/plc_data.c",
        "dnn/dred_rdovae_enc_data.c",
        "dnn/dred_rdovae_dec_data.c",
        "dnn/nnet.h",
        "dnn/pitchdnn_data.h",
        "dnn/fargan_data.h",
        "dnn/plc_data.h",
        "dnn/dred_rdovae_enc_data.h",
        "dnn/dred_rdovae_dec_data.h",
    ] {
        println!("cargo:rerun-if-changed={}", ref_dir.join(rel).display());
    }
    let driver = manifest_dir.join("build").join("gen_weights_blob.c");
    println!("cargo:rerun-if-changed={}", driver.display());

    if !(dnn_dir.join("pitchdnn_data.c").exists()
        && dnn_dir.join("fargan_data.c").exists()
        && dnn_dir.join("plc_data.c").exists()
        && driver.exists())
    {
        // No reference sources available — emit empty blob so
        // `include_bytes!` in the Rust side still works.
        println!(
            "cargo:warning=ropus: reference DNN data files not found under {}, \
             compile-time weights blob left empty. Run `cargo run -p fetch-assets -- \
             weights` to embed neural PLC weights out-of-the-box.",
            dnn_dir.display()
        );
        write_empty_blob(&blob_path);
        return;
    }

    // DRED's RDOVAE data files ship from the same xiph tarball but are
    // optional — older fetches may lack them. Enable their inclusion in
    // the blob only when both files are on disk so a partial tree still
    // builds (without DRED weights embedded).
    let have_dred = dnn_dir.join("dred_rdovae_enc_data.c").exists()
        && dnn_dir.join("dred_rdovae_dec_data.c").exists();

    if let Err(e) = try_build_blob(
        &manifest_dir,
        &ref_dir,
        &driver,
        &blob_path,
        &out_dir,
        have_dred,
    ) {
        println!(
            "cargo:warning=ropus: failed to build embedded weights blob ({e}); \
             leaving blob empty. Neural PLC will require an explicit \
             `OpusDecoder::set_dnn_blob` call."
        );
        write_empty_blob(&blob_path);
    }
}

fn write_empty_blob(blob_path: &Path) {
    fs::write(blob_path, []).expect("write empty weights_blob.bin");
}

/// Attempt to compile + run the blob generator. On any error (missing C
/// toolchain, link failure, runtime failure) we bail out and the caller
/// writes an empty blob — the crate stays buildable without weights.
fn try_build_blob(
    manifest_dir: &Path,
    ref_dir: &Path,
    driver: &Path,
    blob_path: &Path,
    out_dir: &Path,
    have_dred: bool,
) -> Result<(), String> {
    // Locate the host C toolchain via `cc`. Using a fresh `cc::Build`
    // per this script (rather than `cargo:rustc-link-lib`) keeps this
    // step contained — we link an exe, not a staticlib, and we don't
    // want this object to end up in the final rlib.
    let mut builder = cc::Build::new();
    builder
        .cargo_metadata(false) // don't spam cargo with -L/-l for a one-off exe
        .warnings(false)
        .include(ref_dir.join("include"))
        .include(ref_dir.join("celt"))
        .include(ref_dir.join("dnn"))
        .define("DUMP_BINARY_WEIGHTS", None);
    if have_dred {
        builder.define("ROPUS_HAVE_DRED_WEIGHTS", None);
    }
    let tool = builder
        .try_get_compiler()
        .map_err(|e| format!("cc::Build::try_get_compiler failed: {e}"))?;

    let exe_stem = "ropus_gen_weights_blob";
    let exe_name = if cfg!(windows) {
        format!("{exe_stem}.exe")
    } else {
        exe_stem.to_string()
    };
    let exe_path = out_dir.join(&exe_name);
    let obj_path = out_dir.join(if tool.is_like_msvc() {
        "gen_weights_blob.obj"
    } else {
        "gen_weights_blob.o"
    });

    // Compile the driver to an object file.
    let mut compile = tool.to_command();
    if tool.is_like_msvc() {
        compile
            .arg("/nologo")
            .arg("/c")
            .arg(driver)
            .arg(format!("/Fo{}", obj_path.display()));
    } else {
        compile
            .arg("-c")
            .arg(driver)
            .arg("-o")
            .arg(&obj_path);
    }
    let status = compile
        .status()
        .map_err(|e| format!("failed to launch C compiler: {e}"))?;
    if !status.success() {
        return Err(format!("C compile of gen_weights_blob.c exited {status}"));
    }

    // Link the object into an executable. Use the same underlying tool
    // so we get MSVC's /link semantics on Windows and ld on Unix.
    let mut link = tool.to_command();
    if tool.is_like_msvc() {
        link.arg("/nologo")
            .arg(&obj_path)
            .arg(format!("/Fe{}", exe_path.display()));
    } else {
        link.arg(&obj_path).arg("-o").arg(&exe_path);
    }
    let status = link
        .status()
        .map_err(|e| format!("failed to launch C linker: {e}"))?;
    if !status.success() {
        return Err(format!("link of gen_weights_blob exited {status}"));
    }

    // Run the generator, targeting our OUT_DIR blob path.
    let status = std::process::Command::new(&exe_path)
        .arg(blob_path)
        .status()
        .map_err(|e| format!("failed to launch {}: {e}", exe_path.display()))?;
    if !status.success() {
        return Err(format!("{} exited {status}", exe_path.display()));
    }

    // Sanity-check the blob grew.
    let meta = fs::metadata(blob_path)
        .map_err(|e| format!("stat {}: {e}", blob_path.display()))?;
    if meta.len() == 0 {
        return Err(format!(
            "{} was written as an empty file — the generator couldn't locate any arrays",
            blob_path.display()
        ));
    }
    // Informational only — goes to stderr / build logs, not cargo:warning=,
    // so it doesn't surface as a top-level warning on every build.
    eprintln!(
        "ropus: embedded DNN weight blob ({} bytes) — neural PLC activates out-of-the-box.",
        meta.len()
    );
    let _ = (manifest_dir,); // silence unused-import warning if this goes away later

    Ok(())
}
