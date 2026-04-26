//! `fetch-assets` — download external build assets into `./reference/`.
//!
//! Two assets are needed to build the harness and conformance suite:
//!
//! 1. **xiph/opus C reference**, pinned to a specific commit — cloned with `git`.
//! 2. **DNN model weights** (FARGAN, LPCNet PLC, PitchDNN, DRED, OSCE, Lossgen),
//!    pinned to a specific sha256 — downloaded with `curl`, sha256-verified,
//!    and extracted with `tar` into `reference/dnn/`.
//!
//! Usage:
//!
//! ```text
//! cargo run -p fetch-assets -- reference   # clone + checkout pinned commit
//! cargo run -p fetch-assets -- weights     # download + verify + extract tarball
//! cargo run -p fetch-assets -- all         # both (reference first, weights second)
//! ```
//!
//! Idempotent: each subcommand short-circuits when its sentinel file is already
//! present. Delete `reference/` or the sentinel marker to force a refetch.
//!
//! Tool dependencies (expected on PATH): `git`, `curl`, `tar`. All three ship
//! with Windows 10 1803+, macOS, and every modern Linux distro.

use std::env;
use std::fs;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use sha2::{Digest, Sha256};

const OPUS_REF_URL: &str = "https://github.com/xiph/opus.git";
const OPUS_REF_COMMIT: &str = "788cc89ce4f2c42025d8c70ec1b4457dc89cd50f";

const WEIGHTS_SHA256: &str = "a5177ec6fb7d15058e99e57029746100121f68e4890b1467d4094aa336b6013e";
const WEIGHTS_URL: &str = "https://media.xiph.org/opus/models/opus_data-\
                           a5177ec6fb7d15058e99e57029746100121f68e4890b1467d4094aa336b6013e.tar.gz";

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("");

    let repo_root = match find_repo_root() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("fetch-assets: {e}");
            return ExitCode::from(2);
        }
    };

    let result = match cmd {
        "reference" => fetch_reference(&repo_root),
        "weights" => fetch_weights(&repo_root),
        "all" => fetch_reference(&repo_root).and_then(|_| fetch_weights(&repo_root)),
        _ => {
            print_usage();
            return ExitCode::from(2);
        }
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("fetch-assets: {e}");
            ExitCode::FAILURE
        }
    }
}

fn print_usage() {
    eprintln!(
        "Usage: cargo run -p fetch-assets -- <reference|weights|all>\n\
         \n\
         Subcommands:\n\
           reference  Clone xiph/opus into ./reference/ at pinned commit {commit}\n\
           weights    Download DNN model weights (~135 MB) into ./reference/dnn/\n\
           all        Both, in order\n",
        commit = &OPUS_REF_COMMIT[..10],
    );
}

fn find_repo_root() -> Result<PathBuf, String> {
    // When run via `cargo run -p fetch-assets`, CARGO_MANIFEST_DIR points at
    // this crate's directory (tools/fetch-assets/). Repo root is two levels up.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .map(Path::to_path_buf)
        .ok_or_else(|| "cannot locate repo root from CARGO_MANIFEST_DIR".to_string())
}

// --- reference (xiph/opus) ---------------------------------------------------

fn fetch_reference(repo_root: &Path) -> Result<(), String> {
    let ref_dir = repo_root.join("reference");
    let sentinel = ref_dir.join("celt").join("bands.c");

    if sentinel.exists() {
        let head = git_head(&ref_dir).unwrap_or_else(|_| "unknown".to_string());
        if head == OPUS_REF_COMMIT {
            println!("[reference] already present at pinned commit, skipping");
        } else {
            println!(
                "[reference] already present (HEAD={}), expected {}.\n\
                 [reference] not touching it; delete reference/ and rerun to force a clean clone.",
                &head[..head.len().min(10)],
                &OPUS_REF_COMMIT[..10],
            );
        }
        return Ok(());
    }

    require_tool("git")?;

    println!(
        "[reference] cloning {OPUS_REF_URL} into {}...",
        ref_dir.display()
    );
    run(Command::new("git")
        .arg("clone")
        .arg("--no-checkout")
        .arg(OPUS_REF_URL)
        .arg(&ref_dir))?;

    println!("[reference] checking out {OPUS_REF_COMMIT}...");
    run(Command::new("git")
        .arg("-C")
        .arg(&ref_dir)
        .arg("checkout")
        .arg(OPUS_REF_COMMIT))?;

    println!("[reference] OK (commit {})", &OPUS_REF_COMMIT[..10]);
    Ok(())
}

fn git_head(dir: &Path) -> Result<String, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(dir)
        .arg("rev-parse")
        .arg("HEAD")
        .output()
        .map_err(|e| format!("git rev-parse failed: {e}"))?;
    if !output.status.success() {
        return Err("git rev-parse returned non-zero".to_string());
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

// --- weights (DNN model data) ------------------------------------------------

fn fetch_weights(repo_root: &Path) -> Result<(), String> {
    let ref_dir = repo_root.join("reference");
    if !ref_dir.join("celt").join("bands.c").exists() {
        return Err(
            "reference/ not populated; run `cargo run -p fetch-assets -- reference` first \
             (the weights tarball extracts into reference/dnn/)"
                .to_string(),
        );
    }

    let marker = ref_dir.join(format!(".opus_data-{WEIGHTS_SHA256}.installed"));
    if marker.exists() {
        println!(
            "[weights] already installed (marker {} present)",
            marker.display()
        );
        return Ok(());
    }

    require_tool("curl")?;
    require_tool("tar")?;

    let cache_dir = ref_dir.join(".cache");
    fs::create_dir_all(&cache_dir)
        .map_err(|e| format!("failed to create cache dir {}: {e}", cache_dir.display()))?;
    let tarball = cache_dir.join(format!("opus_data-{WEIGHTS_SHA256}.tar.gz"));

    let need_download = match verify_sha256(&tarball, WEIGHTS_SHA256) {
        Ok(true) => {
            println!("[weights] reusing cached tarball {}", tarball.display());
            false
        }
        _ => true,
    };

    if need_download {
        println!("[weights] downloading {WEIGHTS_URL}");
        println!("[weights] (~135 MB; curl will show progress below)");
        run(Command::new("curl")
            .arg("--fail")
            .arg("--location")
            .arg("--retry")
            .arg("3")
            .arg("--output")
            .arg(&tarball)
            .arg(WEIGHTS_URL))?;

        println!("[weights] verifying sha256...");
        match verify_sha256(&tarball, WEIGHTS_SHA256) {
            Ok(true) => {}
            Ok(false) => {
                let _ = fs::remove_file(&tarball);
                return Err(format!(
                    "sha256 mismatch for downloaded tarball (expected {WEIGHTS_SHA256}); \
                     deleted and aborting"
                ));
            }
            Err(e) => return Err(format!("sha256 verification failed: {e}")),
        }
    }

    println!("[weights] extracting into {}/...", ref_dir.display());
    run(Command::new("tar")
        .arg("-xzf")
        .arg(&tarball)
        .arg("-C")
        .arg(&ref_dir))?;

    // Touch marker so subsequent runs short-circuit without re-hashing the tarball.
    fs::write(&marker, WEIGHTS_SHA256)
        .map_err(|e| format!("failed to write marker {}: {e}", marker.display()))?;

    println!("[weights] OK");
    Ok(())
}

// --- helpers -----------------------------------------------------------------

fn require_tool(name: &str) -> Result<(), String> {
    // `<tool> --version` is the universal probe for presence; all three tools
    // we call (git, curl, tar) support it.
    Command::new(name)
        .arg("--version")
        .output()
        .map(|_| ())
        .map_err(|e| {
            format!("required tool `{name}` not found on PATH ({e}); install it and retry")
        })
}

fn run(cmd: &mut Command) -> Result<(), String> {
    let status = cmd
        .status()
        .map_err(|e| format!("failed to launch `{}`: {e}", fmt_cmd(cmd)))?;
    if !status.success() {
        return Err(format!("command `{}` exited with {status}", fmt_cmd(cmd)));
    }
    Ok(())
}

fn fmt_cmd(cmd: &Command) -> String {
    let program = cmd.get_program().to_string_lossy().into_owned();
    let args: Vec<String> = cmd
        .get_args()
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    format!("{program} {}", args.join(" "))
}

fn verify_sha256(path: &Path, expected_hex: &str) -> io::Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let file = fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 256 * 1024];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let got = hasher.finalize();
    let mut got_hex = String::with_capacity(got.len() * 2);
    for b in got.iter() {
        use std::fmt::Write;
        let _ = write!(got_hex, "{b:02x}");
    }
    Ok(got_hex.eq_ignore_ascii_case(expected_hex))
}
