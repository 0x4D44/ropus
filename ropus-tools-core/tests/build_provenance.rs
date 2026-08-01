//! Oracles for the shared CLI build-provenance policy.

#[path = "../src/build_provenance.rs"]
mod build_provenance;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use build_provenance::discover;

fn temp_root(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "ropus-build-provenance-{label}-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).expect("create temporary test root");
    root
}

fn git(cwd: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .current_dir(cwd)
        .env("GIT_AUTHOR_NAME", "ropus provenance test")
        .env("GIT_AUTHOR_EMAIL", "provenance@example.invalid")
        .env("GIT_COMMITTER_NAME", "ropus provenance test")
        .env("GIT_COMMITTER_EMAIL", "provenance@example.invalid")
        .args(args)
        .output()
        .expect("run git");
    assert!(
        output.status.success(),
        "git {:?} failed: {}",
        args,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("git output is UTF-8")
        .trim()
        .to_string()
}

fn init_repo(label: &str) -> (PathBuf, PathBuf) {
    let root = temp_root(label);
    git(&root, &["init", "-q"]);
    let package = root.join("ropusenc");
    std::fs::create_dir_all(&package).expect("create package directory");
    std::fs::write(package.join("marker"), "initial").expect("write package marker");
    git(&root, &["add", "ropusenc/marker"]);
    git(&root, &["commit", "-q", "-m", "initial"]);
    (root, package)
}

#[test]
fn normal_clone_tracks_head_and_same_branch_ref() {
    let (root, package) = init_repo("normal");
    let first = discover(&package, &root);
    assert!(
        first.sha.is_some(),
        "normal clone must expose its commit SHA"
    );
    assert!(
        first.watch_paths.iter().any(|path| path.ends_with("HEAD")),
        "HEAD must be watched: {:?}",
        first.watch_paths
    );
    assert!(
        first
            .watch_paths
            .iter()
            .any(|path| path.to_string_lossy().contains("refs")),
        "symbolic branch target must be watched: {:?}",
        first.watch_paths
    );

    std::fs::write(package.join("marker"), "second").expect("update marker");
    git(&root, &["add", "ropusenc/marker"]);
    git(&root, &["commit", "-q", "-m", "second"]);
    let second = discover(&package, &root);
    assert_ne!(
        first.sha, second.sha,
        "same-branch commits need fresh SHA data"
    );

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn detached_head_still_reports_commit_and_watches_head() {
    let (root, package) = init_repo("detached");
    git(&root, &["checkout", "-q", "--detach"]);
    let discovery = discover(&package, &root);
    assert!(
        discovery.sha.is_some(),
        "detached HEAD still identifies a commit"
    );
    assert!(
        discovery
            .watch_paths
            .iter()
            .any(|path| path.ends_with("HEAD")),
        "detached builds must watch HEAD: {:?}",
        discovery.watch_paths
    );

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn linked_worktree_resolves_common_ref_without_using_root_git_path() {
    let (root, _package) = init_repo("linked-source");
    let worktree = temp_root("linked-worktree").join("checkout");
    git(
        &root,
        &[
            "worktree",
            "add",
            "-q",
            "-b",
            "provenance-linked",
            worktree.to_str().expect("worktree path UTF-8"),
        ],
    );
    let worktree_package = worktree.join("ropusenc");
    let discovery = discover(&worktree_package, &worktree);
    assert!(
        discovery.sha.is_some(),
        "linked worktree must expose its commit SHA"
    );
    assert!(
        discovery
            .watch_paths
            .iter()
            .any(|path| path.to_string_lossy().contains("worktrees")),
        "linked worktree HEAD must use its per-worktree git dir: {:?}",
        discovery.watch_paths
    );
    assert!(
        discovery
            .watch_paths
            .iter()
            .any(|path| path.to_string_lossy().contains("refs")),
        "linked worktree branch ref must use the common git dir: {:?}",
        discovery.watch_paths
    );

    git(
        &root,
        &["worktree", "remove", "-f", worktree.to_str().unwrap()],
    );
    let _ = std::fs::remove_dir_all(root);
    let _ = std::fs::remove_dir_all(worktree.parent().expect("worktree parent"));
}

#[test]
fn vendored_and_packaged_layouts_return_unknown() {
    let consumer = temp_root("vendored");
    git(&consumer, &["init", "-q"]);
    let vendor_root = consumer.join("vendor");
    let package = vendor_root.join("ropusenc");
    std::fs::create_dir_all(&package).expect("create vendored package");
    std::fs::write(package.join("marker"), "vendored").expect("write vendored marker");
    git(&consumer, &["add", "vendor/ropusenc/marker"]);
    git(&consumer, &["commit", "-q", "-m", "consumer"]);

    let vendored = discover(&package, &vendor_root);
    assert_eq!(vendored.sha, None, "consumer repository SHA must not leak");
    assert!(vendored.watch_paths.is_empty());

    let packaged_root = temp_root("packaged");
    let packaged = packaged_root.join("ropusenc");
    std::fs::create_dir_all(&packaged).expect("create packaged package");
    let packaged_discovery = discover(&packaged, &packaged_root);
    assert_eq!(packaged_discovery.sha, None);
    assert!(packaged_discovery.watch_paths.is_empty());

    let _ = std::fs::remove_dir_all(consumer);
    let _ = std::fs::remove_dir_all(packaged_root);
}
