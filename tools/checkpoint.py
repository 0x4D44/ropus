"""Safe checkpoint commits for the autonomous debugging loops."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Sequence
from pathlib import Path, PurePath


# The debugging agents are instructed to change the Rust codec only.  Keep this
# list explicit so a checkpoint cannot sweep the repository or capture logs,
# fixtures, or another agent's work.
CHECKPOINT_PATHS: tuple[str, ...] = ("ropus/src",)
_GIT_CONFIG = (
    "-c",
    "user.name=0x4D44",
    "-c",
    "user.email=martingdavidson@gmail.com",
)


def _run_git(
    repo_root: Path,
    args: Sequence[str],
    action: str,
    log: logging.Logger,
) -> subprocess.CompletedProcess[str] | None:
    """Run one bounded Git command and report execution failures."""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.error("Checkpoint git %s could not run: %s", action, exc)
        return None


def _git_output(result: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(
        part.strip()
        for part in (result.stdout, result.stderr)
        if part and part.strip()
    )


def _report_git_failure(
    result: subprocess.CompletedProcess[str] | None,
    action: str,
    log: logging.Logger,
) -> bool:
    if result is None:
        return False
    if result.returncode == 0:
        return True
    detail = _git_output(result) or "no diagnostic output"
    log.error("Checkpoint git %s failed with exit code %d: %s", action, result.returncode, detail)
    return False


def _resolve_git_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path.strip())
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def ensure_dedicated_worktree(repo_root: Path, log: logging.Logger) -> bool:
    repo_root = repo_root.resolve()

    top_level = _run_git(
        repo_root,
        ["rev-parse", "--show-toplevel"],
        "repository-root check",
        log,
    )
    if not _report_git_failure(top_level, "repository-root check", log):
        return False
    assert top_level is not None
    if _resolve_git_path(repo_root, top_level.stdout) != repo_root:
        log.error("Checkpoint refused: Git top level is not the script worktree")
        return False

    worktree_git_dir = _run_git(
        repo_root,
        ["rev-parse", "--git-dir"],
        "worktree check",
        log,
    )
    common_git_dir = _run_git(
        repo_root,
        ["rev-parse", "--git-common-dir"],
        "common-directory check",
        log,
    )
    if not _report_git_failure(worktree_git_dir, "worktree check", log):
        return False
    if not _report_git_failure(common_git_dir, "common-directory check", log):
        return False
    assert worktree_git_dir is not None
    assert common_git_dir is not None

    if _resolve_git_path(repo_root, worktree_git_dir.stdout) == _resolve_git_path(
        repo_root, common_git_dir.stdout
    ):
        log.error(
            "Checkpoint refused: run the autonomous loop from a dedicated linked worktree"
        )
        return False
    return True


def _valid_allowlist(paths: Sequence[str], log: logging.Logger) -> tuple[str, ...] | None:
    normalized: list[str] = []
    for raw_path in paths:
        path = str(raw_path).replace("\\", "/").strip("/")
        parts = PurePath(path).parts
        if (
            not path
            or path == "."
            or Path(path).is_absolute()
            or ".." in parts
            or path.startswith(":")
            or any(token in path for token in ("*", "?", "["))
        ):
            log.error("Checkpoint refused: invalid relative allowlist path %r", raw_path)
            return None
        normalized.append(path)
    if not normalized:
        log.error("Checkpoint refused: the allowlist is empty")
        return None
    return tuple(normalized)


def _path_is_allowed(path: str, allowlist: Sequence[str]) -> bool:
    normalized = path.replace("\\", "/")
    return any(normalized == allowed or normalized.startswith(f"{allowed}/") for allowed in allowlist)


def _reject_out_of_scope_staged_paths(
    repo_root: Path,
    allowlist: Sequence[str],
    log: logging.Logger,
) -> bool:
    result = _run_git(
        repo_root,
        ["diff", "--cached", "--name-only", "-z", "--"],
        "staged-path check",
        log,
    )
    if not _report_git_failure(result, "staged-path check", log):
        return False
    assert result is not None
    staged = [path for path in result.stdout.split("\0") if path]
    out_of_scope = [path for path in staged if not _path_is_allowed(path, allowlist)]
    if out_of_scope:
        log.error(
            "Checkpoint refused: staged paths outside the allowlist: %s",
            ", ".join(out_of_scope),
        )
        return False
    return True


def commit_checkpoint(
    repo_root: Path,
    allowlist: Sequence[str],
    message: str,
    log: logging.Logger,
) -> bool:
    """Commit only tracked changes under ``allowlist`` in a linked worktree."""
    normalized_allowlist = _valid_allowlist(allowlist, log)
    if normalized_allowlist is None or not ensure_dedicated_worktree(repo_root, log):
        return False

    if not _reject_out_of_scope_staged_paths(repo_root, normalized_allowlist, log):
        return False

    add = _run_git(
        repo_root,
        [*_GIT_CONFIG, "add", "--update", "--", *normalized_allowlist],
        "add",
        log,
    )
    if not _report_git_failure(add, "add", log):
        return False

    # Recheck after staging.  This guards the commit even if Git's path
    # handling or a future edit to the add command changes unexpectedly.
    if not _reject_out_of_scope_staged_paths(repo_root, normalized_allowlist, log):
        return False

    commit = _run_git(
        repo_root,
        [*_GIT_CONFIG, "commit", "-m", message],
        "commit",
        log,
    )
    if not _report_git_failure(commit, "commit", log):
        return False
    return True
