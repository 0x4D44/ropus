from __future__ import annotations

import io
import logging
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tools.bisect_fix as bisect_fix
from tools.checkpoint import commit_checkpoint
import tools.trace_fix as trace_fix


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    if check and result.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed:\n{result.stderr}")
    return result


def _logger() -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    log = logging.Logger("checkpoint-test")
    log.addHandler(logging.StreamHandler(stream))
    return log, stream


class CheckpointTests(unittest.TestCase):
    def _linked_worktree(self) -> tuple[tempfile.TemporaryDirectory[str], Path, Path]:
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        repo = root / "repo"
        repo.mkdir()
        _git(repo, "init")
        _git(repo, "config", "user.name", "test")
        _git(repo, "config", "user.email", "test@example.invalid")
        (repo / "ropus" / "src").mkdir(parents=True)
        (repo / "ropus" / "src" / "allowed.rs").write_text("fn value() -> i32 { 1 }\n", encoding="utf-8")
        (repo / "unrelated.txt").write_text("baseline\n", encoding="utf-8")
        hooks = repo / "hooks"
        hooks.mkdir()
        _git(repo, "config", "core.hooksPath", str(hooks))
        _git(repo, "add", "--", "ropus/src/allowed.rs", "unrelated.txt")
        _git(repo, "commit", "-m", "initial")

        worktree = root / "linked"
        _git(repo, "worktree", "add", "--detach", str(worktree), "HEAD")
        return temp, repo, worktree

    def test_only_allowlisted_tracked_changes_are_committed(self) -> None:
        temp, _repo, worktree = self._linked_worktree()
        self.addCleanup(temp.cleanup)
        (worktree / "ropus" / "src" / "allowed.rs").write_text(
            "fn value() -> i32 { 2 }\n", encoding="utf-8"
        )
        (worktree / "unrelated.txt").write_text("changed\n", encoding="utf-8")
        (worktree / "untracked-secret.txt").write_text("do not commit\n", encoding="utf-8")
        log, stream = _logger()

        committed = commit_checkpoint(worktree, ("ropus/src",), "safe checkpoint", log)
        staged = _git(worktree, "diff", "--cached", "--name-only").stdout.splitlines()
        self.assertNotIn("unrelated.txt", staged, "unrelated tracked work was staged")
        self.assertNotIn("untracked-secret.txt", staged, "unrelated untracked work was staged")
        self.assertTrue(committed)

        changed = _git(worktree, "diff", "HEAD^", "HEAD", "--name-only").stdout.splitlines()
        self.assertEqual(changed, ["ropus/src/allowed.rs"])
        status = _git(worktree, "status", "--short", "--untracked-files=all").stdout
        self.assertIn(" M unrelated.txt", status)
        self.assertIn("?? untracked-secret.txt", status)
        self.assertNotIn("Committed checkpoint", stream.getvalue())

    def test_rejects_main_worktree_before_staging(self) -> None:
        temp, _repo, worktree = self._linked_worktree()
        self.addCleanup(temp.cleanup)
        (worktree / "ropus" / "src" / "allowed.rs").write_text("changed\n", encoding="utf-8")
        log, stream = _logger()

        # The repository returned by _linked_worktree is the primary checkout.
        primary = Path(temp.name) / "repo"
        self.assertFalse(commit_checkpoint(primary, ("ropus/src",), "unsafe", log))
        self.assertEqual(_git(primary, "diff", "--cached", "--name-only").stdout, "")
        self.assertIn("dedicated linked worktree", stream.getvalue())

    def test_autonomous_loops_reject_primary_before_building(self) -> None:
        log, _stream = _logger()
        for module in (bisect_fix, trace_fix):
            with self.subTest(module=module.__name__):
                with (
                    patch.object(module, "setup_logging", return_value=log),
                    patch.object(module, "ensure_dedicated_worktree", return_value=False),
                    patch.object(module, "build") as build,
                ):
                    self.assertEqual(module.cmd_run(None), 1)
                    build.assert_not_called()

    def test_reports_commit_failure_without_claiming_success(self) -> None:
        temp, _repo, worktree = self._linked_worktree()
        self.addCleanup(temp.cleanup)
        (worktree / "ropus" / "src" / "allowed.rs").write_text("changed\n", encoding="utf-8")
        hooks = Path(_git(worktree, "rev-parse", "--git-path", "hooks").stdout.strip())
        (hooks / "pre-commit").write_text("#!/bin/sh\necho forced failure >&2\nexit 17\n", encoding="utf-8")
        log, stream = _logger()

        self.assertFalse(commit_checkpoint(worktree, ("ropus/src",), "will fail", log))
        self.assertIn("git commit failed with exit code", stream.getvalue())
        self.assertIn("forced failure", stream.getvalue())
        self.assertNotIn("Committed checkpoint", stream.getvalue())
        self.assertEqual(_git(worktree, "log", "-1", "--format=%s").stdout.strip(), "initial")
        self.assertEqual(_git(worktree, "diff", "--cached", "--name-only").stdout.strip(), "ropus/src/allowed.rs")


if __name__ == "__main__":
    unittest.main()
