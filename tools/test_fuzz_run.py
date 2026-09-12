from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FUZZ_RUN = ROOT / "tools" / "fuzz_run.sh"
FUZZ_TARGETS = ROOT / "tools" / "fuzz_targets.sh"
FUZZ_MANIFEST = ROOT / "tests" / "fuzz" / "Cargo.toml"

FAKE_CARGO = r'''#!/usr/bin/env bash
set -u
printf '%s\n' "$*" >> "${FAKE_CARGO_LOG:?}"

if [[ "$1" == "fuzz" && "$2" == "--version" ]]; then
    exit 0
fi
if [[ "$1" == "+nightly" && "$2" == "--version" ]]; then
    exit 0
fi
if [[ "$1" == "+nightly" && "$2" == "fuzz" && "$3" == "build" ]]; then
    exit "${FAKE_CARGO_BUILD_EXIT:-0}"
fi
if [[ "$1" == "+nightly" && "$2" == "fuzz" && "$3" == "run" ]]; then
    prefix=""
    for arg in "$@"; do
        case "$arg" in
            -artifact_prefix=*) prefix="${arg#-artifact_prefix=}" ;;
        esac
    done
    if [[ "${FAKE_CARGO_MODE:-pass}" == "artifact" ]]; then
        mkdir -p "$prefix"
        : > "${prefix}crash-fake"
    fi
    exit "${FAKE_CARGO_RUN_EXIT:-0}"
fi
exit 99
'''


class FuzzRunTests(unittest.TestCase):
    @staticmethod
    def _bash_path(path: Path) -> str:
        if os.name != "nt":
            return str(path)
        drive = path.drive.rstrip(":").lower()
        return f"/mnt/{drive}{path.as_posix()[2:]}"

    def _run(self, mode: str = "pass", run_exit: int = 0, *args: str) -> tuple[subprocess.CompletedProcess[str], str]:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            tools_dir = root / "tools"
            fuzz_dir = root / "tests" / "fuzz"
            tools_dir.mkdir(parents=True)
            fuzz_dir.mkdir(parents=True)
            script = tools_dir / "fuzz_run.sh"
            script.write_bytes(FUZZ_RUN.read_bytes().replace(b"\r\n", b"\n"))
            (tools_dir / "fuzz_targets.sh").write_bytes(FUZZ_TARGETS.read_bytes().replace(b"\r\n", b"\n"))
            (fuzz_dir / "Cargo.toml").write_bytes(FUZZ_MANIFEST.read_bytes())

            fake_bin = Path(temp) / "bin"
            fake_bin.mkdir()
            fake_cargo = fake_bin / "cargo"
            fake_cargo.write_text(FAKE_CARGO, encoding="utf-8", newline="\n")
            fake_cargo.chmod(0o755)
            cargo_log = Path(temp) / "cargo.log"
            env = os.environ.copy()
            fake_bin_bash = self._bash_path(fake_bin)
            cargo_log_bash = self._bash_path(cargo_log)
            root_bash = self._bash_path(root)
            script_bash = self._bash_path(script)
            command = "; ".join(
                (
                    f"export PATH={shlex.quote(fake_bin_bash)}:/usr/local/bin:/usr/bin:/bin",
                    f"export FAKE_CARGO_LOG={shlex.quote(cargo_log_bash)}",
                    f"export FAKE_CARGO_MODE={shlex.quote(mode)}",
                    f"export FAKE_CARGO_RUN_EXIT={shlex.quote(str(run_exit))}",
                    f"cd {shlex.quote(root_bash)}",
                    shlex.join(["exec", "bash", script_bash, *args]),
                )
            )

            result = subprocess.run(
                [shutil.which("bash") or "bash", "-c", command],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
            log = cargo_log.read_text(encoding="utf-8") if cargo_log.exists() else ""
            return result, log

    def test_pass_without_artifacts_is_clear(self) -> None:
        result, _log = self._run("pass", 0, "--target", "fuzz_decode", "--duration", "0")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("All clear - no issues found in this run.", result.stdout)
        self.assertIn("Target failures: 0", result.stdout)

    def test_artifact_finding_is_nonzero(self) -> None:
        result, _log = self._run("artifact", 0, "--target", "fuzz_decode", "--duration", "0")

        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("FOUND 1 new findings!", result.stdout)
        self.assertNotIn("All clear - no issues found in this run.", result.stdout)

    def test_child_failure_without_artifact_is_nonzero(self) -> None:
        result, _log = self._run("pass", 7, "--target", "fuzz_decode", "--duration", "0")

        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("exit=7", result.stdout)
        self.assertIn("Target failures: 1", result.stdout)
        self.assertNotIn("All clear - no issues found in this run.", result.stdout)

    def test_no_diff_is_rejected_before_starting_cargo(self) -> None:
        result, cargo_log = self._run("pass", 0, "--no-diff")

        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--no-diff is not supported", result.stderr)
        self.assertEqual(cargo_log, "")


if __name__ == "__main__":
    unittest.main()
