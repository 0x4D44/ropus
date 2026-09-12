from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "tools" / "bench_sweep.sh"

VALID_TABLE = """\
┌──────────────┬────────┬──────────────┬───────────────┐
│ Operation    │  Iters │  ms/iter     │  frames/sec   │
├──────────────┼────────┼──────────────┼───────────────┤
│ C encode     │      1 │      1.000ms │          10   │
│ Rust encode  │      1 │      2.000ms │           5   │
│ C decode     │      1 │      2.000ms │           5   │
│ Rust decode  │      1 │      8.000ms │           1   │
└──────────────┴────────┴──────────────┴───────────────┘
"""


class BenchSweepTests(unittest.TestCase):
    @staticmethod
    def _bash_path(path: Path) -> str:
        if os.name != "nt":
            return str(path)
        drive = path.drive.rstrip(":").lower()
        return f"/mnt/{drive}{path.as_posix()[2:]}"

    @staticmethod
    def _write_executable(path: Path, contents: str) -> None:
        path.write_text(contents, encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def _run(self, output: str, *, failing_vector: str | None = None) -> subprocess.CompletedProcess[str]:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is not installed")

        with tempfile.TemporaryDirectory() as temp:
            temp_root = Path(temp)
            tools_dir = temp_root / "tools"
            tools_dir.mkdir()
            script = tools_dir / SWEEP.name
            script.write_bytes(SWEEP.read_bytes().replace(b"\r\n", b"\n"))

            vectors_dir = temp_root / "tests" / "vectors"
            vectors_dir.mkdir(parents=True)
            (vectors_dir / "48k_sine1k_loud.wav").write_bytes(b"fixture\n")
            if failing_vector is not None:
                (vectors_dir / failing_vector).write_bytes(b"fixture\n")

            target_dir = temp_root / "target" / "release"
            target_dir.mkdir(parents=True)
            fake_bin = target_dir / "ropus-compare"
            output_file = temp_root / "bench-output.txt"
            output_file.write_text(output, encoding="utf-8", newline="\n")
            output_bash = self._bash_path(output_file)
            fail_bash = self._bash_path(vectors_dir / failing_vector) if failing_vector else ""
            self._write_executable(
                fake_bin,
                "#!/usr/bin/env bash\n"
                'if [[ -n "${FAKE_FAIL_WAV:-}" && "$2" == "$FAKE_FAIL_WAV" ]]; then\n'
                "  printf '%s\\n' 'simulated benchmark failure'\n"
                "  exit 7\n"
                "fi\n"
                'cat "${FAKE_BENCH_OUTPUT:?}"\n',
            )

            root_bash = self._bash_path(temp_root)
            script_bash = self._bash_path(script)
            command = "; ".join(
                (
                    f"export PATH={shlex.quote(self._bash_path(target_dir))}:/usr/local/bin:/usr/bin:/bin",
                    f"export FAKE_BENCH_OUTPUT={shlex.quote(output_bash)}",
                    f"export FAKE_FAIL_WAV={shlex.quote(fail_bash)}",
                    f"cd {shlex.quote(root_bash)}",
                    shlex.join(["exec", "bash", script_bash, "--iters=1"]),
                )
            )
            return subprocess.run(
                [bash, "-c", command],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )

    def test_current_table_is_parsed_into_summary(self) -> None:
        result = self._run(VALID_TABLE)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("CELT 48k mono sine 1k loud", result.stdout)
        self.assertIn("enc_ratio=2.000", result.stdout)
        self.assertIn("dec_ratio=4.000", result.stdout)
        self.assertNotIn("SWEEP FAILED", result.stdout + result.stderr)

    def test_missing_timing_row_fails_the_sweep(self) -> None:
        malformed = VALID_TABLE.replace(
            "│ Rust decode  │      1 │      8.000ms │           1   │\n", ""
        )
        result = self._run(malformed)

        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("expected exactly one finite timing row", result.stdout)
        self.assertIn("SWEEP FAILED: 1 vector(s) failed.", result.stderr)

    def test_nonfinite_timing_fails_the_sweep(self) -> None:
        malformed = VALID_TABLE.replace("8.000ms", "NaNms")
        result = self._run(malformed)

        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("expected exactly one finite timing row", result.stdout)
        self.assertIn("SWEEP FAILED: 1 vector(s) failed.", result.stderr)

    def test_mixed_success_and_command_failure_returns_nonzero(self) -> None:
        result = self._run(VALID_TABLE, failing_vector="48k_sweep.wav")

        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("CELT 48k mono sine 1k loud", result.stdout)
        self.assertIn("FAIL CELT 48k mono sweep", result.stdout)
        self.assertIn("SWEEP FAILED: 1 vector(s) failed.", result.stderr)


if __name__ == "__main__":
    unittest.main()
