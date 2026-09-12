from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = (
    ROOT / "tools" / "overnight_fuzz_launch.sh",
    ROOT / "tools" / "fuzz_24h_launch_v2.sh",
)

FAKE_CARGO = r'''#!/usr/bin/env bash
set -u
printf '%s\n' "$*" >> "${FAKE_CARGO_LOG:?}"

if [[ "$1" == "+nightly" && "$2" == "fuzz" && "$3" == "build" ]]; then
    exit "${FAKE_CARGO_BUILD_EXIT:-0}"
fi
if [[ "$1" == "+nightly" && "$2" == "fuzz" && "$3" == "run" ]]; then
    target=""
    for arg in "$@"; do
        case "$arg" in
            fuzz_*) target="$arg" ;;
        esac
    done
    if [[ "${FAKE_FAIL_TARGET:-}" == "$target" ]]; then
        exit 7
    fi
    exit 0
fi
exit 99
'''


class FuzzLauncherTests(unittest.TestCase):
    @staticmethod
    def _bash_path(path: Path) -> str:
        if os.name != "nt":
            return str(path)
        drive = path.drive.rstrip(":").lower()
        return f"/mnt/{drive}{path.as_posix()[2:]}"

    def _run(
        self,
        launcher: Path,
        duration: str,
        fail_target: str = "",
    ) -> tuple[subprocess.CompletedProcess[str], str]:
        with tempfile.TemporaryDirectory() as temp:
            temp_root = Path(temp)
            root = temp_root / "repo"
            tools_dir = root / "tools"
            tools_dir.mkdir(parents=True)
            script = tools_dir / launcher.name
            script.write_bytes(launcher.read_bytes().replace(b"\r\n", b"\n"))
            wrapper = tools_dir / "invoke-launcher.sh"
            wrapper.write_text(
                "#!/usr/bin/env bash\n"
                f"FAKE_DURATION={shlex.quote(duration)}\n"
                'exec bash "$1" "$2" "$FAKE_DURATION"\n',
                encoding="utf-8",
                newline="\n",
            )
            wrapper.chmod(0o755)

            fake_bin = temp_root / "bin"
            fake_bin.mkdir()
            fake_cargo = fake_bin / "cargo"
            fake_cargo.write_text(FAKE_CARGO, encoding="utf-8", newline="\n")
            fake_cargo.chmod(0o755)
            cargo_log = temp_root / "cargo.log"
            campaign = temp_root / "campaign"

            fake_bin_bash = self._bash_path(fake_bin)
            cargo_log_bash = self._bash_path(cargo_log)
            root_bash = self._bash_path(root)
            script_bash = self._bash_path(script)
            wrapper_bash = self._bash_path(wrapper)
            campaign_bash = self._bash_path(campaign)
            command = "; ".join(
                (
                    f"export PATH={shlex.quote(fake_bin_bash)}:/usr/local/bin:/usr/bin:/bin",
                    f"export FAKE_CARGO_LOG={shlex.quote(cargo_log_bash)}",
                    f"export FAKE_FAIL_TARGET={shlex.quote(fail_target)}",
                    f"cd {shlex.quote(root_bash)}",
                    shlex.join(["exec", "bash", wrapper_bash, script_bash, campaign_bash]),
                )
            )
            result = subprocess.run(
                [shutil.which("bash") or "bash", "-c", command],
                cwd=ROOT,
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
            log = cargo_log.read_text(encoding="utf-8") if cargo_log.exists() else ""
            return result, log

    def test_mixed_worker_outcomes_fail_for_both_launchers(self) -> None:
        for launcher in LAUNCHERS:
            with self.subTest(launcher=launcher.name):
                result, log = self._run(launcher, "1", fail_target="fuzz_multistream")

                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("Worker failures", result.stdout)
                self.assertIn("fuzz_multistream", result.stdout)
                self.assertIn("exit=7", result.stdout)
                self.assertNotIn("completed successfully", result.stdout)
                self.assertIn(" fuzz run", log)

    def test_success_waits_for_all_workers_for_both_launchers(self) -> None:
        for launcher in LAUNCHERS:
            with self.subTest(launcher=launcher.name):
                result, log = self._run(launcher, "00001")

                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("Finished:", result.stdout)
                self.assertNotIn("Worker failures:", result.stdout)
                self.assertIn(" fuzz run", log)

    def test_duration_rejects_zero_and_values_above_the_day_limit(self) -> None:
        for launcher in LAUNCHERS:
            for duration in ("0", "86401"):
                with self.subTest(launcher=launcher.name, duration=duration):
                    result, log = self._run(launcher, duration)

                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("decimal integer", result.stderr.lower())
                    self.assertEqual(log, "")

    def test_duration_rejects_command_substitution_without_executing_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / "payload-created"
            duration = f"$(touch {self._bash_path(marker)})"
            for launcher in LAUNCHERS:
                with self.subTest(launcher=launcher.name):
                    result, log = self._run(launcher, duration)

                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("decimal integer", result.stderr.lower())
                    self.assertEqual(log, "")
                    self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
