from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools" / "pgo_build.sh"


class PgoBuildTests(unittest.TestCase):
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

    def test_builds_and_measures_the_selected_pgo_binary_directly(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is not installed")

        with tempfile.TemporaryDirectory() as temp:
            temp_root = Path(temp)
            tools_dir = temp_root / "tools"
            tools_dir.mkdir()
            script = tools_dir / LAUNCHER.name
            script.write_bytes(LAUNCHER.read_bytes().replace(b"\r\n", b"\n"))

            (temp_root / "harness").mkdir()
            (temp_root / "harness" / "Cargo.toml").write_text(
                "[package]\nname = \"ropus-harness\"\n", encoding="utf-8"
            )
            vectors = temp_root / "tests" / "vectors"
            vectors.mkdir(parents=True)
            (vectors / "48k_sine1k_loud.wav").write_bytes(b"fixture\n")

            fake_bin = temp_root / "fake-bin"
            fake_bin.mkdir()
            sysroot = temp_root / "sysroot"
            llvm_profdata_dir = sysroot / "lib" / "rustlib" / "x86_64-unknown-linux-gnu" / "bin"
            llvm_profdata_dir.mkdir(parents=True)
            llvm_profdata = llvm_profdata_dir / "llvm-profdata"
            cargo_log = temp_root / "cargo.log"
            binary_log = temp_root / "binary.log"
            llvm_log = temp_root / "llvm-profdata.log"

            sysroot_bash = self._bash_path(sysroot)
            self._write_executable(
                fake_bin / "rustc",
                "#!/usr/bin/env bash\n"
                'if [[ "${1:-}" == "--print" && "${2:-}" == "sysroot" ]]; then\n'
                f"  printf '%s\\n' {shlex.quote(sysroot_bash)}\n"
                "  exit 0\n"
                "fi\n"
                "exit 1\n",
            )
            self._write_executable(
                llvm_profdata,
                "#!/usr/bin/env bash\n"
                'printf \'%s\\n\' "$*" >> "${FAKE_LLVM_LOG:?}"\n'
                'output=""\n'
                'while [[ $# -gt 0 ]]; do\n'
                '  if [[ "$1" == "-o" ]]; then output="$2"; shift 2; else shift; fi\n'
                "done\n"
                '[[ -n "$output" ]] || exit 1\n'
                'printf \'profile\\n\' > "$output"\n',
            )
            self._write_executable(
                fake_bin / "cargo",
                "#!/usr/bin/env bash\n"
                'printf \'RUSTFLAGS=%s ARGS=%s\\n\' "${RUSTFLAGS-}" "$*" >> "${FAKE_CARGO_LOG:?}"\n'
                'case "${1:-}" in\n'
                '  build)\n'
                '    mkdir -p target/release\n'
                '    cat > target/release/ropus-compare <<\'EOF\'\n'
                "#!/usr/bin/env bash\n"
                'printf \'%s\\n\' "$*" >> "${FAKE_BINARY_LOG:?}"\n'
                "exit 0\n"
                "EOF\n"
                "    chmod +x target/release/ropus-compare\n"
                "    ;;\n"
                '  run) exit 0 ;;\n'
                '  *) exit 99 ;;\n'
                "esac\n",
            )

            fake_bin_bash = self._bash_path(fake_bin)
            root_bash = self._bash_path(temp_root)
            script_bash = self._bash_path(script)
            cargo_log_bash = self._bash_path(cargo_log)
            binary_log_bash = self._bash_path(binary_log)
            llvm_log_bash = self._bash_path(llvm_log)
            command = "; ".join(
                (
                    f"export PATH={shlex.quote(fake_bin_bash)}:/usr/local/bin:/usr/bin:/bin",
                    f"export FAKE_CARGO_LOG={shlex.quote(cargo_log_bash)}",
                    f"export FAKE_BINARY_LOG={shlex.quote(binary_log_bash)}",
                    f"export FAKE_LLVM_LOG={shlex.quote(llvm_log_bash)}",
                    "unset RUSTFLAGS",
                    f"cd {shlex.quote(root_bash)}",
                    shlex.join(["exec", "bash", script_bash, "--iters=1"]),
                )
            )
            result = subprocess.run(
                [bash, "-c", command],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Baseline binary identity:", result.stdout)
            self.assertIn("Baseline binary unchanged after benchmark:", result.stdout)
            self.assertIn("Instrumented binary identity:", result.stdout)
            self.assertIn("PGO binary identity:", result.stdout)
            self.assertIn("PGO binary unchanged after benchmark:", result.stdout)

            cargo_lines = cargo_log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(cargo_lines), 3, cargo_lines)
            self.assertTrue(
                all(
                    "ARGS=build --release" in line
                    and "harness/Cargo.toml" in line
                    and "--target-dir " in line
                    and "--package ropus-harness --bin ropus-compare" in line
                    for line in cargo_lines
                ),
                cargo_lines,
            )
            self.assertNotIn("ARGS=run", "\n".join(cargo_lines))
            self.assertIn("RUSTFLAGS=-Cprofile-generate=", "\n".join(cargo_lines))
            self.assertIn("RUSTFLAGS=-Cprofile-use=", "\n".join(cargo_lines))

            binary_lines = binary_log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(binary_lines), 6, binary_lines)
            self.assertEqual(binary_lines[0].split()[0], "bench")
            self.assertEqual(binary_lines[-1].split()[0], "bench")
            self.assertEqual(sum(line.startswith("roundtrip ") for line in binary_lines), 4)
            self.assertEqual(sum(line.startswith("bench ") for line in binary_lines), 2)

            llvm_lines = llvm_log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(llvm_lines), 1, llvm_lines)
            self.assertIn("merge -o", llvm_lines[0])


if __name__ == "__main__":
    unittest.main()
