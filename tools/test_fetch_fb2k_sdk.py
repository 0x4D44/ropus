from __future__ import annotations

import http.server
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FETCHER = ROOT / "tools" / "fetch-fb2k-sdk.ps1"
EXPECTED_LENGTH = 765947


class _ArchiveHandler(http.server.BaseHTTPRequestHandler):
    payload = b""

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, _format: str, *_args: object) -> None:
        return


class FetchFb2kSdkTests(unittest.TestCase):
    def test_same_length_wrong_archive_fails_before_replacing_sdk(self) -> None:
        if os.name != "nt":
            self.skipTest("fetch-fb2k-sdk.ps1 requires Windows")

        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell is None:
            self.skipTest("PowerShell is not installed")
        if shutil.which("curl.exe") is None:
            self.skipTest("curl.exe is not installed")

        wrong_archive = (b"not a trusted foobar2000 SDK\n" * EXPECTED_LENGTH)[:EXPECTED_LENGTH]
        _ArchiveHandler.payload = wrong_archive
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ArchiveHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                temp_root = Path(temp)
                tools_dir = temp_root / "tools"
                tools_dir.mkdir()
                script = tools_dir / FETCHER.name
                source = FETCHER.read_text(encoding="utf-8")
                url = f"http://127.0.0.1:{server.server_address[1]}/SDK-2025-03-07.7z"
                script.write_text(
                    source.replace(
                        "https://www.foobar2000.org/downloads/SDK-2025-03-07.7z", url
                    ),
                    encoding="utf-8",
                    newline="\n",
                )

                sdk_dir = temp_root / "foo_ropus" / "sdk"
                sdk_dir.mkdir(parents=True)
                marker = sdk_dir / "existing-sdk-marker.txt"
                marker.write_text("keep this SDK", encoding="utf-8")

                fake_bin = temp_root / "fake-bin"
                fake_bin.mkdir()
                # Hash rejection must happen before this executable is invoked.
                (fake_bin / "7z.exe").write_text("must not run", encoding="utf-8")

                env = os.environ.copy()
                env["PATH"] = os.pathsep.join((str(fake_bin), env.get("PATH", "")))
                env["NO_PROXY"] = "127.0.0.1,localhost"
                env["no_proxy"] = env["NO_PROXY"]
                result = subprocess.run(
                    [powershell, "-NoProfile", "-File", str(script), "-Force"],
                    cwd=temp_root,
                    env=env,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=30,
                )

                output = result.stdout + result.stderr
                self.assertNotEqual(result.returncode, 0, output)
                self.assertIn("SHA-256 mismatch", output)
                self.assertIn("Aborting without touching", output)
                self.assertNotIn("extracting", output.lower())
                self.assertEqual(marker.read_text(encoding="utf-8"), "keep this SDK")
                self.assertEqual(sorted(path.name for path in sdk_dir.iterdir()), [marker.name])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
