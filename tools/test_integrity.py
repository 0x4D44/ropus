from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tools.bisect_fix as bisect_fix
import tools.integrate as integrate


def _logger() -> logging.Logger:
    return logging.Logger("integrity-test")


class IntegrateIntegrityTests(unittest.TestCase):
    def test_missing_configured_wav_is_a_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with (
                patch.object(integrate, "CORPUS_DIR", Path(temp)),
                patch.object(
                    integrate,
                    "TEST_CASES",
                    [{"name": "required", "sample_rate": 48000, "channels": 1}],
                ),
                patch.object(integrate, "BITRATES", [64000]),
                patch.object(integrate, "run_comparison") as comparator,
            ):
                results = integrate.run_all_tests(_logger())

        self.assertEqual(len(results), 1)
        self.assertIs(results[0]["passed"], False)
        self.assertEqual(results[0]["returncode"], 1)
        self.assertIn("Missing required WAV fixture", results[0]["output"])
        comparator.assert_not_called()

    def test_empty_results_are_a_failure(self) -> None:
        self.assertEqual(integrate.summarize_results([], _logger()), (0, 1, 0))

    def test_fix_loop_stops_on_missing_fixture(self) -> None:
        missing = [{"passed": False, "error": "missing_fixture", "output": "missing"}]
        with patch.object(integrate, "invoke_claude") as invoke:
            self.assertFalse(integrate.fix_loop(missing, _logger()))
        invoke.assert_not_called()

    def test_fix_loop_updates_caller_results_after_retest(self) -> None:
        results = [{"passed": False, "output": "before fix"}]
        retested = [{"passed": True, "output": "after fix"}]
        with (
            patch.object(integrate, "MAX_FIX_ITERATIONS", 1),
            patch.object(integrate, "invoke_claude", return_value=(True, "fixed")),
            patch.object(integrate, "build_harness", return_value=True),
            patch.object(integrate, "run_all_tests", return_value=retested),
        ):
            self.assertTrue(integrate.fix_loop(results, _logger()))

        self.assertEqual(results, retested)

    def test_test_command_propagates_comparator_failure(self) -> None:
        failed = [{"mode": "encode", "wav": "fixture.wav", "passed": False, "output": "FAIL"}]
        with (
            patch.object(integrate, "setup_logging", return_value=_logger()),
            patch.object(integrate, "build_harness", return_value=True),
            patch.object(integrate, "run_all_tests", return_value=failed),
            patch.object(integrate, "save_results"),
        ):
            self.assertEqual(integrate.cmd_test(None), 1)

    def test_run_command_propagates_comparator_failure(self) -> None:
        failed = [{"mode": "encode", "wav": "fixture.wav", "passed": False, "output": "FAIL"}]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            harness = root / "tests" / "harness"
            harness.mkdir(parents=True)
            (harness / "main.rs").write_text("ready\n", encoding="utf-8")
            with (
                patch.object(integrate, "ROOT", root),
                patch.object(integrate, "setup_logging", return_value=_logger()),
                patch.object(integrate, "build_harness", return_value=True),
                patch.object(integrate, "generate_corpus"),
                patch.object(integrate, "run_all_tests", return_value=failed),
                patch.object(integrate, "save_results"),
                patch.object(integrate, "fix_loop", return_value=False),
            ):
                self.assertEqual(integrate.cmd_run(None), 1)


class BisectIntegrityTests(unittest.TestCase):
    def test_missing_wav_is_a_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(bisect_fix, "CORPUS_DIR", Path(temp)):
                result = bisect_fix.run_test("required", 64000)

        self.assertIs(result["passed"], False)
        self.assertEqual(result["returncode"], 1)
        self.assertIn("File not found", result["output"])

    def test_scan_command_propagates_comparator_failure(self) -> None:
        failed = [{"name": "fixture", "bitrate": 64000, "passed": False, "output": "FAIL"}]
        with (
            patch.object(bisect_fix, "setup_logging", return_value=_logger()),
            patch.object(bisect_fix, "build", return_value=True),
            patch.object(bisect_fix, "scan_all", return_value=failed),
        ):
            self.assertEqual(bisect_fix.cmd_scan(None), 1)

    def test_test_command_propagates_comparator_failure(self) -> None:
        failed = {"passed": False, "output": "FAIL"}
        with (
            patch.object(bisect_fix, "setup_logging", return_value=_logger()),
            patch.object(bisect_fix, "build", return_value=True),
            patch.object(bisect_fix, "BITRATES", [64000]),
            patch.object(bisect_fix, "run_test", return_value=failed),
        ):
            self.assertEqual(bisect_fix.cmd_test(type("Args", (), {"wav": "fixture.wav"})()), 1)

    def test_empty_scan_does_not_count_as_success(self) -> None:
        with (
            patch.object(bisect_fix, "MAX_FIX_ITERATIONS", 1),
            patch.object(bisect_fix, "scan_all", return_value=[]),
        ):
            self.assertFalse(bisect_fix.surgical_fix_loop(_logger()))

    def test_surgical_loop_stops_on_missing_fixture(self) -> None:
        missing = [{"passed": False, "error": "missing_fixture", "output": "missing"}]
        with (
            patch.object(bisect_fix, "scan_all", return_value=missing),
            patch.object(bisect_fix, "invoke_claude") as invoke,
        ):
            self.assertFalse(bisect_fix.surgical_fix_loop(_logger()))
        invoke.assert_not_called()

    def test_run_command_rejects_empty_scan(self) -> None:
        with (
            patch.object(bisect_fix, "setup_logging", return_value=_logger()),
            patch.object(bisect_fix, "ensure_dedicated_worktree", return_value=True),
            patch.object(bisect_fix, "build", return_value=True),
            patch.object(bisect_fix, "scan_all", return_value=[]),
        ):
            self.assertEqual(bisect_fix.cmd_run(None), 1)


if __name__ == "__main__":
    unittest.main()
