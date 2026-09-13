from __future__ import annotations

import contextlib
import io
import json
import logging
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tools.coordinator as coordinator


def _logger() -> logging.Logger:
    return logging.Logger("coordinator-test")


class CoordinatorLayoutTests(unittest.TestCase):
    def test_rust_source_constant_points_to_the_ropus_crate(self) -> None:
        rust_src = coordinator.ROOT / "ropus" / "src"

        self.assertEqual(coordinator.SRC, rust_src)
        self.assertTrue(rust_src.is_dir())
        self.assertNotEqual(coordinator.SRC, coordinator.ROOT / "src")

    def test_rust_prompts_use_the_current_crate_path(self) -> None:
        for prompt_name in ("implement_module", "review_module", "fix_errors"):
            with self.subTest(prompt=prompt_name):
                prompt = coordinator.PROMPTS[prompt_name]

                self.assertIn("ropus/src/{rust_module}.rs", prompt)
                self.assertNotIn(
                    ": src/{rust_module}.rs",
                    prompt,
                )

    def test_workspace_path_constants_use_current_layout(self) -> None:
        self.assertEqual(coordinator.NOTES, coordinator.ROOT / "wrk_journals")
        self.assertEqual(coordinator.ASSETS, coordinator.ROOT / "wrk_docs" / "design_docs")

    def test_harness_prompt_uses_the_workspace_harness_package(self) -> None:
        prompt = coordinator.PROMPTS["test_harness"]
        for relative_path in (
            "harness/build.rs",
            "harness/src/bindings.rs",
            "harness/src/main.rs",
        ):
            with self.subTest(path=relative_path):
                self.assertTrue((coordinator.ROOT / relative_path).is_file())
                self.assertIn(f"`{relative_path}`", prompt)

        self.assertNotIn("tests/harness", prompt)

    def test_integrate_runs_an_actual_workspace_package(self) -> None:
        state = {"completed_phases": []}
        result = subprocess.CompletedProcess([], 0, stdout="", stderr="")

        with (
            patch.object(coordinator.subprocess, "run", return_value=result) as run,
            patch.object(coordinator, "save_state"),
        ):
            self.assertTrue(coordinator.phase_integrate(state, _logger()))

        command = run.call_args.args[0]
        self.assertEqual(
            command,
            ["cargo", "test", "-p", "ropus-harness", "--", "--nocapture"],
        )
        self.assertNotIn("--test", command)
        self.assertIn('name = "ropus-harness"', (coordinator.ROOT / "harness" / "Cargo.toml").read_text())
        self.assertEqual(run.call_args.kwargs["cwd"], str(coordinator.ROOT))

    def test_codex_review_output_paths_are_unique_within_one_second(self) -> None:
        results = [
            subprocess.CompletedProcess([], 0, stdout="first review\n", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="second review\n", stderr=""),
        ]
        with (
            patch.object(coordinator.subprocess, "run", side_effect=results) as run,
            patch.object(coordinator.time, "time", return_value=1234.0),
        ):
            first = coordinator.invoke_codex("first prompt")
            second = coordinator.invoke_codex("second prompt")

        self.assertEqual(first, (True, "first review\n"))
        self.assertEqual(second, (True, "second review\n"))
        output_paths = [
            call.args[0][call.args[0].index("-o") + 1]
            for call in run.call_args_list
        ]
        self.assertEqual(len(output_paths), 2)
        self.assertNotEqual(output_paths[0], output_paths[1])

    def test_failed_or_empty_review_does_not_advance_checkpoint(self) -> None:
        for review_result in ((False, "codex unavailable"), (True, "  \n")):
            with self.subTest(review_result=review_result):
                state = {
                    "module_status": {"range_coder": "implemented"},
                    "attempts": {},
                }
                build_result = subprocess.CompletedProcess([], 0, stdout="", stderr="")
                with (
                    patch.object(coordinator, "invoke_codex", return_value=review_result),
                    patch.object(coordinator.subprocess, "run", return_value=build_result),
                    patch.object(coordinator, "write_artifact") as write_artifact,
                    patch.object(coordinator, "save_state") as save_state,
                ):
                    result = coordinator.implement_module(
                        coordinator.MODULES[0], state, _logger()
                    )

                self.assertFalse(result)
                self.assertEqual(state["module_status"]["range_coder"], "implemented")
                write_artifact.assert_not_called()
                save_state.assert_not_called()

    def test_save_state_preserves_existing_checkpoint_when_write_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_file = Path(directory) / "coordinator_state.json"
            original = {"phase": "implement", "completed_phases": ["document"]}
            original_bytes = json.dumps(original, indent=2).encode("utf-8")
            state_file.write_bytes(original_bytes)

            with patch.object(coordinator, "STATE_FILE", state_file):
                with patch.object(
                    coordinator.json,
                    "dump",
                    side_effect=OSError("disk full"),
                ):
                    with self.assertRaisesRegex(OSError, "disk full"):
                        coordinator.save_state({"phase": "integrate"})

            self.assertEqual(state_file.read_bytes(), original_bytes)
            self.assertEqual(list(Path(directory).glob(".*.tmp")), [])

    def test_save_state_writes_a_complete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_file = Path(directory) / "coordinator_state.json"
            state = {"phase": "integrate"}

            with patch.object(coordinator, "STATE_FILE", state_file):
                coordinator.save_state(state)

            self.assertEqual(json.loads(state_file.read_text(encoding="utf-8")), state)
            self.assertEqual(list(Path(directory).glob(".*.tmp")), [])

    def test_load_state_reports_corrupt_checkpoint_with_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_file = Path(directory) / "coordinator_state.json"
            state_file.write_text("{", encoding="utf-8")

            with patch.object(coordinator, "STATE_FILE", state_file):
                with self.assertRaisesRegex(
                    coordinator.CoordinatorStateError,
                    r"corrupt.*Restore valid JSON or remove the file",
                ):
                    coordinator.load_state()

    def test_main_reports_corrupt_checkpoint_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_file = Path(directory) / "coordinator_state.json"
            state_file.write_text("{", encoding="utf-8")
            error = io.StringIO()

            with (
                patch.object(coordinator, "STATE_FILE", state_file),
                patch.object(coordinator.sys, "argv", ["coordinator.py", "status"]),
                contextlib.redirect_stderr(error),
            ):
                result = coordinator.main()

            self.assertEqual(result, 1)
            self.assertIn("ERROR:", error.getvalue())
            self.assertIn("Restore valid JSON or remove the file", error.getvalue())
            self.assertNotIn("Traceback", error.getvalue())


if __name__ == "__main__":
    unittest.main()
