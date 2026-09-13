from __future__ import annotations

import logging
import subprocess
import unittest
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


if __name__ == "__main__":
    unittest.main()
