import logging
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import tools.trace_fix as trace_fix


def _logger() -> logging.Logger:
    return logging.Logger("trace-fix-test")


class TraceFixRecoveryTests(unittest.TestCase):
    def _run_recovery_case(self, *, divergence: bool) -> tuple[Mock, Mock, Mock]:
        failing_output = "FAIL at offset 17\n"
        divergence_info = {
            "tag": "GAIN",
            "occurrence": 0,
            "key": "value",
            "c_value": "1",
            "rs_value": "2",
            "c_line": "[C GAIN] value=1",
            "rs_line": "[RS GAIN] value=2",
        }
        comparison_results = [(False, failing_output)]
        if divergence:
            comparison_results.append((True, ""))

        find_next = Mock(side_effect=[("fixture", 16000), None])
        run_comparison = Mock(side_effect=comparison_results)
        build = Mock(side_effect=[False, True])
        invoke_agent = Mock(return_value=(True, "repair succeeded"))
        cargo_build = Mock(return_value=SimpleNamespace(stderr="synthetic compiler error"))
        log = _logger()

        with (
            patch.object(trace_fix, "find_next_failing_test", find_next),
            patch.object(trace_fix, "run_comparison", run_comparison),
            patch.object(trace_fix, "find_first_divergence", return_value=divergence_info if divergence else None),
            patch.object(trace_fix, "build", build),
            patch.object(trace_fix, "invoke_agent", invoke_agent),
            patch.object(trace_fix.subprocess, "run", cargo_build),
        ):
            try:
                self.assertTrue(trace_fix.trace_fix_loop(log))
            except NameError as exc:
                self.fail(f"build recovery raised an unexpected NameError: {exc}")

        self.assertEqual(build.call_count, 2)
        if divergence:
            self.assertEqual(build.call_args_list, [call(log), call(log)])
            self.assertEqual(run_comparison.call_count, 2)
        else:
            self.assertEqual(build.call_args_list, [call(log, clean=True), call(log, clean=True)])
            self.assertEqual(run_comparison.call_count, 1)
        self.assertEqual(cargo_build.call_count, 1)
        self.assertEqual(invoke_agent.call_count, 2)
        return build, invoke_agent, cargo_build

    def test_divergence_recovery_uses_supported_agent_dispatcher(self) -> None:
        self._run_recovery_case(divergence=True)

    def test_instrumentation_recovery_uses_supported_agent_dispatcher(self) -> None:
        self._run_recovery_case(divergence=False)


if __name__ == "__main__":
    unittest.main()
