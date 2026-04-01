# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-04-01
"""Pytest plugin for tdd-guard integration.

Captures test results and writes them to .claude/tdd-guard/data/test.json
in the format expected by tdd-guard's TestResultSchema:

    {
        "testModules": [
            {
                "moduleId": "tests/test_synthesize.py",
                "tests": [
                    {
                        "name": "test_valid_data_returns_dict",
                        "fullName": "tests/test_synthesize.py::TestLoadAgentOutputHappyPath::test_valid_data_returns_dict",
                        "state": "passed|failed|skipped",
                        "errors": [{"message": "...", "stack": "..."}]
                    }
                ]
            }
        ],
        "reason": "passed|failed|interrupted"
    }
"""

import json
import sys
from pathlib import Path

import pytest

# Add MAGI scripts to sys.path so tests can import them directly.
_SCRIPTS_DIR = str(Path(__file__).parent / "skills" / "magi" / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

DATA_DIR = Path(".claude", "tdd-guard", "data")
TEST_RESULTS_FILE = DATA_DIR / "test.json"


def pytest_sessionstart(session: pytest.Session) -> None:
    """Ensure the tdd-guard data directory exists before tests run."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write test results to tdd-guard's expected JSON format.

    Args:
        session: The pytest session object containing all test reports.
        exitstatus: The exit status code from pytest.
    """
    modules: dict[str, list[dict]] = {}

    for item in session.items:
        module_id = item.nodeid.split("::")[0]
        test_name = item.name
        full_name = item.nodeid

        report = getattr(item, "_tdd_guard_report", None)
        state = "skipped"
        errors: list[dict[str, str]] = []

        if report is not None:
            if report.passed:
                state = "passed"
            elif report.failed:
                state = "failed"
                if report.longrepr:
                    errors.append(
                        {
                            "message": str(report.longrepr).split("\n")[-1],
                            "stack": str(report.longrepr),
                        }
                    )
            elif report.skipped:
                state = "skipped"

        test_entry: dict = {
            "name": test_name,
            "fullName": full_name,
            "state": state,
        }
        if errors:
            test_entry["errors"] = errors

        modules.setdefault(module_id, []).append(test_entry)

    reason = "passed"
    if exitstatus == pytest.ExitCode.INTERRUPTED:
        reason = "interrupted"
    elif exitstatus != pytest.ExitCode.OK:
        reason = "failed"

    result = {
        "testModules": [{"moduleId": mid, "tests": tests} for mid, tests in modules.items()],
        "reason": reason,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TEST_RESULTS_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> None:
    """Capture the test result report for each test item.

    Stores the call-phase report on the item so pytest_sessionfinish
    can read it.

    Args:
        item: The test item being reported on.
        call: The call information for this test phase.
    """
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        item._tdd_guard_report = report  # type: ignore[attr-defined]
