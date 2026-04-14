# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-04-01
"""Tests for load_agent_output validation in synthesize.py."""

import json
import os
import tempfile

import pytest

from synthesize import (
    VALID_AGENTS,
    VALID_SEVERITIES,
    VALID_VERDICTS,
    ValidationError,
    determine_consensus,
    format_banner,
    format_report,
    load_agent_output,
)


def _valid_agent_data() -> dict:
    """Return a minimal valid agent output dictionary."""
    return {
        "agent": "melchior",
        "verdict": "approve",
        "confidence": 0.85,
        "summary": "Looks good.",
        "reasoning": "Code is clean.",
        "findings": [
            {"severity": "info", "title": "Style", "detail": "Minor style nit."},
        ],
        "recommendation": "Merge as-is.",
    }


def _write_json(data, *, suffix: str = ".json") -> str:
    """Write *data* to a temporary JSON file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


class TestLoadAgentOutputHappyPath:
    """Verify that well-formed inputs are accepted."""

    def test_valid_data_returns_dict(self):
        path = _write_json(_valid_agent_data())
        try:
            result = load_agent_output(path)
            assert isinstance(result, dict)
            assert result["agent"] == "melchior"
        finally:
            os.unlink(path)

    @pytest.mark.parametrize("agent", sorted(VALID_AGENTS))
    def test_all_valid_agents_accepted(self, agent):
        data = _valid_agent_data()
        data["agent"] = agent
        path = _write_json(data)
        try:
            result = load_agent_output(path)
            assert result["agent"] == agent
        finally:
            os.unlink(path)

    @pytest.mark.parametrize("verdict", sorted(VALID_VERDICTS))
    def test_all_valid_verdicts_accepted(self, verdict):
        data = _valid_agent_data()
        data["verdict"] = verdict
        path = _write_json(data)
        try:
            result = load_agent_output(path)
            assert result["verdict"] == verdict
        finally:
            os.unlink(path)

    @pytest.mark.parametrize("conf", [0.0, 0.5, 1.0])
    def test_boundary_confidence_values(self, conf):
        data = _valid_agent_data()
        data["confidence"] = conf
        path = _write_json(data)
        try:
            result = load_agent_output(path)
            assert result["confidence"] == conf
        finally:
            os.unlink(path)

    def test_empty_findings_list_accepted(self):
        data = _valid_agent_data()
        data["findings"] = []
        path = _write_json(data)
        try:
            result = load_agent_output(path)
            assert result["findings"] == []
        finally:
            os.unlink(path)


class TestFileErrors:
    """Verify behaviour when the file cannot be read or parsed."""

    def test_missing_file_raises_validation_error(self):
        with pytest.raises(ValidationError, match="Cannot read file"):
            load_agent_output("/nonexistent/path/agent.json")

    def test_invalid_json_raises_validation_error(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("{not valid json!}")
        try:
            with pytest.raises(ValidationError, match="Invalid JSON"):
                load_agent_output(path)
        finally:
            os.unlink(path)

    def test_validation_error_contains_filepath(self):
        with pytest.raises(ValidationError) as exc_info:
            load_agent_output("/nonexistent/path/agent.json")
        assert exc_info.value.filepath == "/nonexistent/path/agent.json"


class TestMissingKeys:
    """Verify detection of missing top-level keys."""

    def test_missing_single_key(self):
        data = _valid_agent_data()
        del data["summary"]
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="missing keys"):
                load_agent_output(path)
        finally:
            os.unlink(path)

    def test_missing_multiple_keys(self):
        data = {"agent": "melchior"}
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="missing keys"):
                load_agent_output(path)
        finally:
            os.unlink(path)


class TestAgentValidation:
    """Verify that only known agent names are accepted."""

    def test_unknown_agent_rejected(self):
        data = _valid_agent_data()
        data["agent"] = "nerv"
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="Unknown agent"):
                load_agent_output(path)
        finally:
            os.unlink(path)

    def test_uppercase_agent_rejected(self):
        data = _valid_agent_data()
        data["agent"] = "Melchior"
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="Unknown agent"):
                load_agent_output(path)
        finally:
            os.unlink(path)


class TestVerdictValidation:
    """Verify that only known verdicts are accepted."""

    def test_invalid_verdict_rejected(self):
        data = _valid_agent_data()
        data["verdict"] = "abstain"
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="Invalid verdict"):
                load_agent_output(path)
        finally:
            os.unlink(path)


class TestConfidenceValidation:
    """Verify that confidence is a float in [0.0, 1.0]."""

    def test_confidence_above_one_rejected(self):
        data = _valid_agent_data()
        data["confidence"] = 85
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="between 0.0 and 1.0"):
                load_agent_output(path)
        finally:
            os.unlink(path)

    def test_negative_confidence_rejected(self):
        data = _valid_agent_data()
        data["confidence"] = -0.1
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="between 0.0 and 1.0"):
                load_agent_output(path)
        finally:
            os.unlink(path)

    def test_string_confidence_rejected(self):
        data = _valid_agent_data()
        data["confidence"] = "high"
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="must be a number"):
                load_agent_output(path)
        finally:
            os.unlink(path)

    def test_none_confidence_rejected(self):
        data = _valid_agent_data()
        data["confidence"] = None
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="must be a number"):
                load_agent_output(path)
        finally:
            os.unlink(path)


class TestFindingsValidation:
    """Verify structural validation of the findings list."""

    def test_findings_none_rejected(self):
        data = _valid_agent_data()
        data["findings"] = None
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="must be a list"):
                load_agent_output(path)
        finally:
            os.unlink(path)

    def test_findings_string_rejected(self):
        data = _valid_agent_data()
        data["findings"] = "no issues"
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="must be a list"):
                load_agent_output(path)
        finally:
            os.unlink(path)

    def test_finding_not_a_dict_rejected(self):
        data = _valid_agent_data()
        data["findings"] = ["not a dict"]
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="must be a dict"):
                load_agent_output(path)
        finally:
            os.unlink(path)

    def test_finding_missing_keys_rejected(self):
        data = _valid_agent_data()
        data["findings"] = [{"severity": "info"}]
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="missing keys"):
                load_agent_output(path)
        finally:
            os.unlink(path)

    def test_finding_invalid_severity_rejected(self):
        data = _valid_agent_data()
        data["findings"] = [
            {"severity": "fatal", "title": "Bad", "detail": "Very bad."},
        ]
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="invalid severity"):
                load_agent_output(path)
        finally:
            os.unlink(path)

    @pytest.mark.parametrize("severity", sorted(VALID_SEVERITIES))
    def test_all_valid_severities_accepted(self, severity):
        data = _valid_agent_data()
        data["findings"] = [
            {"severity": severity, "title": "Check", "detail": "Detail."},
        ]
        path = _write_json(data)
        try:
            result = load_agent_output(path)
            assert result["findings"][0]["severity"] == severity
        finally:
            os.unlink(path)

    def test_second_finding_validated(self):
        """Ensure validation covers all findings, not just the first."""
        data = _valid_agent_data()
        data["findings"] = [
            {"severity": "info", "title": "OK", "detail": "Fine."},
            {"severity": "bogus", "title": "Bad", "detail": "Broken."},
        ]
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="index 1"):
                load_agent_output(path)
        finally:
            os.unlink(path)


class TestValidationErrorAttributes:
    """Verify the custom exception class itself."""

    def test_message_without_filepath(self):
        err = ValidationError("something wrong")
        assert str(err) == "something wrong"
        assert err.filepath == ""

    def test_message_with_filepath(self):
        err = ValidationError("bad data", filepath="/tmp/x.json")
        assert "/tmp/x.json" in str(err)
        assert err.filepath == "/tmp/x.json"

    def test_is_exception_subclass(self):
        assert issubclass(ValidationError, Exception)


class TestConstants:
    """Verify that the exported constant sets are correct."""

    def test_valid_agents(self):
        assert VALID_AGENTS == {"melchior", "balthasar", "caspar"}

    def test_valid_verdicts(self):
        assert VALID_VERDICTS == {"approve", "reject", "conditional"}

    def test_valid_severities(self):
        assert VALID_SEVERITIES == {"critical", "warning", "info"}


# ---------------------------------------------------------------------------
# Helpers for consensus tests
# ---------------------------------------------------------------------------


def _valid_agent(agent_name: str, **overrides) -> dict:
    """Return a minimal valid agent dict, optionally overriding fields.

    Args:
        agent_name: One of 'melchior', 'balthasar', or 'caspar'.
        **overrides: Any keys to override in the returned dict.

    Returns:
        Agent dict suitable for passing to ``determine_consensus``.
    """
    base = {
        "agent": agent_name,
        "verdict": "approve",
        "confidence": 0.85,
        "summary": f"{agent_name} summary.",
        "reasoning": f"{agent_name} reasoning.",
        "findings": [],
        "recommendation": f"{agent_name} recommendation.",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# TestDetermineConsensus
# ---------------------------------------------------------------------------


class TestDetermineConsensus:
    """Verify majority voting and confidence calculation."""

    def test_unanimous_approve_is_strong_go(self):
        """Three approve votes produce STRONG GO."""
        agents = [
            _valid_agent("melchior", verdict="approve", confidence=0.9),
            _valid_agent("balthasar", verdict="approve", confidence=0.8),
            _valid_agent("caspar", verdict="approve", confidence=0.85),
        ]
        result = determine_consensus(agents)
        assert result["consensus"] == "STRONG GO"
        assert result["consensus_verdict"] == "approve"

    def test_unanimous_reject_is_strong_no_go(self):
        """Three reject votes produce STRONG NO-GO."""
        agents = [
            _valid_agent("melchior", verdict="reject", confidence=0.9),
            _valid_agent("balthasar", verdict="reject", confidence=0.8),
            _valid_agent("caspar", verdict="reject", confidence=0.7),
        ]
        result = determine_consensus(agents)
        assert result["consensus"] == "STRONG NO-GO"
        assert result["consensus_verdict"] == "reject"

    def test_two_approve_one_reject_is_go_2_1(self):
        """Two approve, one reject produces GO (2-1) with dissent."""
        agents = [
            _valid_agent("melchior", verdict="approve", confidence=0.9),
            _valid_agent("balthasar", verdict="approve", confidence=0.8),
            _valid_agent("caspar", verdict="reject", confidence=0.7),
        ]
        result = determine_consensus(agents)
        assert result["consensus"] == "GO (2-1)"
        assert result["consensus_verdict"] == "approve"
        assert len(result["dissent"]) == 1
        assert result["dissent"][0]["agent"] == "caspar"

    def test_conditional_approve_reject_is_go_with_caveats(self):
        """Conditional + approve + reject produces GO WITH CAVEATS."""
        agents = [
            _valid_agent("melchior", verdict="conditional", confidence=0.8),
            _valid_agent("balthasar", verdict="approve", confidence=0.9),
            _valid_agent("caspar", verdict="reject", confidence=0.7),
        ]
        result = determine_consensus(agents)
        assert result["consensus"] == "GO WITH CAVEATS"
        assert result["consensus_verdict"] == "conditional"
        assert len(result["conditions"]) == 1
        assert result["conditions"][0]["agent"] == "melchior"

    def test_two_reject_one_approve_is_hold(self):
        """Two reject, one approve produces HOLD (2-1)."""
        agents = [
            _valid_agent("melchior", verdict="reject", confidence=0.9),
            _valid_agent("balthasar", verdict="reject", confidence=0.8),
            _valid_agent("caspar", verdict="approve", confidence=0.7),
        ]
        result = determine_consensus(agents)
        assert result["consensus"] == "HOLD (2-1)"
        assert result["consensus_verdict"] == "reject"

    def test_strong_dissent_lowers_confidence(self):
        """A high-confidence reject should lower consensus confidence.

        Compare all-approve at 0.85 each vs two-approve-one-strong-reject.
        The dissenting agent's confidence should reduce the overall score.
        """
        all_approve = [
            _valid_agent("melchior", verdict="approve", confidence=0.85),
            _valid_agent("balthasar", verdict="approve", confidence=0.85),
            _valid_agent("caspar", verdict="approve", confidence=0.85),
        ]
        conf_all_approve = determine_consensus(all_approve)["confidence"]

        with_dissent = [
            _valid_agent("melchior", verdict="approve", confidence=0.85),
            _valid_agent("balthasar", verdict="approve", confidence=0.85),
            _valid_agent("caspar", verdict="reject", confidence=0.95),
        ]
        conf_with_dissent = determine_consensus(with_dissent)["confidence"]

        assert conf_with_dissent < conf_all_approve, (
            f"Dissent confidence {conf_with_dissent} should be lower "
            f"than unanimous confidence {conf_all_approve}"
        )

    def test_confidence_clamped_to_zero_one(self):
        """Confidence must always be in [0.0, 1.0]."""
        agents = [
            _valid_agent("melchior", verdict="approve", confidence=1.0),
            _valid_agent("balthasar", verdict="approve", confidence=1.0),
            _valid_agent("caspar", verdict="approve", confidence=1.0),
        ]
        result = determine_consensus(agents)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_votes_dict_populated(self):
        """The votes dict should map agent name to verdict."""
        agents = [
            _valid_agent("melchior", verdict="approve"),
            _valid_agent("balthasar", verdict="reject"),
            _valid_agent("caspar", verdict="conditional"),
        ]
        result = determine_consensus(agents)
        assert result["votes"] == {
            "melchior": "approve",
            "balthasar": "reject",
            "caspar": "conditional",
        }

    def test_majority_summary_attributed(self):
        """Majority summary should include agent names."""
        agents = [
            _valid_agent("melchior", verdict="approve", summary="All clear."),
            _valid_agent("balthasar", verdict="approve", summary="Ship it."),
            _valid_agent("caspar", verdict="reject", summary="Too risky."),
        ]
        result = determine_consensus(agents)
        assert "Melchior:" in result["majority_summary"]
        assert "Balthasar:" in result["majority_summary"]
        assert "|" in result["majority_summary"]

    def test_no_hardcoded_agent_count(self):
        """Confidence calculation should use len(agents), not hardcoded 3.

        With all-approve, confidence = sum(conf) / num_agents.
        For 3 agents at 0.9 each: (0.9 * 3) / 3 = 0.9.
        """
        agents = [
            _valid_agent("melchior", verdict="approve", confidence=0.9),
            _valid_agent("balthasar", verdict="approve", confidence=0.9),
            _valid_agent("caspar", verdict="approve", confidence=0.9),
        ]
        result = determine_consensus(agents)
        assert result["confidence"] == 0.9

    def test_unanimous_conditional_is_go_with_caveats(self):
        """Three conditional votes should NOT be STRONG GO (bug W1).

        With weight-based scoring: score = (0.5+0.5+0.5)/3 = 0.5.
        Has conditions, score > 0 -> GO WITH CAVEATS.
        """
        agents = [
            _valid_agent("melchior", verdict="conditional", confidence=0.8),
            _valid_agent("balthasar", verdict="conditional", confidence=0.85),
            _valid_agent("caspar", verdict="conditional", confidence=0.9),
        ]
        result = determine_consensus(agents)
        assert result["consensus"] == "GO WITH CAVEATS"
        assert result["consensus_verdict"] == "conditional"
        assert len(result["conditions"]) == 3

    def test_two_agent_approve_reject_is_hold_tie(self):
        """1 approve + 1 reject (2 agents): score = (1-1)/2 = 0.0 -> HOLD — TIE."""
        agents = [
            _valid_agent("melchior", verdict="approve", confidence=0.9),
            _valid_agent("balthasar", verdict="reject", confidence=0.8),
        ]
        result = determine_consensus(agents)
        assert result["consensus"] == "HOLD -- TIE"

    def test_two_agent_approve_conditional_is_caveats(self):
        """1 approve + 1 conditional: score = (1+0.5)/2 = 0.75 -> GO WITH CAVEATS."""
        agents = [
            _valid_agent("melchior", verdict="approve", confidence=0.9),
            _valid_agent("balthasar", verdict="conditional", confidence=0.8),
        ]
        result = determine_consensus(agents)
        assert result["consensus"] == "GO WITH CAVEATS"

    def test_two_agent_both_conditional(self):
        """2x conditional: score = (0.5+0.5)/2 = 0.5 -> GO WITH CAVEATS."""
        agents = [
            _valid_agent("melchior", verdict="conditional", confidence=0.8),
            _valid_agent("balthasar", verdict="conditional", confidence=0.85),
        ]
        result = determine_consensus(agents)
        assert result["consensus"] == "GO WITH CAVEATS"
        assert len(result["conditions"]) == 2

    def test_two_agent_both_reject(self):
        """2x reject: score = (-1-1)/2 = -1.0 -> STRONG NO-GO."""
        agents = [
            _valid_agent("melchior", verdict="reject", confidence=0.9),
            _valid_agent("balthasar", verdict="reject", confidence=0.8),
        ]
        result = determine_consensus(agents)
        assert result["consensus"] == "STRONG NO-GO"

    def test_weight_confidence_unanimous_conditional(self):
        """3x conditional at 0.9: score=0.5, wf=0.75, base=0.9, conf=0.68."""
        agents = [
            _valid_agent("melchior", verdict="conditional", confidence=0.9),
            _valid_agent("balthasar", verdict="conditional", confidence=0.9),
            _valid_agent("caspar", verdict="conditional", confidence=0.9),
        ]
        result = determine_consensus(agents)
        assert result["confidence"] == 0.68

    def test_weight_confidence_hold_is_moderate(self):
        """2 reject + 1 approve: score=-0.33, confidence is moderate.

        With abs(score): weight_factor = (0.33 + 1) / 2 = 0.665
        base_confidence = (0.9 + 0.8) / 3 = 0.567
        confidence = 0.567 * 0.665 = 0.38
        """
        agents = [
            _valid_agent("melchior", verdict="reject", confidence=0.9),
            _valid_agent("balthasar", verdict="reject", confidence=0.8),
            _valid_agent("caspar", verdict="approve", confidence=0.7),
        ]
        result = determine_consensus(agents)
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["confidence"] == 0.38


# ---------------------------------------------------------------------------
# TestFindingsDedup
# ---------------------------------------------------------------------------


class TestFindingsDedup:
    """Verify that findings deduplication merges across agents correctly."""

    def test_same_title_from_two_agents_merged(self):
        """Same finding title from two agents produces one entry with both sources."""
        agents = [
            _valid_agent(
                "melchior",
                findings=[
                    {
                        "severity": "warning",
                        "title": "SQL Injection",
                        "detail": "Found in query.",
                    },
                ],
            ),
            _valid_agent(
                "balthasar",
                findings=[
                    {
                        "severity": "warning",
                        "title": "SQL Injection",
                        "detail": "Param not escaped.",
                    },
                ],
            ),
            _valid_agent("caspar", findings=[]),
        ]
        result = determine_consensus(agents)
        sql_findings = [f for f in result["findings"] if "sql" in f["title"].lower()]
        assert len(sql_findings) == 1
        assert "melchior" in sql_findings[0]["sources"]
        assert "balthasar" in sql_findings[0]["sources"]

    def test_dedup_keeps_highest_severity(self):
        """When same title has different severities, the highest wins."""
        agents = [
            _valid_agent(
                "melchior",
                findings=[
                    {"severity": "info", "title": "Buffer Issue", "detail": "Minor."},
                ],
            ),
            _valid_agent(
                "balthasar",
                findings=[
                    {
                        "severity": "critical",
                        "title": "Buffer Issue",
                        "detail": "Overflow!",
                    },
                ],
            ),
            _valid_agent("caspar", findings=[]),
        ]
        result = determine_consensus(agents)
        buf_findings = [f for f in result["findings"] if "buffer" in f["title"].lower()]
        assert len(buf_findings) == 1
        assert buf_findings[0]["severity"] == "critical"
        assert buf_findings[0]["detail"] == "Overflow!"

    def test_unique_findings_all_kept(self):
        """Findings with different titles are all preserved."""
        agents = [
            _valid_agent(
                "melchior",
                findings=[
                    {"severity": "info", "title": "Style", "detail": "Nit."},
                ],
            ),
            _valid_agent(
                "balthasar",
                findings=[
                    {
                        "severity": "warning",
                        "title": "Performance",
                        "detail": "Slow loop.",
                    },
                ],
            ),
            _valid_agent(
                "caspar",
                findings=[
                    {"severity": "critical", "title": "Security", "detail": "XSS."},
                ],
            ),
        ]
        result = determine_consensus(agents)
        assert len(result["findings"]) == 3

    def test_findings_sorted_by_severity(self):
        """Findings should be sorted: critical first, then warning, then info."""
        agents = [
            _valid_agent(
                "melchior",
                findings=[
                    {"severity": "info", "title": "Style", "detail": "Nit."},
                ],
            ),
            _valid_agent(
                "balthasar",
                findings=[
                    {"severity": "critical", "title": "Security", "detail": "Bad."},
                ],
            ),
            _valid_agent(
                "caspar",
                findings=[
                    {"severity": "warning", "title": "Perf", "detail": "Slow."},
                ],
            ),
        ]
        result = determine_consensus(agents)
        severities = [f["severity"] for f in result["findings"]]
        assert severities == ["critical", "warning", "info"]

    def test_dedup_case_insensitive(self):
        """Title dedup should be case-insensitive."""
        agents = [
            _valid_agent(
                "melchior",
                findings=[
                    {
                        "severity": "warning",
                        "title": "SQL Injection",
                        "detail": "Found.",
                    },
                ],
            ),
            _valid_agent(
                "balthasar",
                findings=[
                    {
                        "severity": "warning",
                        "title": "sql injection",
                        "detail": "Also found.",
                    },
                ],
            ),
            _valid_agent("caspar", findings=[]),
        ]
        result = determine_consensus(agents)
        sql_findings = [f for f in result["findings"] if "sql" in f["title"].lower()]
        assert len(sql_findings) == 1

    def test_sources_key_tracks_all_reporters(self):
        """Each finding has a 'sources' list, no legacy 'source' key."""
        agents = [
            _valid_agent(
                "melchior",
                findings=[
                    {"severity": "info", "title": "Note", "detail": "FYI."},
                ],
            ),
            _valid_agent("balthasar", findings=[]),
            _valid_agent("caspar", findings=[]),
        ]
        result = determine_consensus(agents)
        assert result["findings"][0]["sources"] == ["melchior"]
        assert "source" not in result["findings"][0]


# ---------------------------------------------------------------------------
# TestFormatBanner
# ---------------------------------------------------------------------------


class TestFormatBanner:
    """Verify that the ASCII banner has consistent alignment."""

    def test_banner_lines_equal_width(self):
        """Every line of the banner must have the same character width."""
        agents = [_valid_agent(n) for n in ["melchior", "balthasar", "caspar"]]
        consensus = determine_consensus(agents)
        banner = format_banner(agents, consensus)
        lines = banner.split("\n")
        widths = {len(line) for line in lines}
        assert len(widths) == 1, f"Inconsistent widths: {widths}"

    def test_banner_width_is_52(self):
        """Banner must be exactly 52 characters wide on every row."""
        agents = [_valid_agent(n) for n in ["melchior", "balthasar", "caspar"]]
        consensus = determine_consensus(agents)
        banner = format_banner(agents, consensus)
        for line in banner.split("\n"):
            assert len(line) == 52

    def test_banner_contains_agent_verdicts(self):
        """Banner should display each agent's name and verdict."""
        agents = [_valid_agent(n) for n in ["melchior", "balthasar", "caspar"]]
        consensus = determine_consensus(agents)
        banner = format_banner(agents, consensus)
        assert "Melchior" in banner
        assert "APPROVE" in banner

    def test_banner_verdicts_aligned_to_same_column(self):
        """Each agent's verdict word must start at the same column."""
        agents = [_valid_agent(n) for n in ["melchior", "balthasar", "caspar"]]
        # Give each a different verdict to exercise each word length.
        agents[0]["verdict"] = "approve"
        agents[1]["verdict"] = "conditional"
        agents[2]["verdict"] = "reject"
        consensus = determine_consensus(agents)
        banner = format_banner(agents, consensus)
        lines = banner.split("\n")
        # Agent rows are lines 3, 4, 5 (0-indexed) of the banner.
        columns = [
            lines[3].index("APPROVE"),
            lines[4].index("CONDITIONAL"),
            lines[5].index("REJECT"),
        ]
        assert len(set(columns)) == 1, f"Verdicts not column-aligned: {columns}"

    def test_banner_uses_integer_percentage(self):
        """Confidence must render as integer percentage, not float."""
        agents = [_valid_agent(n) for n in ["melchior", "balthasar", "caspar"]]
        for agent in agents:
            agent["confidence"] = 0.9
        consensus = determine_consensus(agents)
        banner = format_banner(agents, consensus)
        assert "(90%)" in banner
        assert "(0.9)" not in banner

    def test_banner_title_line_present(self):
        """Banner must contain the canonical title line."""
        agents = [_valid_agent(n) for n in ["melchior", "balthasar", "caspar"]]
        consensus = determine_consensus(agents)
        banner = format_banner(agents, consensus)
        assert "MAGI SYSTEM -- VERDICT" in banner

    def test_banner_consensus_line_present(self):
        """Banner must contain the CONSENSUS row."""
        agents = [_valid_agent(n) for n in ["melchior", "balthasar", "caspar"]]
        consensus = determine_consensus(agents)
        banner = format_banner(agents, consensus)
        assert "CONSENSUS:" in banner


# ---------------------------------------------------------------------------
# TestFormatReport
# ---------------------------------------------------------------------------


class TestFormatReport:
    """Verify human-readable report formatting."""

    def test_findings_show_multiple_sources(self):
        """When two agents report the same finding, both names appear."""
        agents = [_valid_agent(n) for n in ["melchior", "balthasar", "caspar"]]
        agents[0]["findings"] = [
            {"severity": "warning", "title": "Race condition", "detail": "In cache"},
        ]
        agents[2]["findings"] = [
            {"severity": "critical", "title": "Race condition", "detail": "Write risk"},
        ]
        consensus = determine_consensus(agents)
        report = format_report(agents, consensus)
        assert "melchior, caspar" in report

    def test_finding_titles_aligned_to_column_22(self):
        """All finding rows must place the title at the same column (1-indexed 22)."""
        agents = [_valid_agent(n) for n in ["melchior", "balthasar", "caspar"]]
        agents[0]["findings"] = [
            {"severity": "critical", "title": "Crit item", "detail": "x"},
            {"severity": "warning", "title": "Warn item", "detail": "y"},
            {"severity": "info", "title": "Info item", "detail": "z"},
        ]
        consensus = determine_consensus(agents)
        report = format_report(agents, consensus)
        finding_rows = [
            line for line in report.split("\n") if line.startswith(("[!!!]", "[!!]", "[i]"))
        ]
        assert len(finding_rows) == 3
        # Title starts at 1-indexed column 22 → 0-indexed position 21.
        for row in finding_rows:
            assert row[20] == " ", f"Column 21 must be a space separator: {row!r}"
            assert row[21] != " ", f"Column 22 must start the title: {row!r}"

    def test_finding_rows_use_bold_severity_label(self):
        agents = [_valid_agent(n) for n in ["melchior", "balthasar", "caspar"]]
        agents[0]["findings"] = [
            {"severity": "critical", "title": "Crit", "detail": "x"},
            {"severity": "warning", "title": "Warn", "detail": "y"},
            {"severity": "info", "title": "Info", "detail": "z"},
        ]
        consensus = determine_consensus(agents)
        report = format_report(agents, consensus)
        assert "**[CRITICAL]**" in report
        assert "**[WARNING]**" in report
        assert "**[INFO]**" in report

    def test_report_has_no_consensus_summary_section(self):
        """The Consensus Summary section was removed from the canonical format."""
        agents = [_valid_agent(n) for n in ["melchior", "balthasar", "caspar"]]
        consensus = determine_consensus(agents)
        report = format_report(agents, consensus)
        assert "## Consensus Summary" not in report

    def test_report_sections_present_when_applicable(self):
        agents = [_valid_agent(n) for n in ["melchior", "balthasar", "caspar"]]
        agents[0]["findings"] = [{"severity": "critical", "title": "X", "detail": "D"}]
        agents[2]["verdict"] = "reject"
        consensus = determine_consensus(agents)
        report = format_report(agents, consensus)
        assert "## Key Findings" in report
        assert "## Dissenting Opinion" in report
        assert "## Recommended Actions" in report

    def test_dissent_shows_summary_only(self):
        """Dissenting Opinion section prints the one-line summary, not reasoning."""
        agents = [_valid_agent(n) for n in ["melchior", "balthasar", "caspar"]]
        agents[2]["verdict"] = "reject"
        agents[2]["summary"] = "Unsafe to merge."
        agents[2]["reasoning"] = "Very long reasoning text that must not appear."
        consensus = determine_consensus(agents)
        report = format_report(agents, consensus)
        assert "Unsafe to merge." in report
        assert "Very long reasoning text" not in report

    def test_recommended_actions_section_always_present(self):
        agents = [_valid_agent(n) for n in ["melchior", "balthasar", "caspar"]]
        consensus = determine_consensus(agents)
        report = format_report(agents, consensus)
        assert "## Recommended Actions" in report
        assert "- **Melchior** (Scientist):" in report


# ---------------------------------------------------------------------------
# TestSkillMdTemplateParity
# ---------------------------------------------------------------------------


class TestSkillMdTemplateParity:
    """Verify that the canonical template in SKILL.md matches reporting.py.

    The MAGI system runs in three modes (Python orchestrator, native
    sub-agents, fallback) and each must produce identical output. These
    tests guard against drift between the hand-written template in
    ``skills/magi/SKILL.md`` and the output of ``format_report``.
    """

    @staticmethod
    def _read_skill_template() -> str:
        """Return the first fenced code block after the canonical header."""
        from pathlib import Path

        skill_md = Path(__file__).resolve().parent.parent / "skills" / "magi" / "SKILL.md"
        content = skill_md.read_text(encoding="utf-8")
        marker = "#### Canonical output template"
        header_idx = content.index(marker)
        fence_open = content.index("```", header_idx)
        body_start = content.index("\n", fence_open) + 1
        fence_close = content.index("```", body_start)
        return content[body_start:fence_close].rstrip("\n")

    def test_template_banner_width_matches_reporting(self):
        """Banner border in SKILL.md must match reporting.py width."""
        template = self._read_skill_template()
        template_border = template.split("\n")[0]

        agents = [_valid_agent(n) for n in ["melchior", "balthasar", "caspar"]]
        consensus = determine_consensus(agents)
        banner = format_banner(agents, consensus)
        generated_border = banner.split("\n")[0]

        assert len(template_border) == len(generated_border)
        assert template_border == generated_border

    def test_template_verdict_column_matches_reporting(self):
        """Melchior's verdict must start at the same column in both."""
        template = self._read_skill_template()
        template_lines = template.split("\n")
        tmpl_line = next(line for line in template_lines if "Melchior" in line)

        agents = [
            _valid_agent("melchior", verdict="approve", confidence=0.90),
            _valid_agent("balthasar", verdict="conditional", confidence=0.85),
            _valid_agent("caspar", verdict="reject", confidence=0.78),
        ]
        consensus = determine_consensus(agents)
        banner = format_banner(agents, consensus)
        gen_line = next(line for line in banner.split("\n") if "Melchior" in line)

        assert tmpl_line.index("APPROVE") == gen_line.index("APPROVE")

    def test_template_excludes_consensus_summary_section(self):
        """The SKILL.md template must never add ## Consensus Summary."""
        template = self._read_skill_template()
        assert "## Consensus Summary" not in template

    def test_template_has_required_sections_in_order(self):
        """Required section headers must appear in the canonical order."""
        template = self._read_skill_template()
        expected_order = [
            "## Key Findings",
            "## Dissenting Opinion",
            "## Conditions for Approval",
            "## Recommended Actions",
        ]
        positions = [template.index(section) for section in expected_order]
        assert positions == sorted(positions), (
            f"Sections are not in canonical order: {dict(zip(expected_order, positions))}"
        )

    def test_template_finding_rows_align_to_column_22(self):
        """Finding rows in the template must put titles at column 22."""
        template = self._read_skill_template()
        finding_lines = [
            line for line in template.split("\n") if line.startswith(("[!!!]", "[!!]", "[i]"))
        ]
        assert len(finding_lines) >= 3
        for line in finding_lines:
            assert line[20] == " ", f"Column 21 must be a separator space: {line!r}"
            assert line[21] != " ", f"Column 22 must start the title: {line!r}"


# ---------------------------------------------------------------------------
# TestFlexibleMain
# ---------------------------------------------------------------------------


class TestFlexibleMain:
    """Verify that main() accepts a flexible number of agents (2-3)."""

    def test_two_agents_produce_consensus(self):
        """determine_consensus works with 2 agents."""
        agents = [_valid_agent("melchior"), _valid_agent("balthasar")]
        result = determine_consensus(agents)
        assert result["consensus"] == "STRONG GO"
        assert result["confidence"] > 0


# ---------------------------------------------------------------------------
# Tests for bugs fixed in MAGI self-review
# ---------------------------------------------------------------------------


class TestConfidenceFormulaFix:
    """Verify that abs(score) produces meaningful confidence for reject."""

    def test_unanimous_reject_has_high_confidence(self):
        """STRONG NO-GO with 3x 0.9 confidence should NOT be 0.0."""
        agents = [
            _valid_agent("melchior", verdict="reject", confidence=0.9),
            _valid_agent("balthasar", verdict="reject", confidence=0.9),
            _valid_agent("caspar", verdict="reject", confidence=0.9),
        ]
        result = determine_consensus(agents)
        assert result["consensus"] == "STRONG NO-GO"
        assert result["confidence"] == 0.9

    def test_all_zero_confidence_produces_zero(self):
        """Degenerate case: all agents at 0.0 confidence produces 0.0."""
        agents = [
            _valid_agent("melchior", verdict="approve", confidence=0.0),
            _valid_agent("balthasar", verdict="approve", confidence=0.0),
            _valid_agent("caspar", verdict="approve", confidence=0.0),
        ]
        result = determine_consensus(agents)
        assert result["confidence"] == 0.0

    def test_unanimous_reject_confidence_matches_approve(self):
        """Symmetric: unanimous reject confidence == unanimous approve confidence."""
        approve_agents = [
            _valid_agent("melchior", verdict="approve", confidence=0.85),
            _valid_agent("balthasar", verdict="approve", confidence=0.85),
            _valid_agent("caspar", verdict="approve", confidence=0.85),
        ]
        reject_agents = [
            _valid_agent("melchior", verdict="reject", confidence=0.85),
            _valid_agent("balthasar", verdict="reject", confidence=0.85),
            _valid_agent("caspar", verdict="reject", confidence=0.85),
        ]
        approve_conf = determine_consensus(approve_agents)["confidence"]
        reject_conf = determine_consensus(reject_agents)["confidence"]
        assert approve_conf == reject_conf


class TestEmptyInputGuard:
    """Verify determine_consensus rejects invalid input lengths."""

    def test_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="at least 2"):
            determine_consensus([])

    def test_single_agent_raises_value_error(self):
        with pytest.raises(ValueError, match="at least 2"):
            determine_consensus([_valid_agent("melchior")])


class TestFindingFieldTypes:
    """Verify that non-string finding fields are rejected."""

    def test_numeric_title_rejected(self):
        data = _valid_agent_data()
        data["findings"] = [
            {"severity": "info", "title": 123, "detail": "Numeric title."},
        ]
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="must be a string"):
                load_agent_output(path)
        finally:
            os.unlink(path)

    def test_null_detail_rejected(self):
        data = _valid_agent_data()
        data["findings"] = [
            {"severity": "info", "title": "OK", "detail": None},
        ]
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="must be a string"):
                load_agent_output(path)
        finally:
            os.unlink(path)


class TestEmptyFindingTitle:
    """Verify that empty or whitespace-only finding titles are rejected."""

    def test_empty_title_rejected(self):
        data = _valid_agent_data()
        data["findings"] = [
            {"severity": "info", "title": "", "detail": "No title."},
        ]
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="empty or whitespace"):
                load_agent_output(path)
        finally:
            os.unlink(path)

    def test_whitespace_title_rejected(self):
        data = _valid_agent_data()
        data["findings"] = [
            {"severity": "info", "title": "   ", "detail": "Blank title."},
        ]
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="empty or whitespace"):
                load_agent_output(path)
        finally:
            os.unlink(path)


class TestDuplicateAgentNameRejection:
    """Verify that duplicate agent names are rejected."""

    def test_duplicate_names_raises_value_error(self):
        agents = [_valid_agent("melchior"), _valid_agent("melchior")]
        with pytest.raises(ValueError, match="Duplicate agent names"):
            determine_consensus(agents)

    def test_three_agents_with_duplicate_raises(self):
        agents = [
            _valid_agent("melchior"),
            _valid_agent("balthasar"),
            _valid_agent("melchior"),
        ]
        with pytest.raises(ValueError, match="Duplicate agent names"):
            determine_consensus(agents)


class TestStringFieldValidation:
    """Verify that top-level string fields are type-checked."""

    def test_numeric_summary_rejected(self):
        data = _valid_agent_data()
        data["summary"] = 42
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="must be a string"):
                load_agent_output(path)
        finally:
            os.unlink(path)

    def test_numeric_reasoning_rejected(self):
        data = _valid_agent_data()
        data["reasoning"] = 123
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="must be a string"):
                load_agent_output(path)
        finally:
            os.unlink(path)

    def test_none_recommendation_rejected(self):
        data = _valid_agent_data()
        data["recommendation"] = None
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="must be a string"):
                load_agent_output(path)
        finally:
            os.unlink(path)

    def test_oversized_field_rejected(self):
        data = _valid_agent_data()
        data["reasoning"] = "x" * 60_000
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="exceeds maximum length"):
                load_agent_output(path)
        finally:
            os.unlink(path)


class TestBoolConfidenceRejected:
    """Verify that boolean values are not accepted as confidence."""

    def test_true_confidence_rejected(self):
        data = _valid_agent_data()
        data["confidence"] = True
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="must be a number, got bool"):
                load_agent_output(path)
        finally:
            os.unlink(path)

    def test_false_confidence_rejected(self):
        data = _valid_agent_data()
        data["confidence"] = False
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="must be a number, got bool"):
                load_agent_output(path)
        finally:
            os.unlink(path)


class TestAgentVerdictTypeGuard:
    """Verify that non-string agent/verdict fields are rejected."""

    def test_list_agent_rejected(self):
        data = _valid_agent_data()
        data["agent"] = ["melchior"]
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="must be a string"):
                load_agent_output(path)
        finally:
            os.unlink(path)

    def test_int_verdict_rejected(self):
        data = _valid_agent_data()
        data["verdict"] = 1
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="must be a string"):
                load_agent_output(path)
        finally:
            os.unlink(path)


class TestZeroWidthUnicodeTitle:
    """Verify that zero-width Unicode characters in titles are rejected."""

    def test_zero_width_space_title_rejected(self):
        data = _valid_agent_data()
        data["findings"] = [
            {"severity": "info", "title": "\u200b", "detail": "Invisible title."},
        ]
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="empty or whitespace"):
                load_agent_output(path)
        finally:
            os.unlink(path)

    def test_bom_only_title_rejected(self):
        data = _valid_agent_data()
        data["findings"] = [
            {"severity": "info", "title": "\ufeff", "detail": "BOM only."},
        ]
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="empty or whitespace"):
                load_agent_output(path)
        finally:
            os.unlink(path)


class TestTitleNormalization:
    """A-2: zero-width characters are stripped before length cap + storage."""

    def test_zero_width_stripped_from_returned_title(self):
        """Returned dict must contain the cleaned title, not the raw form."""
        data = _valid_agent_data()
        data["findings"] = [
            {
                "severity": "info",
                "title": "Hel\u200blo\u200cWo\ufeffrld",
                "detail": "Valid detail.",
            },
        ]
        path = _write_json(data)
        try:
            result = load_agent_output(path)
            assert result["findings"][0]["title"] == "HelloWorld", (
                "Zero-width chars must be stripped from the stored title "
                "to prevent smuggling via invisible Unicode."
            )
        finally:
            os.unlink(path)

    def test_length_cap_applies_to_cleaned_title(self):
        """A title whose cleaned form fits the cap must be accepted even if
        the raw form with zero-width padding exceeds it."""
        # 400 visible + 200 zero-width = raw 600 (> 500), clean 400 (<= 500)
        padded = ("a" * 400) + ("\u200b" * 200)
        data = _valid_agent_data()
        data["findings"] = [
            {"severity": "info", "title": padded, "detail": "OK."},
        ]
        path = _write_json(data)
        try:
            result = load_agent_output(path)
            assert result["findings"][0]["title"] == "a" * 400
        finally:
            os.unlink(path)

    def test_clean_title_over_cap_rejected(self):
        """A cleaned title exceeding the cap must still be rejected."""
        # 501 visible chars, no zero-width — clean length 501 > 500.
        data = _valid_agent_data()
        data["findings"] = [
            {"severity": "info", "title": "a" * 501, "detail": "OK."},
        ]
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="title exceeds maximum"):
                load_agent_output(path)
        finally:
            os.unlink(path)

    def test_title_is_trimmed_of_surrounding_whitespace(self):
        """The stored title must also have its surrounding whitespace stripped."""
        data = _valid_agent_data()
        data["findings"] = [
            {"severity": "info", "title": "  Real title  ", "detail": "OK."},
        ]
        path = _write_json(data)
        try:
            result = load_agent_output(path)
            assert result["findings"][0]["title"] == "Real title"
        finally:
            os.unlink(path)


class TestFindingSubFieldLimits:
    """Verify length limits on finding title and detail."""

    def test_oversized_title_rejected(self):
        data = _valid_agent_data()
        data["findings"] = [
            {"severity": "info", "title": "x" * 600, "detail": "OK."},
        ]
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="title exceeds maximum"):
                load_agent_output(path)
        finally:
            os.unlink(path)

    def test_oversized_detail_rejected(self):
        data = _valid_agent_data()
        data["findings"] = [
            {"severity": "info", "title": "OK", "detail": "x" * 15_000},
        ]
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="detail exceeds maximum"):
                load_agent_output(path)
        finally:
            os.unlink(path)

    def test_too_many_findings_rejected(self):
        data = _valid_agent_data()
        data["findings"] = [
            {"severity": "info", "title": f"Finding {i}", "detail": "Detail."} for i in range(101)
        ]
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="exceeding maximum"):
                load_agent_output(path)
        finally:
            os.unlink(path)


class TestDynamicConsensusLabels:
    """Verify labels reflect actual agent count, not hardcoded (2-1)."""

    def test_three_agent_go_label(self):
        """2 approve + 1 reject = GO (2-1)."""
        agents = [
            _valid_agent("melchior", verdict="approve"),
            _valid_agent("balthasar", verdict="approve"),
            _valid_agent("caspar", verdict="reject"),
        ]
        result = determine_consensus(agents)
        assert result["consensus"] == "GO (2-1)"

    def test_two_agent_hold_label(self):
        """1 approve + 1 reject = HOLD — TIE, not HOLD (2-1)."""
        agents = [
            _valid_agent("melchior", verdict="approve"),
            _valid_agent("balthasar", verdict="reject"),
        ]
        result = determine_consensus(agents)
        assert result["consensus"] == "HOLD -- TIE"
