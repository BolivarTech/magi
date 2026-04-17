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


class TestTopLevelShape:
    """Top-level JSON must be an object; other JSON shapes must surface as ValidationError.

    Without an explicit ``isinstance(data, dict)`` guard, a malformed agent
    output (list, string, null, number) reaches ``set(data.keys())`` and
    raises ``AttributeError`` — bypassing the ``ValidationError`` contract
    the docstring promises and producing an opaque agent failure in
    ``run_orchestrator``.
    """

    @pytest.mark.parametrize(
        "payload",
        [
            [],
            [1, 2, 3],
            "just a string",
            42,
            3.14,
            True,
            None,
        ],
    )
    def test_non_dict_top_level_raises_validation_error(self, payload):
        path = _write_json(payload)
        try:
            with pytest.raises(ValidationError, match="Top-level JSON must be an object"):
                load_agent_output(path)
        finally:
            os.unlink(path)

    def test_non_dict_top_level_preserves_filepath(self):
        path = _write_json([])
        try:
            with pytest.raises(ValidationError) as exc_info:
                load_agent_output(path)
            assert exc_info.value.filepath == path
        finally:
            os.unlink(path)


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
        """Conditional + approve + reject produces GO WITH CAVEATS (2-1)."""
        agents = [
            _valid_agent("melchior", verdict="conditional", confidence=0.8),
            _valid_agent("balthasar", verdict="approve", confidence=0.9),
            _valid_agent("caspar", verdict="reject", confidence=0.7),
        ]
        result = determine_consensus(agents)
        assert result["consensus"] == "GO WITH CAVEATS (2-1)"
        assert result["consensus_verdict"] == "conditional"
        assert len(result["conditions"]) == 1
        assert result["conditions"][0]["agent"] == "melchior"

    def test_conditions_use_summary_not_recommendation(self):
        """Conditions text is sourced from agent.summary, recommendations from agent.recommendation.

        The two fields must be distinct strings in the output so that
        ``## Conditions for Approval`` and ``## Recommended Actions``
        sections do not render the same text twice.
        """
        agents = [
            _valid_agent(
                "melchior",
                verdict="conditional",
                confidence=0.8,
                summary="Add integration tests before merge",
                recommendation="Ship after adding integration tests",
            ),
            _valid_agent("balthasar", verdict="approve", confidence=0.9),
            _valid_agent("caspar", verdict="approve", confidence=0.7),
        ]
        result = determine_consensus(agents)
        assert len(result["conditions"]) == 1
        assert result["conditions"][0]["agent"] == "melchior"
        assert result["conditions"][0]["condition"] == "Add integration tests before merge"
        assert result["recommendations"]["melchior"] == "Ship after adding integration tests"
        assert result["conditions"][0]["condition"] != result["recommendations"]["melchior"]

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
        Has conditions, score > 0 -> GO WITH CAVEATS (3-0): no dissent,
        just unanimous caveats.
        """
        agents = [
            _valid_agent("melchior", verdict="conditional", confidence=0.8),
            _valid_agent("balthasar", verdict="conditional", confidence=0.85),
            _valid_agent("caspar", verdict="conditional", confidence=0.9),
        ]
        result = determine_consensus(agents)
        assert result["consensus"] == "GO WITH CAVEATS (3-0)"
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
        """1 approve + 1 conditional: score = 0.75 -> GO WITH CAVEATS (2-0)."""
        agents = [
            _valid_agent("melchior", verdict="approve", confidence=0.9),
            _valid_agent("balthasar", verdict="conditional", confidence=0.8),
        ]
        result = determine_consensus(agents)
        assert result["consensus"] == "GO WITH CAVEATS (2-0)"

    def test_two_agent_both_conditional(self):
        """2x conditional: score = (0.5+0.5)/2 = 0.5 -> GO WITH CAVEATS (2-0)."""
        agents = [
            _valid_agent("melchior", verdict="conditional", confidence=0.8),
            _valid_agent("balthasar", verdict="conditional", confidence=0.85),
        ]
        result = determine_consensus(agents)
        assert result["consensus"] == "GO WITH CAVEATS (2-0)"
        assert len(result["conditions"]) == 2

    def test_two_approve_one_conditional_is_caveats_3_0(self):
        """B-1: 2 approve + 1 conditional -> GO WITH CAVEATS (3-0).

        All three agents effectively vote ``approve``; the caveat comes
        from the single conditional. The label must surface 3-0 so the
        reader distinguishes this from a split mix like (2-1).
        """
        agents = [
            _valid_agent("melchior", verdict="approve", confidence=0.9),
            _valid_agent("balthasar", verdict="approve", confidence=0.85),
            _valid_agent("caspar", verdict="conditional", confidence=0.8),
        ]
        result = determine_consensus(agents)
        assert result["consensus"] == "GO WITH CAVEATS (3-0)"
        assert result["consensus_verdict"] == "conditional"

    def test_one_approve_two_conditional_is_caveats_3_0(self):
        """B-1: 1 approve + 2 conditional -> GO WITH CAVEATS (3-0)."""
        agents = [
            _valid_agent("melchior", verdict="approve", confidence=0.9),
            _valid_agent("balthasar", verdict="conditional", confidence=0.85),
            _valid_agent("caspar", verdict="conditional", confidence=0.8),
        ]
        result = determine_consensus(agents)
        assert result["consensus"] == "GO WITH CAVEATS (3-0)"

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

    def test_two_agent_conditional_reject_confidence_uses_reject_side(self):
        """Regression: 2-agent ``conditional + reject`` must derive confidence
        from the reject side, not the conditional.

        Pre-fix, the count-based majority picked ``approve`` via alphabetical
        tiebreaking (both sides count 1), so ``majority_agents`` held the
        conditional agent while ``consensus_verdict`` was ``reject``. The
        reported confidence came from the losing side. With the
        consensus-aligned selection, the reject agent's confidence drives
        the number.

        Math: ``score = (0.5 - 1) / 2 = -0.25``; ``weight_factor =
        (0.25 + 1) / 2 = 0.625``; ``base = 0.9 / 2 = 0.45``;
        ``confidence = 0.45 * 0.625 = 0.28125 -> 0.28``.
        """
        agents = [
            _valid_agent("melchior", verdict="conditional", confidence=0.2),
            _valid_agent("balthasar", verdict="reject", confidence=0.9),
        ]
        result = determine_consensus(agents)
        assert result["consensus"] == "HOLD (1-1)"
        assert result["consensus_verdict"] == "reject"
        assert result["confidence"] == 0.28, (
            "Confidence must be computed from the reject agent (0.9), "
            "not the conditional one (0.2), so it matches the "
            "consensus_verdict."
        )
        # Dissent should surface the conditional (non-consensus) agent.
        assert [d["agent"] for d in result["dissent"]] == ["melchior"]

    def test_two_agent_approve_reject_confidence_uses_reject_side(self):
        """Regression: 2-agent ``approve + reject`` tie (score=0) resolves
        to ``reject`` via tie-policy; confidence must follow the reject
        agent, not the approve one picked by alphabetical tiebreaking.

        Math: ``score = 0``; ``weight_factor = 0.5``; ``base = 0.9 / 2
        = 0.45``; ``confidence = 0.45 * 0.5 = 0.225 -> 0.23``.
        """
        agents = [
            _valid_agent("melchior", verdict="approve", confidence=0.2),
            _valid_agent("balthasar", verdict="reject", confidence=0.9),
        ]
        result = determine_consensus(agents)
        assert result["consensus"] == "HOLD -- TIE"
        assert result["consensus_verdict"] == "reject"
        assert result["confidence"] == 0.23
        assert [d["agent"] for d in result["dissent"]] == ["melchior"]

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

    def test_three_agent_two_conditional_one_reject_is_hold_tie(self):
        """Regression (v2.1.1): 2 conditional + 1 reject tie.

        Math: ``score = (0.5 + 0.5 - 1) / 3 = 0`` — exact tie, so the
        banner is ``HOLD -- TIE`` with ``consensus_verdict="reject"``.

        Pre-fix, the count-based majority picked ``approve`` (2 effective
        approve votes vs 1 reject), so ``majority_agents`` held the two
        conditional agents while ``consensus_verdict`` was ``reject``.
        The confidence basis diverged from the verdict basis. This test
        locks the single-source-of-truth invariant: when
        ``consensus_verdict="reject"``, ``majority_agents`` must be the
        agents on the reject side.
        """
        agents = [
            _valid_agent("melchior", verdict="conditional", confidence=0.2),
            _valid_agent("balthasar", verdict="conditional", confidence=0.2),
            _valid_agent("caspar", verdict="reject", confidence=0.9),
        ]
        result = determine_consensus(agents)
        assert result["consensus"] == "HOLD -- TIE"
        assert result["consensus_verdict"] == "reject"
        # Dissent surfaces the two conditionals (the minority side).
        assert sorted(d["agent"] for d in result["dissent"]) == ["balthasar", "melchior"]
        # Confidence derives from the single reject agent (0.9), diluted
        # by the two dissenters and halved by weight_factor=0.5 at tie:
        # base = 0.9 / 3 = 0.3, conf = 0.3 * 0.5 = 0.15.
        assert result["confidence"] == 0.15, (
            "Confidence must come from the reject side (consensus_verdict), "
            "not the conditional side the count-based majority would pick."
        )

    def test_three_agent_two_reject_one_conditional_is_hold_2_1(self):
        """Regression (v2.1.1): 2 reject + 1 conditional.

        Math: ``score = (-1 - 1 + 0.5) / 3 = -0.5`` — negative, not a
        tie. Banner renders ``HOLD (2-1)`` with the split derived from
        the reject side (2) vs the conditional side (1). Confidence uses
        the two reject agents.
        """
        agents = [
            _valid_agent("melchior", verdict="reject", confidence=0.8),
            _valid_agent("balthasar", verdict="reject", confidence=0.9),
            _valid_agent("caspar", verdict="conditional", confidence=0.2),
        ]
        result = determine_consensus(agents)
        assert result["consensus"] == "HOLD (2-1)"
        assert result["consensus_verdict"] == "reject"
        # Majority_agents invariant: on reject consensus, majority is the
        # reject side. Conditions from the conditional agent still appear
        # in ``conditions`` for the operator.
        assert sorted(d["agent"] for d in result["dissent"]) == ["caspar"]
        assert [c["agent"] for c in result["conditions"]] == ["caspar"]

    def test_consensus_side_invariant_across_vectors(self):
        """Banner split, consensus_verdict, majority_agents, and confidence
        basis must all agree with each other on every vector tested.

        This locks the single-source-of-truth fix at the contract level:
        whatever logic decides ``consensus_verdict`` must also drive the
        agents that appear in ``majority_agents`` and the counts that
        appear in the rendered split label.
        """
        vectors = [
            # [verdicts], expected consensus label, expected verdict
            (["conditional", "reject"], "HOLD (1-1)", "reject"),
            (["conditional", "conditional", "reject"], "HOLD -- TIE", "reject"),
            (["reject", "reject", "conditional"], "HOLD (2-1)", "reject"),
            (["approve", "approve", "reject"], "GO (2-1)", "approve"),
            (["approve", "conditional", "reject"], "GO WITH CAVEATS (2-1)", "conditional"),
        ]
        for verdicts, expected_label, expected_short in vectors:
            agents = [
                _valid_agent(f"agent{i}", verdict=v)
                # agent0/1/2 are not valid agent names, so remap to real ones.
                for i, v in enumerate(verdicts)
            ]
            real_names = ["melchior", "balthasar", "caspar"][: len(verdicts)]
            for agent, name in zip(agents, real_names):
                agent["agent"] = name
            result = determine_consensus(agents)
            assert result["consensus"] == expected_label, (
                f"Vector {verdicts}: expected {expected_label}, got {result['consensus']}"
            )
            assert result["consensus_verdict"] == expected_short
            # Partition invariant: every agent is in exactly one bucket.
            majority_names = set(result["votes"].keys()) - {d["agent"] for d in result["dissent"]}
            dissent_names = {d["agent"] for d in result["dissent"]}
            assert majority_names.isdisjoint(dissent_names)
            assert majority_names | dissent_names == set(result["votes"].keys())
            # Side invariant: majority names have verdicts consistent with
            # consensus_verdict ("conditional" effectively maps to approve).
            side = "reject" if expected_short == "reject" else "approve"
            for agent in agents:
                eff = "approve" if agent["verdict"] == "conditional" else agent["verdict"]
                is_majority = agent["agent"] in majority_names
                assert (eff == side) == is_majority, (
                    f"Vector {verdicts}: agent {agent['agent']} "
                    f"(verdict={agent['verdict']}) partition disagrees with "
                    f"consensus_side={side}"
                )


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

    def test_dedup_merges_nfc_and_nfd_equivalents(self):
        """A-3: titles that differ only in Unicode normalization must merge.

        ``Café`` written as NFC (precomposed U+00E9) and NFD (``e`` + U+0301)
        are canonically equivalent and must be treated as the same finding.
        """
        agents = [
            _valid_agent(
                "melchior",
                findings=[
                    {
                        "severity": "warning",
                        "title": "Caf\u00e9 de-sync",  # NFC
                        "detail": "Precomposed.",
                    },
                ],
            ),
            _valid_agent(
                "balthasar",
                findings=[
                    {
                        "severity": "warning",
                        "title": "Cafe\u0301 de-sync",  # NFD
                        "detail": "Decomposed.",
                    },
                ],
            ),
            _valid_agent("caspar", findings=[]),
        ]
        result = determine_consensus(agents)
        cafe_findings = [f for f in result["findings"] if "caf" in f["title"].lower()]
        assert len(cafe_findings) == 1, (
            f"NFC and NFD equivalents must merge, got {len(cafe_findings)}: "
            f"{[f['title'] for f in cafe_findings]}"
        )
        assert sorted(cafe_findings[0]["sources"]) == ["balthasar", "melchior"]

    def test_dedup_uses_full_unicode_casefold(self):
        """A-3: German eszett ``ß`` must casefold to ``ss``.

        ``str.lower()`` leaves ``ß`` as-is, so ``STRASSE`` and ``straße``
        are distinct under ``lower()``. Under ``str.casefold()`` both
        become ``strasse`` and the findings must merge.
        """
        agents = [
            _valid_agent(
                "melchior",
                findings=[
                    {
                        "severity": "warning",
                        "title": "STRASSE parser",
                        "detail": "Upper.",
                    },
                ],
            ),
            _valid_agent(
                "balthasar",
                findings=[
                    {
                        "severity": "warning",
                        "title": "stra\u00dfe parser",  # straße
                        "detail": "With eszett.",
                    },
                ],
            ),
            _valid_agent("caspar", findings=[]),
        ]
        result = determine_consensus(agents)
        strasse_findings = [f for f in result["findings"] if "parser" in f["title"].lower()]
        assert len(strasse_findings) == 1, "STRASSE and straße must merge under full casefold"
        assert sorted(strasse_findings[0]["sources"]) == ["balthasar", "melchior"]

    def test_dedup_merges_nfkc_compatibility_forms(self):
        """A-3: NFKC compatibility forms (fullwidth vs halfwidth) must merge."""
        agents = [
            _valid_agent(
                "melchior",
                findings=[
                    {
                        "severity": "warning",
                        "title": "\uff33\uff31\uff2c Injection",  # ＳＱＬ
                        "detail": "Fullwidth.",
                    },
                ],
            ),
            _valid_agent(
                "balthasar",
                findings=[
                    {
                        "severity": "warning",
                        "title": "SQL Injection",
                        "detail": "Halfwidth.",
                    },
                ],
            ),
            _valid_agent("caspar", findings=[]),
        ]
        result = determine_consensus(agents)
        sql_findings = [f for f in result["findings"] if "injection" in f["title"].lower()]
        assert len(sql_findings) == 1, "Fullwidth and halfwidth variants must merge under NFKC"
        assert sorted(sql_findings[0]["sources"]) == ["balthasar", "melchior"]

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

    def test_banner_width_guard_truncates_overlong_label(self):
        """Regression (v2.1.1): a label that exceeds ``_BANNER_INNER``
        must be truncated with ``...`` so the border column does not
        slide.

        Before the fix, ``str.ljust`` never truncated, so a long custom
        agent role name produced a row wider than the ``+===+`` borders
        and silently violated the MANDATORY FINAL OUTPUT CONTRACT.
        """
        from reporting import AGENT_TITLES

        # Patch the title dict for one agent to a pathologically long
        # role, run the banner, then undo the patch.
        original = AGENT_TITLES.get("melchior")
        try:
            AGENT_TITLES["melchior"] = (
                "VeryLongMelchior",
                "AnExtremelyDescriptiveAndExcessivelyLongRoleName",
            )
            agents = [_valid_agent(n) for n in ["melchior", "balthasar", "caspar"]]
            consensus = determine_consensus(agents)
            banner = format_banner(agents, consensus)
            lines = banner.split("\n")
            # Every row must still be exactly 52 characters — the width
            # invariant the rest of the banner tests assert on.
            for line in lines:
                assert len(line) == 52, (
                    f"Overlong label broke banner width invariant: "
                    f"got len={len(line)} line={line!r}"
                )
            # The overlong row must end with the ellipsis marker just
            # inside the closing ``|``.
            assert "..." in lines[3]
        finally:
            if original is None:
                AGENT_TITLES.pop("melchior", None)
            else:
                AGENT_TITLES["melchior"] = original

    def test_banner_overlong_label_preserves_verdict_and_confidence(self):
        """Regression (v2.1.3): when a label overflows ``_BANNER_INNER``
        the truncation must eat the label, not the verdict.

        Pre-2.1.3 the ``_fit_content`` tail-truncation erased the
        verdict/confidence suffix, producing rows like ``| Long (Long
        Role): APPRO...|`` where the operator could no longer read the
        agent's decision. The new contract: the ``{VERDICT} ({NN%})``
        suffix is always preserved; the label is truncated with ``...``
        instead.
        """
        from reporting import AGENT_TITLES

        original = AGENT_TITLES.get("melchior")
        try:
            AGENT_TITLES["melchior"] = (
                "VeryLongMelchior",
                "AnExtremelyDescriptiveAndExcessivelyLongRoleName",
            )
            agents = [_valid_agent(n) for n in ["melchior", "balthasar", "caspar"]]
            agents[0]["verdict"] = "approve"
            agents[0]["confidence"] = 0.85
            consensus = determine_consensus(agents)
            banner = format_banner(agents, consensus)
            lines = banner.split("\n")
            # Width invariant: every line must remain exactly 52 chars.
            for line in lines:
                assert len(line) == 52, (
                    f"Overlong label broke width invariant: len={len(line)} line={line!r}"
                )
            melchior_row = lines[3]
            # Verdict and confidence tokens must both survive intact.
            assert "APPROVE" in melchior_row, f"Verdict token was truncated away: {melchior_row!r}"
            assert "(85%)" in melchior_row, f"Confidence token was truncated away: {melchior_row!r}"
            # Truncation marker must sit inside the label half, not at
            # the row tail.
            assert "..." in melchior_row
            assert not melchior_row.rstrip(" |").endswith("..."), (
                f"Ellipsis landed on the tail, erasing the verdict: {melchior_row!r}"
            )
        finally:
            if original is None:
                AGENT_TITLES.pop("melchior", None)
            else:
                AGENT_TITLES["melchior"] = original

    def test_banner_width_guard_truncates_overlong_consensus_label(self):
        """Regression (v2.1.1): a consensus label longer than the budget
        must also be truncated so the CONSENSUS row stays inside the
        border column.
        """
        agents = [_valid_agent(n) for n in ["melchior", "balthasar", "caspar"]]
        consensus = determine_consensus(agents)
        # Simulate a pathologically long consensus label (future custom
        # formats, operator-supplied overrides, etc.).
        consensus["consensus"] = "SOME_EXTREMELY_LONG_CONSENSUS_LABEL_" * 3
        banner = format_banner(agents, consensus)
        for line in banner.split("\n"):
            assert len(line) == 52
        cons_line = [ln for ln in banner.split("\n") if "CONSENSUS:" in ln][0]
        assert cons_line.endswith("...|")


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


class TestCleanTitlePublicAPI:
    """``clean_title`` must be exposed as a public symbol on ``validate``.

    ``consensus.py`` builds its dedup key from the same normalization
    source as :func:`load_agent_output`. Importing a private underscored
    symbol across modules is a code smell and hides the contract from
    third-party consumers, so the helper must be public.
    """

    def test_clean_title_is_importable_from_validate(self):
        from validate import clean_title  # must not raise

        assert callable(clean_title)
        assert clean_title("  hello\u200bworld  ") == "helloworld"

    def test_clean_title_is_reexported_from_synthesize(self):
        from synthesize import clean_title  # must not raise

        assert callable(clean_title)


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

    def test_newline_in_title_collapsed_to_space(self):
        """Regression (v2.1.1): a newline in a finding title must never
        reach the rendered output — otherwise it corrupts the fixed-column
        marker/severity/title layout of ``_format_finding_line`` and can
        smuggle content that reads as an extra finding row.
        """
        from validate import clean_title

        assert clean_title("Broken line\ninjected second row") == (
            "Broken line injected second row"
        )

    def test_all_control_whitespace_stripped_from_title(self):
        """``\\t``, ``\\n``, ``\\r``, ``\\v``, ``\\f``, and NEL (U+0085)
        must all be removed/collapsed by :func:`clean_title`.
        """
        from validate import clean_title

        # Each control character separately would otherwise land in the
        # rendered row and break alignment.
        raw = "a\tb\nc\rd\ve\ff\u0085g"
        cleaned = clean_title(raw)
        assert "\t" not in cleaned and "\n" not in cleaned and "\r" not in cleaned
        assert "\v" not in cleaned and "\f" not in cleaned and "\u0085" not in cleaned
        # All the visible letters survive in order.
        assert cleaned.replace(" ", "") == "abcdefg"

    def test_title_with_only_control_whitespace_rejected(self):
        """A title that is purely control whitespace must fail validation
        (empty after cleaning), not pass through as a malformed row.
        """
        data = _valid_agent_data()
        data["findings"] = [
            {"severity": "info", "title": "\t\n\r", "detail": "Invisible title."},
        ]
        path = _write_json(data)
        try:
            with pytest.raises(ValidationError, match="empty or whitespace"):
                load_agent_output(path)
        finally:
            os.unlink(path)

    def test_word_joiner_stripped(self):
        """Regression (v2.1.2): ``U+2060`` (WORD JOINER) is invisible
        and must be stripped. Without this, two titles that look
        identical in the rendered report can still produce different
        dedup keys, smuggling a duplicate finding past
        :func:`consensus._dedup_key`.
        """
        from validate import clean_title

        assert clean_title("Hel\u2060lo") == "Hello"

    def test_invisible_math_operators_stripped(self):
        """Regression (v2.1.2): ``U+2061`` (FUNCTION APPLICATION),
        ``U+2062`` (INVISIBLE TIMES), ``U+2063`` (INVISIBLE
        SEPARATOR), and ``U+2064`` (INVISIBLE PLUS) must all be
        stripped. Each is Cf-category invisible and exposes the
        same dedup-key smuggling surface as the zero-width spaces
        already covered.
        """
        from validate import clean_title

        raw = "a\u2061b\u2062c\u2063d\u2064e"
        assert clean_title(raw) == "abcde"

    def test_full_2060_to_206f_block_stripped(self):
        """Regression (v2.1.2): every character in ``U+2060-U+206F``
        must be stripped. The block contains the word joiner, the
        four invisible mathematical operators, the deprecated
        formatting characters (``U+2065-U+2069``), and the four
        deprecated language-tag controls (``U+206A-U+206F``). All
        are Cf-category invisible and share the same smuggling
        surface — testing the boundary catches a partial regex
        regression that would otherwise let only the unmentioned
        codepoints leak through.
        """
        from validate import clean_title

        block = "".join(chr(cp) for cp in range(0x2060, 0x2070))
        assert clean_title(f"x{block}y") == "xy"

    def test_title_with_newline_stored_in_cleaned_form(self):
        """The stored title must be the cleaned form, not the raw one —
        consumers writing ``_format_finding_line`` must never see the
        control whitespace.
        """
        data = _valid_agent_data()
        data["findings"] = [
            {
                "severity": "info",
                "title": "Row\nwith\tinjection",
                "detail": "OK.",
            },
        ]
        path = _write_json(data)
        try:
            result = load_agent_output(path)
            assert "\n" not in result["findings"][0]["title"]
            assert "\t" not in result["findings"][0]["title"]
            assert result["findings"][0]["title"] == "Row with injection"
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
