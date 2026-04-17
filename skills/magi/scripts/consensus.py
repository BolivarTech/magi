#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 2.1.1
# Date: 2026-04-17
"""MAGI consensus engine.

Applies weight-based scoring to agent verdicts and produces a unified
consensus with confidence calculation, findings deduplication, and
dissent tracking.
"""

from __future__ import annotations

import unicodedata
from typing import Any

from validate import clean_title

VERDICT_WEIGHT: dict[str, float] = {
    "approve": 1,
    "conditional": 0.5,
    "reject": -1,
}

_SEVERITY_ORDER: dict[str, int] = {"critical": 0, "warning": 1, "info": 2}
_UNKNOWN_SEVERITY_RANK = 99
_EPSILON: float = 1e-9


def _severity_rank(severity: str) -> int:
    """Return the sort rank of *severity* (0=critical, 2=info, 99=unknown)."""
    return _SEVERITY_ORDER.get(severity, _UNKNOWN_SEVERITY_RANK)


def _dedup_key(title: str) -> str:
    """Return the canonical key used to merge findings with the same title.

    Applies, in order:

    1. :func:`validate.clean_title` — strips invisible characters (zero-width,
       bidi marks, BOM, soft hyphen) and surrounding whitespace.
    2. ``unicodedata.normalize("NFKC", ...)`` — collapses compatibility forms
       (fullwidth/halfwidth, ligatures) and combines canonically equivalent
       sequences (precomposed vs combining accents).
    3. :meth:`str.casefold` — full Unicode case folding, strictly stronger
       than ``str.lower`` (e.g. ``ß`` → ``ss``).

    The result is an internal lookup key; the displayed title preserves the
    original form of the first finding seen under each key.
    """
    return unicodedata.normalize("NFKC", clean_title(title)).casefold()


def _consensus_short_verdict(score: float, has_conditions: bool) -> str:
    """Derive the consensus short verdict from *score* alone.

    This is split-independent by design. The caller uses it to pick the
    consensus side (approve/reject), partitions agents based on that
    side, and only then derives the ``(N-M)`` split — guaranteeing that
    the rendered label, ``majority_agents``, and ``_compute_confidence``
    all reference the same side.

    Tie policy: ``score == 0`` maps to ``reject`` (safer default).

    Args:
        score: Normalized weighted score in [-1.0, 1.0].
        has_conditions: Whether any agent voted 'conditional'.

    Returns:
        One of ``"approve"``, ``"reject"``, or ``"conditional"``.
    """
    if abs(score - 1.0) < _EPSILON:
        return "approve"
    if abs(score - (-1.0)) < _EPSILON:
        return "reject"
    is_positive = score > _EPSILON
    if is_positive and has_conditions:
        return "conditional"
    if is_positive:
        return "approve"
    # Tie (abs(score) < eps) and negative both default to reject.
    return "reject"


def _format_consensus_label(
    score: float,
    consensus_short: str,
    split: tuple[int, int],
) -> str:
    """Render the consensus label shown on the banner.

    Args:
        score: Normalized weighted score in [-1.0, 1.0].
        consensus_short: Short verdict from :func:`_consensus_short_verdict`.
        split: ``(majority_count, minority_count)`` derived from the
            agents partitioned by ``consensus_side`` — i.e., the counts
            are always taken from the same side as ``consensus_short``.

    Returns:
        The rendered consensus label (e.g., ``"GO (2-1)"``,
        ``"HOLD -- TIE"``, ``"STRONG NO-GO"``).
    """
    if abs(score - 1.0) < _EPSILON:
        return "STRONG GO"
    if abs(score - (-1.0)) < _EPSILON:
        return "STRONG NO-GO"
    is_tie = abs(score) < _EPSILON
    if is_tie:
        return "HOLD -- TIE"
    split_label = f"({split[0]}-{split[1]})"
    if consensus_short == "conditional":
        return f"GO WITH CAVEATS {split_label}"
    if consensus_short == "approve":
        return f"GO {split_label}"
    return f"HOLD {split_label}"


def _deduplicate_findings(agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge findings across agents, deduplicating by normalized title.

    Two findings are considered the same when their titles collapse to the
    same :func:`_dedup_key` — i.e. they are equivalent under NFKC
    normalization and full Unicode case folding after invisible-character
    stripping. When a collision occurs, the displayed title preserves the
    form first seen in agent iteration order; the highest severity among
    the colliding findings wins and each reporting agent is recorded in a
    ``sources`` list.

    Args:
        agents: List of validated agent output dictionaries.

    Returns:
        Deduplicated findings sorted by severity (critical first).
    """
    findings_by_title: dict[str, dict[str, Any]] = {}
    for a in agents:
        for f in a.get("findings", []):
            title_key = _dedup_key(f["title"])
            existing = findings_by_title.get(title_key)
            if existing is None:
                findings_by_title[title_key] = {**f, "sources": [a["agent"]]}
                continue
            existing["sources"].append(a["agent"])
            if _severity_rank(f["severity"]) < _severity_rank(existing["severity"]):
                existing["severity"] = f["severity"]
                existing["detail"] = f["detail"]

    return sorted(findings_by_title.values(), key=lambda f: _severity_rank(f["severity"]))


def _compute_confidence(
    majority_agents: list[dict[str, Any]],
    num_agents: int,
    score: float,
) -> float:
    """Calculate consensus confidence from majority agent confidences.

    Formula::

        base_confidence = sum(majority_agent.confidence) / num_agents
        weight_factor   = (abs(score) + 1) / 2
        confidence      = clamp(base_confidence * weight_factor, 0.0, 1.0)

    The denominator is ``num_agents`` (not ``len(majority_agents)``) by
    design: a minority that disagrees dilutes the numerator, so a
    unanimous win yields a higher confidence than a bare-majority one
    even when the surviving side's own confidence is identical. This
    is the "dissent dilution" term — operators reading a moderate
    confidence on a narrow win should treat it as "the split itself
    reduces certainty", not as "the majority is individually uncertain".
    ``weight_factor`` then independently scales by how far the score is
    from zero, so a unanimous approve (score=1) and unanimous reject
    (score=-1) both land at the same high confidence, while an exact
    tie (score=0) halves confidence regardless of individual votes.

    Args:
        majority_agents: Agents on the majority side (as determined by
            the consensus-aligned partition in :func:`determine_consensus`).
        num_agents: Total number of agents, including dissenters.
        score: Normalized weighted score in [-1.0, 1.0].

    Returns:
        Confidence value clamped to [0.0, 1.0], rounded to 2 decimals.
    """
    majority_conf: float = sum(a["confidence"] for a in majority_agents)
    base_confidence = majority_conf / num_agents
    weight_factor = (abs(score) + 1) / 2
    return float(round(max(0.0, min(1.0, base_confidence * weight_factor)), 2))


def determine_consensus(agents: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply weight-based scoring to determine consensus.

    Uses VERDICT_WEIGHT to compute a normalized score, then maps to
    consensus labels via thresholds.

    Args:
        agents: List of validated agent output dictionaries (minimum 2).

    Returns:
        Dictionary with keys ``consensus``, ``consensus_verdict``,
        ``confidence``, ``votes``, ``majority_summary``, ``dissent``,
        ``findings``, ``conditions``, and ``recommendations``.

    Raises:
        ValueError: If fewer than 2 agents are provided or agent names
            are not unique.
    """
    num_agents = len(agents)
    if num_agents < 2:
        raise ValueError(f"determine_consensus requires at least 2 agents, got {num_agents}")

    agent_names = [a["agent"] for a in agents]
    if len(agent_names) != len(set(agent_names)):
        raise ValueError(f"Duplicate agent names detected: {agent_names}")

    verdicts = [a["verdict"] for a in agents]
    score = sum(VERDICT_WEIGHT[v] for v in verdicts) / num_agents
    has_conditions = "conditional" in verdicts

    # Invariant: ``consensus_short`` is derived from ``score`` alone, then
    # ``consensus_side`` selects the agents, and ``split`` is derived from
    # that partition. All three — rendered label, ``majority_agents``, and
    # the input to ``_compute_confidence`` — reference the same side. This
    # replaces the pre-2.1.1 flow where ``split`` came from a count-based
    # majority with alphabetical tie-break, which could point at the
    # opposite side from ``consensus_verdict`` on vectors like
    # ``[conditional, reject]`` or ``[conditional, conditional, reject]``.
    consensus_short = _consensus_short_verdict(score, has_conditions)
    consensus_side = "reject" if consensus_short == "reject" else "approve"

    majority_agents = []
    dissent_agents = []
    for a in agents:
        eff = "approve" if a["verdict"] == "conditional" else a["verdict"]
        if eff == consensus_side:
            majority_agents.append(a)
        else:
            dissent_agents.append(a)

    split = (len(majority_agents), len(dissent_agents))
    consensus = _format_consensus_label(score, consensus_short, split)

    all_findings = _deduplicate_findings(agents)

    # Conditions are sourced from ``summary`` (short one-liner stating the
    # blocking condition) so they render distinctly from the
    # ``recommendations`` section, which uses ``recommendation`` (full
    # next-step action).
    conditions = [
        {"agent": a["agent"], "condition": a["summary"]}
        for a in agents
        if a["verdict"] == "conditional"
    ]

    confidence = _compute_confidence(majority_agents, num_agents, score)

    return {
        "consensus": consensus,
        "consensus_verdict": consensus_short,
        "confidence": confidence,
        "votes": {a["agent"]: a["verdict"] for a in agents},
        "majority_summary": " | ".join(
            f"{a['agent'].capitalize()}: {a['summary']}" for a in majority_agents
        ),
        "dissent": [
            {"agent": a["agent"], "summary": a["summary"], "reasoning": a["reasoning"]}
            for a in dissent_agents
        ],
        "findings": all_findings,
        "conditions": conditions,
        "recommendations": {a["agent"]: a["recommendation"] for a in agents},
    }
