#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.1.0
# Date: 2026-04-01
"""MAGI consensus engine.

Applies weight-based scoring to agent verdicts and produces a unified
consensus with confidence calculation, findings deduplication, and
dissent tracking.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

VERDICT_WEIGHT: dict[str, float] = {
    "approve": 1,
    "conditional": 0.5,
    "reject": -1,
}

_SEVERITY_ORDER: dict[str, int] = {"critical": 0, "warning": 1, "info": 2}
_EPSILON: float = 1e-9


def _classify_consensus(
    score: float,
    has_conditions: bool,
    effective_verdicts: list[str],
    majority_verdict: str,
    num_agents: int,
) -> tuple[str, str]:
    """Map weighted score to a consensus label and short verdict.

    Args:
        score: Normalized weighted score in [-1.0, 1.0].
        has_conditions: Whether any agent voted 'conditional'.
        effective_verdicts: Verdicts with 'conditional' mapped to 'approve'.
        majority_verdict: The most common effective verdict.
        num_agents: Total number of agents.

    Returns:
        Tuple of (consensus label, short verdict).
    """
    if abs(score - 1.0) < _EPSILON:
        return "STRONG GO", "approve"
    if abs(score - (-1.0)) < _EPSILON:
        return "STRONG NO-GO", "reject"
    if score > _EPSILON and has_conditions:
        return "GO WITH CAVEATS", "conditional"

    majority_count = sum(1 for v in effective_verdicts if v == majority_verdict)
    minority_count = num_agents - majority_count

    if score > _EPSILON:
        return f"GO ({majority_count}-{minority_count})", "approve"
    if abs(score) < _EPSILON:
        return "HOLD -- TIE", "reject"
    return f"HOLD ({majority_count}-{minority_count})", "reject"


def _deduplicate_findings(agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge findings across agents, deduplicating by title.

    Findings with the same title (case-insensitive) are merged: the
    highest severity is kept, and all reporting agents are tracked in
    a ``sources`` list.

    Args:
        agents: List of validated agent output dictionaries.

    Returns:
        Deduplicated findings sorted by severity (critical first).
    """
    findings_by_title: dict[str, dict[str, Any]] = {}
    for a in agents:
        for f in a.get("findings", []):
            title_key = f["title"].lower().strip()
            if title_key in findings_by_title:
                existing = findings_by_title[title_key]
                existing["sources"].append(a["agent"])
                if _SEVERITY_ORDER.get(f["severity"], 99) < _SEVERITY_ORDER.get(
                    existing["severity"], 99
                ):
                    existing["severity"] = f["severity"]
                    existing["detail"] = f["detail"]
            else:
                findings_by_title[title_key] = {
                    **f,
                    "sources": [a["agent"]],
                }

    return sorted(
        findings_by_title.values(),
        key=lambda f: _SEVERITY_ORDER.get(f["severity"], 99),
    )


def _compute_confidence(
    majority_agents: list[dict[str, Any]],
    num_agents: int,
    score: float,
) -> float:
    """Calculate consensus confidence from majority agent confidences.

    Uses ``abs(score)`` so both unanimous approve and unanimous reject
    produce high confidence.  At score=0 (exact tie), weight_factor=0.5,
    halving confidence — appropriate for an undecided split.

    Args:
        majority_agents: Agents on the majority side.
        num_agents: Total number of agents.
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

    effective_verdicts = ["approve" if v == "conditional" else v for v in verdicts]
    # Sort by count descending, then by verdict name ascending for deterministic
    # tie-breaking when counts are equal (e.g., 1 approve + 1 reject).
    verdict_counts = Counter(effective_verdicts)
    majority_verdict = sorted(verdict_counts.keys(), key=lambda v: (-verdict_counts[v], v))[0]

    consensus, consensus_short = _classify_consensus(
        score, has_conditions, effective_verdicts, majority_verdict, num_agents
    )

    majority_agents = []
    dissent_agents = []
    for a in agents:
        eff = "approve" if a["verdict"] == "conditional" else a["verdict"]
        if eff == majority_verdict:
            majority_agents.append(a)
        else:
            dissent_agents.append(a)

    all_findings = _deduplicate_findings(agents)

    conditions = [
        {"agent": a["agent"], "condition": a["recommendation"]}
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
