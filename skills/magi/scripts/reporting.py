#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 2.0.2
# Date: 2026-04-14
"""MAGI report formatting.

Generates the ASCII verdict banner and the full human-readable
markdown report from agent outputs and consensus data.

All output is ASCII-only (no multi-byte glyphs) so that box alignment
is stable across terminals and the report format is invariant across
parallel and fallback execution modes.
"""

from __future__ import annotations

from typing import Any

AGENT_TITLES: dict[str, tuple[str, str]] = {
    "melchior": ("Melchior", "Scientist"),
    "balthasar": ("Balthasar", "Pragmatist"),
    "caspar": ("Caspar", "Critic"),
}

# Banner layout constants.
_BANNER_WIDTH: int = 52
_BANNER_INNER: int = _BANNER_WIDTH - 2  # 50 characters between the borders

# Findings layout constants.
# Marker column is wide enough for ``[!!!]`` (5 chars); severity column
# is wide enough for ``**[CRITICAL]**`` (14 chars).
_FINDING_MARKER_WIDTH: int = 5
_FINDING_SEVERITY_WIDTH: int = 14

_SEVERITY_MARKERS: dict[str, str] = {
    "critical": "[!!!]",
    "warning": "[!!]",
    "info": "[i]",
}


def _agent_title(agent_name: str) -> tuple[str, str]:
    """Look up agent display name and role, with fallback for unknown agents.

    Args:
        agent_name: Agent identifier (e.g., 'melchior').

    Returns:
        Tuple of (display name, role title).
    """
    return AGENT_TITLES.get(agent_name, (agent_name.capitalize(), "Agent"))


def _agent_label(agent_name: str) -> str:
    """Return the ``Name (Title):`` label used in the banner."""
    name, title = _agent_title(agent_name)
    return f"{name} ({title}):"


def format_banner(agents: list[dict[str, Any]], consensus: dict[str, Any]) -> str:
    """Generate the MAGI verdict banner with consistent alignment.

    Produces an ASCII box of fixed width (52 columns) containing agent
    verdicts and the consensus result. Verdicts are column-aligned by
    padding each agent label to the longest label width so that the
    verdict column starts at the same position on every row.

    Args:
        agents: List of validated agent output dictionaries.
        consensus: Consensus dictionary produced by ``determine_consensus``.

    Returns:
        Multi-line string with the formatted banner. Every line has
        exactly ``_BANNER_WIDTH`` characters.
    """
    labels = [_agent_label(a["agent"]) for a in agents]
    max_label_len = max((len(label) for label in labels), default=0)

    lines: list[str] = []
    border = "+" + "=" * _BANNER_INNER + "+"
    lines.append(border)
    lines.append("|" + "MAGI SYSTEM -- VERDICT".center(_BANNER_INNER) + "|")
    lines.append(border)

    for agent, label in zip(agents, labels):
        verdict_display = agent["verdict"].upper()
        conf_pct = f"{agent['confidence']:.0%}"
        content = f"  {label:<{max_label_len}} {verdict_display} ({conf_pct})"
        lines.append("|" + content.ljust(_BANNER_INNER) + "|")

    lines.append(border)
    cons_content = f"  CONSENSUS: {consensus['consensus']}"
    lines.append("|" + cons_content.ljust(_BANNER_INNER) + "|")
    lines.append(border)

    return "\n".join(lines)


def _format_finding_line(finding: dict[str, Any]) -> str:
    """Format a single finding row with fixed-width marker and severity.

    Layout::

        [!!!] **[CRITICAL]** Title here _(from agent1, agent2)_
        [!!]  **[WARNING]**  Title here _(from agent1)_
        [i]   **[INFO]**     Title here _(from agent1)_

    The marker column is padded to ``_FINDING_MARKER_WIDTH`` and the
    severity label column to ``_FINDING_SEVERITY_WIDTH`` so that the
    title text starts at the same column on every row regardless of
    severity length.

    Args:
        finding: Finding dict with ``severity``, ``title``, and
            optional ``sources`` keys.

    Returns:
        Single-line formatted string (no trailing newline).
    """
    severity = finding["severity"]
    marker = _SEVERITY_MARKERS.get(severity, "[?]")
    severity_label = f"**[{severity.upper()}]**"
    sources = ", ".join(finding.get("sources", ["unknown"]))
    return (
        f"{marker:<{_FINDING_MARKER_WIDTH}} "
        f"{severity_label:<{_FINDING_SEVERITY_WIDTH}} "
        f"{finding['title']} _(from {sources})_"
    )


def format_report(agents: list[dict[str, Any]], consensus: dict[str, Any]) -> str:
    """Generate the full human-readable report.

    The report enforces the canonical MAGI output format:

    1. Banner (from :func:`format_banner`)
    2. ``## Key Findings`` — one aligned row per deduplicated finding
    3. ``## Dissenting Opinion`` — minority agents' one-line summary (if any)
    4. ``## Conditions for Approval`` — conditional agents' ``condition`` text (if any)
    5. ``## Recommended Actions`` — one bullet per agent recommendation

    Sections 2, 3, and 4 are omitted when empty. Section 5 is always
    present.

    Args:
        agents: List of validated agent output dictionaries.
        consensus: Consensus dictionary produced by ``determine_consensus``.

    Returns:
        Multi-line markdown string.
    """
    sections: list[str] = [format_banner(agents, consensus), ""]

    if consensus["findings"]:
        sections.append("## Key Findings")
        for finding in consensus["findings"]:
            sections.append(_format_finding_line(finding))
        sections.append("")

    if consensus["dissent"]:
        sections.append("## Dissenting Opinion")
        for dissent in consensus["dissent"]:
            name, title = _agent_title(dissent["agent"])
            sections.append(f"**{name} ({title})**: {dissent['summary']}")
        sections.append("")

    if consensus["conditions"]:
        sections.append("## Conditions for Approval")
        for cond in consensus["conditions"]:
            name, _ = _agent_title(cond["agent"])
            sections.append(f"- **{name}**: {cond['condition']}")
        sections.append("")

    sections.append("## Recommended Actions")
    for agent_name, rec in consensus["recommendations"].items():
        name, title = _agent_title(agent_name)
        sections.append(f"- **{name}** ({title}): {rec}")

    return "\n".join(sections)
