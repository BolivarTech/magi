#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-04-01
"""MAGI report formatting.

Generates the ASCII verdict banner and the full human-readable
markdown report from agent outputs and consensus data.
"""

from __future__ import annotations

from typing import Any

AGENT_TITLES: dict[str, tuple[str, str]] = {
    "melchior": ("Melchior", "Scientist"),
    "balthasar": ("Balthasar", "Pragmatist"),
    "caspar": ("Caspar", "Critic"),
}


def format_banner(agents: list[dict[str, Any]], consensus: dict[str, Any]) -> str:
    """Generate the MAGI verdict banner with consistent alignment.

    Produces an ASCII box of fixed width containing agent verdicts and the
    consensus result.  Uses only ASCII characters to avoid multi-byte
    alignment issues (e.g. the em dash ``\\u2014`` is wider than one column
    in many terminals).

    Args:
        agents: List of validated agent output dictionaries.
        consensus: Consensus dictionary produced by ``determine_consensus``.

    Returns:
        Multi-line string with the formatted banner.
    """
    width = 52
    inner = width - 2

    lines = []
    lines.append("+" + "=" * inner + "+")
    lines.append("|" + "MAGI SYSTEM -- VERDICT".center(inner) + "|")
    lines.append("+" + "=" * inner + "+")

    for a in agents:
        name, title = AGENT_TITLES[a["agent"]]
        verdict_display = a["verdict"].upper()
        conf = f"{a['confidence']:.0%}"
        content = f"  {name} ({title}):  {verdict_display} ({conf})"
        lines.append("|" + content.ljust(inner) + "|")

    lines.append("+" + "=" * inner + "+")
    cons_content = f"  CONSENSUS: {consensus['consensus']}"
    lines.append("|" + cons_content.ljust(inner) + "|")
    lines.append("+" + "=" * inner + "+")

    return "\n".join(lines)


def format_report(agents: list[dict[str, Any]], consensus: dict[str, Any]) -> str:
    """Generate the full human-readable report.

    Args:
        agents: List of validated agent output dictionaries.
        consensus: Consensus dictionary produced by ``determine_consensus``.

    Returns:
        Multi-line markdown string with banner, findings, dissent,
        conditions, and recommended actions.
    """
    sections = []

    # Banner
    sections.append(format_banner(agents, consensus))
    sections.append("")

    # Consensus summary
    sections.append("## Consensus Summary")
    sections.append(consensus["majority_summary"])
    sections.append("")

    # Key findings
    if consensus["findings"]:
        sections.append("## Key Findings")
        for f in consensus["findings"]:
            icon = {"critical": "[!!!]", "warning": "[!!]", "info": "[i]"}.get(f["severity"], "[?]")
            sources = ", ".join(f.get("sources", ["unknown"]))
            sections.append(f"{icon} **[{f['severity'].upper()}]** {f['title']} _(from {sources})_")
            sections.append(f"   {f['detail']}")
            sections.append("")

    # Dissent
    if consensus["dissent"]:
        sections.append("## Dissenting Opinion")
        for d in consensus["dissent"]:
            name, title = AGENT_TITLES[d["agent"]]
            sections.append(f"**{name} ({title})**: {d['summary']}")
            sections.append(d["reasoning"])
            sections.append("")

    # Conditions
    if consensus["conditions"]:
        sections.append("## Conditions for Approval")
        for c in consensus["conditions"]:
            name, _ = AGENT_TITLES[c["agent"]]
            sections.append(f"- **{name}**: {c['condition']}")
        sections.append("")

    # Recommended action
    sections.append("## Recommended Actions")
    for agent_name, rec in consensus["recommendations"].items():
        name, title = AGENT_TITLES[agent_name]
        sections.append(f"- **{name}** ({title}): {rec}")

    return "\n".join(sections)
