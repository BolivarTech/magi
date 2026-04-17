#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 2.1.3
# Date: 2026-04-17
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


_ELLIPSIS: str = "..."


def _fit_content(content: str, width: int, *, preserve_suffix: str = "") -> str:
    """Truncate *content* to fit inside the ``_BANNER_INNER`` column budget.

    ``str.ljust`` never truncates, so without this guard a label that
    exceeds the inner width produces a row that overruns the border
    column and breaks the MANDATORY FINAL OUTPUT CONTRACT (``+===+``
    borders must align with ``|...|`` rows).

    When ``preserve_suffix`` is provided and there is room for it
    alongside the ellipsis, truncation drops characters from the
    **prefix** (the label half), preserving the suffix (the verdict +
    confidence tokens) verbatim. Pre-2.1.3 the truncation unconditionally
    ate the tail, which could silently erase the verdict itself on a
    pathologically long label — the operator saw a structurally valid
    row that had lost the one token the banner exists to communicate.

    When no suffix is provided, or the suffix is wider than the budget
    allows, truncation falls back to the original tail-cut behaviour so
    no caller that predates ``preserve_suffix`` changes semantics.

    ``content`` shorter than or equal to ``width`` is returned verbatim
    — the caller is responsible for the final ``ljust`` so the output
    is always exactly ``width`` characters before ``"|" + ... + "|"``
    wrapping.

    Args:
        content: Pre-formatted row content (label + verdict + conf).
        width: Column budget (``_BANNER_INNER``).
        preserve_suffix: Trailing slice of ``content`` that must survive
            truncation intact (e.g. `` APPROVE (85%)`` including the
            leading separator). Pass ``""`` to opt out.

    Returns:
        ``content`` if it fits, otherwise a truncated version whose
        length is exactly ``width``.
    """
    if len(content) <= width:
        return content
    # Fallback path: either no suffix contract, or the suffix alone is
    # already at or over the budget (so there is no room for prefix +
    # ellipsis + suffix). Truncate the tail to stay width-valid.
    if not preserve_suffix or len(preserve_suffix) + len(_ELLIPSIS) >= width:
        cutoff = max(1, width - len(_ELLIPSIS))
        return content[:cutoff] + _ELLIPSIS
    # Suffix-preserving path. Budget for the prefix half:
    # total width - ellipsis - suffix. The resulting length is exactly
    # ``width`` (prefix_budget + len(_ELLIPSIS) + len(preserve_suffix)).
    prefix_budget = width - len(_ELLIPSIS) - len(preserve_suffix)
    assert prefix_budget >= 1, "covered by the fallback guard above"
    prefix_source = content[: -len(preserve_suffix)]
    return prefix_source[:prefix_budget] + _ELLIPSIS + preserve_suffix


def format_banner(agents: list[dict[str, Any]], consensus: dict[str, Any]) -> str:
    """Generate the MAGI verdict banner with consistent alignment.

    Produces an ASCII box of fixed width (52 columns) containing agent
    verdicts and the consensus result. Verdicts are column-aligned by
    padding each agent label to the longest label width so that the
    verdict column starts at the same position on every row.

    Any row content that would exceed ``_BANNER_INNER`` (50 characters)
    is truncated with a trailing ``...`` so the ``|`` border column
    never slides — a future longer agent role name or a long consensus
    label cannot silently produce a malformed box.

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
        # The suffix starts at the space between the label column and
        # the verdict, so truncation that eats the label cannot also eat
        # the separator that keeps the verdict legible. On a normal-
        # width label the preserve_suffix is a strict suffix of content,
        # so this is a no-op for the common case.
        verdict_suffix = f" {verdict_display} ({conf_pct})"
        content = f"  {label:<{max_label_len}}{verdict_suffix}"
        fitted = _fit_content(content, _BANNER_INNER, preserve_suffix=verdict_suffix)
        lines.append("|" + fitted.ljust(_BANNER_INNER) + "|")

    lines.append(border)
    cons_content = f"  CONSENSUS: {consensus['consensus']}"
    fitted_cons = _fit_content(cons_content, _BANNER_INNER)
    lines.append("|" + fitted_cons.ljust(_BANNER_INNER) + "|")
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
