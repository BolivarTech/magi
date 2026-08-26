# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-26
"""What the run concluded, rendered for somebody who has to act on it.

The findings themselves are produced by the runner; this module decides how
they read. Two choices carry the weight.

**The cause is printed.** Until this module existed, a ``FAIL`` reached the
reader as its assertion text and nothing else -- the detail every scenario
takes care to write was dropped on the floor. A reader was told that something
had broken and never what, which turns every red into a source-reading
exercise.

**Causes are grouped, findings are not.** Three assertions refused because one
run exceeded its ceiling is one thing that went wrong. Repeating the sentence
three times buries it among the assertions it happened to, so each finding
keeps its own line and the causes are collected once underneath. The group's
key is the run AND the text: two runs that failed the same way failed twice,
and keying on the text alone would report one and hide the other.
"""

from smoke.outcome import Outcome

#: How a finding's line begins, so the report and the certificate cannot drift
#: apart on the one format a reader learns to scan.
FINDING_LINE = "[%s] %s%s - %s"

#: What a finding with a shared run adds to its line.
RUN_SUFFIX = " run=%s"

#: The heading the collected causes sit under.
CAUSES_HEADING = "causes:"

#: How a cause is attributed when the finding declares no shared run.
STANDALONE = "standalone"

#: Bytes per kilobyte, for the environment line. Decimal rather than binary:
#: the number is there to be glanced at, not reconciled with a disk tool.
BYTES_PER_KB = 1000


def render_finding(finding) -> str:
    """Render one finding's line.

    Args:
        finding: The stamped finding.

    Returns:
        str: The line, with the run id when there is one.
    """
    suffix = RUN_SUFFIX % finding.run_id if finding.run_id else ""
    return FINDING_LINE % (finding.outcome.value, finding.scenario, suffix,
                           finding.assertion)


def render_growth(growth) -> str:
    """Render the environment's size for a reader.

    ``active_memories`` of ``None`` means NOT MEASURED and says so. Rendering
    it as zero would be the same lie the harness refuses everywhere else: a
    number nobody took.

    Args:
        growth: The measurement, or None.

    Returns:
        str: The line, or an empty string when there is nothing to report.
    """
    if growth is None:
        return ""
    memories = ("active memories not measured"
                if growth.active_memories is None
                else "%d active memories" % growth.active_memories)
    return "environment: %s, %d KB database, %d KB archived" % (
        memories, growth.db_bytes // BYTES_PER_KB,
        growth.runs_bytes // BYTES_PER_KB)


def summarise(findings) -> str:
    """Render the counting line.

    ``OUT_OF_SCOPE`` is counted as neither passed nor not passed: it was never
    promised, so counting it as a pass inflates the coverage and counting it
    as a failure blocks a gate over something nobody undertook to do. It still
    contributes to the total, which is what keeps the three numbers honest
    about not adding up.

    Args:
        findings: Every stamped finding.

    Returns:
        str: The summary.
    """
    findings = list(findings)
    passed = sum(1 for one in findings if one.outcome is Outcome.PASS)
    blocked = sum(1 for one in findings if one.outcome.blocks_gate)
    return "result: %d passed, %d not passed, %d total" % (passed, blocked,
                                                           len(findings))


class Report:
    """Everything one invocation concluded.

    Example:
        >>> from smoke.outcome import Outcome
        >>> from smoke.runner import StampedFinding
        >>> finding = StampedFinding(scenario="S1", assertion="it parses",
        ...                          outcome=Outcome.PASS, detail="",
        ...                          run_id="R1")
        >>> print(Report([finding]).render().splitlines()[0])
        [PASS] S1 run=R1 - it parses
    """

    def __init__(self, findings, growth=None) -> None:
        """Bind the findings this report covers.

        Args:
            findings: The stamped findings, already in report order.
            growth: The environment's measured size, or None when it was not
                taken.
        """
        self._findings = tuple(findings)
        self._growth = growth

    @property
    def findings(self) -> tuple:
        """The findings, in the order they will be rendered.

        Returns:
            tuple: The stamped findings.
        """
        return self._findings

    def blocking(self) -> list:
        """The findings that forbid a green run.

        Complexity: ``O(findings)``.

        Returns:
            list: Every ``FAIL`` and ``CANNOT_TEST``, in report order.
        """
        return [one for one in self._findings if one.outcome.blocks_gate]

    def grouped_by_run(self) -> dict:
        """Every finding, indexed by the run that produced it.

        Complexity: ``O(findings)``.

        Returns:
            dict: Run id (or None for the standalone scenarios) to the
            findings it produced, each list in report order.
        """
        grouped: dict = {}
        for one in self._findings:
            grouped.setdefault(one.run_id, []).append(one)
        return grouped

    def causes(self) -> list:
        """The distinct causes, each with the assertions it accounts for.

        Complexity: ``O(findings)``.

        Returns:
            list: ``(run_id, detail, [findings])`` in first-seen order.
        """
        order: list = []
        seen: dict = {}
        for one in self._findings:
            if one.outcome is Outcome.PASS or not one.detail:
                continue
            key = (one.run_id, one.detail)
            if key not in seen:
                seen[key] = []
                order.append(key)
            seen[key].append(one)
        return [(run_id, detail, seen[(run_id, detail)])
                for run_id, detail in order]

    def render(self) -> str:
        """Render the whole report.

        Complexity: ``O(findings)``.

        Returns:
            str: One line per finding, the collected causes, the environment's
            size when it was measured, and the summary.
        """
        lines = [render_finding(one) for one in self._findings]
        causes = self.causes()
        if causes:
            lines.append("")
            lines.append(CAUSES_HEADING)
            for run_id, detail, members in causes:
                lines.append("  %s: %s" % (run_id or STANDALONE, detail))
                for member in members:
                    lines.append("    %s - %s"
                                 % (member.scenario, member.assertion))
        environment = render_growth(self._growth)
        lines.append("")
        if environment:
            lines.append(environment)
        lines.append(summarise(self._findings))
        return "\n".join(lines) + "\n"
