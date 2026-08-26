# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-26
"""The one artifact the harness is allowed to leave in the tree.

A certificate that is wrong is the worst defect this harness can produce,
because it is the thing somebody trusts a year later without re-running
anything. Everything here exists to keep it from claiming more than it saw.

**Only one kind of run may emit it** (REQ-S23, REQ-S35, D-17): ``--smoke-2``,
with no ``--profile``, and only when no ``FAIL`` and no ``CANNOT_TEST``
remain. A cheap-profile run never certifies, however green it comes out --
a document that says "product defaults" while the environment named the cheap
models is false at the single point its whole value rests on.

**The name is fixed and it replaces the previous one.** Git's history is the
archive: a version's certificate lives in the commit its tag points at and
comes back with ``git show v<X.Y.Z>:<path>``. That is also why the certified
version goes INSIDE the document -- a versioned filename would force a reader
to know what the old one was called before they could ask for it.

**``rounds needed`` is the cost series.** A release that took three rounds is
information about that release, and ``git log -p`` over the fixed path turns
the number into the series that shows whether the harness is getting dearer --
which is how a cheap gate becomes one nobody runs.

**``contract coverage`` says what it does NOT cover.** Without it, a green
reads as coverage of the whole contract, which is the lie this harness exists
not to tell.
"""

import pathlib

from smoke.registry import DECLARED_SCENARIO_COUNT
from smoke.report import render_finding, render_growth, summarise

#: What the profile line says on a run that may certify. There is no other
#: value: a run with a profile does not get this far.
PRODUCT_DEFAULTS = "product defaults (no --profile)"

#: What this document covers, and what it only covers in part. A declaration
#: rather than a derivation: which specs a scenario set covers is a judgement
#: about the specs, and no arithmetic over the registry can make it.
CONTRACT_COVERAGE = (
    "StructuredVerdicts v0.14.3 (6/6 REQ-EA); "
    "Vault v0.9.0, Headless v0.10.0, MagiCore MS1-MS3, "
    "OperationBudget -- partial"
)

#: What no invocation of this harness can reach, named so a green cannot be
#: read as covering it. Both are properties of artifacts built elsewhere: the
#: harness runs one binary on one machine.
OUT_OF_SCOPE_DECLARATION = (
    "[OUT_OF_SCOPE] - cross-OS linkage and the published crate "
    "(REQ-S26, REQ-S27)"
)

#: The file the round counter lives in, inside the environment. It is not in
#: the tree on purpose -- nothing but the certificate may be -- which means
#: ``--reset-env`` discards it and the next round counts as the first. That is
#: a known and documented limitation, not an oversight: the alternative is a
#: second tracked file whose only content is a number.
ROUNDS_FILENAME = "rounds"


def may_certify(smoke_2: bool, profile, findings, evaluated: int) -> bool:
    """Whether this run is allowed to emit a certificate.

    FOUR conditions, and the fourth is the one the count itself demands. The
    document's headline is "N of N", and nothing checked the first N: drop a
    module from the scenario package and reconciliation still passes, because
    it compares what was invoked against what reported and both shrink
    together. The certificate would then read "18 of 19" and be emitted
    anyway -- honest arithmetic over silently reduced coverage.

    Complexity: ``O(findings)``.

    Args:
        smoke_2: Whether ``--smoke-2`` was passed.
        profile: The model profile in force, or None for the product's own
            defaults.
        findings: Every stamped finding of the run.
        evaluated: How many distinct scenarios reported.

    Returns:
        bool: True only when all four conditions hold at once.
    """
    if not smoke_2 or profile is not None:
        return False
    # Defence in depth, and unreachable in production on purpose:
    # ``require_declared_count`` already exits 3 at startup on a short
    # registry, and the runner's reconciliation guarantees every
    # registered scenario reported. It is here so a future caller that
    # skips either of those cannot emit a document reading "18 of 19".
    if evaluated != DECLARED_SCENARIO_COUNT:
        return False
    return not any(one.outcome.blocks_gate for one in findings)


class RoundCounter:
    """How many attempts this release has needed so far.

    Example:
        >>> import pathlib, tempfile
        >>> with tempfile.TemporaryDirectory() as scratch:
        ...     path = pathlib.Path(scratch) / "rounds"
        ...     (RoundCounter(path).bump(), RoundCounter(path).bump())
        (1, 2)
    """

    def __init__(self, path) -> None:
        """Bind the file the count survives in.

        Args:
            path: Where to keep it. Each iteration is a separate process, so a
                counter held in memory would report 1 forever.
        """
        self._path = pathlib.Path(path)

    def value(self) -> int:
        """What the counter holds now.

        An unreadable or malformed file counts as zero rather than raising. A
        corrupt counter is one wrong line in the document; refusing to run
        over it would cost the whole expensive run it is a footnote to.

        Returns:
            int: The count, or 0 when there is none to read.
        """
        try:
            return max(0, int(self._path.read_text(encoding="utf-8").strip()))
        except (OSError, ValueError):
            return 0

    def bump(self) -> int:
        """Count this round and report which one it is.

        Returns:
            int: The new count.

        Raises:
            OSError: If the count cannot be written. The caller decides what
                that is worth; here it is not swallowed, because a counter
                that silently never advances reports every release as its
                first.
        """
        count = self.value() + 1
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("%d\n" % count, encoding="utf-8")
        return count

    def reset(self) -> None:
        """Start again, once a certificate has been emitted."""
        try:
            self._path.unlink()
        except OSError:
            # Nothing to remove is the state reset wants anyway, and a counter
            # that cannot be cleared is not worth failing an emitted
            # certificate over.
            pass


class Certificate:
    """The rendered claim, and the file it becomes."""

    def __init__(self, version: str, commit: str, date: str,
                 binary_sha256: str, run_count: int, duration_s: float,
                 rounds: int, evaluated: int, growth, findings) -> None:
        """Bind everything the document declares.

        Args:
            version: The product version this certifies.
            commit: The short sha the binary was built from.
            date: The UTC date, as ``YYYY-MM-DD``.
            binary_sha256: The binary's digest -- a record of identity, never
                a proof of derivation.
            run_count: How many backend runs were executed.
            duration_s: What they took, in seconds.
            rounds: How many rounds this release needed.
            evaluated: How many scenarios were evaluated.
            growth: The environment's measured size, or None.
            findings: Every stamped finding, in report order.
        """
        self.version = version
        self.commit = commit
        self.date = date
        self.binary_sha256 = binary_sha256
        self.run_count = run_count
        self.duration_s = duration_s
        self.rounds = rounds
        self.evaluated = evaluated
        self.growth = growth
        self.findings = tuple(findings)

    def render(self) -> str:
        """Render the document.

        Complexity: ``O(findings)``.

        Returns:
            str: The markdown a reader gets.
        """
        lines = [
            "# Smoke Certificate",
            "",
            "- version: %s" % self.version,
            "- commit: %s" % self.commit,
            "- date: %s (UTC)" % self.date,
            "- profile: %s" % PRODUCT_DEFAULTS,
            "- binary: sha256 %s (rebuilt from the commit above)"
            % self.binary_sha256,
            "- real cost: %d backend run(s) in %ds"
            % (self.run_count, round(self.duration_s)),
            "- rounds needed: %d" % self.rounds,
            "- scope: %d of %d scenarios evaluated"
            % (self.evaluated, DECLARED_SCENARIO_COUNT),
            "- contract coverage: %s" % CONTRACT_COVERAGE,
        ]
        environment = render_growth(self.growth)
        if environment:
            lines.append("- %s" % environment)
        lines.append("- %s" % summarise(self.findings))
        lines.append("")
        lines.extend(render_finding(one) for one in self.findings)
        lines.append(OUT_OF_SCOPE_DECLARATION)
        return "\n".join(lines) + "\n"

    def write(self, path) -> None:
        """Write the document, replacing whatever was there.

        Args:
            path: Where the certificate lives. Its directory is created if it
                does not exist yet.

        Raises:
            OSError: If the file cannot be written.
        """
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.render(), encoding="utf-8")
