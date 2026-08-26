# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Scenario invocation, id stamping, and the reconciliation that closes the
two ways a scenario can vanish."""

import dataclasses
import hashlib
import pathlib
import subprocess
import typing

from smoke.errors import HarnessError, ProductOutputError
from smoke.outcome import Finding, Outcome
from smoke.registry import Registry, ScenarioEntry


class _TimedOutRun(typing.Protocol):
    """The only thing the runner needs of a shared run's result.

    ``RunResult`` is produced by Task 21 and this module is written in Task 3,
    so the real type cannot be imported here. A Protocol names the contract
    anyway: the runner reads ``timed_out`` and nothing else, and a type checker
    can verify Task 21's ``RunResult`` satisfies it. Without this the agreement
    between the two tasks lives in a comment, which is exactly the kind of
    contract that drifts silently.
    """

    timed_out: bool


_GIT_TIMEOUT_S = 60
_TRANSLATED_FAILURE = "the product's output could not be interpreted"
_BACKEND_UNREACHABLE = "the backend was not reachable"
_RUN_TIMED_OUT = "the shared run did not finish inside its timeout"

#: The three texts the RUNNER answers with when a scenario never ran: a
#: timed-out run, an unreachable backend, a capture that could not be read.
#: Named as a group because the completeness check has to recognise all three
#: or none -- a scenario that did not run promised nothing on that path.
SUBSTITUTED_ASSERTIONS = frozenset(
    (_TRANSLATED_FAILURE, _BACKEND_UNREACHABLE, _RUN_TIMED_OUT)
)


@dataclasses.dataclass(frozen=True)
class StampedFinding:
    """A Finding once the runner has attached the scenario that produced it.

    Args:
        scenario: The id of the scenario the runner invoked.
        assertion: The English text of what was asserted.
        outcome: What became of it.
        detail: The cause when the outcome is not ``PASS``.
        run_id: The shared run that produced it, or None.
    """

    scenario: str
    assertion: str
    outcome: Outcome
    detail: str
    run_id: str | None


@dataclasses.dataclass(frozen=True)
class TreeSnapshot:
    """What the working tree looked like before the harness touched anything.

    Attributes:
        entries: Relative POSIX path -> sha256 of the file's bytes, for every
            tracked and untracked file git reports. A mapping rather than a
            hash of the whole tree, so S12 can name WHICH file appeared or
            changed instead of only that something did -- a diff nobody can
            read is a red nobody acts on.
    """

    entries: dict[str, str]


@dataclasses.dataclass(frozen=True)
class Ambient:
    """State captured before any run started, handed to scenarios that ask.

    Every field is a plain builtin, deliberately. ``SmokeConfig`` is produced by
    Task 6 and this module is written in Task 3, so holding the config object
    itself would make an earlier task import a later one's type -- the same
    ordering rule ``run_results`` pays for. Carrying the three values S9
    actually reads costs nothing and keeps the direction of the dependency
    right.

    Attributes:
        tree_snapshot: The pre-run snapshot, or None when nothing captured one
            (the CLI before Task 21 wires it, and every unit test that does not
            exercise S12).
        ceiling_fraction: The configured fraction for the same assertion, or
            None when unconfigured.
        memory_settings: The environment's ``magi.toml`` ``[memory]`` block as
            parsed, empty when the section is absent. S9 assertion 3b needs to
            know whether the product declared the fields at all: absent is not
            zero, and treating it as zero would assert against a ceiling the
            product never derived.
    """

    tree_snapshot: TreeSnapshot | None
    ceiling_fraction: float | None
    memory_settings: dict[str, object]


def capture_tree(root: pathlib.Path) -> TreeSnapshot:
    """Snapshot the working tree so S12 can prove the harness left no trace.

    Uses ``git ls-files`` rather than walking the filesystem: the walk would
    have to reimplement ``.gitignore``, and the reimplementation is what would
    drift. ``--exclude-standard`` is not optional -- without it ``--others``
    lists every ignored path, so ``target/``, ``graphify-out/`` and the
    environment's own database land in the snapshot and S12 goes red on every
    run that compiles anything. What S12 asks is whether the harness left a
    trace in the tree a human would notice, and an ignored build artifact is
    not that.

    Complexity: O(total bytes of the listed files), one sha256 pass each.

    Args:
        root: The repository root.

    Returns:
        The snapshot.

    Raises:
        HarnessError: If git is unavailable or fails. This is the harness's own
            dependency, so it is never a product defect.
    """
    try:
        listed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached", "--others",
             "--exclude-standard"],
            capture_output=True, check=True, text=True, timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise HarnessError(f"could not snapshot the tree: {exc}") from exc
    entries: dict[str, str] = {}
    for line in listed.stdout.splitlines():
        path = root / line
        if path.is_file():
            entries[line] = hashlib.sha256(path.read_bytes()).hexdigest()
    return TreeSnapshot(entries=entries)


class Runner:
    """Invokes every registered scenario and reconciles what happened."""

    def __init__(self, registry: Registry, run_results: dict[str, object],
                 backend_reachable: bool, ambient: Ambient) -> None:
        """Create a runner.

        Args:
            registry: The scenarios to invoke.
            run_results: The completed shared runs, keyed by run id. Typed
                ``dict[str, object]`` and not ``dict[str, RunResult]`` for a
                reason that is not laziness: ``RunResult`` is produced by Task
                21 and this module is written in Task 3, so naming the type
                here would make an earlier task import a later one's symbol.
                The looseness is the cost of the ordering rule, and it is paid
                once, here. A
                scenario declaring ``run="R4"`` receives ``run_results["R4"]``.
                It carries the runs and nothing else -- not the config, not the
                environment: two dicts of the same type, one of which silently
                reports every scenario as never invoked.
            backend_reachable: Whether the preflight reached the backend. A
                third argument rather than a defaulted one, and a plain ``bool``
                rather than ``BackendStatus``, so no caller can forget it and no
                earlier phase has to import a type a later one produces.
            ambient: State captured before any run started -- today just the
                working-tree snapshot S12 compares against. ``Ambient`` and
                ``capture_tree`` live in this module rather than beside S12
                because the runner is what hands ambient state to scenarios;
                putting the capture in a scenario module would make the runner
                import from ``scenarios/``, which is the dependency direction
                section 2.1 forbids. It is a fourth
                argument rather than an entry in ``run_results`` for the reason
                that parameter's own docstring gives: a snapshot is not a run,
                and the two have the same Python type, so merging them fails
                silently instead of loudly.
        """
        self._registry = registry
        self._run_results = run_results
        self._backend_reachable = backend_reachable
        self._ambient = ambient

    def run(self) -> list[StampedFinding]:
        """Invoke every registered scenario, in deterministic order.

        Returns:
            Every finding, stamped with the scenario the runner invoked.

        Raises:
            HarnessError: If reconciliation fails either way.
        """
        invoked: set[str] = set()
        reported: set[str] = set()
        findings: list[StampedFinding] = []

        for entry in self._registry.entries():
            invoked.add(entry.scenario_id)
            answered: list[str] = []
            for finding in self._invoke(entry):
                reported.add(entry.scenario_id)
                answered.append(finding.assertion)
                findings.append(
                    StampedFinding(
                        scenario=entry.scenario_id,
                        assertion=finding.assertion,
                        outcome=finding.outcome,
                        detail=finding.detail,
                        run_id=finding.run_id,
                    )
                )
            self.reconcile_completeness(entry, answered)

        self.reconcile(
            registered=set(self._registry.registered_ids()),
            invoked=invoked,
            reported=reported,
        )
        return findings

    @staticmethod
    def reconcile_completeness(entry: ScenarioEntry, answered: list[str]) -> None:
        """Check that a scenario answered everything it promised.

        The other reconciliation checks PRESENCE: a scenario was invoked and it
        reported. That is satisfied by a scenario which returns after two of
        its five findings, and the three it never reached leave no trace -- not
        in the report, not in the exit code, and not in the certificate, which
        counts scenarios rather than assertions.

        Both directions are errors. A missing assertion is silent coverage
        loss; an assertion nobody declared reaches the certificate verbatim,
        so a text that drifted from its module constant is a wrong claim in
        the one document meant to be trusted later.

        Complexity: ``O(declared + answered)``.

        Args:
            entry: The scenario that just ran.
            answered: The assertion texts it yielded, in order. When the runner
                substituted its own single finding the scenario never ran, and
                the check does not apply.

        Raises:
            HarnessError: If the two sets differ. This is a defect in the
                HARNESS -- exit 3 -- never a verdict on the product.
        """
        if not entry.assertions:
            return
        if len(answered) == 1 and answered[0] in SUBSTITUTED_ASSERTIONS:
            # The RUNNER answered, not the scenario. A timed-out run, an
            # unreachable backend and an unreadable capture each collapse the
            # whole scenario into one finding on purpose, and the scenario body
            # never ran -- so it promised nothing on this path and owes nothing.
            return
        declared = set(entry.assertions)
        arrived = set(answered)
        silent = sorted(declared - arrived)
        invented = sorted(arrived - declared)
        if not silent and not invented:
            return
        parts = []
        if silent:
            parts.append("never answered %s" % "; ".join(repr(s) for s in silent))
        if invented:
            parts.append("answered %s, which it never declared"
                         % "; ".join(repr(s) for s in invented))
        raise HarnessError(
            "scenario %s %s" % (entry.scenario_id, " and ".join(parts))
        )

    def _invoke(self, entry: ScenarioEntry) -> list[Finding]:
        """Run one scenario, translating a product-output failure to FAIL.

        Args:
            entry: The registered scenario to invoke.

        Returns:
            The findings it yielded; a single FAIL when the product's output
            could not be interpreted.

        Raises:
            Exception: Anything the harness itself got wrong propagates
                untouched, so it reaches the exit-3 path.
        """
        for run_id, result in self._runs_for(entry).items():
            # Iteration follows the declared tuple order, so a scenario hanging
            # off several runs always reports the FIRST one that timed out and
            # the report never reorders between invocations.
            if result is not None and result.timed_out and not entry.inspects_timeouts:
                # SS5.1 decided this: a timeout is CANNOT_TEST, never FAIL. A slow
                # provider is not a product defect, and calling it one puts the gate
                # red on someone else's load -- the intermittent red that gets
                # rationalised until the gate is ignored. The ONE scenario that can
                # tell a degraded ceiling from a slow provider is S7, and it does it
                # by reading applied_caps out of the partial output, which is
                # evidence. It declares inspects_timeouts and receives the result;
                # everything else stops here.
                return [
                    Finding(
                        assertion=_RUN_TIMED_OUT,
                        outcome=Outcome.CANNOT_TEST,
                        detail=f"run {run_id} exceeded its timeout",
                        run_id=run_id,
                    )
                ]
        if entry.needs_backend and not self._backend_reachable:
            # D-17: a down backend is not a product defect, so this is never FAIL.
            # It still blocks the gate, and the certificate is still not written.
            return [
                Finding(
                    assertion=_BACKEND_UNREACHABLE,
                    outcome=Outcome.CANNOT_TEST,
                    detail="the backend did not answer; this scenario needs it",
                    run_id=entry.run,
                )
            ]
        run = self._resolve_run(entry)
        try:
            findings = (entry.func(run, self._ambient) if entry.needs_ambient
                        else entry.func(run))
            return list(findings)
        except ProductOutputError as exc:
            return [
                Finding(
                    assertion=_TRANSLATED_FAILURE,
                    outcome=Outcome.FAIL,
                    detail=str(exc),
                    run_id=entry.run,
                )
            ]

    def _runs_for(self, entry: ScenarioEntry) -> dict[str, _TimedOutRun | None]:
        """The shared runs this scenario hangs off, keyed by id.

        The annotation is the runner's OWN view, and it is narrower than what
        the scenario receives on purpose: all this class ever reads off a
        result is ``timed_out``. Saying so in the type is what gives the
        Protocol a consumer instead of leaving it as a comment with a colon.

        Args:
            entry: The registered scenario.

        Returns:
            Empty when the scenario is standalone; one pair for a single run;
            one per id for a scenario declaring several.
        """
        if entry.run is None:
            return {}
        ids = (entry.run,) if isinstance(entry.run, str) else entry.run
        return {run_id: self._run_results.get(run_id) for run_id in ids}

    def _resolve_run(self, entry: ScenarioEntry) -> object:
        """What the scenario receives as its ``run`` argument.

        A single declared run arrives as the result itself, which is what all
        but one scenario wants. Several arrive as the whole ``dict`` keyed by
        id, because a scenario comparing R2 against R3 needs both and picking
        one to be "the" run would be arbitrary.

        Args:
            entry: The registered scenario.

        Returns:
            ``None``, one ``RunResult``, or a ``dict[str, RunResult]``.
        """
        runs = self._runs_for(entry)
        if not runs:
            return None
        if isinstance(entry.run, str):
            return runs[entry.run]
        return runs

    @staticmethod
    def reconcile(registered: set[str], invoked: set[str], reported: set[str]) -> None:
        """Check that nothing disappeared, either way.

        Args:
            registered: Every id the decorator recorded.
            invoked: Every id the runner actually called. The RUNNER records
                this as it calls, never derived from who reported.
            reported: Every id that yielded at least one Finding.

        Raises:
            HarnessError: With one message for registered-but-never-invoked,
                and a different one for invoked-but-silent. An empty iterator
                is ALWAYS a harness failure and never a pass: every scenario
                declares at least one assertion, so yielding nothing means the
                body stopped early. A scenario that cannot evaluate its
                assertions yields ``CANNOT_TEST`` with the cause; it does not
                stay quiet.
        """
        never_invoked = sorted(registered - invoked)
        if never_invoked:
            raise HarnessError(
                f"scenario registered but never invoked: {never_invoked}"
            )
        silent = sorted(invoked - reported)
        if silent:
            raise HarnessError(f"scenario invoked but reported nothing: {silent}")
