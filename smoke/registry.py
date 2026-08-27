# Author: Julian Bolivar
# Version: 0.17.0
# Date: 2026-08-27
"""Scenario registration.

Adding a scenario is a decorator and a function (REQ-S33). If adding one ever
requires touching ``runs.py`` or ``certificate.py``, the design has drifted.
"""

import dataclasses
from typing import Callable, Iterator, Optional, Union

from smoke.errors import HarnessError
from smoke.outcome import Finding

#: How many scenarios the harness promises to evaluate. Read by the
#: certificate renderer, which publishes "N of N scenarios evaluated", and by
#: the test that asserts the registry actually holds that many. One constant
#: with two readers is what keeps the headline and the registry from drifting.
DECLARED_SCENARIO_COUNT = 22

#: How many ASSERTIONS those scenarios promise between them, per section 8 of
#: the spec. A separate constant because it answers a separate question, and
#: the reconciliation cannot answer this one: it checks that a scenario
#: ANSWERED what it declared, so deleting an assertion from a module constant
#: and its yield together satisfies it -- both sides shrank. The scenario
#: count stays 22, the certificate still reads "22 of 22", and the coverage is
#: quietly smaller.
#:
#: It went 70 -> 69 on 2026-08-27 when S21 was redesigned. Its third assertion
#: was not lost, it was retired with the forcing recipe it belonged to: an
#: assertion invented to hold the total at 70 would have been padding, which is
#: the defect that redesign exists to remove.
DECLARED_ASSERTION_COUNT = 69

#: A scenario takes the run it declared, and -- only when it declares
#: needs_ambient -- the ambient state as a second argument. The union is
#: honest about a real two-shape call rather than hiding it behind ``...``,
#: which would type-check every wrong signature too.
ScenarioFunc = Union[
    Callable[[Optional[object]], Iterator[Finding]],
    Callable[[Optional[object], "Ambient"], Iterator[Finding]],
]


@dataclasses.dataclass(frozen=True)
class ScenarioEntry:
    """One registered scenario and what it needs in order to run.

    Args:
        scenario_id: The id used in the report, e.g. ``"S7"``.
        func: The scenario body. It yields Findings with no scenario id.
        run: The shared run this scenario reads, or a **tuple** of ids when it
            reads several -- S9 compares R1, R2 and R3, and declaring one of
            them as "the" run would be arbitrary. A single id arrives as the
            result; a tuple arrives as the dict keyed by id. None when the
            scenario is standalone.
        needs_backend: Whether a reachable backend is required. This is what
            lets D-17 classify a scenario without executing it.
        needs_ambient: Whether the scenario reads state captured before any run
            started. Exactly one does: S12 compares the working tree against a
            snapshot taken before the harness touched anything, and that
            snapshot is by definition not the output of a run. Declaring it
            keeps every other scenario's signature at ``(run)``.
        inspects_timeouts: Whether the scenario can extract signal from a run
            that timed out. Exactly one can: S7 reads ``applied_caps`` out of
            what R4 emitted before hanging, which is how SS5.1 tells a degraded
            ceiling from a slow provider -- on evidence, never inferred from
            configuration. Everything else gets ``CANNOT_TEST`` and never sees
            the partial result.
    """

    scenario_id: str
    func: ScenarioFunc
    run: str | tuple[str, ...] | None
    needs_backend: bool
    needs_ambient: bool
    inspects_timeouts: bool
    #: The verbatim texts the scenario promises to answer, for the runner's
    #: completeness check. Empty means it declares none.
    assertions: tuple[str, ...] = ()


def _sort_key(scenario_id: str) -> tuple[str, int]:
    """Order ids numerically inside their letter prefix, so S9 precedes S13.

    Args:
        scenario_id: An id shaped like ``"S13"``.

    Returns:
        The (prefix, number) pair to sort on.

    Raises:
        HarnessError: If the id is not a letter followed by digits.
    """
    prefix, digits = scenario_id[:1], scenario_id[1:]
    if not prefix.isalpha() or not digits.isdigit():
        raise HarnessError(f"malformed scenario id: {scenario_id!r}")
    return (prefix, int(digits))


class Registry:
    """The scenarios known before anything runs."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._entries: dict[str, ScenarioEntry] = {}

    def add(self, entry: ScenarioEntry) -> None:
        """Record a scenario.

        Args:
            entry: The scenario to record.

        Raises:
            HarnessError: If the id is malformed, or already registered. Two
                scenarios under one id means one of them can never be reported.
        """
        _sort_key(entry.scenario_id)
        if entry.scenario_id in self._entries:
            raise HarnessError(f"scenario {entry.scenario_id} is registered twice")
        self._entries[entry.scenario_id] = entry

    def get(self, scenario_id: str) -> ScenarioEntry:
        """Return one registered scenario.

        Args:
            scenario_id: The id to look up.

        Returns:
            The recorded entry.

        Raises:
            HarnessError: If the id was never registered.
        """
        try:
            return self._entries[scenario_id]
        except KeyError as exc:
            raise HarnessError(f"scenario {scenario_id} is not registered") from exc

    def registered_ids(self) -> list[str]:
        """Return every registered id in deterministic report order.

        Returns:
            The ids sorted by letter then number, so the certificate's line
            order never depends on filesystem iteration order.

        Raises:
            HarnessError: If a recorded id is malformed. ``add`` rejects those,
                so reaching this is a harness defect, not a caller's mistake.
        """
        return sorted(self._entries, key=_sort_key)

    def entries(self) -> list[ScenarioEntry]:
        """Return every entry in that same deterministic order.

        Returns:
            The entries, ordered as ``registered_ids``.

        Raises:
            HarnessError: As ``registered_ids``.
        """
        return [self._entries[key] for key in self.registered_ids()]


DEFAULT_REGISTRY = Registry()


def scenario(
    scenario_id: str,
    assertions: tuple[str, ...] = (),
    run: str | tuple[str, ...] | None = None,
    needs_backend: bool = False,
    needs_ambient: bool = False,
    inspects_timeouts: bool = False,
    registry: Registry | None = None,
) -> Callable[[ScenarioFunc], ScenarioFunc]:
    """Register a scenario.

    Args:
        scenario_id: The id used in the report, e.g. ``"S7"``.
        assertions: The verbatim texts this scenario promises to answer. The
            runner reconciles what arrives against them: reconciliation alone
            checks PRESENCE, and a scenario that returns after two of its five
            findings is present, reported, and short three assertions that
            leave no trace in the report, the exit code or the certificate.
            Empty means the scenario declares nothing to check against, which
            is how the tests that predate this register.
        run: One run id, a tuple of them, or None when the scenario is
            standalone. Standalone does not mean it leaves the product alone --
            it reaches it through ``runs.invoke()``.
        needs_backend: Whether a reachable backend is required.
        needs_ambient: Whether the scenario reads pre-run state; see
            ``ScenarioEntry``, which documents all three flags in full rather
            than repeating them here.
        inspects_timeouts: Whether the scenario can classify a timed-out run
            from its partial output.
        registry: The registry to record into; defaults to the process-wide
            one. Tests pass their own so they never mutate global state.

    Returns:
        The decorator, which records the function and returns it unchanged.

    Raises:
        HarnessError: If the id is malformed or already registered. The
            decorator records at import time, so a clash stops the process
            before any scenario runs.

    Example:
        Into a registry of its own, never the default one: the decorator
        records at import time and the default already holds every real
        scenario, so an example that omitted ``registry=`` registered ``S7``
        a second time and raised. It went unnoticed until the doctests were
        wired into the suite, which is the argument for wiring them.

        >>> own = Registry()
        >>> @scenario("S7", run="R4", needs_backend=True, registry=own)
        ... def ceiling_does_not_degrade_the_trio(run):
        ...     yield Finding("the trio is not degraded", Outcome.PASS, "", "R4")
        >>> own.get("S7").run
        'R4'
    """

    def decorate(func: ScenarioFunc) -> ScenarioFunc:
        target = DEFAULT_REGISTRY if registry is None else registry
        target.add(ScenarioEntry(scenario_id, func, run, needs_backend,
                                 needs_ambient, inspects_timeouts,
                                 tuple(assertions)))
        return func

    return decorate
