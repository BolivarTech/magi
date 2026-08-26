# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""S12 -- the harness leaves no trace.

Protects the precondition of the pre-merge gate and of the finishing checklist:
``git status`` clean with respect to the plan's scope. A harness that scattered
files through the working tree would either break that gate or, worse, teach
whoever runs it to stop reading the gate.

**Assertion 1 is a DIFF against a snapshot, never a claim that the tree is
clean.** A clean tree is the normal state at the finishing gate, but the
harness is run mostly DURING development, where uncommitted work is what a
working tree is for. Asserting cleanliness would go red because of the
developer's own edits and report as a product defect something the product did
not cause -- a false positive that also teaches people to ignore the scenario.
The invariant is *the harness leaves no trace*, and that is measured by
subtracting: what git reports afterwards, minus what it reported before, must
be the certificate and nothing else.

The snapshot is taken by the RUNNER before any scenario ran, and arrives here
as ambient state. Taking it inside this scenario would already include whatever
the earlier scenarios wrote, and the subtraction would then cancel out exactly
the traces it exists to find.

**Assertion 2 is about what git SEES, not about what is on disk.** The
environment goes on existing on purpose (REQ-S30); what must never happen is
that any of it becomes committable by accident.

Known limitation, declared rather than defended against: a git operation in
another terminal can move the working tree between the two snapshots. The
harness cannot lock the developer's own repository, and pretending otherwise
would cost more than the rare false red it would avoid.
"""

import subprocess

from smoke import runs
from smoke.config import CERTIFICATE_PATH
from smoke.outcome import Finding, Outcome
from smoke.registry import scenario
from smoke.runner import capture_tree

#: The verbatim assertion texts of the spec's section 8, for S12.
ASSERTIONS = (
    "the run adds no entry to git status --porcelain beyond the certificate",
    "git status --ignored smoke/env/ shows the whole environment on the "
    "ignored side",
)

#: How long git is given to answer, in seconds. It walks the working tree, so
#: the bound is generous enough for a cold filesystem cache and no more.
GIT_TIMEOUT_S = 120

#: The porcelain status prefix git puts on an ignored path. Everything the
#: environment holds must carry it.
IGNORED_PREFIX = "!!"

#: How wide a porcelain status field is, before the space and the path.
STATUS_FIELD_WIDTH = 2


@scenario("S12", assertions=ASSERTIONS, needs_ambient=True)
def the_harness_leaves_no_trace(run, ambient):
    """Subtract the pre-run snapshot from the tree, and ask git about the env.

    Args:
        run: Always ``None``; S12 declares no shared run.
        ambient: State captured before any scenario ran. Its
            ``tree_snapshot`` is what assertion 1 subtracts.

    Yields:
        Finding: One per entry of :data:`ASSERTIONS`, in that order.
    """
    yield _no_new_entries_finding(ambient.tree_snapshot)
    yield _environment_is_ignored_finding()


def _no_new_entries_finding(before):
    """Judge assertion 1 by subtracting the pre-run snapshot from the tree.

    Complexity: ``O(total bytes git lists)`` for the second snapshot, then
    ``O(number of files)`` for the subtraction.

    Args:
        before: The snapshot taken before any scenario ran, or None.

    Returns:
        Finding: PASS when the only difference is the certificate.
    """
    if before is None:
        return _s12(0, Outcome.CANNOT_TEST,
                    "no pre-run snapshot was captured, so there is nothing to "
                    "subtract and a clean tree would prove nothing")
    # capture_tree raises HarnessError when git is unavailable, and that is
    # deliberately NOT caught: snapshotting needs git, which is the HARNESS's
    # own dependency. Turning it into a finding would enter a verdict about the
    # product on the strength of the harness's broken toolchain, so it travels
    # to the exit-3 path untouched.
    after = capture_tree(runs.repo_root())
    changed = sorted(
        set(_differences(before.entries, after.entries)) - {CERTIFICATE_PATH}
    )
    if changed:
        return _s12(0, Outcome.FAIL,
                    "the run left %d entr%s behind: %s"
                    % (len(changed), "y" if len(changed) == 1 else "ies",
                       ", ".join(changed)))
    return _s12(0, Outcome.PASS, "")


def _differences(before, after):
    """Name every path that appeared, vanished or changed between snapshots.

    Complexity: ``O(len(before) + len(after))``.

    Args:
        before: The earlier snapshot's entries.
        after: The later snapshot's entries.

    Returns:
        list[str]: The changed paths. A path present in BOTH with the same
        hash is absent from the result even when it is a dirty, uncommitted
        file -- that is the developer's state, and it is not a trace the
        harness left.
    """
    return [key for key in set(before) | set(after)
            if before.get(key) != after.get(key)]


def _environment_is_ignored_finding():
    """Judge assertion 2 by asking git what it can see of the environment.

    Returns:
        Finding: PASS when git reports the environment and reports all of it as
        ignored; FAIL when any part of it is visible; CANNOT_TEST when the
        environment does not exist or git could not answer.
    """
    environment = runs.workspace_root()
    if not environment.is_dir():
        return _s12(1, Outcome.CANNOT_TEST,
                    "%s does not exist, so git has nothing to report about it"
                    % environment)
    try:
        listed = subprocess.run(
            ["git", "-C", str(runs.repo_root()), "status", "--porcelain",
             "--ignored", "--", str(environment)],
            capture_output=True, text=True, timeout=GIT_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _s12(1, Outcome.CANNOT_TEST, "git could not be run: %s" % exc)
    if listed.returncode != 0:
        return _s12(1, Outcome.CANNOT_TEST,
                    "git status exited %d for %s: %s"
                    % (listed.returncode, environment, listed.stderr.strip()))
    entries = [line for line in listed.stdout.splitlines() if line.strip()]
    if not entries:
        return _s12(1, Outcome.CANNOT_TEST,
                    "git reported nothing at all about %s, so whether it is "
                    "ignored or merely empty cannot be told apart"
                    % environment)
    visible = [line for line in entries
               if line[:STATUS_FIELD_WIDTH] != IGNORED_PREFIX]
    if visible:
        return _s12(1, Outcome.FAIL,
                    "git can see %d entr%s of the environment: %s"
                    % (len(visible), "y" if len(visible) == 1 else "ies",
                       ", ".join(entry.strip() for entry in visible)))
    return _s12(1, Outcome.PASS, "")


def _s12(index, outcome, detail):
    """Build the finding for one entry of :data:`ASSERTIONS`.

    Args:
        index: Position in :data:`ASSERTIONS`.
        outcome: What became of it.
        detail: The cause when the outcome is not PASS.

    Returns:
        Finding: The finding, with no run id -- S12 is standalone.
    """
    return Finding(assertion=ASSERTIONS[index], outcome=outcome, detail=detail,
                   run_id=None)
