"""Derive a nextest filter from the files a commit actually touches.

Why this exists
---------------
The full suite runs 400-600 s, and the agent tooling's command ceiling is
600 s. Agents were therefore backgrounding the run and yielding their turn,
which strands them: a background task only wakes an agent that is still
running. Seven stalls in one milestone traced back to that single squeeze.

Running only the touched modules per commit takes the run well clear of the
ceiling. The full suite still runs once per round, so nothing is skipped --
only deferred.

What protects the gap
---------------------
Per commit this narrows *which tests* run, not *whether* the code compiles:
``cargo clippy --all-targets`` and ``cargo build`` still cover every target,
and they are what catches the common cross-module break -- a changed
signature. What this can miss is a *behavioural* regression in a module the
filter excluded, and the round-closing full suite is what catches that.

Fail-safe by construction
-------------------------
The filter is DERIVED from ``git diff``, never hand-maintained. A hand-written
list is exactly the failure mode ``.config/nextest.toml`` documents twice: a
filter that matches nothing raises no error, so it silently stops applying.
Anything this script cannot map confidently -- a crate-root file, a manifest,
a build script, an unrecognised path -- falls back to the FULL suite. The
default direction is always "run more", never "run less".

Usage
-----
    python scripts/scoped_tests.py            # working tree vs HEAD
    python scripts/scoped_tests.py --range A..B
    python scripts/scoped_tests.py --print    # show the filter, run nothing

Exits with the test runner's own status so it can gate a commit directly.
"""

import argparse
import subprocess
import sys

# Touching any of these invalidates the mapping, so we run everything.
FULL_SUITE_TRIGGERS = (
    "cargo.toml",
    "cargo.lock",
    "build.rs",
    "src/main.rs",
    "src/lib.rs",
    ".config/nextest.toml",
)


def changed_files(rev_range):
    """Return the repository-relative paths the range (or worktree) touches.

    With no range, compares the working tree *and* the index against HEAD, so
    staged-but-uncommitted work is included.
    """
    if rev_range:
        cmd = ["git", "diff", "--name-only", rev_range]
    else:
        cmd = ["git", "diff", "--name-only", "HEAD"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def module_filter(path):
    """Map one repository path to a nextest filter fragment, or None.

    Returns the string ``"full"`` when the path forces a full run.
    """
    lowered = path.lower()
    if lowered in FULL_SUITE_TRIGGERS:
        return "full"

    if path.startswith("tests/") and path.endswith(".rs"):
        # An integration-test file is its own binary; run that binary whole.
        name = path[len("tests/"):-len(".rs")]
        if "/" in name:
            # A helper module under tests/ is compiled into several binaries
            # and we cannot tell which, so widen rather than guess.
            return "full"
        return "binary(%s)" % name

    if path.startswith("src/") and path.endswith(".rs"):
        rest = path[len("src/"):-len(".rs")]
        parts = rest.split("/")
        if parts[-1] == "mod":
            # A module root: its children may depend on it, so take the subtree.
            parts = parts[:-1]
        if not parts:
            return "full"
        return "test(/%s::/)" % "::".join(parts)

    # Documentation, fixtures and config we do not know how to map: those
    # cannot break a test on their own, so they contribute no filter. If the
    # commit touched nothing else, the caller treats that as "nothing to run".
    if path.startswith(("docs/", "dev-docs/", ".superpowers/", "planning/", "sbtdd/")):
        return None
    if path.endswith((".md", ".toml", ".json", ".yml", ".yaml")):
        return None

    # Anything unrecognised: widen, never narrow.
    return "full"


def build_filter(paths):
    """Return (expression, reason). An empty expression means 'run nothing'."""
    if not paths:
        return None, "no changes detected"

    fragments = []
    for path in paths:
        mapped = module_filter(path)
        if mapped == "full":
            return "full", "%s forces a full run" % path
        if mapped and mapped not in fragments:
            fragments.append(mapped)

    if not fragments:
        return None, "only documentation or config changed"
    return " | ".join(fragments), "%d path(s) mapped" % len(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--range", dest="rev_range", default=None,
                        help="git revision range, e.g. HEAD~3..HEAD")
    parser.add_argument("--print", dest="print_only", action="store_true",
                        help="print the derived filter and exit")
    args = parser.parse_args()

    paths = changed_files(args.rev_range)
    expression, reason = build_filter(paths)

    if expression == "full":
        print("[scoped-tests] FULL suite: %s" % reason)
        cmd = ["cargo", "nextest", "run"]
    elif expression is None:
        print("[scoped-tests] nothing to run: %s" % reason)
        if args.print_only:
            return 0
        # Still run the full suite rather than claiming a green gate on nothing.
        print("[scoped-tests] running the full suite anyway, to avoid a vacuous pass")
        cmd = ["cargo", "nextest", "run"]
    else:
        print("[scoped-tests] scoped run (%s): %s" % (reason, expression))
        cmd = ["cargo", "nextest", "run", "-E", expression]

    if args.print_only:
        print(" ".join(cmd))
        return 0

    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
