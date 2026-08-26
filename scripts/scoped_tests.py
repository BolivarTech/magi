"""Derive a nextest filter from the files a commit actually touches.

Why this exists
---------------
The full suite runs 400-600 s, and the agent tooling's command ceiling is
600 s. Agents were therefore backgrounding the run and yielding their turn,
which strands them: a background task only wakes an agent that is still
running. Nine stalls in one milestone traced back to that single squeeze.

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
The filter is DERIVED, never hand-maintained. A hand-written list is exactly
the failure mode ``.config/nextest.toml`` documents twice: a filter that
matches nothing raises no error, so it silently stops applying. Anything git
SEES that this script cannot map confidently -- a workspace manifest, a
lockfile, the nextest config, an unrecognised path -- falls back to the FULL
suite. Where there is doubt, the direction is "run more".

The one exception is an EMPTY diff: when git reports no change at all, nothing
runs. That is not "running less" -- it is not running the same thing twice:
there is nothing to verify, so the previous verification still stands. The case
that matters in practice is editing a GIT-IGNORED file (local notes, agent
config, planning material): git cannot see it, it cannot affect the build, and
spending the whole suite on it is the waste this script exists to remove. The
"never green over nothing" rule applies to a documentation change git *can*
see -- it might touch a doctest -- and not to an empty diff.

Workspaces
----------
Package boundaries come from ``cargo metadata``, not from guessed directory
names such as ``crates/``. Guessing would be the same class of mistake as a
hand-written filter: it works until someone lays the workspace out differently,
and then it silently maps nothing. With the real package list, touching one
member's crate root scopes the run to *that package* rather than to everything.

Adapting this to another language
---------------------------------
This implementation is Rust-specific in exactly three places, and nowhere else:

* ``packages()`` -- how package boundaries are discovered (``cargo metadata``).
* ``module_filter()`` -- how a file path maps to a test selector. Here that is
  Rust's module convention, ``src/foo/bar.rs`` -> ``foo::bar``.
* the runner command -- ``cargo nextest run [-E EXPR]``.

Everything else is language-neutral and is the part actually worth reusing:
derive the selection from ``git diff`` instead of maintaining a list by hand,
and make every case you cannot map confidently widen to the full suite rather
than narrow.

Sketches for the two other stacks this author works in:

* **Python / pytest** -- packages come from the project layout or a monorepo
  manifest; ``pkg/foo/bar.py`` maps to ``tests/foo/test_bar.py`` or to a
  ``-k``/marker expression; the runner is ``pytest``. Touching ``conftest.py``
  or ``pyproject.toml`` is the analogue of touching a crate root, so it widens.
* **C / C++ with CMake + CTest** -- packages come from the CMake target list;
  a source file maps to the target that compiles it, then to that target's
  tests via ``ctest -R`` or a label. Touching ``CMakeLists.txt`` widens.

**Do not port the fail-safe as an afterthought.** It is the reason this is
safe to run at all: a filter that quietly matches nothing looks exactly like a
filter that legitimately selected nothing, and both report success. Whatever
maps paths in your language, the rule stays -- when in doubt, run everything.

Usage
-----
    python scripts/scoped_tests.py            # working tree vs HEAD
    python scripts/scoped_tests.py --range A..B
    python scripts/scoped_tests.py --print    # show the filter, run nothing

Exits with the test runner's own status so it can gate a commit directly.
"""

import argparse
import json
import os
import subprocess
import sys

# Touching any of these invalidates the mapping, so we run everything. These are
# repository-root paths; a *member's* Cargo.toml is handled per package below.
FULL_SUITE_TRIGGERS = (
    "cargo.toml",
    "cargo.lock",
    "build.rs",
    ".config/nextest.toml",
)

# Paths that cannot break a test on their own and so contribute no filter.
#
# ``graphify-out/`` is GENERATED DOCUMENTATION -- a knowledge-graph dump some
# projects track so library consumers get the structure map. Nothing compiles it
# and no test reads it, so it belongs here on the merits. It is listed by prefix
# rather than left to DOC_SUFFIXES because the directory also holds ``.html``,
# which is not a documentation suffix and would therefore force a full run. That
# matters where a post-commit hook refreshes the graph: without this line almost
# every commit widens, and the script quietly stops scoping anything at all --
# it still reports success, which is the failure mode this file is written to
# avoid everywhere else.
DOC_PREFIXES = ("docs/", "dev-docs/", ".superpowers/", "planning/", "sbtdd/",
                "graphify-out/")

# The smoke harness. A THIRD category, and neither of the existing two would do.
#
# It cannot be "full": cargo does not compile Python, so no Rust test can be
# affected by it. It cannot be DOC_PREFIXES either -- that routes to ``None``,
# which widens to the full suite all the same, so the cost would survive the
# classification. And it cannot reuse the empty-diff answer, whose message tells
# the reader the file is gitignored and invisible to git: ``smoke/`` is TRACKED,
# so that message would state the opposite of the truth about their own tree.
#
# This does not weaken the "when in doubt, widen" rule. ``smoke/`` stops being an
# UNKNOWN path and becomes a CLASSIFIED one; what is removed is the doubt, not
# the widening. Its corollary is mandatory and lives in the message below: if
# this script runs nothing for the harness, something else must gate it, or the
# saving is paid for by leaving the harness unverified.
HARNESS_PREFIXES = ("smoke/",)
DOC_SUFFIXES = (".md", ".toml", ".json", ".yml", ".yaml", ".txt")


def changed_files(rev_range, cwd=None):
    """Return the repository-relative paths the range (or worktree) touches.

    UNTRACKED files count. ``git diff`` lists only what git already knows
    about, so a brand-new source file that has not been staged is invisible to
    it -- and the script would then take its one shortcut, "git reports no
    change", and run nothing at all on the commit that adds a file. The empty
    answer has to mean there is no change, never that git was not looking.
    Ignored paths stay out (``--exclude-standard``): widening the whole suite
    for a log file is the other direction of the same mistake.

    A rev range names two commits, and an untracked file belongs to neither,
    so the second listing is taken only for the worktree.
    """
    cmd = ["git", "diff", "--name-only"]
    cmd.append(rev_range if rev_range else "HEAD")
    listed = subprocess.run(cmd, capture_output=True, text=True, check=True,
                            cwd=cwd).stdout.splitlines()
    if not rev_range:
        listed += subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, check=True, cwd=cwd).stdout.splitlines()
    seen, paths = set(), []
    for line in listed:
        path = line.strip().replace("\\", "/")
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def repo_root():
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True, check=True).stdout
    return out.strip().replace("\\", "/")


def packages():
    """Return [(package_name, dir_relative_to_repo_root)], longest dir first.

    Sorting by descending path length lets the caller take the first prefix
    match, which is the innermost package -- the right answer when one member
    lives inside another's directory tree.
    """
    try:
        out = subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"],
            capture_output=True, text=True, check=True).stdout
        meta = json.loads(out)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None  # Not a cargo project, or cargo is unavailable.

    root = repo_root()
    found = []
    for pkg in meta.get("packages", []):
        manifest = pkg.get("manifest_path", "").replace("\\", "/")
        pkg_dir = os.path.dirname(manifest)
        rel = os.path.relpath(pkg_dir, root).replace("\\", "/")
        rel = "" if rel == "." else rel
        found.append((pkg["name"], rel))
    found.sort(key=lambda item: len(item[1]), reverse=True)
    return found


def owning_package(path, pkgs):
    """Return (package_name, path_relative_to_that_package), or None."""
    for name, pkg_dir in pkgs:
        if not pkg_dir:
            return name, path
        if path.startswith(pkg_dir + "/"):
            return name, path[len(pkg_dir) + 1:]
    return None


def scope(name, inner, single_crate):
    """Wrap a filter fragment in a package predicate unless the crate is alone."""
    if single_crate:
        return inner
    if inner is None:
        return "package(%s)" % name
    return "package(%s) & %s" % (name, inner)


def module_filter(path, pkgs=None, single_crate=True):
    """Map one repository path to a nextest filter fragment.

    Returns ``"full"`` when the path forces a full run, ``None`` when it
    contributes no filter at all.
    """
    lowered = path.lower()
    if lowered in FULL_SUITE_TRIGGERS:
        return "full"

    # Before the package lookup and before DOC_SUFFIXES: ``smoke/*.toml`` would
    # otherwise be read as documentation and ``smoke/*.py`` as an unknown path,
    # and both of those end in the full suite.
    if path.startswith(HARNESS_PREFIXES):
        return "harness"

    if pkgs:
        owned = owning_package(path, pkgs)
        if owned is None:
            # Inside the repository but outside every package: documentation and
            # tooling live here, and anything else we do not understand.
            if path.startswith(DOC_PREFIXES) or path.endswith(DOC_SUFFIXES):
                return None
            return "full"
        name, rel = owned
    else:
        name, rel = None, path

    # A member's own manifest or build script: scope to that package.
    if rel.lower() in ("cargo.toml", "build.rs"):
        return "full" if single_crate else scope(name, None, False)

    if rel.startswith("tests/") and rel.endswith(".rs"):
        stem = rel[len("tests/"):-len(".rs")]
        if "/" in stem:
            # A helper module under tests/ is compiled into several binaries and
            # we cannot tell which, so widen rather than guess.
            return "full" if single_crate else scope(name, None, False)
        return scope(name, "binary(%s)" % stem, single_crate)

    if rel.startswith("src/") and rel.endswith(".rs"):
        parts = rel[len("src/"):-len(".rs")].split("/")
        if parts[-1] == "mod":
            # A module root: its children may depend on it, so take the subtree.
            parts = parts[:-1]
        if not parts or parts == ["main"] or parts == ["lib"]:
            # The crate root. In a workspace that means "this package"; in a
            # single crate it means everything, so there is nothing to narrow.
            return "full" if single_crate else scope(name, None, False)
        return scope(name, "test(/%s::/)" % "::".join(parts), single_crate)

    if path.startswith(DOC_PREFIXES) or path.endswith(DOC_SUFFIXES):
        return None

    return "full"


def build_filter(paths, pkgs):
    """Return (expression, reason).

    ``None`` means "documentation or config only" -- widen to the full suite.
    ``"none"`` means git sees no change at all, which is different and must not
    widen: there is nothing to verify, so the previous verification still
    stands. Conflating the two burns a full suite every time someone edits a
    git-ignored file, which is exactly the waste this script exists to remove.
    """
    if not paths:
        return "none", "git reports no change"

    single_crate = not pkgs or len(pkgs) == 1
    fragments = []
    harness_touched = False
    for path in paths:
        mapped = module_filter(path, pkgs, single_crate)
        if mapped == "full":
            return "full", "%s forces a full run" % path
        if mapped == "harness":
            # Contributes no filter, and must not erase one: a change that
            # touches the harness AND Rust still runs the Rust selection.
            harness_touched = True
            continue
        if mapped and mapped not in fragments:
            fragments.append(mapped)

    if not fragments:
        if harness_touched:
            return "harness", "only the smoke harness changed"
        return None, "only documentation or config changed"
    if len(fragments) == 1:
        return fragments[0], "1 path group mapped"
    return " | ".join("(%s)" % f for f in fragments), "%d path groups mapped" % len(fragments)


def main():
    parser = argparse.ArgumentParser(description="Run the tests a change can affect.")
    parser.add_argument("--range", dest="rev_range", default=None,
                        help="git revision range, e.g. HEAD~3..HEAD")
    parser.add_argument("--print", dest="print_only", action="store_true",
                        help="print the derived command and exit")
    args = parser.parse_args()

    paths = changed_files(args.rev_range)
    pkgs = packages()
    if pkgs is None:
        print("[scoped-tests] cargo metadata unavailable: running the full suite")
        expression, reason = "full", "no package metadata"
    else:
        expression, reason = build_filter(paths, pkgs)
        if len(pkgs) > 1:
            print("[scoped-tests] workspace with %d packages" % len(pkgs))

    if expression == "none":
        print("[scoped-tests] %s -- nothing to verify, so nothing is run." % reason)
        print("[scoped-tests] A git-ignored file (CLAUDE.md, .claude/, dev-docs/, planning/) "
              "is invisible to git and cannot affect the build.")
        return 0

    if expression == "harness":
        print("[scoped-tests] %s -- cargo does not compile it, so no Rust test "
              "can be affected." % reason)
        print("[scoped-tests] The harness has its own gate and it is NOT optional: "
              "python -m compileall -q smoke/ && python -m unittest discover smoke/tests -q")
        return 0

    if expression == "full":
        print("[scoped-tests] FULL suite: %s" % reason)
        cmd = ["cargo", "nextest", "run"]
    elif expression is None:
        print("[scoped-tests] %s -- running the full suite anyway, so the gate is "
              "never green over nothing" % reason)
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
