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
matches nothing raises no error, so it silently stops applying. Anything this
script cannot map confidently -- a workspace manifest, a lockfile, the nextest
config, an unrecognised path -- falls back to the FULL suite. The default
direction is always "run more", never "run less".

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
DOC_PREFIXES = ("docs/", "dev-docs/", ".superpowers/", "planning/", "sbtdd/")
DOC_SUFFIXES = (".md", ".toml", ".json", ".yml", ".yaml", ".txt")


def changed_files(rev_range):
    """Return the repository-relative paths the range (or worktree) touches."""
    cmd = ["git", "diff", "--name-only"]
    cmd.append(rev_range if rev_range else "HEAD")
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [line.strip().replace("\\", "/") for line in out.splitlines() if line.strip()]


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
    """Return (expression, reason). ``None`` means 'nothing to run'."""
    if not paths:
        return None, "no changes detected"

    single_crate = not pkgs or len(pkgs) == 1
    fragments = []
    for path in paths:
        mapped = module_filter(path, pkgs, single_crate)
        if mapped == "full":
            return "full", "%s forces a full run" % path
        if mapped and mapped not in fragments:
            fragments.append(mapped)

    if not fragments:
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
