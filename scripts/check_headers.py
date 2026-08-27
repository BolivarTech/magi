# Author: Julian Bolivar
# Version: 0.17.0
# Date: 2026-08-27

"""Sweep the file headers of everything a milestone touched.

Why this exists
---------------
Every source file carries a three-line header::

    // Author: Julian Bolivar
    // Version: 0.17.0
    // Date: 2026-08-27

(``#`` instead of ``//`` in Python.) The Version is the RELEASE version and
the Date is the release date, so both go stale on every release that touches
the file. Nothing catches that decay on its own: a stale header breaks no
build, fails no test, and reads exactly like a fresh one. A manual pass over
the diff is precisely the chore that gets forgotten, so this script is the
mechanism instead of the reminder.

What it enumerates
------------------
The file list is DERIVED from ``git diff`` against a base ref, never typed by
hand -- a hand-kept list omits the file nobody remembered, which is the same
failure the header sweep exists to prevent. Four rules shape the enumeration:

* Deletions are excluded (``--diff-filter=d``). A deleted file has no header
  to stamp, and enumerating it aborts the sweep on a missing path.
* Renames resolve to the NEW path (``-M``); the old one no longer exists.
* Only the extensions that actually carry the header are in scope
  (``.rs``, ``.py``). A changed ``.toml``, ``.md``, ``.yml`` or ``.lock`` is
  not a source file and must not be reported as one missing a header.
* The release version comes from ``Cargo.toml``'s ``[package] version``, so
  the two can never disagree.

Modes
-----
Check (default)
    Exit 0 when every enumerated file's header carries the crate version and a
    ``YYYY-MM-DD`` date; otherwise print every offender and exit non-zero.
Write (``--write``)
    Stamp the release version and date into those headers, creating the header
    where a file has none (after a shebang, which stays on line 1). Running it
    twice is a no-op: the second run finds the header it wrote and rewrites the
    same two fields rather than stacking a second block on top.

Line endings are preserved per file. This tree is mixed -- most files are CRLF
on disk and some are LF -- and rewriting one with the wrong ending shows every
line as changed, which buries the real diff.

Example:
    Check the working tree against ``main``::

        python scripts/check_headers.py

    Stamp today's date and the crate version into everything the branch
    touched::

        python scripts/check_headers.py --write
"""

import argparse
import datetime
import pathlib
import re
import subprocess
import sys
import tomllib

#: Comment marker per extension. The keys double as the extension scope: an
#: extension absent from this map is not a source file for header purposes.
COMMENT_BY_EXTENSION = {".rs": "//", ".py": "#"}

#: The author line is a constant, not a checked field -- only Version and Date
#: decay, and rewriting authorship is not this script's business.
AUTHOR = "Julian Bolivar"

#: Default base ref for the diff. A milestone branches from here.
DEFAULT_BASE_REF = "main"

#: Exit code for "at least one file has a stale or missing header".
EXIT_OFFENDERS = 1

#: Exit code for "the sweep itself could not run" (bad ref, no Cargo.toml).
EXIT_ERROR = 2

#: How many lines from the top may precede the header block. A shebang or a
#: short run of crate attributes plus blank lines is the realistic ceiling;
#: scanning further would start matching an "Author:" mention inside a module
#: docstring or a licence banner.
MAX_PREAMBLE_LINES = 8

_AUTHOR_RE = re.compile(r"^\s*(?://|#)\s*Author:\s*(.+?)\s*$")
_VERSION_RE = re.compile(r"^\s*(?://|#)\s*Version:\s*(\S+)\s*$")
_DATE_RE = re.compile(r"^\s*(?://|#)\s*Date:\s*(\S+)\s*$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class HeaderSweepError(RuntimeError):
    """The sweep could not run at all.

    Raised for conditions that invalidate the whole run -- an unusable base
    ref, an unreadable ``Cargo.toml`` -- as opposed to a single file whose
    header is wrong, which is a finding rather than an error.
    """


class Header:
    """The location and contents of a header block inside one file.

    Attributes:
        start (int): Index of the ``Author`` line within the file's line list.
        version (str): The version string as found, e.g. ``"0.16.0"``.
        date (str): The date string as found, e.g. ``"2026-08-27"``.
        version_line (int): Index of the ``Version`` line.
        date_line (int): Index of the ``Date`` line.
    """

    def __init__(self, start: int, version: str, date: str) -> None:
        """Record a parsed header block.

        Args:
            start: Index of the ``Author`` line within the file's line list.
            version: The version string as found in the header.
            date: The date string as found in the header.
        """
        self.start = start
        self.version = version
        self.date = date

    @property
    def version_line(self) -> int:
        """int: Index of the ``Version`` line within the file's line list."""
        return self.start + 1

    @property
    def date_line(self) -> int:
        """int: Index of the ``Date`` line within the file's line list."""
        return self.start + 2


def run_git(repo_root: pathlib.Path, *args: str) -> str:
    """Run a git command inside a repository and return its stdout.

    Args:
        repo_root: Repository the command runs in.
        *args: Arguments after ``git``, e.g. ``"diff", "--name-status"``.

    Returns:
        str: The command's standard output, decoded as UTF-8.

    Raises:
        HeaderSweepError: If git is missing or exits non-zero. The stderr text
            is carried into the message so the caller sees the real cause
            rather than a bare exit code.
    """
    try:
        completed = subprocess.run(
            ("git",) + args,
            cwd=str(repo_root),
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - git is a hard dep
        raise HeaderSweepError("git is not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", "replace").strip()
        raise HeaderSweepError(f"git {' '.join(args)} failed: {detail}") from exc
    return completed.stdout.decode("utf-8", "replace")


def repo_root_from(start: pathlib.Path) -> pathlib.Path:
    """Locate the repository root containing a directory.

    Args:
        start: Any directory inside the repository.

    Returns:
        pathlib.Path: The absolute repository root.

    Raises:
        HeaderSweepError: If ``start`` is not inside a git repository.
    """
    top = run_git(start, "rev-parse", "--show-toplevel").strip()
    return pathlib.Path(top)


def changed_source_files(repo_root: pathlib.Path,
                         base_ref: str) -> list[pathlib.Path]:
    """Enumerate the source files changed since a base ref.

    Deletions are dropped, renames resolve to their new path, and anything
    outside :data:`COMMENT_BY_EXTENSION` is filtered out -- see the module
    docstring for why each of those three matters.

    Complexity: O(n) in the number of changed paths.

    Args:
        repo_root: Repository to diff.
        base_ref: Ref to compare the working tree against, e.g. ``"main"``.

    Returns:
        list[pathlib.Path]: Repository-relative paths, sorted and de-duplicated.

    Raises:
        HeaderSweepError: If the diff cannot be produced (unknown ref).
    """
    raw = run_git(repo_root, "diff", "--name-status", "--diff-filter=d",
                  "-M", base_ref)
    found: set[pathlib.Path] = set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        status = fields[0]
        # A rename or copy prints "R100<TAB>old<TAB>new": the header lives at
        # the destination, and the source path is already gone from disk.
        if status[:1] in ("R", "C") and len(fields) >= 3:
            path = fields[2]
        else:
            path = fields[-1]
        candidate = pathlib.Path(path)
        if candidate.suffix in COMMENT_BY_EXTENSION:
            found.add(candidate)
    return sorted(found)


def crate_version(repo_root: pathlib.Path) -> str:
    """Read the release version from ``Cargo.toml``.

    Args:
        repo_root: Repository whose ``Cargo.toml`` holds ``[package] version``.

    Returns:
        str: The version string, e.g. ``"0.17.0"``.

    Raises:
        HeaderSweepError: If ``Cargo.toml`` is absent, unparseable, or carries
            no ``[package] version`` key.
    """
    manifest = repo_root / "Cargo.toml"
    try:
        with manifest.open("rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        raise HeaderSweepError(f"cannot read {manifest}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise HeaderSweepError(f"cannot parse {manifest}: {exc}") from exc
    version = data.get("package", {}).get("version")
    if not isinstance(version, str) or not version:
        raise HeaderSweepError(f"{manifest} has no [package] version")
    return version


def read_text(path: pathlib.Path) -> str:
    """Read a file without translating its line endings.

    Args:
        path: File to read.

    Returns:
        str: The file's contents with CR and LF bytes left exactly as stored.

    Raises:
        HeaderSweepError: If the file cannot be read or decoded as UTF-8.
    """
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return handle.read()
    except OSError as exc:
        raise HeaderSweepError(f"cannot read {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise HeaderSweepError(f"{path} is not valid UTF-8: {exc}") from exc


def write_text(path: pathlib.Path, text: str) -> None:
    """Write a file without translating its line endings.

    Args:
        path: File to write.
        text: Contents whose CR and LF bytes are already correct.

    Raises:
        HeaderSweepError: If the file cannot be written.
    """
    try:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
    except OSError as exc:
        raise HeaderSweepError(f"cannot write {path}: {exc}") from exc


def dominant_newline(lines: list[str]) -> str:
    """Pick the line ending a newly inserted line should use.

    The first terminated line wins, which keeps an inserted header consistent
    with the block it is being inserted above rather than with the platform
    the script happens to run on.

    Args:
        lines: The file's lines with their endings kept.

    Returns:
        str: ``"\\r\\n"`` or ``"\\n"``; ``"\\n"`` for a file with no line end.
    """
    for line in lines:
        if line.endswith("\r\n"):
            return "\r\n"
        if line.endswith("\n"):
            return "\n"
    return "\n"


def is_pinned_line(line: str, comment: str, index: int) -> bool:
    """Say whether a line is pinned above the header.

    Two things outrank the header at the top of a file, and both would break if
    a header were inserted over them: a Python shebang, which stops working
    anywhere but line 1, and a Rust crate-level inner attribute such as
    ``#![forbid(unsafe_code)]``, which the compiler rejects after any item.
    ``src/main.rs`` in this tree is exactly the second case, so treating it as
    "no header" would report a file that has one.

    Args:
        line: The line to classify, endings included.
        comment: The comment marker for the file's language.
        index: The line's position, zero-based.

    Returns:
        bool: ``True`` when the header must go below this line.
    """
    if comment == "#":
        return index == 0 and line.startswith("#!")
    return line.lstrip().startswith("#![")


def header_start(lines: list[str], comment: str) -> int:
    """Find the line index where a header may legally begin.

    Skips whatever :func:`is_pinned_line` reports as outranking the header,
    plus any blank lines between it and the header.

    Args:
        lines: The file's lines with their endings kept.
        comment: The comment marker for the file's language.

    Returns:
        int: Index of the first line eligible to be the ``Author`` line.
    """
    index = 0
    while index < len(lines) and index < MAX_PREAMBLE_LINES:
        line = lines[index]
        if line.strip() and not is_pinned_line(line, comment, index):
            break
        index += 1
    return index


def parse_header(lines: list[str], comment: str) -> Header | None:
    """Parse the three-line header at the top of a file.

    Args:
        lines: The file's lines with their endings kept.
        comment: The comment marker for the file's language.

    Returns:
        Header | None: The parsed block, or ``None`` when the file has no
        header -- which write mode treats as "create one" and check mode
        treats as an offence.
    """
    start = header_start(lines, comment)
    if start + 3 > len(lines):
        return None
    author = _AUTHOR_RE.match(lines[start].rstrip("\r\n"))
    version = _VERSION_RE.match(lines[start + 1].rstrip("\r\n"))
    date = _DATE_RE.match(lines[start + 2].rstrip("\r\n"))
    if not (author and version and date):
        return None
    return Header(start, version.group(1), date.group(1))


def check_file(path: pathlib.Path, version: str) -> str | None:
    """Check one file's header against the release version.

    Args:
        path: Absolute path to the file.
        version: The release version the header must carry.

    Returns:
        str | None: A human-readable reason the file fails, or ``None`` when
        it conforms.

    Raises:
        HeaderSweepError: If the file cannot be read.
    """
    comment = COMMENT_BY_EXTENSION[path.suffix]
    lines = read_text(path).splitlines(keepends=True)
    header = parse_header(lines, comment)
    if header is None:
        return "no Author/Version/Date header"
    problems = []
    if header.version != version:
        problems.append(f"Version {header.version!r}, expected {version!r}")
    if not _ISO_DATE_RE.match(header.date):
        problems.append(f"Date {header.date!r} is not YYYY-MM-DD")
    return "; ".join(problems) if problems else None


def stamp_file(path: pathlib.Path, version: str, date: str) -> bool:
    """Stamp the release version and date into one file's header.

    Creates the header when the file has none, placing it after a shebang.
    Idempotent: a file already carrying the right values is not rewritten, so
    a second run stacks nothing and leaves the mtime alone.

    Args:
        path: Absolute path to the file.
        version: Release version to stamp.
        date: Release date to stamp, ``YYYY-MM-DD``.

    Returns:
        bool: ``True`` if the file was modified, ``False`` if already correct.

    Raises:
        HeaderSweepError: If the file cannot be read or written.
    """
    comment = COMMENT_BY_EXTENSION[path.suffix]
    original = read_text(path)
    lines = original.splitlines(keepends=True)
    newline = dominant_newline(lines)
    header = parse_header(lines, comment)

    if header is None:
        start = header_start(lines, comment)
        block = [
            f"{comment} Author: {AUTHOR}{newline}",
            f"{comment} Version: {version}{newline}",
            f"{comment} Date: {date}{newline}",
        ]
        # Separate the new header from whatever it now sits above, but never
        # from a blank line that is already there.
        if start < len(lines) and lines[start].strip():
            block.append(newline)
        if start > 0 and not lines[start - 1].endswith(("\n", "\r")):
            lines[start - 1] = lines[start - 1] + newline
        lines[start:start] = block
    else:
        ending = _line_ending(lines[header.version_line], newline)
        lines[header.version_line] = f"{comment} Version: {version}{ending}"
        ending = _line_ending(lines[header.date_line], newline)
        lines[header.date_line] = f"{comment} Date: {date}{ending}"

    updated = "".join(lines)
    if updated == original:
        return False
    write_text(path, updated)
    return True


def _line_ending(line: str, fallback: str) -> str:
    """Return the line ending of one line, so a rewrite preserves it.

    Args:
        line: The line whose ending is wanted, endings kept.
        fallback: Ending to use when the line has none (final line, no EOL).

    Returns:
        str: ``"\\r\\n"``, ``"\\n"``, or the fallback.
    """
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith("\n"):
        return "\n"
    return fallback


def sweep(repo_root: pathlib.Path, base_ref: str, version: str,
          date: str, write: bool) -> tuple[list[str], list[pathlib.Path]]:
    """Run the sweep over every changed source file.

    Args:
        repo_root: Repository to sweep.
        base_ref: Ref the working tree is diffed against.
        version: Release version the headers must carry.
        date: Release date used by write mode.
        write: ``True`` to stamp headers, ``False`` to only report.

    Returns:
        tuple[list[str], list[pathlib.Path]]: The offender messages (empty in
        write mode, which fixes rather than reports) and the paths modified.

    Raises:
        HeaderSweepError: If the diff, a read, or a write fails.
    """
    offenders: list[str] = []
    modified: list[pathlib.Path] = []
    for relative in changed_source_files(repo_root, base_ref):
        absolute = repo_root / relative
        if not absolute.is_file():
            # Belt and braces: --diff-filter=d already dropped deletions.
            continue
        if write:
            if stamp_file(absolute, version, date):
                modified.append(relative)
        else:
            problem = check_file(absolute, version)
            if problem is not None:
                offenders.append(f"{relative.as_posix()}: {problem}")
    return offenders, modified


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Check or stamp the Author/Version/Date header of every "
                    "source file changed since a base ref.")
    parser.add_argument("--base-ref", default=DEFAULT_BASE_REF,
                        help="ref to diff the working tree against "
                             f"(default: {DEFAULT_BASE_REF})")
    parser.add_argument("--write", action="store_true",
                        help="stamp the headers instead of only checking them")
    parser.add_argument("--version", default=None,
                        help="release version to require or stamp "
                             "(default: [package] version from Cargo.toml)")
    parser.add_argument("--date", default=None,
                        help="release date to stamp, YYYY-MM-DD "
                             "(default: today)")
    parser.add_argument("--repo-root", default=None,
                        help="repository to sweep (default: the one "
                             "containing the current directory)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list without the program name; ``None`` uses
            ``sys.argv[1:]``.

    Returns:
        int: ``0`` when everything conforms, :data:`EXIT_OFFENDERS` when a
        file's header is stale or missing, :data:`EXIT_ERROR` when the sweep
        could not run.
    """
    args = build_parser().parse_args(argv)
    try:
        if args.repo_root is not None:
            root = pathlib.Path(args.repo_root).resolve()
        else:
            root = repo_root_from(pathlib.Path.cwd())
        version = args.version or crate_version(root)
        date = args.date or datetime.date.today().isoformat()
        if args.date is not None and not _ISO_DATE_RE.match(date):
            raise HeaderSweepError(f"--date {date!r} is not YYYY-MM-DD")
        offenders, modified = sweep(root, args.base_ref, version, date,
                                    args.write)
    except HeaderSweepError as exc:
        print(f"check_headers: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if args.write:
        for path in modified:
            print(f"stamped {path.as_posix()}")
        print(f"{len(modified)} file(s) stamped with "
              f"Version {version}, Date {date}")
        return 0

    if offenders:
        print(f"{len(offenders)} file(s) with a stale or missing header "
              f"(expected Version {version}):")
        for offender in offenders:
            print(f"  {offender}")
        return EXIT_OFFENDERS

    print(f"all changed source files carry Version {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
