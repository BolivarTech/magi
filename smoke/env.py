# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""The test environment's lifecycle.

The environment is PERSISTENT on purpose (REQ-S30). The real user's case is a
database that already holds data, and an environment rebuilt on every run only
ever exercises first startup -- the one case that never fails. Keeping it is
what stops S4's never-delete check and S9's memory checks from being vacuous.

``scratch/`` is the single declared exception: ``magi init`` can only be
exercised against a directory that does NOT yet carry ``.magi/``, which S2 and
S14 need.

Only the CONFIGURATION is ever normalised, never the data. That distinction is
the whole of :meth:`Environment.normalize_magi_toml`.
"""

import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass

from smoke.config import ModelProfile
from smoke.errors import HarnessError, PreflightError

#: The environment's own gitignore. ``*`` hides everything that lands here and
#: ``!.gitignore`` exempts the file itself, so the rule is self-protecting: a
#: ``.gitkeep`` would make the directory exist without stopping anybody from
#: committing the database that grows inside it.
ENV_GITIGNORE = "*\n!.gitignore\n"

GITIGNORE_NAME = ".gitignore"
MAGI_DIR_NAME = ".magi"
MAGI_TOML_NAME = "magi.toml"
DATABASE_NAME = ".magi-rs-memory.db"
RUNS_DIR_NAME = "runs"
PAYLOAD_DIR_NAME = "payload"
SCRATCH_DIR_NAME = "scratch"

#: The product subcommand that writes a starting ``magi.toml``.
INIT_SUBCOMMAND = "init"

#: How long the product is given to scaffold one temporary tree, in seconds.
#: ``magi init`` writes a handful of small files and touches no network, so a
#: run that has not finished by now is hung rather than slow.
INIT_TIMEOUT_SECONDS = 60

#: The product's own table inside the environment's ``magi.toml``. It is NOT
#: ``[calibration]``, which is the harness's own table in ``smoke.toml``: two
#: related sets of numbers in two files owned by two programs.
MEMORY_SECTION = "memory"

#: The keys the cheap profile overrides, by table. ``[openai].model`` is the
#: main agent's model for both ``ollama`` and ``openai-compat`` -- they share
#: the completions protocol -- and the three ``[magi]`` seats are the trio.
PROFILE_MODEL_KEY = ("openai", "model")

#: The three seats, in the order a profile's trio declares them. Each one owns
#: TWO keys in the product's file -- ``<seat>_model`` and ``<seat>_lineage`` --
#: and the harness rewrites both or neither. Since v0.13.0 the product treats a
#: seat that declares a model without its lineage as a load error, so writing
#: one and leaving the other is not a smaller change: it is a broken file.
PROFILE_SEATS = ("melchior", "balthasar", "caspar")


#: The product's startup diagnostics line. The active-memory count cannot be
#: read off the filesystem -- the database is encrypted -- so the product's own
#: report is the only source, and that makes a RUN its producer. Anchored on
#: the counts rather than on the whole sentence so a change to the index-size
#: suffix does not silently stop it being found.
STARTUP_LINE = re.compile(
    rb"memory:\s*(\d+)\s+active,\s*(\d+)\s+archived,\s*(\d+)\s+pending re-embed"
)


def active_memories(capture):
    """How many memories the product reported active, or None.

    Complexity: ``O(len(capture))``.

    Args:
        capture: Bytes a run printed.

    Returns:
        int | None: The count from the LAST startup line -- runs execute in
        order, so the newest is the current one -- or None when no line was
        found. None means NOT MEASURED, never zero.
    """
    found = STARTUP_LINE.findall(capture)
    return int(found[-1][0]) if found else None


@dataclass(frozen=True)
class Growth:
    """How large the persistent environment has become.

    Nothing here enforces a limit. Growth is made VISIBLE rather than capped,
    because the threshold at which it starts to hurt has not been measured, and
    an invented limit is a number defended rather than known. ``--reset-env``
    is the recovery.

    Attributes:
        db_bytes: Size of the product's encrypted database, or 0 if absent.
        runs_bytes: Total bytes of every archived run artifact.
        active_memories: The product's active-memory count when a run has
            reported one, otherwise ``None``. It cannot be read off the
            filesystem: the database is encrypted, so the only source is the
            product's own diagnostics line. ``None`` means NOT MEASURED, never
            zero -- the same distinction the harness draws between
            ``CANNOT_TEST`` and ``FAIL``.
    """

    db_bytes: int
    runs_bytes: int
    active_memories: int | None


class Environment:
    """The persistent test environment under ``smoke/env/``.

    Example:
        Under a temporary root, never ``smoke/env`` itself. The examples in
        this package are executed by the suite, and ``init`` removes the root
        before it rebuilds -- so an example naming the real environment would
        make running the unit tests destroy the accumulated history the
        environment exists to hold, taking the decision that :meth:`init`
        raises a ``PreflightError`` precisely to leave with a human.

        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as scratch:
        ...     env = Environment(pathlib.Path(scratch) / "env")
        ...     env.exists()
        False
    """

    def __init__(self, root: pathlib.Path | str) -> None:
        """Bind to one environment root without touching the filesystem.

        Args:
            root: The environment directory, normally ``smoke/env``. It is not
                created here: construction is cheap and total, so the CLI can
                build one before deciding whether to init, reset or run.
        """
        self._root = pathlib.Path(root)

    @property
    def root(self) -> pathlib.Path:
        """The environment directory itself.

        Returns:
            The root path, whether or not it exists.
        """
        return self._root

    @property
    def magi_dir(self) -> pathlib.Path:
        """The product's workspace inside the environment.

        Returns:
            The ``.magi/`` directory holding ``magi.toml`` and the database.
        """
        return self._root / MAGI_DIR_NAME

    @property
    def runs_dir(self) -> pathlib.Path:
        """Where each run's scrubbed command and output are archived.

        Returns:
            The ``runs/`` directory.
        """
        return self._root / RUNS_DIR_NAME

    @property
    def payload_dir(self) -> pathlib.Path:
        """Where the generated large payload lives.

        Returns:
            The ``payload/`` directory.
        """
        return self._root / PAYLOAD_DIR_NAME

    @property
    def scratch_dir(self) -> pathlib.Path:
        """The fixture area for the scenarios that need a virgin tree.

        It is a SIBLING of the environment root, never a child, and that is
        forced by what it is for. The root carries the product's own
        ``.magi/``, and ``magi init`` refuses to nest a second one inside an
        existing one -- so a scratch area under the root fails at exactly the
        thing it was created for, and every scenario seeding a workspace there
        reads the product's correct refusal as its own inability to run.

        Returns:
            The ``scratch/`` directory beside the environment.
        """
        return self._root.parent / SCRATCH_DIR_NAME

    def exists(self) -> bool:
        """Report whether the environment has been INITIALISED.

        Not whether the directory is present, and the difference is the whole
        point. ``smoke/env/.gitignore`` is tracked, so the directory exists in
        every clone before anything has been built -- reading existence off the
        directory made ``--init-env`` refuse on a fresh checkout and left the
        harness unusable out of the box. The marker is the product's workspace,
        which is what ``init`` creates and what every later step needs.

        Returns:
            bool: ``True`` when the environment carries the product workspace.
        """
        return self.magi_dir.is_dir()

    def init(self) -> None:
        """Create the environment, all of it or none of it.

        The tree is built in a temporary SIBLING and renamed into place, which
        is the same discipline ``magi init`` uses: a failure half way through
        leaves the staging directory behind and the environment simply absent,
        rather than a directory that exists, looks initialised, and is missing
        the one subdirectory a scenario needs.

        Raises:
            PreflightError: If the environment already exists. Destroying it
                silently would throw away the accumulated history that makes S4
                and S9 meaningful, so the message names ``--reset-env`` and the
                choice stays with the operator.
            HarnessError: If the tree cannot be built or moved into place.
        """
        if self.exists():
            raise PreflightError(
                "%s already exists; run --reset-env to destroy and rebuild it. "
                "The accumulated history is not discarded by accident."
                % self._root
            )
        self._root.parent.mkdir(parents=True, exist_ok=True)
        staging = pathlib.Path(
            tempfile.mkdtemp(prefix=".%s." % self._root.name, dir=self._root.parent)
        )
        try:
            for name in (MAGI_DIR_NAME, RUNS_DIR_NAME, PAYLOAD_DIR_NAME):
                (staging / name).mkdir()
            (staging / GITIGNORE_NAME).write_text(ENV_GITIGNORE, encoding="utf-8")
            # The root may already be present and empty: its .gitignore is
            # TRACKED, so a fresh clone carries the directory before anything is
            # built. os.replace onto an existing directory is refused on Windows
            # (WinError 5), so the empty root is removed first.
            #
            # That trades one guarantee for another rather than dropping it. The
            # window is now "root gone, staging not yet renamed", and a crash
            # there leaves the environment ABSENT -- which git restores and
            # --init-env rebuilds. The guarantee kept is the one that matters:
            # never a directory that exists, looks initialised, and is missing
            # the subdirectory a scenario needs.
            if self._root.is_dir():
                shutil.rmtree(self._root)
            os.replace(staging, self._root)
        except OSError as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise HarnessError(
                "could not create the test environment at %s: %s" % (self._root, exc)
            ) from exc
        self.prepare_scratch()

    def prepare_scratch(self) -> None:
        """Create the scratch area beside the environment and ignore it.

        It is not built in the staging tree and renamed with the rest, because
        it is not part of the rename's atomicity argument: the guarantee there
        is that the ENVIRONMENT is never half-built, and the scratch area holds
        only throwaway workspaces a scenario rebuilds anyway. It carries its
        own self-protecting ``.gitignore`` because the environment's covers the
        environment and nothing else.

        Raises:
            HarnessError: If the directory or its ignore rule cannot be
                written.
        """
        try:
            self.scratch_dir.mkdir(parents=True, exist_ok=True)
            (self.scratch_dir / GITIGNORE_NAME).write_text(
                ENV_GITIGNORE, encoding="utf-8")
        except OSError as exc:
            raise HarnessError(
                "could not create the scratch area at %s: %s"
                % (self.scratch_dir, exc)
            ) from exc

    def reset(self) -> None:
        """Destroy the environment and build it again.

        The scratch area goes with it. It is not part of the environment's
        directory any more, but it is part of what a run leaves behind -- one
        throwaway workspace per seeding scenario per run -- and a reset that
        left it standing would be a partial one.

        Raises:
            HarnessError: If the existing tree cannot be removed, or the
                rebuild fails.
        """
        for directory in (self._root if self.exists() else None,
                          self.scratch_dir if self.scratch_dir.is_dir()
                          else None):
            if directory is None:
                continue
            try:
                shutil.rmtree(directory)
            except OSError as exc:
                raise HarnessError(
                    "could not remove %s: %s" % (directory, exc)
                ) from exc
        self.init()

    def growth(self) -> Growth:
        """Measure how large the environment has become.

        Complexity: ``O(number of archived files)`` -- one ``stat`` per file
        under ``runs/``, walked once per run, never per finding.

        Returns:
            Growth: The database size, the archived bytes, and ``None`` for the
            active-memory count, which no run reports yet.
        """
        database = self.magi_dir / DATABASE_NAME
        try:
            db_bytes = database.stat().st_size
        except OSError:
            db_bytes = 0
        runs_bytes = 0
        for path in self.runs_dir.rglob("*"):
            try:
                if path.is_file():
                    runs_bytes += path.stat().st_size
            except OSError:
                # A file archived by a run that is still writing can vanish
                # between the walk and the stat. Missing bytes understate the
                # total; failing the measurement would lose all of it.
                continue
        return Growth(db_bytes=db_bytes, runs_bytes=runs_bytes,
                      active_memories=None)

    #: Where a model name can appear in the product's configuration: the
    #: main agent under ``[openai]``, and the embedder under ``[embedding]``.
    #: The three ``[magi]`` seats are read separately because their keys carry
    #: the seat name.
    MODEL_TABLES = ("openai", "embedding")

    def declared_models(self):
        """Every model this environment asks the product to reach.

        ``[[fallback]]`` entries are deliberately EXCLUDED. A missing fallback
        degrades a rotation rather than stopping a run, and the product already
        reports one as unmeasured instead of refusing; requiring them to exist
        would cut a run over a spare nobody has reached for.

        Read off the file the product wrote, never from a copy of its
        defaults: the copy is the one that forgets to be updated.

        Complexity: ``O(size of the configuration)``.

        Returns:
            set[str]: The declared names, empty when the file cannot be read.
        """
        try:
            document = tomllib.loads(
                (self.magi_dir / MAGI_TOML_NAME).read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            return set()
        names = set()
        for table in self.MODEL_TABLES:
            value = (document.get(table) or {}).get("model")
            if isinstance(value, str) and value:
                names.add(value)
        magi = document.get("magi") or {}
        for seat in PROFILE_SEATS:
            value = magi.get("%s_model" % seat)
            if isinstance(value, str) and value:
                names.add(value)
        return names

    def declared_base_url(self):
        """The root ``base_url`` the environment's configuration declares.

        This is where the RUNS go. ``[backend].base_url`` in ``smoke.toml`` is
        a separate setting read only by the reachability probe, and the two
        drifting apart is invisible: the probe reports a healthy backend about
        a host nothing talks to.

        Returns:
            str | None: The declared endpoint, or None when the file is absent
            or unreadable. None means NOT MEASURABLE, and the caller treats it
            as such rather than as a disagreement.
        """
        try:
            document = tomllib.loads(
                (self.magi_dir / MAGI_TOML_NAME).read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            return None
        value = document.get("base_url")
        return value if isinstance(value, str) else None

    def memory_settings(self) -> dict[str, object]:
        """Read the product's ``[memory]`` table out of the environment.

        A field the product wrote **commented out** counts, and that is the
        point. ``magi init`` renders two of the three memory fields as comments
        with their real default values interpolated, so the file carries the
        product's own defaults as data. Reading only the live keys left S9's
        ceiling underivable forever -- on the certifying run too -- and the
        alternative, copying the defaults into the harness, is the second
        source of truth this project rejects everywhere else.

        An unreadable or absent file yields an EMPTY mapping rather than an
        error, and empty is not zero: S9 turns it into ``CANNOT_TEST``, because
        asserting a derived ceiling against defaults that were never declared
        would test a computation that did not happen. A key that appears
        nowhere -- not even as a comment -- is still absent.

        Returns:
            The ``[memory]`` table, or ``{}`` when the file is missing, does not
            parse, or declares no such table.
        """
        path = self.magi_dir / MAGI_TOML_NAME
        try:
            with path.open("rb") as handle:
                document = tomllib.load(handle)
            text = path.read_text(encoding="utf-8")
        except (OSError, tomllib.TOMLDecodeError):
            return {}
        section = document.get(MEMORY_SECTION)
        settings = _commented_defaults(text, MEMORY_SECTION)
        if isinstance(section, dict):
            # A live key always wins: an operator who uncommented a line and
            # changed it meant it, and the commented one is now history.
            settings.update(section)
        return settings

    def normalize_magi_toml(self, profile: ModelProfile | None) -> None:
        """Rewrite the environment's ``magi.toml``, and nothing else.

        With a *profile* the file names that profile's model and trio. With
        ``None`` it names the PRODUCT's own defaults, obtained by running the
        product's ``magi init`` and taking what it wrote: the product is the
        single source of truth for its own defaults, and a copy kept here would
        be the copy that forgets to be updated. That is what keeps a certifying
        run from claiming product defaults over an environment still pointing at
        the cheap models left behind by the previous run.

        The data is untouched. Only the configuration is normalised.

        Args:
            profile: The cheap profile, or ``None`` for the product's defaults.

        Raises:
            HarnessError: If ``magi init`` cannot be run, writes nothing, or the
                transplanted text still carries the temporary directory's path.
                All three are defects in the harness, reported as such and never
                as a verdict on the product.
        """
        generated = self._product_default_toml()
        text = generated if profile is None else _apply_profile(generated, profile)
        self.magi_dir.mkdir(parents=True, exist_ok=True)
        (self.magi_dir / MAGI_TOML_NAME).write_text(text, encoding="utf-8")

    def _product_default_toml(self) -> str:
        """Ask the product to write a default ``magi.toml`` and read it back.

        ``magi init`` runs in a PRIVATE temporary directory, never in
        ``env/scratch/``: it refuses on a tree that already carries ``.magi/``,
        so it cannot be aimed at the environment itself, and ``scratch/`` is the
        declared fixture area for S2 and S14, where an extra ``.magi/`` would
        change what those two scenarios find.

        Returns:
            The generated file's text.

        Raises:
            HarnessError: If the binary cannot be run, exits non-zero, writes no
                file, or the generated text embeds the temporary path. That last
                check is not paranoia: a config carrying an absolute path into a
                directory about to be deleted would leave the environment
                pointing at nothing, and the failure would surface much later as
                an opaque product error.
        """
        # Imported here rather than at module scope on purpose. This is the one
        # method that needs the product binary; importing it at the top would
        # make every consumer of Environment -- including the lifecycle commands
        # that must work before anything is built -- depend on the locator.
        from smoke.binary import ReleaseBinary

        repo_root = pathlib.Path(__file__).resolve().parent.parent
        binary = ReleaseBinary(repo_root)
        scratch = pathlib.Path(tempfile.mkdtemp())
        try:
            try:
                completed = subprocess.run(
                    [str(binary.path), INIT_SUBCOMMAND],
                    cwd=str(scratch),
                    capture_output=True,
                    timeout=INIT_TIMEOUT_SECONDS,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise HarnessError(
                    "could not run the product's %s to obtain its defaults: %s"
                    % (INIT_SUBCOMMAND, exc)
                ) from exc
            if completed.returncode != 0:
                raise HarnessError(
                    "the product's %s exited %d while obtaining its defaults"
                    % (INIT_SUBCOMMAND, completed.returncode)
                )
            written = scratch / MAGI_DIR_NAME / MAGI_TOML_NAME
            try:
                text = written.read_text(encoding="utf-8")
            except OSError as exc:
                raise HarnessError(
                    "the product's %s wrote no %s to read defaults from"
                    % (INIT_SUBCOMMAND, MAGI_TOML_NAME)
                ) from exc
            _reject_transplanted_path(text, scratch)
            return text
        finally:
            shutil.rmtree(scratch, ignore_errors=True)


def _reject_transplanted_path(text: str, scratch: pathlib.Path) -> None:
    """Refuse a generated config that names the temporary directory.

    Both separator spellings are checked because a config may normalise a
    Windows path to forward slashes on its way through a serialiser, and a
    check that only knows one spelling passes on exactly the file it exists to
    catch.

    Args:
        text: The generated configuration.
        scratch: The temporary directory it was generated in.

    Raises:
        HarnessError: If any spelling of the temporary path appears in *text*.
    """
    spellings = set()
    for candidate in (scratch, scratch.resolve()):
        rendered = str(candidate)
        spellings.add(rendered)
        spellings.add(rendered.replace("\\", "/"))
    for spelling in spellings:
        if spelling in text:
            raise HarnessError(
                "the generated %s carries the temporary directory it was built "
                "in, so the environment would point at a directory that no "
                "longer exists" % MAGI_TOML_NAME
            )


def _apply_profile(text: str, profile: ModelProfile) -> str:
    """Overwrite the model and trio keys of a generated configuration.

    The rewrite is line-wise and table-aware rather than a parse-and-serialise
    round trip, because the standard library reads TOML and does not write it,
    and REQ-S02 rules out the package that would. Working on the product's own
    generated file keeps every other key -- including each seat's lineage --
    exactly as the product wrote it.

    Each seat's model and lineage are rewritten **together**. The profile
    declares both because the operator declares both: a lineage is a
    user-chosen failure domain, never inferred from a model name. Rewriting the
    model alone would leave the file describing a failure domain the seat no
    longer belongs to, and the product's diversity check would then pass on
    three labels about models that are not there -- a diversity guarantee that
    is true of the file and false of the run.

    Complexity: ``O(lines)``.

    Args:
        text: The generated configuration.
        profile: The profile whose model and trio should win.

    Returns:
        The rewritten configuration.
    """
    replacements = {PROFILE_MODEL_KEY: profile.model}
    for seat_name, seat in zip(PROFILE_SEATS, profile.trio):
        replacements[("magi", "%s_model" % seat_name)] = seat.model
        replacements[("magi", "%s_lineage" % seat_name)] = seat.lineage

    table = ""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            table = stripped[1:-1].strip()
            lines.append(line)
            continue
        key = stripped.split("=", 1)[0].strip() if "=" in stripped else ""
        replacement = replacements.get((table, key))
        if replacement is None or stripped.startswith("#"):
            lines.append(line)
            continue
        lines.append('%s = "%s"' % (key, _toml_value(key, replacement)))
    return "\n".join(lines) + "\n"


#: Characters a TOML basic string cannot carry raw. A model tag or a lineage
#: label never legitimately holds either.
_UNQUOTABLE = ('"', "\\")


def _toml_value(key: str, value: str) -> str:
    """Return *value*, or refuse it when it cannot be interpolated verbatim.

    The rewrite writes ``key = "value"`` by interpolation, so a value holding
    a double quote or a backslash produces a file that is malformed at best
    and carries keys nobody wrote at worst. What discovers that is the
    PRODUCT, refusing to parse a configuration the HARNESS generated -- a
    failure about as far from its cause as this design gets.

    Refused rather than escaped, deliberately. An escaped tag would be
    written faithfully and then not exist on the daemon, so the run would die
    later and further away; a tag carrying either character is simply wrong,
    and saying so at the profile line is the shortest path to the fix.

    Args:
        key: The configuration key being written, for the message.
        value: The value the profile declared.

    Returns:
        str: The value unchanged, when it is safe to interpolate.

    Raises:
        PreflightError: If it carries a character a basic string cannot hold.

    Example:
        >>> _toml_value("melchior_model", "kimi-k2.6:cloud")
        'kimi-k2.6:cloud'
    """
    for character in _UNQUOTABLE:
        if character in value:
            raise PreflightError(
                "the profile's %s is %r, which carries a %s. A model tag or a "
                "lineage never does, and interpolating it would generate a "
                "magi.toml the product cannot parse -- so the failure would "
                "surface as the product refusing a file the harness wrote. "
                "Fix the value in the profile."
                % (key, value,
                   "double quote" if character == '"' else "backslash")
            )
    return value


def _commented_defaults(text: str, section: str) -> dict:
    """Recover the values the product wrote as commented-out defaults.

    Table-aware on purpose: a commented key under another table is not a
    setting of this one just because it is commented.

    Complexity: ``O(lines)``.

    Args:
        text: The configuration file's text.
        section: The table whose commented keys to recover.

    Returns:
        dict: Key to value for every commented assignment inside *section*
        whose right-hand side parses as TOML. A comment that is prose rather
        than an assignment is skipped rather than guessed at.
    """
    recovered: dict = {}
    table = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            table = stripped[1:-1].strip()
            continue
        if table != section or not stripped.startswith("#"):
            continue
        body = stripped.lstrip("#").strip()
        if "=" not in body:
            continue
        key = body.split("=", 1)[0].strip()
        if not key.isidentifier():
            continue
        try:
            parsed = tomllib.loads(body)
        except tomllib.TOMLDecodeError:
            continue
        if key in parsed:
            recovered[key] = parsed[key]
    return recovered
