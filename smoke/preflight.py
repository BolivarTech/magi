# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""The nine hard cuts that run before any scenario.

A cut is a step that can raise ``PreflightError``, which is exit 2. Counted
by following each step into what it delegates to, there are nine: the
interpreter, the lock, the credential, the permissions, the environment, the
leftover rotation, the binary, the endpoint agreement and the declared
models. Two steps are NOT cuts -- normalising ``magi.toml`` raises
``HarnessError`` (exit 3, a defect of the harness rather than a refusal to
start), and the reachability probe cannot fail at all, by D-17.

The count was wrong twice, in both directions, and the reason is worth
keeping: the steps that cut mostly do not raise here. ``_settle_binary``
raises nothing of its own and delegates to ``binary.py``, which raises
``PreflightError`` in three methods; the lock raises from ``lock.py`` and the
configuration from ``config.py``. Counting the ``raise`` statements in THIS
module answers a different question, and answers it confidently.

Steps 1 to 8 are the spec's own numbering (section 4). ``7b`` (the
normalisation) and ``7c`` (endpoint agreement) are additions that came from
findings rather than from the spec's list, which is why they carry letters:
inserting them as whole numbers would renumber everything after them, and
the published documentation names these steps.

The order is the design, not a convenience. Two boundaries carry a defect each
if moved, and both were found by review rather than by a test that failed:

- **The lock comes before every mutation of the environment.** Normalising
  ``magi.toml`` outside it lets a concurrent run overwrite the file between the
  write and its use, so the run that must certify can end up on the cheap
  models under a certificate claiming the product's defaults.
- **The normalisation comes after the binary is settled**, because with no
  profile the defaults are obtained by running the product's own ``init``, and
  a certifying run rebuilds the binary first. Normalising earlier would take
  the previous build's defaults, so the certificate would name one version's
  defaults over a run that used another's.
"""

import dataclasses
import http.client
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

from smoke.config import PLACEHOLDER_ENTRIES, ROTATION_MARKER
from smoke.errors import PreflightError, ProductOutputError

#: How long the backend probe waits before calling it unreachable. A backend
#: that does not answer is not a failure (D-17): everything that does not need
#: it still runs, and what does is reported CANNOT_TEST.
BACKEND_PROBE_TIMEOUT_S = 10

#: The flag that tells the product WHICH workspace to open.
#:
#: Without it the product walks up from the harness's own working directory,
#: which is the repository root and not ``smoke/env/``. Step 6 then listed a
#: vault that does not exist, answered "cannot tell", and never detected the
#: rotation marker -- and had any ancestor of the launch directory carried a
#: ``.magi/``, the restore would have written the operator's real backend
#: credential into that unrelated workspace.
WORKDIR_FLAG = "-w"

#: The variable the passphrase travels in. Never ``-p``: that is a global flag
#: and would ride in the archived command line of every run.
PASSPHRASE_VAR = "MAGI_PASSPHRASE"

#: How long a vault command gets. Local work behind an Argon2 derivation.
VAULT_TIMEOUT_S = 60

#: How long ``icacls`` gets. It reads a local ACL; anything slower than this is
#: a broken system, not a slow one.
ICACLS_TIMEOUT_S = 30

#: POSIX bits that must all be clear on smoke.toml.
_FORBIDDEN_BITS = stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH

#: A DACL's access-control entries, as SDDL spells them. Well-known SID
#: abbreviations, identical in every locale -- which is the whole reason the
#: check reads SDDL and never the human listing icacls prints by default.
_SDDL_ACE = re.compile(r"\(A;[^)]*;([A-Z]{2})\)")

#: SIDs that must not appear in smoke.toml's DACL. ``WD`` is Everyone, ``BU``
#: is Builtin\\Users, ``AU`` is Authenticated Users.
_FORBIDDEN_SIDS = frozenset({"WD", "BU", "AU"})

#: The minimum Python the harness runs on.
MINIMUM_PYTHON = (3, 11)


@dataclasses.dataclass(frozen=True)
class BackendStatus:
    """Whether the backend answered, and why if it did not.

    Attributes:
        reachable: True when the backend answered the probe.
        cause: Empty when reachable; otherwise what went wrong, so a
            ``CANNOT_TEST`` finding can name it instead of saying nothing.
    """

    reachable: bool
    cause: str


def _read_sddl(path: pathlib.Path) -> str | None:
    """Read *path*'s security descriptor as SDDL.

    Args:
        path: The file whose ACL to read.

    Returns:
        str | None: The saved SDDL text, or ``None`` when it could not be
        obtained. ``None`` means **not measurable**, never "fine": the caller
        turns it into a refusal rather than a pass, because checking something
        easier and calling it green is the failure this whole module is written
        against.
    """
    # A TEMP directory, never beside the file. The config lives in the tracked
    # smoke/ tree and .gitignore covers smoke.toml and .lock and nothing else,
    # so a failed unlink here left a file S12 correctly reports as a trace the
    # harness had no business creating.
    holder = pathlib.Path(tempfile.mkdtemp())
    saved = holder / (path.name + ".sddl")
    try:
        completed = subprocess.run(
            ["icacls", str(path), "/save", str(saved)],
            capture_output=True, timeout=ICACLS_TIMEOUT_S,
        )
        if completed.returncode != 0 or not saved.exists():
            return None
        # icacls writes UTF-16LE. Decoding it wrongly yields text that parses
        # to no ACE at all, which would look exactly like a permissive file.
        text = saved.read_bytes().decode("utf-16-le", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        shutil.rmtree(holder, ignore_errors=True)
    return text if path.name in text else None


def check_config_permissions(path: pathlib.Path) -> None:
    """Refuse a ``smoke.toml`` that anyone but its owner can read.

    The file holds the environment's vault passphrase in the clear. On a shared
    runner, world-readable hands it over.

    Args:
        path: The configuration file.

    Raises:
        PreflightError: If the permissions are permissive, or -- on Windows --
            if the ACL could not be read at all. The second case is a refusal
            and not a pass on purpose: the harness reports what it could not
            measure rather than certifying that it looked.
    """
    if sys.platform == "win32":
        sddl = _read_sddl(path)
        if sddl is None:
            raise PreflightError(
                "the ACL of %s could not be read, so the harness cannot tell "
                "whether the passphrase is exposed. It refuses rather than "
                "assume: restrict the file with icacls and try again." % path
            )
        exposed = sorted(set(_SDDL_ACE.findall(sddl)) & _FORBIDDEN_SIDS)
        if exposed:
            raise PreflightError(
                "%s grants access to %s. Remove those entries with "
                "icacls %s /inheritance:r /grant:r \"%%USERNAME%%:F\""
                % (path, ", ".join(exposed), path)
            )
        return

    mode = os.stat(path).st_mode
    if mode & _FORBIDDEN_BITS:
        raise PreflightError(
            "%s is readable or writable beyond its owner. Fix it with "
            "chmod 600 %s" % (path, path)
        )


#: Where an Ollama daemon lists what it has. The OpenAI-compatible surface
#: publishes no equivalent, so a backend of any other kind cannot be asked and
#: the check degrades to NOT PERFORMED rather than to a guess.
TAGS_PATH = "/api/tags"

#: The backend kind that can be asked at all.
INTROSPECTABLE_KIND = "ollama"


def _tagged(name: str) -> str:
    """Spell a model name the way the daemon lists it.

    Ollama always answers ``/api/tags`` fully tagged, and an operator does not
    type the tag: ``ollama run nomic-embed-text-v2-moe`` is ordinary usage and
    the daemon reports it as ``nomic-embed-text-v2-moe:latest``. Comparing the
    two spellings literally reported a model as missing and refused the whole
    run, telling somebody to pull what they already had.

    The tag is looked for after the last ``/`` so a registry-qualified name
    like ``host/org/model`` is not mistaken for a tagged one.

    Args:
        name: A model name, tagged or not.

    Returns:
        str: The name with an explicit tag.
    """
    return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"


def require_declared_models(declared, available) -> None:
    """Step 8: every model the environment names exists on the backend.

    This is failure #5 -- a run that reaches a healthy backend and asks it for
    something it does not have. It belongs here rather than in a scenario
    because every trio run would otherwise spend its whole ceiling finding out.

    Complexity: ``O(len(declared))``.

    Args:
        declared: The model names the environment's configuration asks for.
        available: What the backend lists, or None when it could not be asked.
            None is NOT "nothing exists": every backend but Ollama publishes no
            tag listing, and cutting on silence would refuse all of them.

    Raises:
        PreflightError: Naming the models that are missing, and only those.
    """
    if available is None or not declared:
        return
    listed = {_tagged(name) for name in available}
    missing = sorted(name for name in declared if _tagged(name) not in listed)
    if not missing:
        return
    raise PreflightError(
        "the environment asks for %s, which the backend does not have. Pull "
        "%s, or point the configuration at a model that is installed."
        % (", ".join(missing),
           " and ".join("`ollama pull %s`" % name for name in missing))
    )


class Preflight:
    """The nine cuts, in order, before a single scenario runs.

    Attributes:
        config: The harness configuration.
        env: The persistent test environment.
        binary: The release binary under test.
        lock: The single-run lock, taken at step 2 and held for the run.
    """

    def __init__(self, config, env, binary, lock) -> None:
        """Store the collaborators.

        Args:
            config: The harness configuration.
            env: The persistent test environment.
            binary: The release binary under test.
            lock: The single-run lock.
        """
        self.config = config
        self.env = env
        self.binary = binary
        self.lock = lock

    def run(self, certifying: bool, profile) -> BackendStatus:
        """Run every cut in order and report whether the backend answered.

        Args:
            certifying: True on a ``--smoke-2`` run with no profile. It forces
                the rebuild at step 7 and decides which models step 8 checks.
            profile: The profile whose models should win, or ``None`` for the
                product's own defaults.

        Returns:
            BackendStatus: Whether the backend answered, with the cause when it
            did not. **A backend that does not answer does not cut** (D-17):
            what does not need it still runs, and what does is reported
            ``CANNOT_TEST``.

        Raises:
            PreflightError: On any of the hard cuts, with the fix in the
                message.
        """
        self._require_python()
        self.lock.acquire()
        self._require_config()
        self._require_environment()
        self._restore_rotation_if_left_over()
        self._settle_binary(certifying)
        self.env.normalize_magi_toml(profile)
        self._require_one_endpoint()
        backend = self._probe_backend()
        # Step 8 runs only when the backend answered, which is why the probe
        # above carries no number of its own.
        if backend.reachable:
            require_declared_models(self.env.declared_models(),
                                    self._available_models())
        return backend

    def _require_python(self) -> None:
        """Step 1: the interpreter is new enough.

        Raises:
            PreflightError: If it is older than :data:`MINIMUM_PYTHON`.
        """
        if sys.version_info[:2] < MINIMUM_PYTHON:
            raise PreflightError(
                "the harness needs Python %d.%d or newer" % MINIMUM_PYTHON
            )

    def _require_config(self) -> None:
        """Steps 3 and 4: the configuration resolves and is not exposed.

        The credential is resolved HERE rather than where it is spent. R7
        rotates it and needs the real value to put back, and the executor that
        performs the rotation runs after every cheaper run has already
        completed -- so a variable that resolves to nothing surfaced as a
        harness failure with the whole backend bill already paid. Nothing is
        read out of the credential and nothing is printed: only whether it
        resolves at all.

        Raises:
            PreflightError: If the backend credential does not resolve, or the
                permissions are permissive or unreadable.
        """
        # Read as an ATTRIBUTE, never through a getattr default. The default
        # is what turned "the config does not declare where it came from" into
        # a silently skipped check on every real run, on the one file that
        # holds the vault passphrase in cleartext. A config without a path is
        # now an AttributeError at startup, which is the loud version.
        check_config_permissions(pathlib.Path(self.config.path))
        self._require_backend_credential()

    def _require_backend_credential(self) -> None:
        """Step 3: the variable naming the backend credential holds something.

        Every read of the configuration in this module is a plain attribute,
        never ``getattr`` with a default. The default is what turned a missing
        ``path`` into a permission check that silently never ran, and the same
        five reads sat one rename away from the same failure. A configuration
        that does not declare what this module needs should raise here, at
        startup, where it is loud.

        Raises:
            PreflightError: Naming the variable and how to set it. The value
                itself never reaches the message.
        """
        key = self.config.backend_key_env
        if not isinstance(key, str) or not key:
            raise PreflightError(
                "smoke.toml declares no backend credential variable; set "
                "[backend].key_env to the variable holding it"
            )
        if not os.environ.get(key, "").strip():
            raise PreflightError(
                "%s carries no backend credential. Export it before running: "
                "R7 rotates that credential and has to put the real one back, "
                "so a run started without it spends every other run first and "
                "then fails." % key
            )

    def _require_environment(self) -> None:
        """Step 5: the environment exists.

        Raises:
            PreflightError: Naming ``--init-env``, which is the fix. This runs
                BEFORE the normalisation so a missing environment produces this
                message rather than a low-level error from writing into a
                directory that is not there.
        """
        if not self.env.exists():
            raise PreflightError(
                "the test environment is missing; create it with --init-env"
            )

    def _restore_rotation_if_left_over(self) -> None:
        """Step 6: undo a rotation a previous run died in the middle of.

        Detection is by NAME, through ``vault ls``: the product never prints a
        stored value, so recognising the sentinel by reading it is not merely
        unspecified, it is impossible. A successful restore is announced --
        a rotation that recovers silently every run is indistinguishable from
        one that never happens, and the thing worth investigating stays
        invisible precisely because the recovery works.
        """
        listed = self._vault_names()
        # The ROTATION first, the placeholders second. A leftover placeholder
        # is inert -- step 7b rewrites the environment's magi.toml every run,
        # so no base_url declares the placeholders that would read the entries
        # back. A rotated credential is not: it breaks every backend run that
        # follows. Sweeping first put the recoverable failure ahead of the one
        # that matters.
        if ROTATION_MARKER in listed:
            self._restore_rotation(listed)
        self._sweep_placeholders(listed)

    def _restore_rotation(self, listed: list[str]) -> None:
        """Put the real backend credential back and drop the sentinel.

        Args:
            listed: The vault entry names, which carry the marker.

        Raises:
            PreflightError: If nothing holds a credential to restore from, or
                a vault write fails.
        """
        key = self.config.backend_key_env
        credential = os.environ.get(key, "")
        if not credential:
            raise PreflightError(
                "a previous run died mid-rotation: the vault still holds %s, "
                "and %s carries no credential to restore from. Set it, or run "
                "--reset-env to discard the environment."
                % (ROTATION_MARKER, key or "the configured key variable")
            )
        # Restore FIRST, drop the marker second. A crash between the two leaves
        # the marker with the credential already correct, so the next preflight
        # restores a correct value over itself -- a no-op. The other order would
        # leave the credential rotated with nothing left to say so.
        self._vault_write(["vault", WORKDIR_FLAG, str(self.env.root),
                           "set", key, "--force"],
                          "could not restore the rotated backend credential",
                          stdin=credential.encode("utf-8"))
        self._vault_write(["vault", WORKDIR_FLAG, str(self.env.root),
                           "rm", ROTATION_MARKER, "--force"],
                          "could not drop the rotation marker")
        print(
            "[preflight] a previous run died mid-rotation; %s has been "
            "restored and %s removed. Something is killing R7 -- the recovery "
            "working is not the same as nothing being wrong."
            % (key, ROTATION_MARKER)
        )

    def _sweep_placeholders(self, listed: list[str]) -> None:
        """Remove any placeholder entry a killed R6 left in the vault.

        R6 removes its own in a ``finally``, which covers the run that fails
        and the run that times out but not the one that is killed -- there is
        no ``finally`` to reach. The removal warns rather than raising, on the
        grounds that the next preflight sweeps them; this is that sweep.

        Complexity: ``O(len(PLACEHOLDER_ENTRIES))`` vault calls, and only for
        entries that are actually there.

        Args:
            listed: The vault entry names, as ``vault ls`` gave them.
        """
        left = [name for name in PLACEHOLDER_ENTRIES if name in listed]
        for name in left:
            self._vault_write(["vault", WORKDIR_FLAG, str(self.env.root),
                               "rm", name, "--force"],
                              "could not remove the placeholder entry %s that "
                              "a killed run left behind" % name)
        if left:
            print(
                "[preflight] a previous run was killed mid-R6; removed the "
                "placeholder entries it left: %s." % ", ".join(left)
            )

    def _vault_names(self) -> list[str]:
        """List the vault entry names, or cut naming why they cannot be read.

        It used to answer ``None`` and let the caller return quietly, which
        the spec forbids for a reason worth restating: a run killed
        mid-rotation leaves a sentinel credential in the environment, and a
        preflight that cannot read the vault cannot tell. Starting anyway
        means every backend invocation authenticates with the sentinel and
        dies of an opaque auth error, while S16 renders a verdict over a
        half-rotated database -- and the report says nothing, because the
        announcement covers only the branch where the restore succeeded.

        Returns:
            list[str]: The entry names. Nothing here reads a stored VALUE:
            the product does not expose one, and the sentinel is recognised
            by its name. Only ``stdout`` is parsed -- ``raw()`` appends
            stderr, and a diagnostic there naming an entry would invent one.

        Raises:
            PreflightError: If the vault could not be listed, naming the
                cause. Not silence: this is one of the nine hard cuts.
        """
        try:
            completed = self.binary.invoke(
                ["vault", WORKDIR_FLAG, str(self.env.root), "ls"],
                env={PASSPHRASE_VAR: self.config.passphrase},
                timeout=VAULT_TIMEOUT_S,
            )
        except (OSError, ProductOutputError) as exc:
            # Narrow on purpose. A bare `except Exception` here swallowed a
            # TypeError from calling invoke with the wrong keyword, so this
            # method answered "cannot tell" on every run and the marker was
            # never detected at all -- the silence looked exactly like a clean
            # vault.
            raise PreflightError(
                "the environment's vault could not be listed, so the harness "
                "cannot tell whether a previous run died mid-rotation: %s. "
                "Build the binary and check the passphrase in smoke.toml, or "
                "run --reset-env to discard the environment." % exc
            ) from exc
        if completed.exit_code != 0:
            raise PreflightError(
                "the environment's vault could not be listed (vault ls exited "
                "%d), so the harness cannot tell whether a previous run died "
                "mid-rotation: %s. Check the passphrase in smoke.toml, or run "
                "--reset-env to discard the environment."
                % (completed.exit_code,
                   completed.stderr.decode("utf-8", errors="replace").strip())
            )
        return completed.stdout.decode("utf-8", errors="replace").split()

    def _vault_write(self, argv, subject, stdin=None) -> None:
        """Run one vault mutation, refusing to continue if it fails.

        Args:
            argv: The product arguments, starting with ``vault``.
            subject: What this write is doing, for the failure message. It is
                a REQUIRED argument rather than a default: the message used to
                say "could not restore the rotated credential" for every
                caller, so an operator whose placeholder removal failed was
                sent to look at R7's rotation, which had not happened.
            stdin: Bytes to feed the command, for a value that must never
                appear in a command line.

        Raises:
            PreflightError: If the command cannot run or exits non-zero. A
                half-restored environment produces authentication failures in
                scenarios that have nothing to do with it, and those get
                diagnosed for hours.
        """
        try:
            completed = self.binary.invoke(
                argv, stdin=stdin,
                env={PASSPHRASE_VAR: self.config.passphrase},
                timeout=VAULT_TIMEOUT_S,
            )
        except (OSError, ProductOutputError) as exc:
            raise PreflightError("%s: %s" % (subject, exc)) from exc
        if completed.exit_code != 0:
            raise PreflightError(
                "%s (%s exited %d)"
                % (subject, " ".join(argv[:2]), completed.exit_code)
            )

    def _settle_binary(self, certifying: bool) -> None:
        """Step 7: the binary answers, and a certifying run always rebuilds.

        Args:
            certifying: Whether this run may emit a certificate.

        Raises:
            PreflightError: If the binary cannot be produced or does not run.
        """
        if certifying:
            self.binary.rebuild()
        self.binary.version()

    def _require_one_endpoint(self) -> None:
        """Step 7c: the endpoint this preflight probes is the one the runs use.

        Runs after step 7b, because 7b is what writes the file this reads.

        Raises:
            PreflightError: If the two disagree, naming both. A probe aimed at
                a different host certifies a backend nothing talks to, and it
                stays invisible for as long as both hosts happen to serve the
                same models.
        """
        declared = self.env.declared_base_url()
        if declared is None:
            return
        probed = self.config.backend_base_url
        if declared.rstrip("/") == probed.rstrip("/"):
            return
        raise PreflightError(
            "the runs go to %s, which is what the environment's configuration "
            "declares, but smoke.toml points the backend probe at %s. Make "
            "[backend].base_url match, or the probe certifies a host nothing "
            "talks to." % (declared, probed)
        )

    def _available_models(self):
        """What the backend says it has, or None when it cannot be asked.

        Returns:
            set[str] | None: The tags, or None for a backend kind that
            publishes no listing, an unparseable answer, or a failed request.
            Every one of those is "not measurable", and the caller treats it
            as such rather than as an empty backend.
        """
        if self.config.backend_kind != INTROSPECTABLE_KIND:
            return None
        root = self.config.backend_base_url.rstrip("/")
        # The tag endpoint sits beside the OpenAI-compatible surface, not
        # inside it: /v1 is the completions root and /api/tags is the daemon's.
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        try:
            with urllib.request.urlopen(root + TAGS_PATH,
                                        timeout=BACKEND_PROBE_TIMEOUT_S) as answer:
                document = json.loads(answer.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, ValueError,
                http.client.HTTPException):
            # HTTPException is not an OSError: a daemon that truncates
            # its response would otherwise take the whole harness to
            # exit 3 rather than to "could not be asked".
            return None
        models = document.get("models")
        if not isinstance(models, list):
            return None
        return {entry.get("name") for entry in models
                if isinstance(entry, dict) and isinstance(entry.get("name"), str)}

    def _probe_backend(self) -> BackendStatus:
        """Ask the backend whether it is there.

        NOT a numbered cut, and the spec is right not to give it one: by D-17
        a backend that does not answer never cuts. It is the CONDITION step 8
        hangs on -- asking a host that is down which models it has produces
        silence, and silence there means "could not be asked", never "nothing
        exists". Numbering it pushed the model check to 9 and left the
        published README pointing at a step that had moved.

        Returns:
            BackendStatus: Reachable, or not with the cause recorded.
        """
        url = self.config.backend_base_url
        if not url:
            return BackendStatus(reachable=False, cause="no backend base_url configured")
        try:
            with urllib.request.urlopen(url, timeout=BACKEND_PROBE_TIMEOUT_S):
                return BackendStatus(reachable=True, cause="")
        except urllib.error.HTTPError:
            # An HTTP error is an ANSWER: something is listening and speaking
            # HTTP, which is all this step establishes.
            return BackendStatus(reachable=True, cause="")
        except (urllib.error.URLError, OSError, ValueError,
                http.client.HTTPException) as exc:
            # HTTPException for the same reason the model listing names it,
            # and this is the call that MATTERS: this step runs first and
            # decides whether the listing runs at all, so a guard only there
            # can never fire -- the daemon dies here, several statements
            # earlier.
            return BackendStatus(reachable=False, cause=str(exc))
