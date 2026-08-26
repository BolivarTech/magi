# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""The nine hard cuts that run before any scenario.

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
import os
import pathlib
import re
import stat
import subprocess
import sys
import urllib.error
import urllib.request

from smoke.config import ROTATION_MARKER
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
    saved = path.parent / (path.name + ".sddl")
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
        try:
            saved.unlink(missing_ok=True)
        except OSError:
            pass
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
        return self._probe_backend()

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

        Raises:
            PreflightError: Naming the variable and how to set it. The value
                itself never reaches the message.
        """
        key = getattr(self.config, "backend_key_env", "")
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
        if listed is None or ROTATION_MARKER not in listed:
            return
        key = getattr(self.config, "backend_key_env", "")
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
                          stdin=credential.encode("utf-8"))
        self._vault_write(["vault", WORKDIR_FLAG, str(self.env.root),
                           "rm", ROTATION_MARKER, "--force"])
        print(
            "[preflight] a previous run died mid-rotation; %s has been "
            "restored and %s removed. Something is killing R7 -- the recovery "
            "working is not the same as nothing being wrong."
            % (key, ROTATION_MARKER)
        )

    def _vault_names(self) -> list[str] | None:
        """List the vault entry names, or ``None`` when they cannot be read.

        Returns:
            list[str] | None: The names, or ``None`` if the vault could not be
            listed. Nothing here reads a stored VALUE: the product does not
            expose one, and the sentinel is recognised by its name.
        """
        try:
            completed = self.binary.invoke(
                ["vault", WORKDIR_FLAG, str(self.env.root), "ls"],
                env={PASSPHRASE_VAR: getattr(self.config, "passphrase", "")},
                timeout=VAULT_TIMEOUT_S,
            )
        except (OSError, ProductOutputError):
            # Narrow on purpose. A bare `except Exception` here swallowed a
            # TypeError from calling invoke with the wrong keyword, so this
            # method answered "cannot tell" on every run and the marker was
            # never detected at all -- the silence looked exactly like a clean
            # vault.
            return None
        if completed.exit_code != 0:
            return None
        return completed.raw().decode("utf-8", errors="replace").split()

    def _vault_write(self, argv, stdin=None) -> None:
        """Run one vault mutation, refusing to continue if it fails.

        Args:
            argv: The product arguments, starting with ``vault``.
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
                env={PASSPHRASE_VAR: getattr(self.config, "passphrase", "")},
                timeout=VAULT_TIMEOUT_S,
            )
        except (OSError, ProductOutputError) as exc:
            raise PreflightError(
                "could not restore the rotated credential: %s" % exc
            ) from exc
        if completed.exit_code != 0:
            raise PreflightError(
                "restoring the rotated credential failed (%s exited %d)"
                % (" ".join(argv[:2]), completed.exit_code)
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

    def _probe_backend(self) -> BackendStatus:
        """Step 8: ask the backend whether it is there.

        Returns:
            BackendStatus: Reachable, or not with the cause recorded.
        """
        url = getattr(self.config, "backend_base_url", None)
        if not url:
            return BackendStatus(reachable=False, cause="no backend base_url configured")
        try:
            with urllib.request.urlopen(url, timeout=BACKEND_PROBE_TIMEOUT_S):
                return BackendStatus(reachable=True, cause="")
        except urllib.error.HTTPError:
            # An HTTP error is an ANSWER: something is listening and speaking
            # HTTP, which is all this step establishes.
            return BackendStatus(reachable=True, cause="")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return BackendStatus(reachable=False, cause=str(exc))
