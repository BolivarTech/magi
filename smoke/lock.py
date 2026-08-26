# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""The single-run lock.

The environment is persistent and shared, so two simultaneous runs over the
same checkout corrupt each other: R7 rotates a credential while R1 plants
facts in the same database, and the run-artifact directories collide. The
result is not an error but a FALSE VERDICT.

This is a one-process invariant, not a coordination mechanism: no waiting, no
queue, no retry. Whoever arrives second leaves.
"""

import os
import pathlib

from smoke.errors import PreflightError

WINDOWS_LOCK_BYTES = 1

#: The Windows lock sits at this offset, PAST every byte the file will ever
#: hold, and the reason is measured rather than assumed. ``msvcrt.locking``
#: refuses any access to the locked range from a second handle -- including a
#: plain read, and including one opened by the very process that holds the
#: lock. Locking byte 0, where the diagnostic pid lives, therefore makes the pid
#: unreadable: the contender's error message degrades to "pid unknown" on the
#: one platform it was written for. A region beyond end-of-file is lockable on
#: Windows and excludes a second holder exactly the same way, so the two
#: purposes -- exclusion and diagnosis -- stop sharing a byte.
WINDOWS_LOCK_OFFSET = 4096

try:
    import fcntl

    HAS_FCNTL = True
except ImportError:
    import msvcrt

    HAS_FCNTL = False


class RunLock:
    """An advisory lock the OS holds for the life of the process."""

    def __init__(self, path: pathlib.Path | str) -> None:
        """Create a lock over one path.

        Args:
            path: Where the lock file lives. It must sit OUTSIDE
                ``smoke/env/``: ``--init-env`` creates that directory and
                ``--reset-env`` destroys it.
        """
        self._path = pathlib.Path(path)
        self._handle = None

    def acquire(self) -> None:
        """Take the lock, or refuse.

        Raises:
            PreflightError: If another process holds it. The message names the
                pid recorded in the file, which is diagnostic text only.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_lock_file(self._path)
        handle = open(self._path, "r+b")
        try:
            self._take(handle)
        except OSError as exc:
            # The close goes in a finally rather than after the read: a lock
            # file left behind by a foreign writer can hold bytes that are not
            # UTF-8, and the resulting UnicodeDecodeError is not an OSError, so
            # on the straight-line version it would escape past the close and
            # leak the handle on the one path taken when things are already
            # going wrong.
            try:
                owner = self._recorded_owner()
            finally:
                handle.close()
            raise PreflightError(
                f"another smoke run holds {self._path} (pid {owner}); "
                "run one at a time"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n".encode("utf-8"))
        handle.flush()
        self._handle = handle

    @staticmethod
    def _take(handle) -> None:  # noqa: ANN001
        """Take the platform's non-blocking lock.

        Args:
            handle: The open lock file.

        Raises:
            OSError: If the lock is already held.
        """
        if HAS_FCNTL:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        # Windows locks a byte RANGE, not a descriptor, so where that range
        # sits is a design decision and not a detail. It sits past end-of-file
        # (see WINDOWS_LOCK_OFFSET) so it never overlaps the diagnostic pid.
        # Nothing is written here: the file already exists, created once by
        # _ensure_lock_file, so this method only ever takes the lock. Writing
        # before the lock is held is the race that would let two first-ever
        # runs both write, and the loser would land inside the winner's range.
        handle.seek(WINDOWS_LOCK_OFFSET)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, WINDOWS_LOCK_BYTES)

    @staticmethod
    def _ensure_lock_file(path: pathlib.Path) -> None:
        """Create the lock file, exactly once and without writing to it.

        Exclusive creation is what makes this safe: every racing process but one
        gets ``FileExistsError``, and none of them writes a byte, so no write
        ever lands in a range another process has locked. The file is created
        rather than left to :func:`open` because ``acquire`` opens it ``r+b`` --
        a mode that must not create -- so that the ordering "exists first, lock
        second, write third" holds for every process, including the first.

        Args:
            path: The lock file's path.
        """
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return
        os.close(descriptor)

    def _recorded_owner(self) -> str:
        """Read the recorded pid, for the error message only.

        Returns:
            The recorded text, or ``"unknown"`` if it cannot be read.
        """
        try:
            return self._path.read_text(encoding="utf-8").strip() or "unknown"
        except (OSError, UnicodeDecodeError):
            # UnicodeDecodeError is a ValueError, not an OSError. The file may
            # have been written by something else entirely, or truncated
            # mid-write, and this read exists only to name a holder in a
            # message -- so a file it cannot decode makes the holder unknown,
            # never the failure the caller reports.
            return "unknown"

    def release(self) -> None:
        """Release the lock if held. Safe to call twice."""
        if self._handle is None:
            return
        try:
            if HAS_FCNTL:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            else:
                self._handle.seek(WINDOWS_LOCK_OFFSET)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, WINDOWS_LOCK_BYTES)
        finally:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "RunLock":
        """Take the lock.

        Returns:
            This lock.
        """
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001
        """Release the lock."""
        self.release()
