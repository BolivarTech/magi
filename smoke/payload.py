# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Generation of the large payload, at a size declared in bytes.

**Bytes are chosen; tokens are asserted.** The harness declares a size in bytes
because that is what a generator can measure without a tokenizer, and the
scenario asserts the token floor at run time against the count the product
reports. The asymmetry is the design: the token count is the load-bearing
property but not the controllable magnitude -- it depends on the model's
tokenizer, which this harness does not have and should not carry. Predicting
the second from the first would need a bytes-per-token ratio; asserting it
needs nothing.

That is also why no such ratio lives here. An earlier design kept one, along
with a cache keyed on it, a plausibility band and an invalidation rule -- all
of it there to make a PREDICTION accurate. With no prediction there is nothing
to cache, nothing to invalidate, and no "I measured wrong and stored it" state
that fails to correct itself.

What survives from that design is the deterministic ORDER, for a different
reason: so two runs of the same commit send the same bytes and their results
are comparable. Filesystem order will not do -- one filesystem normalises
Unicode in names, another is case-insensitive, and iteration order is not
guaranteed anywhere -- so the rule is to sort by the byte order of the
POSIX-normalised relative path and to read every file in BINARY, never as
text, because text mode translates line endings and on a CRLF checkout that
changes the content actually sent.
"""

import pathlib

from smoke.errors import HarnessError

#: Where the material comes from. The product's own Rust sources: they are
#: large enough, they are in every clone, and they are the kind of text a
#: review payload is actually made of.
SOURCE_GLOB = "src/**/*.rs"

#: How much is read at a time. The concatenation is bounded by the caller's
#: target, but a single source file is not bounded by anything this module
#: controls, so it is read in pieces.
READ_CHUNK_BYTES = 65536


class PayloadBuilder:
    """Builds a payload of a declared byte size from one checkout.

    Example:
        >>> builder = PayloadBuilder(pathlib.Path("."))
        >>> len(builder.build(1024))
        1024
    """

    def __init__(self, repo_root: pathlib.Path | str) -> None:
        """Bind to one checkout without touching the filesystem.

        Args:
            repo_root: The tree the payload is drawn from.
        """
        self._repo_root = pathlib.Path(repo_root)

    def sources(self) -> list[pathlib.Path]:
        """The files the payload is drawn from, in the order it uses them.

        The key is the UTF-8 bytes of the relative path spelled with forward
        slashes. Normalising the separator first is not cosmetic: ``/`` is
        0x2F and ``\\`` is 0x5C, with letters between them, so sorting the
        native spelling puts a nested file on one side of a sibling on Windows
        and the other side on Linux -- and the same commit would then produce
        two different payloads.

        Complexity: ``O(n log n)`` over the number of source files.

        Returns:
            list[pathlib.Path]: Absolute paths, deterministically ordered.
        """
        found = [path for path in self._repo_root.glob(SOURCE_GLOB)
                 if path.is_file()]
        return sorted(
            found,
            key=lambda path: path.relative_to(
                self._repo_root).as_posix().encode("utf-8"),
        )

    def available_bytes(self) -> int:
        """How much material the tree holds.

        Complexity: ``O(n)`` stats over the number of source files.

        Returns:
            int: The total size of every source file, in bytes.

        Raises:
            HarnessError: If a file cannot be measured. Reporting a smaller
                total than the tree holds would turn into a refusal to build a
                payload the tree could have supplied, which reads as a product
                defect and is not one.
        """
        total = 0
        for path in self.sources():
            try:
                total += path.stat().st_size
            except OSError as exc:
                raise HarnessError(
                    "could not measure %s while sizing the payload: %s"
                    % (path, exc)
                ) from exc
        return total

    def build(self, target_bytes: int) -> bytes:
        """Concatenate the sources and cut at exactly *target_bytes*.

        Complexity: ``O(target_bytes)`` in time and memory -- it stops reading
        once the target is reached rather than concatenating the whole tree.

        Args:
            target_bytes: How many bytes the payload must hold.

        Returns:
            bytes: Exactly *target_bytes* bytes, identical across calls over
            an unchanged tree.

        Raises:
            HarnessError: If *target_bytes* is negative, if the tree does not
                hold that much material, or if a source file cannot be read.
                The scenario turns this into ``CANNOT_TEST``: it never got
                evaluated, and saying so is what stops the harness accusing
                the product of a size it was never sent.
        """
        if target_bytes < 0:
            raise HarnessError(
                "a payload of %d bytes was asked for; a size is not negative"
                % target_bytes
            )
        collected = bytearray()
        for path in self.sources():
            if len(collected) >= target_bytes:
                break
            self._append(path, collected, target_bytes)
        if len(collected) < target_bytes:
            raise HarnessError(
                "the tree holds %d bytes under %s and the payload asked for "
                "%d; the scenario cannot be evaluated on a short payload"
                % (len(collected), SOURCE_GLOB, target_bytes)
            )
        return bytes(collected)

    @staticmethod
    def _append(path: pathlib.Path, collected: bytearray,
                target_bytes: int) -> None:
        """Read one source file into *collected*, stopping at the target.

        Args:
            path: The file to read.
            collected: The buffer to extend, in place.
            target_bytes: Where to stop.

        Raises:
            HarnessError: If the file cannot be read.
        """
        try:
            with path.open("rb") as handle:
                while len(collected) < target_bytes:
                    chunk = handle.read(READ_CHUNK_BYTES)
                    if not chunk:
                        break
                    collected.extend(chunk[:target_bytes - len(collected)])
        except OSError as exc:
            raise HarnessError(
                "could not read %s while building the payload: %s" % (path, exc)
            ) from exc
