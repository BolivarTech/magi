# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Secret hygiene for the harness's own artifacts.

The harness knows the values to scrub because it planted them. This is a
literal search over known secrets, NOT a heuristic credential detector: a
heuristic fails in both directions and the expensive one is the miss.

Two detectors live here and they fail for **different** reasons, which is the
whole reason there are two. :func:`find_secret` looks for a value it planted;
:func:`find_unredacted_authorities` looks for a *shape*, and would still fire on
a credential nobody planted. Sharing one derived set would make them fail
together -- one omission producing both a secret on disk and a green assertion.
"""

import dataclasses
import re
import secrets as stdlib_secrets
import string
import urllib.parse

#: How long a minted marker is. Short or pronounceable markers occur by chance
#: in a model's own output, and then the security scenario goes red with no leak
#: at all -- the kind of red that gets rationalised until someone disables the
#: assertion, which is worse than a miss.
MARKER_CHARS = 32

#: Reserved characters a minted credential embeds, so the product walks the
#: percent-encoding path the security scenario hunts.
RESERVED_SAMPLE = "@/?#"

_MARKER_ALPHABET = string.ascii_lowercase + string.digits
_AUTHORITY = re.compile(rb"[a-zA-Z][a-zA-Z0-9+.-]*://([^/\s@]+)@")
_REDACTED_USERINFO = re.compile(rb"^\*+$")


@dataclasses.dataclass(frozen=True)
class PlantedSecret:
    """A value the harness planted, and the label it is announced under.

    Attributes:
        value: The secret itself.
        label: What to call it in a scrub marker and in a finding's detail.
    """

    value: str
    label: str

    @property
    def marker(self) -> str:
        """The encoding-invariant substring of this value, if it has one.

        Returns:
            str: The longest alphanumeric run in the value. For a minted
            credential that is the random marker; for a hand-written one it may
            be short, which is why the shape assertion does not depend on it.
        """
        runs = re.findall(r"[A-Za-z0-9]+", self.value)
        return max(runs, key=len) if runs else ""

    def forms(self) -> list[bytes]:
        """Every form the product may write this value in.

        The marker is included, and it is what makes the set complete rather
        than merely long: it is alphanumeric, so no encoder ever changes it,
        and searching for it finds the secret at any level of encoding --
        including levels nobody derived. Deriving encodings instead would be an
        arms race with no end, since ``%2540`` shares no substring with either
        the raw value or ``%40``.

        The length guard is not defensive padding. A short or pronounceable
        marker can occur by chance in a model's own output, and then the search
        reports a leak that never happened. :func:`mint_credential` always
        builds one long enough; a hand-made secret that does not reach the
        floor is searched by its value alone.

        Returns:
            list[bytes]: The raw bytes, the fully percent-encoded bytes and the
            marker, longest first so a nested shorter form cannot leave a
            fragment behind.
        """
        encoded = urllib.parse.quote(self.value, safe="")
        unique = {self.value, encoded}
        if len(self.marker) >= MARKER_CHARS:
            unique.add(self.marker)
        return sorted(
            (form.encode("utf-8") for form in unique), key=len, reverse=True
        )


def mint_credential(reserved: str = RESERVED_SAMPLE) -> PlantedSecret:
    """Build a credential that exercises the bug and survives any encoding.

    Args:
        reserved: Reserved characters to embed, so the product walks its
            percent-encoding path.

    Returns:
        PlantedSecret: A secret whose marker is alphanumeric and at least
        :data:`MARKER_CHARS` long, so it reads identically raw, encoded or
        double-encoded.
    """
    marker = "".join(
        stdlib_secrets.choice(_MARKER_ALPHABET) for _ in range(MARKER_CHARS)
    )
    return PlantedSecret("%s%s%s" % (marker, reserved, marker), "backend credential")


def _pattern_for(form: bytes) -> re.Pattern:
    """Compile a case-insensitive literal matcher for one derived form.

    Case matters here for one specific reason: ``%2F`` and ``%2f`` are the same
    byte, and which one appears depends on the product's encoder rather than on
    the harness. Matching case-insensitively covers both without deriving a
    list of spellings -- the same mistake that cost the product its redaction,
    trusting a value to arrive spelled one way, applied one level in.

    Args:
        form: The derived form to match literally.

    Returns:
        re.Pattern: The compiled matcher.
    """
    return re.compile(re.escape(form), re.IGNORECASE)


def find_secret(data: bytes, secrets) -> list[str]:
    """Report which planted secrets appear in *data*, in any derived form.

    Complexity: ``O(len(data) * total derived forms)``.

    Args:
        data: The bytes to search, at full fidelity and unscrubbed.
        secrets: The :class:`PlantedSecret` values to look for.

    Returns:
        list[str]: The label of every secret found, in the order given. A
        secret found in several forms is reported once: the answer to *did this
        leak* is not a count.
    """
    found = []
    for secret in secrets:
        if any(_pattern_for(form).search(data) for form in secret.forms()):
            found.append(secret.label)
    return found


def scrub(data: bytes, secrets) -> bytes:
    """Replace every planted secret with a marker naming what was removed.

    A silent deletion makes the artifact useless for the one thing it is kept
    for. The replacement says *what* was found and *how often*, so a reader can
    tell a scrubbed artifact from one that never carried a secret.

    Complexity: ``O(len(data) * total derived forms)``.

    Args:
        data: The bytes about to be persisted.
        secrets: The :class:`PlantedSecret` values to remove.

    Returns:
        bytes: The scrubbed bytes. Surrounding text is untouched.
    """
    for secret in secrets:
        patterns = [_pattern_for(form) for form in secret.forms()]
        # Count across every form BEFORE replacing any of them, so the marker
        # reports what the artifact actually held rather than what happened to
        # survive the previous substitution.
        total = sum(len(pattern.findall(data)) for pattern in patterns)
        if not total:
            continue
        replacement = b"<redacted: %s, %d occurrence%s>" % (
            secret.label.encode("utf-8"),
            total,
            b"" if total == 1 else b"s",
        )
        # Longest first: a shorter form nested inside a longer one would
        # otherwise leave the remainder of the longer one behind.
        for pattern in patterns:
            data = pattern.sub(replacement, data)
    return data


def find_unredacted_authorities(data: bytes) -> list[str]:
    """Report URL authorities that still carry userinfo.

    This is the second detector and it asserts on **shape**, never on a value.
    It fires on a credential the harness never planted -- one nobody derived a
    form for -- which is exactly the hole the first detector cannot close.

    Complexity: ``O(len(data))``.

    Args:
        data: The bytes to search.

    Returns:
        list[str]: One entry per authority whose userinfo is present and not
        already the redaction marker.
    """
    reported = []
    for match in _AUTHORITY.finditer(data):
        userinfo = match.group(1)
        if _REDACTED_USERINFO.match(userinfo):
            continue
        reported.append(match.group(0).decode("utf-8", errors="replace"))
    return reported
