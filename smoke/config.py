# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Parsing of ``smoke.toml``, the harness's own configuration.

The file carries the environment's vault passphrase in clear, so every error
path in this module is written under one rule: **name the file and the failing
field, never the value**. That is not caution for its own sake. The product this
harness tests once printed a credential straight to stderr because a decoder's
error text reproduces the offending line (REQ-A16c), and the first section of
this file is the passphrase.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from smoke.errors import PreflightError

#: The product's own floor. It also enforces ``zxcvbn >= 3``, which the harness
#: cannot check without a dependency it is not allowed to have; that half of the
#: rule is declared in ``smoke/README.md`` rather than silently dropped.
MIN_PASSPHRASE_CHARS = 12

#: A MAGI trio is three seats. A profile naming any other number is a
#: configuration error, not a smaller trio.
TRIO_SIZE = 3

#: Backends the harness knows how to point the product at.
BACKEND_KINDS = ("ollama", "openai-compat", "anthropic")

#: Payload defaults. They live here and not in ``payload.py`` because the config
#: parses ``[payload]`` and needs fallbacks, and ``payload.py`` belongs to a
#: later task: defaults in the later module would invert the dependency.
PAYLOAD_TARGET_BYTES = 250_000
PAYLOAD_TOKEN_FLOOR = 50_000

#: Where the certificate is written, relative to the repository root and spelled
#: with forward slashes because that is how git reports a path, and S12
#: subtracts it from what git reports. A FIXED name that replaces the previous
#: one: git history is already the archive, and a versioned filename would force
#: a reader to know what the old one was called.
#:
#: Declared here rather than in ``certificate.py`` for the same reason as
#: ROTATION_MARKER below: both of its readers are later tasks, and a constant
#: living in the later module would invert the dependency.
CERTIFICATE_PATH = "docs/test/smoke-certificate.md"

#: The vault entry R7 creates before rotating and removes after restoring. The
#: preflight finds it by NAME through ``vault ls``, because the product never
#: prints a stored value. Declared here because both readers are later tasks.
ROTATION_MARKER = "SMOKE_R7_ROTATION"

#: The vault entries the substitution reads. BOTH pairs, and that is measured
#: rather than assumed: with only the root pair the product asked for
#: ``MAGI_BASE_URL_USER``, and with only the prefixed pair it asked for
#: ``BASE_URL_USER``. The trio section inherits the root endpoint, so both
#: resolutions happen and both need their entry.
PLACEHOLDER_ENTRIES = ("BASE_URL_USER", "BASE_URL_PASSWORD",
                       "MAGI_BASE_URL_USER", "MAGI_BASE_URL_PASSWORD")


def _read(path):
    """Parse a TOML file without ever echoing its contents.

    Args:
        path: The file to read.

    Returns:
        dict: The parsed document.

    Raises:
        PreflightError: If the file is missing or does not parse. The message
            names the file and nothing else, and the context is suppressed so
            no traceback can surface the decoder's text either.

            The reason is worth stating precisely, because the obvious one is
            wrong. Measured on Python 3.11, ``tomllib`` reports position only
            -- ``(at line N, column M)`` -- and echoes no content, including
            when the syntax error sits on the passphrase's own line. The leak
            REQ-A16c records belongs to the PRODUCT's Rust decoder, which does
            reproduce the offending line. What this suppression buys is
            independence from that measurement continuing to hold: the file
            being parsed here opens with a passphrase, and the cost of not
            depending on a third party's error format is one lost line of
            diagnostic text.
    """
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        raise PreflightError(
            "%s does not exist; copy smoke/smoke.toml.example and fill it in" % path
        ) from None
    except tomllib.TOMLDecodeError:
        raise PreflightError(
            "%s is not valid TOML. The parser's own message is withheld on "
            "purpose: it reproduces the offending line, and this file holds a "
            "passphrase." % path.name
        ) from None


def _require(document, section, key, path):
    """Fetch one declared field, naming what is missing rather than guessing.

    Args:
        document: The parsed TOML document.
        section: The table the field should live in.
        key: The field name.
        path: The file, for the message.

    Returns:
        object: The value.

    Raises:
        PreflightError: If the section or the field is absent.
    """
    table = document.get(section)
    if not isinstance(table, dict) or key not in table:
        raise PreflightError("%s is missing [%s].%s" % (path.name, section, key))
    return table[key]


class SmokeConfig:
    """The harness's configuration, minus anything it must not carry.

    There is deliberately **no** ``profile`` attribute. ``--profile`` names this
    same file and ``[profile.cheap]`` lives inside it, so a config that parsed
    the section eagerly would carry the cheap trio into the run that must
    certify with the product's defaults, and the certificate would claim
    product defaults over a run that used cheap models. That is a false green
    on the one line whose entire value is being true. The profile is read only
    where it was asked for, by :meth:`ModelProfile.load`. The bad state is not
    forbidden by a rule someone can forget; there is no field to read.

    Attributes:
        passphrase: The environment's vault passphrase.
        backend_kind: One of :data:`BACKEND_KINDS`.
        backend_base_url: Where the product should reach the backend.
        backend_key_env: The environment variable holding the backend key.
        ceiling_fraction: S9's fraction; 0.0 until phase 5 measures it.
        payload_target_bytes: Bytes the payload builder generates.
        payload_token_floor: The token floor S8 asserts against.

    Example:
        >>> config = SmokeConfig.load(Path("smoke/smoke.toml"))
        >>> config.backend_kind
        'ollama'
    """

    def __init__(self, path, passphrase, backend_kind, backend_base_url,
                 backend_key_env, ceiling_fraction,
                 payload_target_bytes, payload_token_floor):
        """Store the parsed configuration.

        Args:
            passphrase: The environment's vault passphrase.
            backend_kind: One of :data:`BACKEND_KINDS`.
            backend_base_url: Where the product should reach the backend.
            backend_key_env: The variable holding the backend key.
            ceiling_fraction: S9's fraction.
            payload_target_bytes: Bytes to generate.
            payload_token_floor: The floor S8 asserts.
        """
        self.path = path
        self.passphrase = passphrase
        self.backend_kind = backend_kind
        self.backend_base_url = backend_base_url
        self.backend_key_env = backend_key_env
        self.ceiling_fraction = ceiling_fraction
        self.payload_target_bytes = payload_target_bytes
        self.payload_token_floor = payload_token_floor

    @classmethod
    def load(cls, path):
        """Read and validate ``smoke.toml``.

        Args:
            path: The configuration file.

        Returns:
            SmokeConfig: The parsed configuration.

        Raises:
            PreflightError: If the file is missing, does not parse, omits a
                required field, declares a backend kind outside
                :data:`BACKEND_KINDS`, or carries a passphrase shorter than
                :data:`MIN_PASSPHRASE_CHARS`. No message repeats a value it
                rejected.
        """
        path = Path(path)
        document = _read(path)

        passphrase = _require(document, "env", "passphrase", path)
        if not isinstance(passphrase, str) or len(passphrase) < MIN_PASSPHRASE_CHARS:
            raise PreflightError(
                "the passphrase in %s is shorter than %d characters. The "
                "rejected value is not repeated here."
                % (path.name, MIN_PASSPHRASE_CHARS)
            )

        kind = _require(document, "backend", "kind", path)
        if kind not in BACKEND_KINDS:
            raise PreflightError(
                "%s declares an unknown [backend].kind; expected one of %s"
                % (path.name, ", ".join(BACKEND_KINDS))
            )

        calibration = document.get("calibration", {})
        payload = document.get("payload", {})
        return cls(
            path=path,
            passphrase=passphrase,
            backend_kind=kind,
            backend_base_url=_require(document, "backend", "base_url", path),
            backend_key_env=_require(document, "backend", "key_env", path),
            ceiling_fraction=calibration.get("ceiling_fraction", 0.0),
            payload_target_bytes=payload.get("target_bytes", PAYLOAD_TARGET_BYTES),
            payload_token_floor=payload.get("token_floor", PAYLOAD_TOKEN_FLOOR),
        )


@dataclass(frozen=True)
class Seat:
    """One MAGI seat: a model and the failure domain it belongs to.

    The pair is inseparable, and that is the point. The product made it
    breaking in v0.13.0 that a seat declaring a model must also declare its
    lineage, because a lineage is a **user-chosen failure domain** that cannot
    be read off a model name -- the same two models are legitimately one domain
    for one operator and two for another. Carrying the model without the
    lineage would let the harness rewrite a seat and leave behind a label
    describing a model that is no longer there, and the product's diversity
    check would then pass on three descriptions of the wrong thing.

    Attributes:
        model: The model tag the seat runs.
        lineage: The failure domain the operator assigned to it.
    """

    model: str
    lineage: str


class ModelProfile:
    """The cheap model profile, read only when ``--profile`` asked for it.

    Attributes:
        model: The main agent's model.
        trio: Exactly :data:`TRIO_SIZE` seats, each a :class:`Seat`.
    """

    def __init__(self, model, trio):
        """Store the profile.

        Args:
            model: The main agent's model.
            trio: The three seats.
        """
        self.model = model
        self.trio = trio

    @classmethod
    def load(cls, path):
        """Read ``[profile.cheap]`` from *path*.

        Args:
            path: The configuration file naming the profile.

        Returns:
            ModelProfile: The declared profile.

        Raises:
            PreflightError: If the file does not parse, the section is absent,
                or the trio does not name exactly :data:`TRIO_SIZE` models. An
                absent section **cuts** rather than falling back to the
                product's defaults, because that fallback would certify by
                accident, which is the failure REQ-S35 exists to prevent.
        """
        path = Path(path)
        document = _read(path)
        section = document.get("profile", {}).get("cheap")
        if not isinstance(section, dict):
            raise PreflightError(
                "%s declares no [profile.cheap], and --profile asked for one. "
                "Falling back to the product's defaults here would certify by "
                "accident." % path.name
            )
        trio = section.get("trio")
        if not isinstance(trio, list) or len(trio) != TRIO_SIZE:
            raise PreflightError(
                "[profile.cheap].trio in %s must name exactly %d seats"
                % (path.name, TRIO_SIZE)
            )
        seats = []
        for position, entry in enumerate(trio):
            if not isinstance(entry, dict) or "model" not in entry:
                raise PreflightError(
                    "[profile.cheap].trio[%d] in %s must declare a model"
                    % (position, path.name)
                )
            if "lineage" not in entry:
                raise PreflightError(
                    "[profile.cheap].trio[%d] in %s declares a model without a "
                    "lineage. A lineage is a failure domain you choose, so the "
                    "harness will not infer one: guessing would look exactly "
                    "like a declaration." % (position, path.name)
                )
            seats.append(Seat(model=entry["model"], lineage=entry["lineage"]))
        return cls(model=section.get("model"), trio=seats)
