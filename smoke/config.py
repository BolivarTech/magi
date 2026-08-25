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

#: The vault entry R7 creates before rotating and removes after restoring. The
#: preflight finds it by NAME through ``vault ls``, because the product never
#: prints a stored value. Declared here because both readers are later tasks.
ROTATION_MARKER = "SMOKE_R7_ROTATION"


def _read(path):
    """Parse a TOML file without ever echoing its contents.

    Args:
        path: The file to read.

    Returns:
        dict: The parsed document.

    Raises:
        PreflightError: If the file is missing or does not parse. The message
            names the file and nothing else: the decoder's own text reproduces
            the offending line, and in this file that line may be the
            passphrase. The exception context is suppressed for the same
            reason, so no traceback can surface it either.
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
        margin_tokens: S9's margin; 0 until phase 5 measures it.
        ceiling_fraction: S9's fraction; 0.0 until phase 5 measures it.
        payload_target_bytes: Bytes the payload builder generates.
        payload_token_floor: The token floor S8 asserts against.

    Example:
        >>> config = SmokeConfig.load(Path("smoke/smoke.toml"))
        >>> config.backend_kind
        'ollama'
    """

    def __init__(self, passphrase, backend_kind, backend_base_url,
                 backend_key_env, margin_tokens, ceiling_fraction,
                 payload_target_bytes, payload_token_floor):
        """Store the parsed configuration.

        Args:
            passphrase: The environment's vault passphrase.
            backend_kind: One of :data:`BACKEND_KINDS`.
            backend_base_url: Where the product should reach the backend.
            backend_key_env: The variable holding the backend key.
            margin_tokens: S9's margin.
            ceiling_fraction: S9's fraction.
            payload_target_bytes: Bytes to generate.
            payload_token_floor: The floor S8 asserts.
        """
        self.passphrase = passphrase
        self.backend_kind = backend_kind
        self.backend_base_url = backend_base_url
        self.backend_key_env = backend_key_env
        self.margin_tokens = margin_tokens
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
            passphrase=passphrase,
            backend_kind=kind,
            backend_base_url=_require(document, "backend", "base_url", path),
            backend_key_env=_require(document, "backend", "key_env", path),
            margin_tokens=calibration.get("margin_tokens", 0),
            ceiling_fraction=calibration.get("ceiling_fraction", 0.0),
            payload_target_bytes=payload.get("target_bytes", PAYLOAD_TARGET_BYTES),
            payload_token_floor=payload.get("token_floor", PAYLOAD_TOKEN_FLOOR),
        )


class ModelProfile:
    """The cheap model profile, read only when ``--profile`` asked for it.

    Attributes:
        model: The main agent's model.
        trio: Exactly :data:`TRIO_SIZE` models, one per MAGI seat.
    """

    def __init__(self, model, trio):
        """Store the profile.

        Args:
            model: The main agent's model.
            trio: The three seat models.
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
                "[profile.cheap].trio in %s must name exactly %d models"
                % (path.name, TRIO_SIZE)
            )
        return cls(model=section.get("model"), trio=trio)
