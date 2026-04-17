#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 2.1.1
# Date: 2026-04-17
"""MAGI agent output validation.

Loads and validates JSON output files produced by the three MAGI agents
(Melchior, Balthasar, Caspar) against the expected schema.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any


class ValidationError(Exception):
    """Raised when agent output fails validation.

    Attributes:
        message: Human-readable description of the validation failure.
        filepath: Path to the file that failed validation, if applicable.
    """

    def __init__(self, message: str, filepath: str = "") -> None:
        self.filepath = filepath
        super().__init__(f"{filepath}: {message}" if filepath else message)


VALID_AGENTS: set[str] = {"melchior", "balthasar", "caspar"}
VALID_VERDICTS: set[str] = {"approve", "reject", "conditional"}
VALID_SEVERITIES: set[str] = {"critical", "warning", "info"}

_REQUIRED_KEYS = frozenset(
    {
        "agent",
        "verdict",
        "confidence",
        "summary",
        "reasoning",
        "findings",
        "recommendation",
    }
)

_REQUIRED_FINDING_KEYS = frozenset({"severity", "title", "detail"})
#: Upper bound (bytes) for any on-disk input MAGI will ingest — user prompts,
#: Claude CLI outputs, and agent JSON files. Centralised here so every module
#: enforces the same ceiling; bump it in one place and the whole pipeline
#: follows.
MAX_INPUT_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB
_MAX_FINDINGS_PER_AGENT: int = 100
_MAX_FIELD_LENGTH: int = 50_000  # 50,000 characters per top-level string field
_MAX_TITLE_LENGTH: int = 500
_MAX_DETAIL_LENGTH: int = 10_000
# Invisible characters that can smuggle hidden content into a finding
# title: zero-width spaces/joiners (U+200B-U+200D), bidi marks and embeds
# (U+200E-U+200F, U+202A-U+202E), line/paragraph separators (U+2028-U+2029),
# narrow no-break space (U+202F), the byte-order mark (U+FEFF), and the
# soft hyphen (U+00AD). These span categories Cf, Zl, Zp, and Zs rather
# than Cf alone.
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u2028-\u202f\ufeff\u00ad]")
# ASCII control whitespace (``\t``, ``\n``, ``\r``, vertical tab, form feed)
# plus the NEL (U+0085) that terminals treat as a line break. These are not
# "invisible" in the Cf/Zl/Zp sense — they render as column breaks — so they
# are stripped by a separate pass. Left in place, they would corrupt the
# fixed-column marker/severity/title layout of ``_format_finding_line`` and
# the banner row width contract.
_CONTROL_WHITESPACE_RE = re.compile(r"[\t\n\v\f\r\x85]")


def clean_title(raw: str) -> str:
    """Return *raw* with invisible characters and edge whitespace removed.

    Applies, in order:

    1. :data:`_ZERO_WIDTH_RE` — strips Cf/Zl/Zp-category invisibles and
       bidi marks that would otherwise let a title smuggle content
       through length checks without rendering.
    2. :data:`_CONTROL_WHITESPACE_RE` — strips ASCII control whitespace
       (``\\t``, ``\\n``, ``\\v``, ``\\f``, ``\\r``) and the NEL
       (``U+0085``). Without this, a title like
       ``"Broken\\ninjected row"`` passes validation (non-empty, under
       the length cap) and then corrupts the fixed-column layout in
       :func:`reporting._format_finding_line` and the banner.
    3. :meth:`str.strip` — trims surrounding whitespace.

    Used by :func:`load_agent_output` to produce the canonical title form
    that is both length-capped and stored on the finding. Exposed as a
    public helper so that downstream consumers (notably the finding
    dedup in ``consensus.py``) can derive a normalization key from the
    same source of truth.
    """
    stripped_invisibles = _ZERO_WIDTH_RE.sub("", raw)
    without_breaks = _CONTROL_WHITESPACE_RE.sub(" ", stripped_invisibles)
    return without_breaks.strip()


def load_agent_output(filepath: str) -> dict[str, Any]:
    """Load and validate a single agent's JSON output.

    Reads a JSON file produced by one of the three MAGI agents and
    validates its structure before returning the parsed data.

    Args:
        filepath: Path to the agent JSON file.

    Returns:
        Validated agent output dictionary containing at least the keys
        ``agent``, ``verdict``, ``confidence``, ``summary``,
        ``reasoning``, ``findings``, and ``recommendation``.

    Raises:
        ValidationError: If the file cannot be read, is not valid JSON,
            or its content fails any structural / value check.
    """
    try:
        file_size = os.path.getsize(filepath)
        if file_size > MAX_INPUT_FILE_SIZE:
            raise ValidationError(
                f"File exceeds maximum size of {MAX_INPUT_FILE_SIZE} bytes "
                f"(got {file_size} bytes).",
                filepath,
            )
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON: {exc}", filepath) from exc
    except OSError as exc:
        raise ValidationError(f"Cannot read file: {exc}", filepath) from exc

    # --- top-level key check ---
    missing = _REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValidationError(f"Agent output missing keys: {sorted(missing)}", filepath)

    # --- agent name ---
    agent = data["agent"]
    if not isinstance(agent, str):
        raise ValidationError(
            f"Field 'agent' must be a string, got {type(agent).__name__}.",
            filepath,
        )
    if agent not in VALID_AGENTS:
        raise ValidationError(
            f"Unknown agent '{agent}'. Must be one of {sorted(VALID_AGENTS)}.",
            filepath,
        )

    # --- verdict ---
    verdict = data["verdict"]
    if not isinstance(verdict, str):
        raise ValidationError(
            f"Field 'verdict' must be a string, got {type(verdict).__name__}.",
            filepath,
        )
    if verdict not in VALID_VERDICTS:
        raise ValidationError(
            f"Invalid verdict '{verdict}'. Must be one of {sorted(VALID_VERDICTS)}.",
            filepath,
        )

    # --- confidence ---
    confidence = data["confidence"]
    if isinstance(confidence, bool):
        raise ValidationError(
            "Confidence must be a number, got bool.",
            filepath,
        )
    if not isinstance(confidence, (int, float)):
        raise ValidationError(
            f"Confidence must be a number, got {type(confidence).__name__}.",
            filepath,
        )
    if not (0.0 <= confidence <= 1.0):
        raise ValidationError(
            f"Confidence must be between 0.0 and 1.0, got {confidence}.",
            filepath,
        )

    # --- string fields ---
    for field in ("summary", "reasoning", "recommendation"):
        value = data[field]
        if not isinstance(value, str):
            raise ValidationError(
                f"Field '{field}' must be a string, got {type(value).__name__}.",
                filepath,
            )
        if len(value) > _MAX_FIELD_LENGTH:
            raise ValidationError(
                f"Field '{field}' exceeds maximum length of {_MAX_FIELD_LENGTH} characters.",
                filepath,
            )

    # --- findings ---
    findings = data["findings"]
    if not isinstance(findings, list):
        raise ValidationError(
            f"Findings must be a list, got {type(findings).__name__}.",
            filepath,
        )
    if len(findings) > _MAX_FINDINGS_PER_AGENT:
        raise ValidationError(
            f"Findings list has {len(findings)} items, "
            f"exceeding maximum of {_MAX_FINDINGS_PER_AGENT}.",
            filepath,
        )
    for idx, finding in enumerate(findings):
        if not isinstance(finding, dict):
            raise ValidationError(
                f"Finding at index {idx} must be a dict, got {type(finding).__name__}.",
                filepath,
            )
        f_missing = _REQUIRED_FINDING_KEYS - set(finding.keys())
        if f_missing:
            raise ValidationError(
                f"Finding at index {idx} missing keys: {sorted(f_missing)}.",
                filepath,
            )
        for field in ("severity", "title", "detail"):
            if not isinstance(finding[field], str):
                raise ValidationError(
                    f"Finding at index {idx} field '{field}' must be a string, "
                    f"got {type(finding[field]).__name__}.",
                    filepath,
                )
        if finding["severity"] not in VALID_SEVERITIES:
            raise ValidationError(
                f"Finding at index {idx} has invalid severity "
                f"'{finding['severity']}'. "
                f"Must be one of {sorted(VALID_SEVERITIES)}.",
                filepath,
            )
        cleaned = clean_title(finding["title"])
        if not cleaned:
            raise ValidationError(
                f"Finding at index {idx} has empty or whitespace-only title.",
                filepath,
            )
        if len(cleaned) > _MAX_TITLE_LENGTH:
            raise ValidationError(
                f"Finding at index {idx} title exceeds maximum length "
                f"of {_MAX_TITLE_LENGTH} characters.",
                filepath,
            )
        # Replace the raw title with the cleaned form so downstream consumers
        # (dedup, rendering) never see smuggled zero-width characters.
        finding["title"] = cleaned
        if len(finding["detail"]) > _MAX_DETAIL_LENGTH:
            raise ValidationError(
                f"Finding at index {idx} detail exceeds maximum length "
                f"of {_MAX_DETAIL_LENGTH} characters.",
                filepath,
            )

    return dict(data)  # type-narrow from Any
