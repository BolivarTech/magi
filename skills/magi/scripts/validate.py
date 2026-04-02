#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-04-01
"""MAGI agent output validation.

Loads and validates JSON output files produced by the three MAGI agents
(Melchior, Balthasar, Caspar) against the expected schema.
"""

from __future__ import annotations

import json
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
    if agent not in VALID_AGENTS:
        raise ValidationError(
            f"Unknown agent '{agent}'. Must be one of {sorted(VALID_AGENTS)}.",
            filepath,
        )

    # --- verdict ---
    verdict = data["verdict"]
    if verdict not in VALID_VERDICTS:
        raise ValidationError(
            f"Invalid verdict '{verdict}'. Must be one of {sorted(VALID_VERDICTS)}.",
            filepath,
        )

    # --- confidence ---
    confidence = data["confidence"]
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
    _MAX_FIELD_LENGTH = 50_000  # 50 KB per field
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
        if not finding["title"].strip():
            raise ValidationError(
                f"Finding at index {idx} has empty or whitespace-only title.",
                filepath,
            )

    return dict(data)  # type-narrow from Any
