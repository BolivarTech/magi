#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.1.0
# Date: 2026-04-25
"""MAGI model registry.

Single source of truth for the Claude model short names accepted on the
MAGI command line and the Anthropic model IDs they resolve to. Bumping a
model (e.g. from ``claude-opus-4-6`` to a future ``claude-opus-5``) is a
one-line edit here; the orchestrator in :mod:`run_magi` imports the
mapping without needing to change.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

_MODEL_IDS_MUTABLE: dict[str, str] = {
    "opus": "claude-opus-4-7",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
}

#: Read-only view of the short-name → Anthropic model-ID mapping.
#: Exposed as a :class:`~types.MappingProxyType` so downstream code cannot
#: accidentally mutate the canonical registry at runtime.
MODEL_IDS: Mapping[str, str] = MappingProxyType(_MODEL_IDS_MUTABLE)

#: Tuple of accepted short names, kept in lockstep with :data:`MODEL_IDS`.
VALID_MODELS: tuple[str, ...] = tuple(MODEL_IDS.keys())

#: Per-mode default short name (2.2.3). When ``--model`` is not given on
#: the CLI, :func:`run_magi.parse_args` looks the analysis mode up here
#: to pick the default. Explicit ``--model X`` always wins. Rationale:
#:
#: * ``code-review`` and ``design`` keep the historical opus default
#:   because their outputs typically drive PR-blocking decisions where
#:   the marginal $0.50/run cost over sonnet is justified.
#: * ``analysis`` ships a sonnet default because sonnet matches opus
#:   quality on exploratory questions at roughly 4× lower cost. The
#:   captured ``magi-report.json`` corpus showed no measurable consensus
#:   delta between models on analysis-mode runs.
#:
#: Every key MUST be in :data:`run_magi.VALID_MODES` and every value
#: MUST be in :data:`MODEL_IDS`; the test suite enforces both invariants.
_MODE_DEFAULTS_MUTABLE: dict[str, str] = {
    "code-review": "opus",
    "design": "opus",
    "analysis": "sonnet",
}
MODE_DEFAULT_MODELS: Mapping[str, str] = MappingProxyType(_MODE_DEFAULTS_MUTABLE)


def resolve_model(short_name: str) -> str:
    """Return the Anthropic model ID for *short_name*.

    Args:
        short_name: A MAGI model short name (e.g. ``"opus"``).

    Returns:
        The corresponding Anthropic model identifier.

    Raises:
        ValueError: If *short_name* is not a registered model.
    """
    try:
        return MODEL_IDS[short_name]
    except KeyError:
        raise ValueError(
            f"Unknown model '{short_name}'. Must be one of {sorted(MODEL_IDS.keys())}."
        ) from None
