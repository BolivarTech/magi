# Author: Julian Bolivar
# Version: 0.23.0
# Date: 2026-10-04
"""D1-D13 -- ``magi decide`` against real backends, its refusals and its agent use.

``decide`` asks a System One model for a typed decision. What the unit suite
cannot see is the part this module covers: a real Ollama and a real OpenRouter
answering the same protocol, the account-policy 404 a hosted provider returns,
a base URL one path segment short, a key that must never travel to the wrong
host, and an old ``magi.toml`` meeting the new binary.

Three rules shape every scenario here.

**Properties of the product, never of the model.** No assertion reads which
option a model chose. They read the envelope's shape, the exit code, the chain
of attempts and the causes the product names.

**One configuration per scenario.** Scenarios that need a chain of their own
(an absent primary, a threshold nothing can reach, a base URL without its API
prefix) seed a scratch workspace with ``magi init`` and rewrite only its
``[decision]`` section. The product's own default text stays the base, so a
failure cannot come from this module's idea of the rest of the schema.

**What needs a key says so.** OpenRouter scenarios read the key from the
``DECISION_API_KEY`` variable of the harness's own environment, at run time. No
key means ``CANNOT_TEST``, never ``PASS``: the property was promised and the
environment refused to let it run. The key is declared as planted, so the
archive scrubs it.

The only server the harness runs is the existing error backend (D9) and a
listener that accepts a connection and never answers (D10), which is not a
model mock: it says nothing, so there is nothing to disagree with.
"""

import contextlib
import dataclasses
import json
import os
import pathlib
import re
import shutil
import socket
import socketserver
import tempfile
import threading
import time
import tomllib

from smoke import runs
from smoke.env import (DECISION_FALLBACK_HEADER, DECISION_HEADER,
                       table_header)
from smoke.outcome import Finding, Outcome
from smoke.registry import scenario
from smoke.secrets import PlantedSecret, find_secret, mint_credential

DECIDE_SUBCOMMAND = "decide"
VAULT_SUBCOMMAND = "vault"
LS_SUBCOMMAND = "ls"
INIT_SUBCOMMAND = "init"
QUERY_SUBCOMMAND = "query"
WORKDIR_FLAG = "-w"
NO_MEMORY_FLAG = "--no-memory"
OUTPUT_FLAG = "-o"
QUESTIONS_FLAG = "-q"
TIMEOUT_FLAG = "--timeout"
MAGI_DIR_NAME = ".magi"
MAGI_TOML_NAME = "magi.toml"
LOG_DIR_PARTS = (".magi", "logs")

#: The variable the product reads the decision endpoint's key from, and the one
#: the OpenRouter scenarios read it from in the harness's own environment.
KEY_VARIABLE = "DECISION_API_KEY"

#: Blank means absent to the product, so this is how a scenario that must not
#: send a key stops an inherited one from travelling.
BLANK_KEY = {KEY_VARIABLE: ""}

#: A model name nobody ships. D3's primary, written into its own workspace so
#: the preflight (which reads only the environment's toml) never asks for it.
ABSENT_DECISION_MODEL = "smoke-absent-decision-model:0"

#: The hosted endpoint and the models the OpenRouter scenarios use. The first
#: is the cheap one the probe measured; the second is excluded by the account
#: policy of the debug account (a real 404 the client must report properly).
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "typesafe/jev-1.13"
OPENROUTER_POLICY_MODEL = "inception/mercury-decide:free"

#: The same host with the API prefix missing, for D7.
OPENROUTER_NO_PREFIX_URL = "https://openrouter.ai"
NO_PREFIX_TRIED_URL = "https://openrouter.ai/v1/systemone"

#: What a provider's account-policy refusal says. It must never be copied into
#: the product's output.
POLICY_PROVIDER_TEXT = b"guardrail restrictions and data policy"

#: The closed vocabulary of ``error.kind`` and ``chain[].cause`` the scenarios
#: read (spec REQ-D10).
KIND_MISSING_KEY = "missing_key"
KIND_UNAVAILABLE = "decision_unavailable"
KIND_INVALID_INPUT = "invalid_input"
CAUSE_MODEL_NOT_FOUND = "model_not_found"
CAUSE_ACCOUNT_POLICY = "account_policy"
CAUSE_NOT_FOUND = "not_found"
CAUSE_INVALID_RESPONSE = "invalid_response"
PREFIX_HINT_NEEDLE = "API prefix"
CAUSE_TRANSPORT = "transport"
CAUSE_TIMEOUT = "timeout"
OUTCOME_ANSWERED = "answered"
OUTCOME_FAILED = "failed"
OUTCOME_LOW_CONFIDENCE = "low_confidence"

#: Exit codes (README, REQ-D22): a decision was made; no decision was made; the
#: caller or its configuration was wrong.
EXIT_DECIDED = 0
EXIT_NO_DECISION = 1
EXIT_BAD_INPUT = 2

#: How long the harness waits for one invocation. A cold local model load takes
#: up to ~90 s when the machine is busy (measured), and the product's own
#: per-attempt bound is 120 s, so the harness bound sits above it.
INVOCATION_TIMEOUT_S = 300
INIT_TIMEOUT_S = 120
VAULT_TIMEOUT_S = 120

#: D10: the override the slow-endpoint assertion passes, the toml value it must
#: beat, and the wall clock inside which the attempt has to end. The bound is a
#: factor of ten over the override, far below the toml value, so a product that
#: ignored the flag cannot end inside it.
OVERRIDE_TIMEOUT_S = 1
TOML_TIMEOUT_S = 120
OVERRIDE_WALL_CLOCK_S = 10

#: A threshold no honest distribution reaches, paired with a question whose two
#: options mean the same thing, so a model splits its probability between them.
UNREACHABLE_THRESHOLD = 0.99

SCHEMA_VERSION_KEY = "schema_version"

#: One question of each type (spec REQ-D23, example 1). The score question is
#: one the ticket answers unambiguously: with the scaffold's
#: ``min_confidence = 0.7`` an ambiguous one ("how urgent is this ticket?",
#: nimble 0.176, tev1 0.294) exhausts the chain. Measured 2026-10-04 on nimble
#: and tev1, three calls each (identical every time); lowest Magi confidence
#: per question: nimble team 0.956, refund 0.996, score 0.967; tev1 team
#: 0.935, refund 0.984, score 0.878.
THREE_TYPES_REQUEST = {
    "state": {"ticket": ("I was charged twice this month. Please refund the "
                         "extra payment.")},
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {"billing": "Payments, invoices and refunds",
                         "technical": "Bugs, errors and integrations",
                         "other": "Anything else"},
        },
        "refund": {
            "type": "noul",
            "instructions": "Does the customer explicitly ask for a refund?",
        },
        "urgency": {
            "type": "score",
            "instructions": ("How likely is it that this ticket is about a "
                             "billing problem?"),
            "criteria": ["Very unlikely", "Possible", "Certain"],
        },
    },
}
LEVEL_COUNT = 3

#: A question no model can answer with confidence: both options are the same.
AMBIGUOUS_REQUEST = {
    "state": "The ticket mentions nothing at all.",
    "questions": {
        "pick": {
            "type": "choice",
            "instructions": "Pick one.",
            "criteria": {"first": "Either one is fine",
                         "second": "Either one is fine"},
        },
    },
}

#: D6 sends a well-formed request, so a refusal is about the configuration and
#: not about the input.
QUESTIONS_ONLY_REQUEST = THREE_TYPES_REQUEST

#: Questions-only file for D10's ``-q``.
QUESTIONS_ONLY = {"refund": THREE_TYPES_REQUEST["questions"]["refund"]}


# ---------------------------------------------------------------------------
# D1 -- S1 against local Ollama
# ---------------------------------------------------------------------------

D1_ASSERTIONS = (
    "decide answers one question of each type against local Ollama with "
    "exit 0",
    "every answer carries a Magi confidence between 0 and 1",
    "the envelope names the model that answered and its usage",
)


@scenario("D1", assertions=D1_ASSERTIONS, needs_backend=True)
def decide_answers_against_local_ollama(run):
    """Run the three question types through the persistent environment.

    The environment's own ``magi.toml`` is used on purpose: it is the file the
    preflight normalised (cheap profile or product defaults), so this is the
    scenario that proves the profile rewrite left a loadable ``[decision]``.

    Args:
        run: Always ``None``; D1 declares no shared run.

    Yields:
        Finding: One per entry of :data:`D1_ASSERTIONS`, in that order.
    """
    attempt = _decide_in_environment(THREE_TYPES_REQUEST, "d1-ollama")
    envelope = _envelope_of(attempt)
    if envelope is None:
        yield from _cannot_or_fail(D1_ASSERTIONS, attempt)
        return
    yield _finding(D1_ASSERTIONS, 0, _answered_all_types(attempt, envelope))
    yield _finding(D1_ASSERTIONS, 1, _confidences_in_range(envelope))
    yield _finding(D1_ASSERTIONS, 2, _names_model_and_usage(envelope))


# ---------------------------------------------------------------------------
# D2 -- S1 against OpenRouter
# ---------------------------------------------------------------------------

D2_ASSERTIONS = (
    "decide answers one question of each type against OpenRouter with exit 0",
    "the envelope reports a numeric cost",
    "the score answer has a legend for every level",
)


@scenario("D2", assertions=D2_ASSERTIONS)
def decide_answers_against_openrouter(run):
    """Run the three question types against the hosted provider.

    Args:
        run: Always ``None``.

    Yields:
        Finding: One per entry of :data:`D2_ASSERTIONS`, in that order.
            ``CANNOT_TEST`` when the harness's environment carries no key.
    """
    key = _hosted_key()
    if key is None:
        yield from _all(D2_ASSERTIONS, Outcome.CANNOT_TEST, _NO_KEY_DETAIL)
        return
    attempt = _in_scratch_workspace(
        "d2", _decision_section(OPENROUTER_BASE_URL, OPENROUTER_MODEL),
        THREE_TYPES_REQUEST, env={KEY_VARIABLE: key.value}, planted=(key,))
    envelope = _envelope_of(attempt)
    if envelope is None:
        yield from _cannot_or_fail(D2_ASSERTIONS, attempt)
        return
    yield _finding(D2_ASSERTIONS, 0, _answered_all_types(attempt, envelope))
    yield _finding(D2_ASSERTIONS, 1, _numeric_cost(envelope))
    yield _finding(D2_ASSERTIONS, 2, _legend_covers_levels(envelope))


# ---------------------------------------------------------------------------
# D3 -- S2, a missing primary rotates
# ---------------------------------------------------------------------------

D3_ASSERTIONS = (
    "a missing primary model rotates to the declared fallback",
    "the chain records the primary as model_not_found",
)


@scenario("D3", assertions=D3_ASSERTIONS, needs_backend=True)
def a_missing_primary_rotates_to_the_fallback(run):
    """Declare an absent primary in front of the real local model.

    Args:
        run: Always ``None``.

    Yields:
        Finding: One per entry of :data:`D3_ASSERTIONS`, in that order.
    """
    target = _local_target()
    if target is None:
        yield from _all(D3_ASSERTIONS, Outcome.CANNOT_TEST, _NO_TARGET_DETAIL)
        return
    base_url, model = target
    section = _decision_section(base_url, ABSENT_DECISION_MODEL,
                                fallbacks=(model,))
    attempt = _in_scratch_workspace("d3", section, THREE_TYPES_REQUEST,
                                    env=BLANK_KEY)
    envelope = _envelope_of(attempt)
    if envelope is None:
        yield from _cannot_or_fail(D3_ASSERTIONS, attempt)
        return
    chain = _chain(envelope)
    rotated = (attempt.output.exit_code == EXIT_DECIDED
               and envelope.get("model") == model and len(chain) >= 2
               and chain[-1].get("outcome") == OUTCOME_ANSWERED)
    yield _finding(D3_ASSERTIONS, 0, "" if rotated else (
        "exit %d, answering model %r, chain %s; expected %r to answer after "
        "the absent primary" % (attempt.output.exit_code,
                                envelope.get("model"), _brief(chain), model)))
    first = chain[0] if chain else {}
    yield _finding(D3_ASSERTIONS, 1, "" if (
        first.get("model") == ABSENT_DECISION_MODEL
        and first.get("cause") == CAUSE_MODEL_NOT_FOUND) else (
        "the first hop was %s, expected the absent primary with cause %s"
        % (_brief(first), CAUSE_MODEL_NOT_FOUND)))


# ---------------------------------------------------------------------------
# D4 -- S9, account policy
# ---------------------------------------------------------------------------

D4_ASSERTIONS = (
    "a model excluded by account policy rotates to the next one",
    "the hop names account_policy and repeats no provider text",
)


@scenario("D4", assertions=D4_ASSERTIONS)
def an_account_policy_refusal_rotates(run):
    """Put a model the account excludes in front of a working one.

    Args:
        run: Always ``None``.

    Yields:
        Finding: One per entry of :data:`D4_ASSERTIONS`, in that order.
            ``CANNOT_TEST`` without a key, or when the account no longer
            excludes the model (nothing was refused, so nothing rotated).
    """
    key = _hosted_key()
    if key is None:
        yield from _all(D4_ASSERTIONS, Outcome.CANNOT_TEST, _NO_KEY_DETAIL)
        return
    section = _decision_section(OPENROUTER_BASE_URL, OPENROUTER_POLICY_MODEL,
                                fallbacks=(OPENROUTER_MODEL,))
    attempt = _in_scratch_workspace(
        "d4", section, THREE_TYPES_REQUEST, env={KEY_VARIABLE: key.value},
        planted=(key,))
    envelope = _envelope_of(attempt)
    if envelope is None:
        yield from _cannot_or_fail(D4_ASSERTIONS, attempt)
        return
    chain = _chain(envelope)
    first = chain[0] if chain else {}
    if first.get("outcome") == OUTCOME_ANSWERED:
        yield from _all(D4_ASSERTIONS, Outcome.CANNOT_TEST,
                        "the account answered %s, so no policy refusal "
                        "happened to rotate on" % OPENROUTER_POLICY_MODEL)
        return
    rotated = (attempt.output.exit_code == EXIT_DECIDED
               and envelope.get("model") == OPENROUTER_MODEL)
    yield _finding(D4_ASSERTIONS, 0, "" if rotated else (
        "exit %d, answering model %r, chain %s"
        % (attempt.output.exit_code, envelope.get("model"), _brief(chain))))
    leaked = POLICY_PROVIDER_TEXT in attempt.output.raw().lower()
    named = first.get("cause") == CAUSE_ACCOUNT_POLICY
    yield _finding(D4_ASSERTIONS, 1, "" if named and not leaked else (
        "the first hop was %s (cause expected %s)%s"
        % (_brief(first), CAUSE_ACCOUNT_POLICY,
           "; the provider's own text reached the output" if leaked else "")))


# ---------------------------------------------------------------------------
# D5 -- S4, no key for a hosted endpoint
# ---------------------------------------------------------------------------

D5_ASSERTIONS = (
    "a hosted endpoint without DECISION_API_KEY exits 2 with missing_key",
    "the message names DECISION_API_KEY",
    "with the same configuration vault ls still runs",
)


@scenario("D5", assertions=D5_ASSERTIONS)
def a_hosted_endpoint_without_a_key_is_missing_key(run):
    """Point decide at OpenRouter with the key blanked and the principal local.

    Args:
        run: Always ``None``.

    Yields:
        Finding: One per entry of :data:`D5_ASSERTIONS`, in that order.
    """
    root = _seed_scratch("d5")
    if root is None:
        yield from _all(D5_ASSERTIONS, Outcome.CANNOT_TEST, _NO_SEED_DETAIL)
        return
    try:
        if not _install_decision(
                root, _decision_section(OPENROUTER_BASE_URL,
                                        OPENROUTER_MODEL)):
            yield from _all(D5_ASSERTIONS, Outcome.CANNOT_TEST,
                            _NO_INSTALL_DETAIL)
            return
        attempt = _decide(root, THREE_TYPES_REQUEST, "d5-no-key",
                          env=BLANK_KEY, memory=True)
        envelope = _envelope_of(attempt)
        reached = _first_hop_cause(envelope) not in (None, CAUSE_TRANSPORT)
        if envelope is None or not reached:
            why = (attempt.failure if not attempt.ok else
                   "the endpoint was not reached (%s), so the refusal that "
                   "names the missing key could not happen"
                   % _brief(_chain(envelope or {})))
            yield from _all(D5_ASSERTIONS[:2], Outcome.CANNOT_TEST, why)
        else:
            error = envelope.get("error") or {}
            yield _finding(D5_ASSERTIONS, 0, "" if (
                attempt.output.exit_code == EXIT_BAD_INPUT
                and error.get("kind") == KIND_MISSING_KEY) else (
                "exit %d, error kind %r, expected exit %d with %s"
                % (attempt.output.exit_code, error.get("kind"),
                   EXIT_BAD_INPUT, KIND_MISSING_KEY)))
            yield _finding(D5_ASSERTIONS, 1, "" if (
                KEY_VARIABLE in str(error.get("message", ""))) else (
                "the message does not name %s: %s"
                % (KEY_VARIABLE, error.get("message"))))
        yield _vault_ls_finding(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _vault_ls_finding(root):
    """Judge D5's third assertion: the vault is untouched by the bad endpoint.

    Args:
        root: The scratch workspace whose toml points at the hosted endpoint.

    Returns:
        Finding: PASS when ``vault ls`` exits 0.
    """
    attempt = runs.attempt(
        [VAULT_SUBCOMMAND, LS_SUBCOMMAND, WORKDIR_FLAG, str(root)], stdin=b"",
        timeout_s=VAULT_TIMEOUT_S, label="d5-vault-ls",
        env={"MAGI_PASSPHRASE": runs.passphrase()})
    if not attempt.ok:
        return _finding(D5_ASSERTIONS, 2, None, attempt.failure)
    return _finding(D5_ASSERTIONS, 2, "" if attempt.output.exit_code == 0 else (
        "vault ls exited %d with this configuration"
        % attempt.output.exit_code))


# ---------------------------------------------------------------------------
# D6 -- S6 and S19, old and empty configurations
# ---------------------------------------------------------------------------

D6_ASSERTIONS = (
    "a v0.22.0 magi.toml is refused naming the missing [decision] section",
    "an empty magi.toml is refused offering to delete it or run magi-rs init",
)


@scenario("D6", assertions=D6_ASSERTIONS)
def an_old_or_empty_config_is_refused(run):
    """Install a ``magi.toml`` without ``[decision]``, then an empty one.

    Args:
        run: Always ``None``.

    Yields:
        Finding: One per entry of :data:`D6_ASSERTIONS`, in that order.
    """
    root = _seed_scratch("d6")
    if root is None:
        yield from _all(D6_ASSERTIONS, Outcome.CANNOT_TEST, _NO_SEED_DETAIL)
        return
    try:
        generated = _read_toml(root)
        if generated is None:
            yield from _all(D6_ASSERTIONS, Outcome.CANNOT_TEST,
                            "the scaffolded %s could not be read"
                            % MAGI_TOML_NAME)
            return
        old = _refusal(root, _without_decision(generated), "d6-old")
        yield _refusal_finding(
            D6_ASSERTIONS, 0, old, (b"[decision]", b"magi-rs init"))
        empty = _refusal(root, "", "d6-empty")
        yield _refusal_finding(
            D6_ASSERTIONS, 1, empty, (b"delete", b"magi-rs init"))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _refusal(root, text, label):
    """Install *text* as the toml and ask for a decision.

    Args:
        root: The scratch workspace.
        text: The configuration to install.
        label: What to call the invocation in the archive.

    Returns:
        Attempt: The capture, or why there is none.
    """
    if not _write_toml(root, text):
        return runs.Attempt(output=None, failure=_NO_INSTALL_DETAIL)
    return _decide(root, QUESTIONS_ONLY_REQUEST, label, env=BLANK_KEY,
                   memory=False)


def _refusal_finding(assertions, index, attempt, needles):
    """Judge one refusal: exit 2, the needles named, and no decision printed.

    Args:
        assertions: The scenario's assertion tuple.
        index: Which assertion.
        attempt: The capture.
        needles: Lowercase byte strings the message has to carry.

    Returns:
        Finding: PASS, FAIL naming what was missing, or CANNOT_TEST.
    """
    if not attempt.ok:
        return _finding(assertions, index, None, attempt.failure)
    raw = attempt.output.raw().lower()
    missing = [n.decode() for n in needles if n not in raw]
    if attempt.output.exit_code != EXIT_BAD_INPUT or missing:
        return _finding(assertions, index, (
            "exit %d, the refusal does not carry %s: %s"
            % (attempt.output.exit_code, ", ".join(missing) or "nothing",
               _excerpt(attempt.output))))
    return _finding(assertions, index, "")


# ---------------------------------------------------------------------------
# D7 -- S13 and S20, a dead endpoint and a base_url without the API prefix
# ---------------------------------------------------------------------------

D7_ASSERTIONS = (
    "a dead endpoint exits 1 with decision_unavailable and no answers",
    "a base_url without the API prefix is diagnosed with the URL that was "
    "tried",
)


@scenario("D7", assertions=D7_ASSERTIONS)
def a_dead_or_misaddressed_endpoint_is_diagnosed(run):
    """Aim decide at a closed port, then at a host missing its API prefix.

    Args:
        run: Always ``None``.

    Yields:
        Finding: One per entry of :data:`D7_ASSERTIONS`, in that order.
    """
    target = _local_target()
    if target is None:
        yield from _all(D7_ASSERTIONS, Outcome.CANNOT_TEST, _NO_TARGET_DETAIL)
        return
    _, model = target
    dead = "http://127.0.0.1:%d" % _closed_port()
    first = _in_scratch_workspace(
        "d7-dead", _decision_section(dead, model), THREE_TYPES_REQUEST,
        env=BLANK_KEY)
    yield _dead_endpoint_finding(first)
    second = _in_scratch_workspace(
        "d7-prefix", _decision_section(OPENROUTER_NO_PREFIX_URL,
                                       OPENROUTER_MODEL),
        THREE_TYPES_REQUEST, env=BLANK_KEY)
    yield _prefix_finding(second)


def _dead_endpoint_finding(attempt):
    """Judge D7's first assertion.

    Args:
        attempt: The closed-port capture.

    Returns:
        Finding: PASS on exit 1, ``decision_unavailable`` and no answers.
    """
    envelope = _envelope_of(attempt)
    if envelope is None:
        return _cannot_or_fail_one(D7_ASSERTIONS, 0, attempt)
    error = envelope.get("error") or {}
    ok = (attempt.output.exit_code == EXIT_NO_DECISION
          and error.get("kind") == KIND_UNAVAILABLE
          and "answers" not in envelope
          and not _any_answers(envelope))
    return _finding(D7_ASSERTIONS, 0, "" if ok else (
        "exit %d, error kind %r, top-level answers %s, chain %s"
        % (attempt.output.exit_code, error.get("kind"),
           "present" if "answers" in envelope else "absent",
           _brief(_chain(envelope)))))


def _prefix_finding(attempt):
    """Judge D7's second assertion.

    Args:
        attempt: The capture against the host without its API prefix.

    Returns:
        Finding: PASS when the tried URL is named; CANNOT_TEST when the host
        was never reached (no network).
    """
    envelope = _envelope_of(attempt)
    if envelope is None:
        return _cannot_or_fail_one(D7_ASSERTIONS, 1, attempt)
    cause = _first_hop_cause(envelope)
    if cause == CAUSE_TRANSPORT:
        return _finding(D7_ASSERTIONS, 1, None,
                        "the host was not reachable from here, so the 404 "
                        "that carries the diagnosis never happened")
    text = json.dumps(envelope)
    # A real OpenRouter answers the prefix-less path with a non-JSON 200, so the cause may be
    # invalid_response; what the scenario guards is that the URL and the prefix hint are named.
    ok = (cause in (CAUSE_NOT_FOUND, CAUSE_INVALID_RESPONSE) and NO_PREFIX_TRIED_URL in text
          and PREFIX_HINT_NEEDLE in text)
    return _finding(D7_ASSERTIONS, 1, "" if ok else (
        "first hop cause %r; the tried URL %s %s the output"
        % (cause, NO_PREFIX_TRIED_URL,
           "is in" if NO_PREFIX_TRIED_URL in text else "is not in")))


# ---------------------------------------------------------------------------
# D8 -- S17, a threshold nothing reaches
# ---------------------------------------------------------------------------

D8_ASSERTIONS = (
    "when no model reaches its min_confidence the call exits 1",
    "every hop is reported as low_confidence with its confidence and "
    "threshold",
)


@scenario("D8", assertions=D8_ASSERTIONS, needs_backend=True)
def an_unreachable_threshold_ends_the_chain(run):
    """Ask an ambiguous question of a model with a threshold it cannot meet.

    Args:
        run: Always ``None``.

    Yields:
        Finding: One per entry of :data:`D8_ASSERTIONS`, in that order.
    """
    target = _local_target()
    if target is None:
        yield from _all(D8_ASSERTIONS, Outcome.CANNOT_TEST, _NO_TARGET_DETAIL)
        return
    base_url, model = target
    section = _decision_section(base_url, model,
                                min_confidence=UNREACHABLE_THRESHOLD)
    attempt = _in_scratch_workspace("d8", section, AMBIGUOUS_REQUEST,
                                    env=BLANK_KEY)
    envelope = _envelope_of(attempt)
    if envelope is None:
        yield from _cannot_or_fail(D8_ASSERTIONS, attempt)
        return
    chain = _chain(envelope)
    if any(h.get("outcome") == OUTCOME_FAILED for h in chain):
        yield from _all(D8_ASSERTIONS, Outcome.CANNOT_TEST,
                        "a hop failed outright (%s), so the model never "
                        "answered for the threshold to judge" % _brief(chain))
        return
    yield _finding(D8_ASSERTIONS, 0, "" if (
        attempt.output.exit_code == EXIT_NO_DECISION) else (
        "exit %d, expected %d" % (attempt.output.exit_code,
                                  EXIT_NO_DECISION)))
    every = bool(chain) and all(
        h.get("outcome") == OUTCOME_LOW_CONFIDENCE
        and _is_number(h.get("confidence"))
        and _is_number(h.get("min_confidence")) for h in chain)
    yield _finding(D8_ASSERTIONS, 1, "" if every else (
        "hops: %s" % _brief(chain)))


# ---------------------------------------------------------------------------
# D9 -- S22, a planted credential reaches no channel
# ---------------------------------------------------------------------------

D9_ASSERTIONS = (
    "the planted credential does not reach stdout",
    "the planted credential does not reach stderr",
    "the planted credential does not reach the run log",
)


@scenario("D9", assertions=D9_ASSERTIONS)
def a_planted_credential_reaches_no_channel(run):
    """Send a key to an endpoint that answers 401 and echoes it back.

    The echoing backend is the existing one S10 uses: it repeats the
    ``Authorization`` header in its error body, so a product that copied the
    provider's text into its output would carry the key out.

    Args:
        run: Always ``None``.

    Yields:
        Finding: One per entry of :data:`D9_ASSERTIONS`, in that order.
    """
    root = _seed_scratch("d9")
    if root is None:
        yield from _all(D9_ASSERTIONS, Outcome.CANNOT_TEST, _NO_SEED_DETAIL)
        return
    secret = mint_credential()
    try:
        with runs.error_backend() as base_url:
            if not _install_decision(root, _decision_section(
                    base_url, "smoke-echo-model")):
                yield from _all(D9_ASSERTIONS, Outcome.CANNOT_TEST,
                                _NO_INSTALL_DETAIL)
                return
            attempt = _decide(root, THREE_TYPES_REQUEST, "d9-echo",
                              env={KEY_VARIABLE: secret.value},
                              planted=(secret,), memory=False)
        if not attempt.ok:
            yield from _all(D9_ASSERTIONS, Outcome.CANNOT_TEST,
                            attempt.failure)
            return
        if attempt.output.exit_code == EXIT_DECIDED:
            # The backend refused and the product still reported a decision:
            # that is the product's defect, not a reason to skip the checks.
            yield _finding(D9_ASSERTIONS, 0,
                           "the backend answered 401 and decide still "
                           "exited 0, so the error path was never walked")
        else:
            yield _leak_finding(0, attempt.output.stdout, secret, "stdout")
        yield _leak_finding(1, attempt.output.stderr, secret, "stderr")
        log = _read_logs(root)
        if log is None:
            yield _finding(D9_ASSERTIONS, 2, None,
                           "the product wrote no run log under %s, so there "
                           "was nothing to search"
                           % pathlib.Path(*LOG_DIR_PARTS))
        else:
            yield _leak_finding(2, log, secret, "the run log")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _leak_finding(index, data, secret, channel):
    """Search one channel for every derived form of the planted secret.

    Args:
        index: Which D9 assertion.
        data: The bytes to search, unscrubbed.
        secret: The planted credential.
        channel: What to call the channel in the detail.

    Returns:
        Finding: FAIL naming the form found, PASS otherwise.
    """
    found = find_secret(data, (secret,))
    if found:
        return _finding(D9_ASSERTIONS, index,
                        "%s carries %s" % (channel, ", ".join(sorted(set(found)))))
    return _finding(D9_ASSERTIONS, index, "")


# ---------------------------------------------------------------------------
# D10 -- S23, the CLI's inputs and flags
# ---------------------------------------------------------------------------

D10_ASSERTIONS = (
    "-q with questions also in the input exits 2 before any request",
    "-o writes the envelope and leaves stdout empty",
    "--timeout overrides timeout_secs per attempt",
    "an explicit model is tried alone",
)


@scenario("D10", assertions=D10_ASSERTIONS, needs_backend=True)
def the_cli_inputs_and_flags_hold(run):
    """Exercise ``-q``, ``-o``, ``--timeout`` and an explicit ``model``.

    Args:
        run: Always ``None``.

    Yields:
        Finding: One per entry of :data:`D10_ASSERTIONS`, in that order.
    """
    target = _local_target()
    if target is None:
        yield from _all(D10_ASSERTIONS, Outcome.CANNOT_TEST, _NO_TARGET_DETAIL)
        return
    base_url, model = target
    root = _seed_scratch("d10")
    if root is None:
        yield from _all(D10_ASSERTIONS, Outcome.CANNOT_TEST, _NO_SEED_DETAIL)
        return
    try:
        yield _questions_conflict_finding(root, base_url, model)
        yield _output_file_finding(root, base_url, model)
        yield _timeout_override_finding(root, model)
        yield _explicit_model_finding(root, base_url, model)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _questions_conflict_finding(root, base_url, model):
    """D10 assertion 1: ``-q`` plus ``questions`` in the input is refused.

    The endpoint is a closed port: an empty chain proves no request was made,
    and a request that was made would end in a transport hop instead.

    Args:
        root: The scratch workspace.
        base_url: The local endpoint (unused: the refusal precedes it).
        model: The local model.

    Returns:
        Finding: PASS on exit 2, ``invalid_input`` and an empty chain.
    """
    dead = "http://127.0.0.1:%d" % _closed_port()
    if not _install_decision(root, _decision_section(dead, model)):
        return _finding(D10_ASSERTIONS, 0, None, _NO_INSTALL_DETAIL)
    questions = _write_json(root, "questions.json", QUESTIONS_ONLY)
    attempt = _decide(root, THREE_TYPES_REQUEST, "d10-conflict",
                      env=BLANK_KEY, memory=False,
                      extra=(QUESTIONS_FLAG, str(questions)))
    envelope = _envelope_of(attempt)
    if envelope is None:
        return _cannot_or_fail_one(D10_ASSERTIONS, 0, attempt)
    error = envelope.get("error") or {}
    ok = (attempt.output.exit_code == EXIT_BAD_INPUT
          and error.get("kind") == KIND_INVALID_INPUT
          and _chain(envelope) == [])
    return _finding(D10_ASSERTIONS, 0, "" if ok else (
        "exit %d, error kind %r, chain %s"
        % (attempt.output.exit_code, error.get("kind"),
           _brief(_chain(envelope)))))


def _output_file_finding(root, base_url, model):
    """D10 assertion 2: ``-o`` carries the envelope and stdout stays empty.

    Args:
        root: The scratch workspace.
        base_url: The local endpoint.
        model: The local model.

    Returns:
        Finding: PASS when the file parses as an envelope and stdout is empty.
    """
    if not _install_decision(root, _decision_section(base_url, model)):
        return _finding(D10_ASSERTIONS, 1, None, _NO_INSTALL_DETAIL)
    target = root / "out.json"
    target.unlink(missing_ok=True)
    attempt = _decide(root, THREE_TYPES_REQUEST, "d10-output", env=BLANK_KEY,
                      memory=False, extra=(OUTPUT_FLAG, str(target)))
    if not attempt.ok:
        return _finding(D10_ASSERTIONS, 1, None, attempt.failure)
    try:
        written = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return _finding(D10_ASSERTIONS, 1,
                        "the output file did not hold an envelope: %s" % exc)
    ok = (isinstance(written, dict) and SCHEMA_VERSION_KEY in written
          and not attempt.output.stdout.strip())
    return _finding(D10_ASSERTIONS, 1, "" if ok else (
        "stdout held %d bytes, schema_version %s in the file"
        % (len(attempt.output.stdout.strip()),
           "present" if isinstance(written, dict)
           and SCHEMA_VERSION_KEY in written else "absent")))


def _timeout_override_finding(root, model):
    """D10 assertion 3: ``--timeout`` beats ``timeout_secs`` per attempt.

    The endpoint accepts the connection and never answers. The toml says 120
    seconds; the flag says one. The attempt has to end as a timeout well
    inside ten.

    Args:
        root: The scratch workspace.
        model: The model name to declare.

    Returns:
        Finding: PASS on a ``timeout`` hop inside the wall-clock bound.
    """
    with _silent_endpoint() as base_url:
        if not _install_decision(root, _decision_section(
                base_url, model, timeout_secs=TOML_TIMEOUT_S)):
            return _finding(D10_ASSERTIONS, 2, None, _NO_INSTALL_DETAIL)
        started = time.monotonic()
        attempt = _decide(root, THREE_TYPES_REQUEST, "d10-timeout",
                          env=BLANK_KEY, memory=False,
                          extra=(TIMEOUT_FLAG, str(OVERRIDE_TIMEOUT_S)))
        elapsed = time.monotonic() - started
    envelope = _envelope_of(attempt)
    if envelope is None:
        return _cannot_or_fail_one(D10_ASSERTIONS, 2, attempt)
    cause = _first_hop_cause(envelope)
    ok = cause == CAUSE_TIMEOUT and elapsed < OVERRIDE_WALL_CLOCK_S
    return _finding(D10_ASSERTIONS, 2, "" if ok else (
        "first hop cause %r after %.1f s; expected %s inside %d s with "
        "timeout_secs = %d in the toml"
        % (cause, elapsed, CAUSE_TIMEOUT, OVERRIDE_WALL_CLOCK_S,
           TOML_TIMEOUT_S)))


def _explicit_model_finding(root, base_url, model):
    """D10 assertion 4: an explicit ``model`` is tried and nothing else.

    The declared primary is absent, so a product that ignored the explicit
    model would record that absent primary as a first hop.

    Args:
        root: The scratch workspace.
        base_url: The local endpoint.
        model: The real local model, declared as the fallback and requested.

    Returns:
        Finding: PASS when the chain holds exactly that model.
    """
    section = _decision_section(base_url, ABSENT_DECISION_MODEL,
                                fallbacks=(model,))
    if not _install_decision(root, section):
        return _finding(D10_ASSERTIONS, 3, None, _NO_INSTALL_DETAIL)
    request = dict(THREE_TYPES_REQUEST, model=model)
    attempt = _decide(root, request, "d10-explicit-model", env=BLANK_KEY,
                      memory=False)
    envelope = _envelope_of(attempt)
    if envelope is None:
        return _cannot_or_fail_one(D10_ASSERTIONS, 3, attempt)
    chain = _chain(envelope)
    ok = len(chain) == 1 and chain[0].get("model") == model
    return _finding(D10_ASSERTIONS, 3, "" if ok else (
        "chain was %s, expected one hop for %r" % (_brief(chain), model)))


# ---------------------------------------------------------------------------
# D11 -- S11 and S24, the agent calls decide on its own in the default tier
# ---------------------------------------------------------------------------

D11_ASSERTIONS = (
    "when the agent calls decide in the default tier the call is approved",
)

#: The tool's name as the agent calls it, and the key of the run record.
DECIDE_TOOL_NAME = "decide"
TOOL_CALLS_KEY = "tool_calls"
#: What the product records for a call the authorization tier refused
#: (``src/agent/mod.rs``); a call that ran and failed carries other text.
TIER_DENIAL_TEXT = "denied: not authorized in the current authorization tier"
QUERY_TIMEOUT_S = 420
CONSULT_SUBCOMMAND = "consult"
CONSULT_TIMEOUT_S = 600

#: A request that is plainly a closed choice and that names the tool, so the
#: only open question is whether the main model reaches for it. Whether it
#: does is the model's behaviour, never the product's contract.
D11_PROMPT = (
    "Use the decide tool, and nothing else, to classify this issue: 'The "
    "application crashes when I click Save'. Ask ONE choice question named "
    "kind with the options bug, feature and question. Then tell me the "
    "result in one sentence.\n")


@scenario("D11", assertions=D11_ASSERTIONS, needs_backend=True)
def the_agent_calls_decide_in_the_default_tier(run):
    """Ask the agent a closed question without ``--auto`` and read its record.

    ``decide`` is a read-only tool, so the default tier approves it with no
    flag. The choice to call it belongs to the main model: when it answers
    without calling the tool there is nothing to judge, and the assertion is
    ``CANNOT_TEST`` with the observation recorded.

    Args:
        run: Always ``None``.

    Yields:
        Finding: One per entry of :data:`D11_ASSERTIONS`.
    """
    attempt = _query_in_environment(D11_PROMPT, "d11-query")
    records = _decide_records(attempt)
    if records is None:
        yield from _cannot_or_fail(D11_ASSERTIONS, attempt)
        return
    if not records:
        yield _finding(D11_ASSERTIONS, 0, None, _NOT_CALLED_DETAIL)
        return
    denied = [record for record in records
              if TIER_DENIAL_TEXT in str(record.get("result"))]
    failed = [record for record in records
              if record not in denied and record.get("ok") is not True]
    observation = "" if not failed else (
        "%d of %d decide calls were approved and executed but returned an "
        "error (the request the model built, recorded as an observation): %s"
        % (len(failed), len(records),
           "; ".join(_excerpt_text(str(r.get("result"))) for r in failed)))
    if denied:
        yield _finding(D11_ASSERTIONS, 0, (
            "%d of %d decide calls were denied by the tier: %s"
            % (len(denied), len(records),
               "; ".join(_excerpt_text(str(r.get("result")))
                         for r in denied))))
        return
    finding = _finding(D11_ASSERTIONS, 0, "")
    yield dataclasses.replace(finding, detail=observation) if observation else finding


# ---------------------------------------------------------------------------
# D12 -- S12, a dead decision endpoint costs a consult nothing
# ---------------------------------------------------------------------------

D12_ASSERTIONS = (
    "a consult with a dead decision endpoint still completes",
    "the run log records one mode fallback with its cause",
    "nothing about the fallback reaches stderr or stdout",
)

#: What the product writes for an internal fallback (its log target and
#: message). Neither may reach a screen.
FALLBACK_LOG_TARGET = "magi_rs::decision::fallback"
FALLBACK_MESSAGE = "fell back to the LLM"
FALLBACK_MODE_FIELD = "consumer=mode"
_FALLBACK_CAUSE = re.compile(r"cause=([a-z0-9_]+)")
D12_PAYLOAD = (b"Is a single-line change worth a full review? Answer "
               b"briefly.\n")


@scenario("D12", assertions=D12_ASSERTIONS, needs_backend=True)
def a_dead_decision_endpoint_costs_a_consult_nothing(run):
    """Run a consult whose decision endpoint refuses connections.

    The consult names no mode, so the product infers it: it asks ``decide``
    first, fails, and must fall back to the main model, leave exactly one run
    log entry with the cause, and say nothing anywhere else. The workspace is
    the environment's own configuration with only ``[decision]`` replaced.

    Args:
        run: Always ``None``.

    Yields:
        Finding: One per entry of :data:`D12_ASSERTIONS`, in that order.
    """
    root = _seed_scratch("d12")
    if root is None:
        yield from _all(D12_ASSERTIONS, Outcome.CANNOT_TEST, _NO_SEED_DETAIL)
        return
    try:
        environment_toml = _read_environment_toml()
        dead = "http://127.0.0.1:%d" % _closed_port()
        if environment_toml is None or not _write_toml(root, (
                _without_decision(environment_toml) + "\n"
                + _decision_section(dead, "smoke-dead-decision-model"))):
            yield from _all(D12_ASSERTIONS, Outcome.CANNOT_TEST,
                            _NO_INSTALL_DETAIL)
            return
        attempt = runs.attempt(
            [CONSULT_SUBCOMMAND, "--output-format", "json", WORKDIR_FLAG,
             str(root), NO_MEMORY_FLAG],
            stdin=D12_PAYLOAD, timeout_s=CONSULT_TIMEOUT_S, label="d12-consult",
            env={"MAGI_PASSPHRASE": runs.passphrase(), **BLANK_KEY})
        if not attempt.ok:
            yield from _all(D12_ASSERTIONS, Outcome.CANNOT_TEST,
                            attempt.failure)
            return
        yield _consult_completed_finding(attempt)
        yield _one_fallback_entry_finding(_read_logs(root))
        yield _fallback_is_silent_finding(attempt)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _consult_completed_finding(attempt):
    """Judge D12's first assertion.

    Args:
        attempt: The consult capture.

    Returns:
        Finding: PASS on exit 0 with a JSON object on stdout.
    """
    envelope = _envelope_of(attempt)
    ok = attempt.output.exit_code == EXIT_DECIDED and envelope is not None
    return _finding(D12_ASSERTIONS, 0, "" if ok else (
        "exit %d, stdout %s: %s" % (
            attempt.output.exit_code,
            "is a JSON object" if envelope is not None else "is not JSON",
            _excerpt(attempt.output))))


def _one_fallback_entry_finding(log):
    """Judge D12's second assertion over the run log.

    Args:
        log: The log bytes, or None.

    Returns:
        Finding: PASS on exactly one mode entry carrying a cause.
    """
    if log is None:
        return _finding(D12_ASSERTIONS, 1,
                        "the product wrote no run log, so no fallback was "
                        "recorded")
    lines = [line for line in log.decode("utf-8", errors="replace").splitlines()
             if FALLBACK_LOG_TARGET in line and FALLBACK_MODE_FIELD in line]
    causes = [_FALLBACK_CAUSE.search(line) for line in lines]
    ok = len(lines) == 1 and causes[0] is not None
    return _finding(D12_ASSERTIONS, 1, "" if ok else (
        "%d mode fallback entries in the run log, causes %s"
        % (len(lines), [c.group(1) if c else None for c in causes])))


def _fallback_is_silent_finding(attempt):
    """Judge D12's third assertion over both screens.

    Args:
        attempt: The consult capture.

    Returns:
        Finding: PASS when neither stream names the fallback.
    """
    leaks = [name for name, data in (("stdout", attempt.output.stdout),
                                     ("stderr", attempt.output.stderr))
             for needle in (FALLBACK_LOG_TARGET, FALLBACK_MESSAGE)
             if needle.encode("utf-8") in data]
    return _finding(D12_ASSERTIONS, 2, "" if not leaks else (
        "the fallback is named on %s" % ", ".join(sorted(set(leaks)))))


# ---------------------------------------------------------------------------
# D13 -- S16 and S21, the run record of a decide call keeps no state
# ---------------------------------------------------------------------------

D13_ASSERTIONS = (
    "the decide call in tool_calls stores the state as <omitted>",
    "the decide call in tool_calls keeps its questions and its result",
    "the state marker reaches neither the decide record nor the decide log lines",
)

STATE_OMITTED = "<omitted>"
DECISION_LOG_TARGET = "magi_rs::decision"


@scenario("D13", assertions=D13_ASSERTIONS, needs_backend=True)
def a_decide_call_leaves_no_state_in_the_record(run):
    """Make the agent pass a unique marker as the state and search for it.

    The marker is minted per run and appears nowhere else in the prompt than
    as the state, so finding it in the decide record or the decide log lines
    can only mean the product stored the state. ``CANNOT_TEST`` when the main
    model did not call ``decide``, or put the marker somewhere else than the
    state: both are the model's behaviour.

    Args:
        run: Always ``None``.

    Yields:
        Finding: One per entry of :data:`D13_ASSERTIONS`, in that order.
    """
    marker = mint_credential()
    prompt = (
        "Use the decide tool, and nothing else. Pass EXACTLY this text as the "
        "state, character for character: %s . Ask ONE choice question named "
        "kind with the options bug, feature and question, whose instructions "
        "do not repeat that text. Then tell me the result in one sentence.\n"
        % marker.marker)
    attempt = _query_in_environment(prompt, "d13-query", planted=(marker,))
    records = _decide_records(attempt)
    if records is None:
        yield from _cannot_or_fail(D13_ASSERTIONS, attempt)
        return
    if not records:
        yield from _all(D13_ASSERTIONS, Outcome.CANNOT_TEST,
                        _NOT_CALLED_DETAIL)
        return
    record = records[0]
    questions = (record.get("input") or {}).get("questions")
    if marker.marker in json.dumps(questions):
        yield from _all(D13_ASSERTIONS, Outcome.CANNOT_TEST,
                        "the model put the marker in the questions, so the "
                        "state could not be told from them")
        return
    yield _state_omitted_finding(record)
    yield _questions_and_result_finding(record, questions)
    yield _marker_absent_finding(records, marker.marker)


def _state_omitted_finding(record):
    """Judge D13's first assertion.

    Args:
        record: The first decide record of ``tool_calls``.

    Returns:
        Finding: PASS when ``input.state`` is the omission marker.
    """
    state = (record.get("input") or {}).get("state")
    return _finding(D13_ASSERTIONS, 0, "" if state == STATE_OMITTED else (
        "input.state is %s, expected %r" % (_excerpt_text(repr(state)),
                                            STATE_OMITTED)))


def _questions_and_result_finding(record, questions):
    """Judge D13's second assertion.

    Args:
        record: The first decide record.
        questions: Its ``input.questions``.

    Returns:
        Finding: PASS on non-empty questions and a result that parses as the
        decide envelope (it carries its ``schema_version``).
    """
    result = _result_envelope(record.get("result"))
    ok = (isinstance(questions, dict) and bool(questions)
          and isinstance(result, dict) and SCHEMA_VERSION_KEY in result)
    return _finding(D13_ASSERTIONS, 1, "" if ok else (
        "questions %s, result %s" % (
            "present" if isinstance(questions, dict) and questions else "missing",
            _excerpt_text(str(record.get("result"))))))


def _result_envelope(text):
    """Parse the envelope a decide call returned to the agent.

    The agent loop frames a tool error as ``Execution failed: <text>``, so the
    envelope starts at the first ``{`` and a success has no prefix at all.

    Args:
        text: The record's ``result``.

    Returns:
        dict | None: The envelope, or None when the text carries none.
    """
    start = str(text or "").find("{")
    if start < 0:
        return None
    try:
        return json.loads(str(text)[start:])
    except ValueError:
        return None


def _marker_absent_finding(records, marker):
    """Judge D13's third assertion over the record and the log.

    Args:
        records: Every decide record of the run.
        marker: The planted state marker.

    Returns:
        Finding: PASS when neither the records nor the decide log lines carry
        it; CANNOT_TEST when no run log exists to search.
    """
    if marker in json.dumps(records):
        return _finding(D13_ASSERTIONS, 2,
                        "the marker is in the decide record of tool_calls")
    log = _read_logs(runs.workspace_root())
    if log is None:
        return _finding(D13_ASSERTIONS, 2, None,
                        "the product wrote no run log to search")
    lines = [line for line in log.decode("utf-8", errors="replace").splitlines()
             if DECISION_LOG_TARGET in line]
    return _finding(D13_ASSERTIONS, 2, "" if not any(
        marker in line for line in lines) else (
        "the marker is in a decide log line"))


_NOT_CALLED_DETAIL = (
    "the main model answered without calling decide; whether it reaches for "
    "the tool is its behaviour, not the product's contract (recorded as an "
    "observation)")


def _read_environment_toml():
    """The persistent environment's ``magi.toml``.

    Returns:
        str | None: Its text, or None when unreadable.
    """
    try:
        return (runs.workspace_root() / MAGI_DIR_NAME
                / MAGI_TOML_NAME).read_text(encoding="utf-8")
    except OSError:
        return None


def _query_in_environment(prompt, label, *, planted=()):
    """Run ``magi query`` at the default tier against the environment.

    Args:
        prompt: What to ask the agent.
        label: The archive label.
        planted: Secrets the prompt carries.

    Returns:
        Attempt: The capture, or why there is none.
    """
    return runs.attempt(
        [QUERY_SUBCOMMAND, "--output-format", "json", WORKDIR_FLAG,
         str(runs.workspace_root()), NO_MEMORY_FLAG],
        stdin=prompt.encode("utf-8"), timeout_s=QUERY_TIMEOUT_S, label=label,
        planted=planted,
        env={"MAGI_PASSPHRASE": runs.passphrase(), **BLANK_KEY})


def _decide_records(attempt):
    """The ``decide`` entries of a query's ``tool_calls``.

    Args:
        attempt: The query capture.

    Returns:
        list[dict] | None: The records (possibly empty), or None when the
        output carries no readable ``tool_calls``.
    """
    envelope = _envelope_of(attempt)
    calls = envelope.get(TOOL_CALLS_KEY) if envelope else None
    if not isinstance(calls, list):
        return None
    return [call for call in calls
            if isinstance(call, dict) and call.get("name") == DECIDE_TOOL_NAME]


def _excerpt_text(text, limit=200):
    """Shorten a text for a detail.

    Args:
        text: Any text.
        limit: Characters to keep.

    Returns:
        str: The start of *text*.
    """
    return text[:limit]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_NO_KEY_DETAIL = (
    "no %s in the harness's environment, so the hosted provider could not be "
    "reached with a key. Export a valid OpenRouter key under that name and "
    "re-run" % KEY_VARIABLE)
_NO_TARGET_DETAIL = (
    "the environment's magi.toml declares no usable [decision] model and "
    "endpoint, so there is no local target to run against")
_NO_SEED_DETAIL = "the product's init did not scaffold a scratch workspace"
_NO_INSTALL_DETAIL = "the scratch workspace's magi.toml could not be written"

_DECISION_HEADERS = (DECISION_HEADER, DECISION_FALLBACK_HEADER)


def _hosted_key():
    """Read the hosted provider's key from the harness's own environment.

    Returns:
        PlantedSecret | None: The key declared as planted (so the archive
        scrubs it), or None when it is absent or blank.
    """
    value = os.environ.get(KEY_VARIABLE, "").strip()
    return PlantedSecret(value, "hosted decision key") if value else None


def _local_target():
    """The local endpoint and model the persistent environment declares.

    Read off the file the preflight normalised, never from a copy of the
    product's defaults: with the cheap profile it names the cheap model, and
    with the product's defaults it names the shipped one.

    Returns:
        tuple[str, str] | None: ``(base_url, model)``, or None when the file
        cannot be read or declares no decision model.
    """
    try:
        document = tomllib.loads((runs.workspace_root() / MAGI_DIR_NAME
                                  / MAGI_TOML_NAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return None
    section = document.get("decision") or {}
    model = section.get("model")
    base_url = section.get("base_url") or document.get("base_url")
    if isinstance(model, str) and model and isinstance(base_url, str):
        return base_url, model
    return None


def _decision_section(base_url, model, fallbacks=(), min_confidence=None,
                      timeout_secs=None):
    """Render one ``[decision]`` section.

    Args:
        base_url: The endpoint.
        model: The primary model.
        fallbacks: Fallback model names, in order.
        min_confidence: The primary's threshold, or None for none.
        timeout_secs: The per-attempt timeout, or None for the default.

    Returns:
        str: TOML text.
    """
    lines = ["[decision]", 'base_url = "%s"' % base_url,
             'model = "%s"' % model]
    if timeout_secs is not None:
        lines.append("timeout_secs = %d" % timeout_secs)
    if min_confidence is not None:
        lines.append("min_confidence = %s" % min_confidence)
    for name in fallbacks:
        lines += ["", "[[decision.fallback]]", 'model = "%s"' % name]
    return "\n".join(lines) + "\n"


def _without_decision(text):
    """Remove the ``[decision]`` section and its fallback blocks.

    Complexity: ``O(lines)``.

    Args:
        text: A ``magi.toml``.

    Returns:
        str: The text as a v0.22.0 file would have carried it.
    """
    kept = []
    dropping = False
    for line in text.splitlines():
        header = table_header(line)
        if header is not None:
            dropping = header in _DECISION_HEADERS
        if not dropping:
            kept.append(line)
    return "\n".join(kept) + "\n"


def _seed_scratch(label):
    """Scaffold one scratch workspace with the product's own ``init``.

    Args:
        label: A short name for the directory and the archive.

    Returns:
        pathlib.Path | None: The directory holding ``.magi/``, or None.
    """
    root = pathlib.Path(tempfile.mkdtemp(prefix="%s-" % label,
                                         dir=str(runs.scratch_root())))
    seeded = runs.attempt([INIT_SUBCOMMAND], stdin=b"",
                          timeout_s=INIT_TIMEOUT_S, label="%s-init" % label,
                          cwd=root, env={"MAGI_PASSPHRASE": runs.passphrase()})
    if not seeded.ok or not (root / MAGI_DIR_NAME).is_dir():
        shutil.rmtree(root, ignore_errors=True)
        return None
    return root


def _read_toml(root):
    """Read a scratch workspace's ``magi.toml``.

    Args:
        root: The workspace.

    Returns:
        str | None: The text, or None when unreadable.
    """
    try:
        return (root / MAGI_DIR_NAME / MAGI_TOML_NAME).read_text(
            encoding="utf-8")
    except OSError:
        return None


def _write_toml(root, text):
    """Replace a scratch workspace's ``magi.toml``.

    Args:
        root: The workspace.
        text: The new content.

    Returns:
        bool: Whether it was written.
    """
    try:
        (root / MAGI_DIR_NAME / MAGI_TOML_NAME).write_text(text,
                                                           encoding="utf-8")
    except OSError:
        return False
    return True


def _install_decision(root, section):
    """Install the product's generated text with *section* as its decision.

    Args:
        root: The scratch workspace.
        section: The ``[decision]`` text to use.

    Returns:
        bool: Whether the file was rewritten.
    """
    current = _read_toml(root)
    if current is None:
        return False
    return _write_toml(root, _without_decision(current) + "\n" + section)


def _write_json(root, name, document):
    """Write a JSON file beside a scratch workspace's ``.magi/``.

    Args:
        root: The workspace.
        name: The file name.
        document: The JSON-serialisable content.

    Returns:
        pathlib.Path: The file.
    """
    path = root / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _decide(root, request, label, *, env=None, planted=(), memory=True,
            extra=()):
    """Run ``magi decide`` against a workspace, request on stdin.

    Args:
        root: The workspace (``-w``).
        request: The request document.
        label: What to call it in the archive.
        env: Variables overlaid on the harness's environment.
        planted: Secrets this invocation puts in front of the product.
        memory: Whether the vault is opened (the passphrase is supplied).
            When False the invocation carries ``--no-memory``.
        extra: More arguments.

    Returns:
        Attempt: The capture, or why there is none.
    """
    argv = [DECIDE_SUBCOMMAND, WORKDIR_FLAG, str(root)]
    overlay = dict(env or {})
    if memory:
        overlay["MAGI_PASSPHRASE"] = runs.passphrase()
    else:
        argv.append(NO_MEMORY_FLAG)
    argv += list(extra)
    return runs.attempt(argv, stdin=json.dumps(request).encode("utf-8"),
                        timeout_s=INVOCATION_TIMEOUT_S, label=label,
                        planted=planted, env=overlay)


def _decide_in_environment(request, label):
    """Run ``magi decide`` against the persistent environment.

    Args:
        request: The request document.
        label: The archive label.

    Returns:
        Attempt: The capture, or why there is none.
    """
    return _decide(runs.workspace_root(), request, label, env=BLANK_KEY)


def _in_scratch_workspace(label, section, request, *, env=None, planted=()):
    """Seed a workspace, install *section*, run decide, remove the workspace.

    Args:
        label: Short name for the directory and archive.
        section: The ``[decision]`` text.
        request: The request document.
        env: Variables overlaid on the harness's environment.
        planted: Secrets declared to the archive scrubber.

    Returns:
        Attempt: The capture, or the reason there is none.
    """
    root = _seed_scratch(label)
    if root is None:
        return runs.Attempt(output=None, failure=_NO_SEED_DETAIL)
    try:
        if not _install_decision(root, section):
            return runs.Attempt(output=None, failure=_NO_INSTALL_DETAIL)
        return _decide(root, request, "%s-decide" % label, env=env,
                       planted=planted, memory=False)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _envelope_of(attempt):
    """Parse an attempt's stdout as the decide envelope.

    Args:
        attempt: The capture.

    Returns:
        dict | None: The envelope, or None when there is no capture or stdout
        is not a JSON object.
    """
    if not attempt.ok:
        return None
    try:
        document = json.loads(attempt.output.stdout.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    return document if isinstance(document, dict) else None


def _chain(envelope):
    """The envelope's chain of attempts, or an empty list.

    Args:
        envelope: A parsed envelope.

    Returns:
        list: The hops that are objects.
    """
    chain = envelope.get("chain")
    if not isinstance(chain, list):
        return []
    return [hop for hop in chain if isinstance(hop, dict)]


def _first_hop_cause(envelope):
    """The cause of the first hop, when there is one.

    Args:
        envelope: A parsed envelope, or None.

    Returns:
        str | None: The cause.
    """
    chain = _chain(envelope or {})
    return chain[0].get("cause") if chain else None


def _any_answers(envelope):
    """Whether any hop of the chain carries answers.

    Args:
        envelope: A parsed envelope.

    Returns:
        bool: True when some hop holds an ``answers`` key.
    """
    return any("answers" in hop for hop in _chain(envelope))


def _answered_all_types(attempt, envelope):
    """Judge "answers one question of each type with exit 0".

    Args:
        attempt: The capture.
        envelope: The parsed envelope.

    Returns:
        str: Empty on success, otherwise the detail.
    """
    answers = envelope.get("answers")
    expected = {name: q["type"]
                for name, q in THREE_TYPES_REQUEST["questions"].items()}
    got = ({k: v.get("type") for k, v in answers.items()
            if isinstance(v, dict)} if isinstance(answers, dict) else {})
    if attempt.output.exit_code == EXIT_DECIDED and got == expected:
        return ""
    error = envelope.get("error") or {}
    return "exit %d, answer types %s, expected %s%s" % (
        attempt.output.exit_code, got, expected,
        (", error %s: %s" % (error.get("kind"), error.get("message")))
        if error else "")


def _confidences_in_range(envelope):
    """Judge "every answer carries a Magi confidence between 0 and 1".

    Args:
        envelope: The parsed envelope.

    Returns:
        str: Empty on success, otherwise the detail.
    """
    answers = envelope.get("answers")
    if not isinstance(answers, dict) or not answers:
        return "the envelope carries no answers"
    bad = [name for name, a in answers.items()
           if not (isinstance(a, dict) and _is_number(a.get("confidence"))
                   and 0.0 <= a["confidence"] <= 1.0)]
    return "" if not bad else (
        "no confidence in [0, 1] on: %s" % ", ".join(sorted(bad)))


def _names_model_and_usage(envelope):
    """Judge "the envelope names the model that answered and its usage".

    Args:
        envelope: The parsed envelope.

    Returns:
        str: Empty on success, otherwise the detail.
    """
    usage = envelope.get("usage")
    model = envelope.get("model")
    tokens = (isinstance(usage, dict)
              and all(isinstance(usage.get(k), int)
                      and not isinstance(usage.get(k), bool)
                      for k in ("input_tokens", "output_tokens")))
    if isinstance(model, str) and model and tokens:
        return ""
    return "model %r, usage %r" % (model, usage)


def _numeric_cost(envelope):
    """Judge "the envelope reports a numeric cost".

    Args:
        envelope: The parsed envelope.

    Returns:
        str: Empty on success, otherwise the detail.
    """
    usage = envelope.get("usage") or {}
    return "" if _is_number(usage.get("cost")) else (
        "usage.cost was %r" % (usage.get("cost"),))


def _legend_covers_levels(envelope):
    """Judge "the score answer has a legend for every level".

    Args:
        envelope: The parsed envelope.

    Returns:
        str: Empty on success, otherwise the detail.
    """
    answers = envelope.get("answers") or {}
    score = answers.get("urgency") if isinstance(answers, dict) else None
    legend = score.get("legend") if isinstance(score, dict) else None
    wanted = {str(i) for i in range(LEVEL_COUNT)}
    if isinstance(legend, dict) and set(legend) == wanted and all(
            isinstance(v, str) and v for v in legend.values()):
        return ""
    return "legend was %r, expected a name for each of %s" % (
        legend, sorted(wanted))


def _is_number(value):
    """Whether *value* is a real number (a bool is not).

    Args:
        value: Anything.

    Returns:
        bool: True for an int or float that is not a bool.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finding(assertions, index, detail, cannot=None):
    """Build one finding.

    Args:
        assertions: The scenario's assertion tuple.
        index: Position in it.
        detail: Empty for PASS, otherwise the failure. Ignored when *cannot*
            is given.
        cannot: A reason the assertion could not be evaluated, which makes the
            outcome ``CANNOT_TEST``.

    Returns:
        Finding: With no run id; every decide scenario is standalone.
    """
    if cannot is not None:
        return Finding(assertion=assertions[index],
                       outcome=Outcome.CANNOT_TEST, detail=cannot, run_id=None)
    return Finding(assertion=assertions[index],
                   outcome=Outcome.PASS if not detail else Outcome.FAIL,
                   detail=detail, run_id=None)


def _all(assertions, outcome, detail):
    """Yield one finding of the same outcome for every assertion.

    Args:
        assertions: The scenario's assertion tuple.
        outcome: The outcome for all of them.
        detail: The cause.

    Yields:
        Finding: One per assertion, in order.
    """
    for text in assertions:
        yield Finding(assertion=text, outcome=outcome, detail=detail,
                      run_id=None)


def _cannot_or_fail(assertions, attempt):
    """Report every assertion when no envelope could be read.

    No capture is the environment's refusal (``CANNOT_TEST``). A capture whose
    stdout is not an envelope is the product's output failing its contract
    (``FAIL``).

    Args:
        assertions: The scenario's assertion tuple.
        attempt: The capture.

    Yields:
        Finding: One per assertion.
    """
    if not attempt.ok:
        yield from _all(assertions, Outcome.CANNOT_TEST, attempt.failure)
    else:
        yield from _all(assertions, Outcome.FAIL, _not_an_envelope(attempt))


def _cannot_or_fail_one(assertions, index, attempt):
    """The single-assertion form of :func:`_cannot_or_fail`.

    Args:
        assertions: The scenario's assertion tuple.
        index: Which assertion.
        attempt: The capture.

    Returns:
        Finding: The finding.
    """
    if not attempt.ok:
        return _finding(assertions, index, None, attempt.failure)
    return _finding(assertions, index, _not_an_envelope(attempt))


def _not_an_envelope(attempt):
    """Describe a stdout that is not a decide envelope.

    Args:
        attempt: A capture whose stdout did not parse.

    Returns:
        str: The exit code and the start of what was printed.
    """
    return "exit %d and stdout is not a JSON envelope: %s" % (
        attempt.output.exit_code, _excerpt(attempt.output))


def _excerpt(output, limit=300):
    """The beginning of both streams, decoded leniently.

    Args:
        output: A capture.
        limit: Bytes to keep.

    Returns:
        str: The excerpt.
    """
    return output.raw()[:limit].decode("utf-8", errors="replace").strip()


def _brief(value):
    """Render a hop or a chain compactly for a detail.

    Args:
        value: A hop (dict) or a chain (list).

    Returns:
        str: ``model/outcome/cause`` per hop.
    """
    hops = value if isinstance(value, list) else [value]
    return "[%s]" % ", ".join(
        "%s/%s/%s" % (h.get("model"), h.get("outcome"), h.get("cause"))
        for h in hops if isinstance(h, dict))


def _closed_port():
    """Find a port nothing listens on.

    Returns:
        int: A port the operating system has just released.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _read_logs(root):
    """Concatenate every file under a workspace's log directory.

    Args:
        root: The workspace.

    Returns:
        bytes | None: The bytes, or None when the directory does not exist or
        holds nothing readable.
    """
    directory = root.joinpath(*LOG_DIR_PARTS)
    if not directory.is_dir():
        return None
    chunks = []
    for path in sorted(directory.rglob("*")):
        try:
            if path.is_file():
                chunks.append(path.read_bytes())
        except OSError:
            continue
    return b"\n".join(chunks) if chunks else None


class _SilentHandler(socketserver.BaseRequestHandler):
    """Holds a connection open and never answers it."""

    def handle(self) -> None:
        """Wait until the server shuts down; say nothing."""
        self.server.release.wait()


class _SilentServer(socketserver.ThreadingTCPServer):
    """A listener that accepts and stays quiet, for D10's timeout.

    Attributes:
        release: Set on shutdown so the parked handler threads end.
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address):
        """Bind on *address*.

        Args:
            address: ``(host, port)``.
        """
        self.release = threading.Event()
        super().__init__(address, _SilentHandler)


@contextlib.contextmanager
def _silent_endpoint():
    """Serve a listener that accepts connections and never replies.

    Yields:
        str: The base URL to point ``[decision].base_url`` at.
    """
    server = _SilentServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d" % server.server_address[1]
    finally:
        server.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)
