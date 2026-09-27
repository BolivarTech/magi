# Reasoning control and the output budget

> How `[magi] reasoning`, `reasoning_spelling`, `max_tokens` and `reasoning_trace` work, what
> each one costs, and what the clock-coverage warning is telling you when it fires.

A reasoning model can spend its whole completion budget thinking and hand back nothing: a
`length` finish with empty content, no verdict, and the tokens are billed anyway. v0.20.0 adds
four `[magi]` keys and one warning, checked on every consult, so a stalled seat is visible and adjustable
instead of a silent, expensive `length` in the consult report. None of this changes which models
the trio talks to or what a default installation does: every value below is what `magi init`
already writes, commented out, in `.magi/magi.toml`.

---

## `reasoning`

```
# reasoning = "default"  # "default" | "disabled" | "enabled"
```

The exact vocabulary of magi-core's `ReasoningControl`, applied to all three seats (magi-core
declined a per-seat override, so there is one value for the whole trio, not three). Left absent,
`"default"` sends nothing on the wire and the backend's own behaviour applies.

- **`default`** (the built-in choice). Nothing is sent; the model decides for itself whether to
  reason.
- **`disabled`** asks the backend to skip its reasoning channel. Measured against the native
  Ollama wire, 11 of the 12 families magi-core checked honour it; `gpt-oss:120b` does not, and a
  completion attempt against it reports `Unsupported` rather than a false zero. Turning reasoning
  off does make a stalled attempt finish faster, but this repository's own measurement
  (`planning/experiments/think-quality-2026-09-23/`) found it costing 20-48% of defect detection
  at every output cap tried. **It is not recommended for a review gate**: a judge that misses a
  defect and reads exactly like one that didn't is a worse failure than a judge that runs long.
- **`enabled`** forces the channel on, which is the lever for a backend whose own default might
  change without notice.

There is no environment variable for this key, matching `[magi].kind`/`base_url`. An unrecognised
value is a configuration error at load, naming the key and the three accepted spellings.

## `reasoning_spelling`

```
# reasoning_spelling = "effort-none"  # (no default)
```

Only `openai-compat` seats speak this vocabulary; `ollama` and `anthropic` ignore it because
neither has an OpenAI-shaped reasoning field to spell. Accepted values: `effort-none`,
`effort-minimal`, `reasoning-enabled-object`. There is no default: the key does nothing unless
declared, and declaring it while every seat runs on a different kind produces a startup notice
saying so rather than a silent no-op.

If `reasoning` is set to anything other than `default` on an `openai-compat` trio with no
spelling declared, magi-rs warns at startup that the control cannot reach the wire, and every
completion record for that trio will report its reasoning as `Unsupported`. An `anthropic` trio
gets the same warning for any control other than `default`, because its seats send no reasoning
switch at all; that includes the trio a TUI `/login` rebuilds, which always runs on `anthropic`.

**The hazard, stated plainly: a spelling the pinned model rejects returns HTTP 400, and
magi-core treats a 400 as lineage-condemning; all three seats sharing that lineage are lost for
the run, not just the one request.** Verify the spelling against the model you have pinned before
relying on it, and verify it again after any update the provider makes to that model: a spelling
that worked yesterday can start failing with no change on the magi-rs side at all. This key is
not exercised against `api.openai.com`; it exists for OpenRouter/Ollama-style OpenAI-compatible
endpoints.

## `max_tokens`

```
# max_tokens = 65536  # tokens; no upper bound of magi-rs's own
```

The output cap sent on every completion attempt, per seat (billed as completion tokens whether
or not the model used the budget to produce a usable answer). Left absent, magi-rs sends its own
declared value, `65536` as of v0.21.0 (`DECLARED_COMPLETION_CAP`; magi-core's own crate default
moved to `32768` in 4.2.0, but magi-rs still sets this field explicitly on every request, so the
crate default is never inherited). The value comes from replay B of REQ-EE-6: against the E-E
bundle, the slowest titular seat, `glm-5.3`, used 56007 tokens of reasoning and answer — 85% of
the cap, tight but enough to finish without a `length` cut.

There is no upper bound magi-rs imposes: raise it as far as a seat's own limits allow. Only zero
or a negative value is rejected at configuration load, along with anything above 4294967295
(`u32::MAX`), the largest number the wire field can carry. A value above what the pinned model
actually accepts as output comes back as an HTTP 400. As with the spelling hazard above, a 400
condemns the whole lineage for that run, not just the oversized request.

**The 65536 default is measured only against the Ollama cloud pool replay B used on
2026-09-27: `glm-5.3`, `kimi-k2.6`, and `deepseek-v4-pro` all completed under it with no HTTP
400.** It is unmeasured for `mistral-large-3`, for any `openai-compat` endpoint, and for
`kind = "anthropic"` — a model whose own output maximum sits below 65536 answers this default
with HTTP 400 and loses its lineage for the run. Check a pinned model's own output limit before
relying on the default.

The number has a direct cost in wall-clock time, and the [clock-coverage warning](#the-clock-coverage-warning)
below exists because of it. At a reference speed of 55 tokens/second (magi-core's own measured
floor across its checked model families), a full 65536-token completion takes roughly 1192
seconds by itself, before rotation or retries. Raising `max_tokens` without also raising
`agent_timeout_secs` (interactive) or `--timeout` (headless) buys a larger budget the clock cannot
reach.

## The clock-coverage warning

Every time the MAGI panel runs (the `consult` tool the agent dispatches on its own, the TUI's
`/consult`, `magi-rs consult`, or `magi-rs query --consult`), magi-rs compares the client timeout
that consult will actually run under against `max_tokens` at 55 tokens/second. When the timeout
would cut the completion off before it could plausibly finish, a `WARN` reaches both the screen
and the log, naming what fraction of the cap the clock covers and what `--timeout` or
`agent_timeout_secs` would cover it instead. The consult still runs: the warning informs, it
does not block, and it never suggests a value the configuration would reject.

**In v0.21.0, with the shipped defaults, this warning still fires on every consult — by design,
not by oversight.** The default ceiling (`agent_timeout_secs = 2335`) derives a 700-second
per-request client timeout sized to cover the trio's *measured convergence* (`glm-5.3` needed up
to 56007 tokens in replay B, roughly 38500 tokens' worth of runway at the 55 tok/s reference
speed), not the full 65536-token cap at that reference speed. Covering the cap in full needs
`--timeout 28619` on the headless path, or `agent_timeout_secs = 3974` on the interactive one —
figures [replay B](#measured-defaults-v0210) recorded but did not run, because a consult that
actually needs the whole cap is rare. The warning carries no cause fields of its own, and it is
emitted under its own tracing target, so it can be filtered as a class if it becomes noise on an
installation that has already tuned both knobs.

## `reasoning_trace`

```
# reasoning_trace = false
```

Off by default. When enabled, only for a **cut attempt** (one that finished with `length` or
came back with empty content), the daily log file gets a bounded head and tail of the model's
reasoning text: 4096 characters from each end (a trace of 8192 characters or fewer is written
whole), cut on a character boundary, sanitized and passed through the same redaction every
foreign string gets before it is written. It goes to the log file only, at `INFO`: never the
consult JSON, never stderr, never the TUI. If the log's own file filter sits above `INFO`, a
startup notice says the flag has no effect at that level.

The trace is untrusted model text, and that has one residual worth stating rather than
discovering later: a credential that was never registered with magi-rs (one pasted directly
into a prompt, say) and that is shorter than 32 characters can still reach the log unmasked if
the model happens to echo it inside a cut attempt's trace. This is why the flag defaults to off.

## Reading `reasoning` and `control` in a consult's completion records

Every `completions[<seat>][k]` record in the consult JSON now carries two more keys alongside the
five it already had:

- `reasoning`: the reasoning state exactly as magi-core reports it, `"NotMeasured"` for a wire
  with no reasoning channel, `{"Measured": {"chars": N, "text": null}}` when a channel was
  measured, or `{"Unsupported": {"backend": ..., "chars": ..., "text": null}}` when the model
  ignored a control it was asked to honour. `text` is always `null`: the model's reasoning text
  never enters this envelope, only its length and state. `"NotMeasured"` is not the same fact as
  zero characters measured; keep the two apart when reading a report.
- `control`: the value `reasoning` was actually sent with for that attempt: `default`,
  `disabled`, or `enabled`.

An attempt that ran until the client timeout killed it, rather than finishing on its own, carries
no reasoning measurement at all (`"NotMeasured"`, with `finish` also `null`): the clock cut it
before there was anything to measure. Every attempt that instead finished by exhausting its
output budget (`finish == "length"`) or came back with empty content also produces one `WARN`
line in the daily log naming the seat, the model, the cap, the token counts, and the reasoning
state that attempt measured. Those lines come from the consult's report, so they are only written
when the consult produces one: if too few seats answer and the whole consult fails, there is no
report to read them from, and the log holds only the failure events magi-core itself emits.

## `agent_timeout_secs`

```
agent_timeout_secs = 2335
```

The per-mage ceiling for the TUI path, and the fallback the headless path uses when no explicit
`--timeout` is given. Its floor is still 30 seconds, below which nothing legitimate fits, but it
has **no upper bound**: an interactive consult is sized for genuine deliberation, not for
chat-speed responsiveness, and a model that needs minutes to finish is a normal case this release
makes room for rather than a violation of the ceiling. The two internal timeout layers derived
from it (the retry budget and the per-request client timeout) still follow the same relation they
always have; there is still exactly one knob.

The 2335-second default derives a 700-second per-request client timeout and a 1401-second
operation budget — the clock basis [replay B](#measured-defaults-v0210) needed for the trio's
measured convergence. Raising the default this far carries three consequences worth stating
plainly rather than discovering later:

- **A TUI `/consult` can block the whole session for up to 2335 seconds per mage, and it cannot
  be cancelled mid-flight.** Cancelling an in-flight consult is tracked as its own backlog item
  under REQ-TUI-1 and did not ship with this release.
- **`magi consult` with no explicit `--timeout` now runs under a roughly 16818-second (~4.7 hour)
  deadline**, derived from the default ceiling with rotation enabled, in place of the 654 seconds
  v0.20.0 derived. This release documents the change rather than shortening it: pass an explicit
  `--timeout` to get a shorter deadline back.
- **`magi query --auto` and `--full-auto` with no explicit `--timeout` now warn `below_formula`
  and publish it in the JSON.** Their tier default deadline (900 seconds,
  `FULL_AUTO_TIMEOUT_SECS`) sits below the roughly 16818-second minimum the derivation formula
  asks for at this ceiling, where at the 90-second default it used to sit above it. The run still
  proceeds and is still obeyed; the remedy is an explicit `--timeout` or `[headless]
  timeout_secs`.

---

## Measured defaults (v0.21.0)

The 65536-token cap, the `kimi-k2.6` Balthasar seat, and the 2335-second `agent_timeout_secs`
default all trace back to two replays run against the published v0.20.0 binary, on the E-E
bundle, on 2026-09-27 between 00:21 and 00:37 UTC, against `http://localhost:11434/v1` (a local
daemon, `:cloud` models) — one attempt per replay, no failures that needed a retry.

**Replay A — what v0.20.0 ships** (cap 16384, the v0.20.0 trio, `--timeout 1800`):

| Seat | Model | finish | completion_tokens | reasoning | control |
|---|---|---|---|---|---|
| Melchior | glm-5.3:cloud | timeout | — | NotMeasured | default |
| Melchior | kimi-k2.6:cloud | timeout | — | NotMeasured | default |
| Melchior | nemotron-3-super:cloud | stop | 7893 | Measured 30836 chars | default |
| Balthasar | gpt-oss:120b-cloud | stop | 1691 | Measured 5914 chars | default |
| Caspar | deepseek-v4-pro:cloud | timeout | — | NotMeasured | default |
| Caspar | minimax-m3:cloud | timeout | — | NotMeasured | default |
| Caspar | mistral-large-3:675b-cloud | stop | 1800 | Measured 0 chars | default |

Three of three seats reached a verdict (two of them through a fallback), zero `length` cuts, and
four timeouts carrying no reasoning measurement. The clock-coverage warning fired, naming
`--timeout 7163` to cover the 16384-token cap. Wall clock: 444 seconds.

**Replay B — the MS2 candidates** (cap 65536, Balthasar `kimi-k2.6`, `--timeout 16820`):

| Seat | Model | finish | completion_tokens | reasoning | control |
|---|---|---|---|---|---|
| Melchior | glm-5.3:cloud | stop | 56007 | Measured 211124 chars | default |
| Balthasar | kimi-k2.6:cloud | stop | 22612 | Measured 88477 chars | default |
| Caspar | deepseek-v4-pro:cloud | stop | 19426 | Measured 67339 chars | default |

Three of three titular seats reached a verdict, with no rotation, zero `length` cuts, and zero
timeouts. The clock-coverage warning still fired, naming `--timeout 28619` to cover the full
65536-token cap at the 55 tok/s reference speed — the run's own 700-second clock covered only
the roughly 38500 tokens the seats actually used. Wall clock: 531 seconds (the three seats ran in
parallel; `glm-5.3`, the slowest at 56007 tokens, still finished inside that window).

**What this decided.** The 65536 cap holds, though with little margin — `glm-5.3` used 85% of
it. `kimi-k2.6` held the Balthasar seat as a titular with room to spare. Every titular's
reasoning channel ran large under `reasoning = "default"`, so the default behaves like
`"enabled"` for this pool; none of E-E's original cuts trace to a daemon that spent tokens
without reasoning to show for it. No 65536-token request produced an HTTP 400 against `glm-5.3`,
`kimi-k2.6`, or `deepseek-v4-pro`.

**Limits, stated rather than hidden.** Each replay ran once (n = 1), in a single evening slot for
US-based cloud endpoints; the Ollama cloud pool is known to behave differently across the day, so
a slower slot could push convergence past what this evidence shows. The full replay evidence,
including the classification of E-E's original cuts, lives in
`planning/milestones/replays-record.md`.

### Recommended `--timeout` for a review gate

For a headless review gate, pass `--timeout 16820`. It derives exactly the default ceiling
(2335 seconds) that replay B ran under, so a gate configured this way needs no `[magi]`
overrides. It also sits below the sanity threshold this release raised `CEILING_SANITY_SECS` to
(2400 seconds): the "extra digit" warning does not fire for `--timeout 16820`, while a classic
typo such as `--timeout 18000` (typed for an intended `1800`) still derives 2499 seconds and
still triggers it.

---

## Two starting points

Copy either fragment into `.magi/magi.toml` and adjust from there; see
[`docs/magi.toml.example`](magi.toml.example) for the complete annotated reference.

The shipped defaults, written out explicitly instead of left commented:

```toml
[magi]
reasoning = "default"
max_tokens = 65536
reasoning_trace = false
agent_timeout_secs = 2335
```

Writing these out pins `max_tokens` at 65536 and `agent_timeout_secs` at 2335 — this release's
values. This release changed both from v0.20.0's (16384 and 90): a file that spells out the old
numbers restores v0.20.0's behaviour exactly; leave them commented to follow whichever release's
defaults you are running instead.

A faster-terminating profile for an `openai-compat` trio (OpenRouter, a self-hosted
OpenAI-compatible gateway, and similar). Read the hazard above before adopting it, and verify
`reasoning_spelling` against the model you have pinned:

```toml
[magi]
reasoning = "disabled"
reasoning_spelling = "effort-none"
```
