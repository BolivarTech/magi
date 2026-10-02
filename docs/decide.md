# Typed decisions: `magi-rs decide`

`magi-rs decide` asks a small decision model a closed question and returns the answer together with the probability of every option. The call is one forward pass. No text is generated, so a decision usually comes back in well under a second once the model is loaded.

The models speak the System One protocol: you send a `state` (text or JSON) and a set of questions, and each question has one of three types.

| Type | You provide | You get back |
|---|---|---|
| `choice` | `criteria`: option name → description (at least 2 options) | the chosen option and a probability for each option |
| `noul` | optional `instructions` and `criteria` (`true` / `false` descriptions) | one number in [0, 1], the probability of "yes" |
| `score` | `criteria`: a list of levels (at least 2) | a `score` in level units, a probability per level index, and a `legend` naming the levels |

## Providers and models

| Provider | `base_url` | Key | Notes |
|---|---|---|---|
| Ollama (default, certified) | `http://localhost:11434` | none | Needs Ollama 0.35 or later. Local, free, no rate limits. |
| OpenRouter (hosted, certified) | `https://openrouter.ai/api/v1` | `DECISION_API_KEY` | Billed per input token; output is free. |
| TypeSafe directly | `https://api.typesafe.ai` | `DECISION_API_KEY` | Same protocol; not part of the certified set. |

`magi-rs` builds the request URL from `base_url` by one rule, without looking at which provider it is: trailing slashes and spaces are dropped, then `/systemone` is appended if the URL already ends in `/v1`, and `/v1/systemone` otherwise. So `http://localhost:11434` and `http://localhost:11434/v1` both reach `http://localhost:11434/v1/systemone`.

Models used by the scaffold that `magi init` writes:

| Model | Provider | Role |
|---|---|---|
| `nimble` | Ollama | primary |
| `tev1` | Ollama | first fallback (4B) |
| `tev1:0.8b` | Ollama | second fallback, smallest and fastest to load |
| `typesafe/jev-1.13` | OpenRouter | primary in the commented example |
| `togethercomputer/tev1-4b-experimental` | OpenRouter | fallback in the example |
| `upstage/solar-decide` | OpenRouter | fallback in the example |

The Ollama models have to be pulled once:

```bash
ollama pull nimble
ollama pull tev1
ollama pull tev1:0.8b
```

Without them, `magi-rs decide` fails with an error that names the missing model, and the chain moves on to the next one. Fallbacks always live on the same endpoint as `[decision]`; there is no `base_url` per fallback.

## Configuration

Everything lives in the `[decision]` section of `.magi/magi.toml`; see `docs/magi.toml.example` for the commented block `magi init` writes.

| Key | Where | Default | Meaning |
|---|---|---|---|
| `base_url` | `[decision]` | the root `base_url` | Endpoint of the decision models. |
| `model` | `[decision]` | none (required) | Primary model. |
| `timeout_secs` | `[decision]` | `120` | Seconds to wait for each model, per attempt. |
| `keep_alive` | `[decision]` | not sent | Ollama only: how long to keep the model loaded (`"10m"`, `"2h"`, `-1`). |
| `auto_approve` | `[decision]` | `true` | Whether the agent's `decide` tool runs without asking. |
| `min_confidence` | `[decision]` | no threshold | Per-model confidence floor (0 to 1) for the primary. |
| `model` | `[[decision.fallback]]` | none (required in each entry) | A fallback model, tried in order. |
| `min_confidence` | `[[decision.fallback]]` | no threshold | Confidence floor for that fallback. |

Unknown keys are a load error, and so is an `api_key` in the file. A model may appear only once in the chain. `[[decision.fallback]]` entries are optional.

### A `magi.toml` from an earlier version

A `magi.toml` that exists but has no `[decision]` section does not load, and neither does one that is empty or holds only comments. No command runs until the file is fixed. If the file is absent altogether, the built-in defaults apply, and they include `[decision]`.

To upgrade a workspace:

1. Move the whole `.magi/` directory aside (`mv .magi .magi.bak-0.22`) and check that the backup holds `.magi-rs-memory.db` and, if they existed, its `-wal` and `-shm` files.
2. Run `magi-rs init` to create a fresh `.magi/`.
3. Copy your own settings from the old `magi.toml` into the new one. To keep history, memories and vault secrets, restore the database from the backup too.

Conversation history, memories and the vault's stored secrets live only in that database. Without it and its original passphrase they are gone.

## Keys

The key is read once, at startup, after the vault is opened. It is chosen in this order:

1. `DECISION_API_KEY`, from the environment or the vault (`magi-rs vault set DECISION_API_KEY`). A blank value counts as absent.
2. The main model's key, only when `[decision]` points at the same origin as the main model: same scheme, host and effective port. The path and any credentials in the URL do not count.
3. No authentication header.

A key is never sent to a different host. Because the test is on the origin, a gateway that routes several services by URL path counts as one origin, and it would receive the main model's key. If that is not what you want, set `DECISION_API_KEY`, which takes precedence. A missing key is not a load error; if the endpoint needs one, it answers 401 or 403 and `magi-rs decide` reports `missing_key`.

Credentials in the URL itself work as in the other sections: write `[user]` and `[password]` placeholders in `base_url` and store the values with `magi-rs vault set DECISION_BASE_URL_USER` and `magi-rs vault set DECISION_BASE_URL_PASSWORD`. A literal credential in the URL is a load error. If one of the two entries is missing, `magi-rs decide` stops before calling any model and names the entry.

Keys never go in `magi.toml`.

## Input

`magi-rs decide` reads one JSON request from `-i FILE` or, without it, from standard input:

```json
{
  "state": {"ticket": "I was charged twice this month. Please refund the extra payment."},
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this ticket?",
      "criteria": {
        "billing": "Payments, invoices and refunds",
        "technical": "Bugs, errors and integrations",
        "other": "Anything else"
      }
    },
    "refund": {"type": "noul", "instructions": "Does the customer explicitly ask for a refund?"},
    "urgency": {"type": "score", "instructions": "How urgent is this ticket?", "criteria": ["Routine", "Soon", "Urgent"]}
  }
}
```

`state` may be a string, an object or an array, and it is forwarded as it is. An optional `"model"` replaces the primary model for that run; only that model is tried, with no fallbacks. Fields the protocol does not define are rejected. The `noul` question carries `instructions` in every example here; the code accepts it without them, but the provider documentation is not consistent about whether it is required, so include it.

To reuse the same questions with a changing state, put just the question map in a file and pass it with `-q`; the request then carries only `state` and, optionally, `model`. Sending `questions` in both places is an error.

## Limits

Ollama enforces its own limits, and `magi-rs` passes its rejection back to you:

- 1 to 64 questions per request;
- 2 to 26 options per `choice`;
- a request body of at most 64 KiB.

A rejected request (HTTP 400, 413 or 422) is reported as `invalid_input` and the chain does not move on, because every model on the endpoint would reject it the same way. Locally, `magi-rs` refuses a request with no questions, a `choice` or `score` without `criteria`, or fewer than 2 options or levels.

## Timeouts

`timeout_secs` applies to each attempt, so with three models in the chain a dead endpoint can cost three timeouts. The default of 120 seconds is generous on purpose: a cold local load takes from a few seconds to well over a minute depending on the model and how busy the machine is. `--timeout SECONDS` replaces it for one run. Set `keep_alive` if you have a machine dedicated to these models and want them to stay loaded.

## Confidence and thresholds

Every answer carries two numbers:

- `confidence` is computed by `magi-rs` from the returned probabilities, with one formula for all three types: `(n * p_max - 1) / (n - 1)`, where `n` is the number of options (or levels) and `p_max` the largest probability, clamped to the range 0 to 1. For `noul`, `n` is 2 and the formula reduces to `|2 * noul - 1|`. 0 means the model is indifferent; 1 means it is certain.
- `provider_confidence` is whatever the provider reported, renamed. Each model computes it differently, so it is informational only. `noul` answers have none.

Only `confidence` decides anything. Probabilities from different models are not comparable, so a threshold is always set per model with `min_confidence`.

With a threshold set, a model whose answer falls below it is skipped and the next model in the chain is tried. When there are several questions, every one of them has to reach the threshold, `score` included; otherwise the whole attempt counts as `low_confidence`. A run with no `min_confidence` always takes the top option.

Some models return slightly different numbers when the same request is repeated. A threshold set very close to a model's usual value can therefore accept in one call and rotate in the next, so leave a margin.

The values in the scaffold are commented out on purpose. They were measured on the maintainer's own prompts, which may not look like yours. Measure on your data before turning one on.

## Output

The answer is one JSON object, on standard output or in the `-o` file. Field by field:

| Field | Meaning |
|---|---|
| `schema_version` | Layout version, currently `1`; always the first key. |
| `run_id` | Identifier of the run. |
| `model` | The model that answered; `null` when none did. |
| `endpoint` | The `base_url` of `[decision]`, with any credentials masked. |
| `answers` | One entry per question; present only when a decision was made. |
| `chain` | Every attempt in order; see below. |
| `error` | `kind` and `message`; present only on failure. |
| `usage` | `input_tokens`, `output_tokens` and `cost` summed over all attempts. `cost` is `null` when no provider reported one. |
| `timings` | `total_ms` for the whole run. |

Each entry in `chain` has the `model` and an `outcome`:

- `answered`;
- `failed`, with a `cause` and sometimes a `detail`;
- `low_confidence`, with the attempt's `confidence`, the `min_confidence` it missed and the rejected `answers`.

Rejected answers appear only inside `chain`, never in the top-level `answers`, so a script that reads `answers` and the exit code cannot mistake a discarded answer for a decision. The `state` is never repeated in the output.

`score` is in level units: with three levels it runs from 0 to 2, and it equals the sum of each level index times its probability. Its `probabilities` are keyed by level index (`"0"`, `"1"`, ...) and `legend` maps each index to its name.

### `error.kind`

| `kind` | When |
|---|---|
| `invalid_input` | The request is invalid, or the provider rejected it (400, 413, 422). Change the request. |
| `missing_key` | The endpoint answered 401 or 403, or a vault entry for the URL credentials is missing. |
| `model_unavailable` | The model you named with `"model"` failed or missed its threshold. |
| `decision_unavailable` | The whole chain failed, or the client could not be built. |

### `cause` of a failed attempt

| `cause` | Situation | Does the chain move on? |
|---|---|---|
| `unauthorized` | HTTP 401 or 403 | no |
| `rejected` | HTTP 400, 413 or 422 | no |
| `model_not_found` | HTTP 404, model does not exist | yes |
| `account_policy` | HTTP 404, refused by the account's policy | yes |
| `not_found` | HTTP 404, anything else | yes |
| `rate_limited` | HTTP 429 | yes |
| `http_5xx` | HTTP 5xx | yes |
| `redirected` | HTTP 3xx (redirects are not followed) | yes |
| `transport` | connection refused, DNS, TLS | yes |
| `timeout` | `timeout_secs` ran out | yes |
| `invalid_response` | the response did not validate | yes |

A 404 that is not recognizably about the model or the account, or an answer that is not JSON at all (OpenRouter answers a path without its API prefix that way), usually means the URL is wrong. The attempt's `detail` names the exact URL that was tried and the two likely causes: a `base_url` without the provider's API prefix (for OpenRouter that is `https://openrouter.ai/api/v1`), or a server that does not implement System One (Ollama before 0.35).

### Exit codes

| Code | Meaning |
|---|---|
| `0` | a decision was made |
| `1` | no decision: `model_unavailable` or `decision_unavailable` |
| `2` | caller error: `invalid_input` or `missing_key` |

Failures before any model is called (an old `magi.toml`, a wrong passphrase, an unreadable input file) behave as they do for `magi query` and `magi consult`.

## What leaves your machine

`magi-rs decide` sends the whole `state` you give it, the questions and the model name to the endpoint in `[decision]`. With the default local Ollama nothing leaves the machine. With a hosted provider, that data goes to it, in the same way that prompts go to your main model's provider. The `state` is not written to the log or repeated in the output.

## Costs

Ollama is free. OpenRouter charges for input tokens only, at roughly 0.00000002 to 0.00000005 USD per token for the models listed above, and output is free; `usage.cost` in the output reports what it charged.
