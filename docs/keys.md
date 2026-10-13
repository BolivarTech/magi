# API keys

magi-rs can talk to four endpoints: the main model, the MAGI trio, the embedder and the decision models. Each may live at its own address, so each has its own key, and a key goes only where you mean it to go.

| Section | Key | Used for |
|---|---|---|
| main model | `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, depending on `provider` | the chat and the agent |
| `[magi]` | `MAGI_API_KEY` | the three seats of the MAGI trio |
| `[embedding]` | `EMBEDDING_API_KEY` | the memory embedder |
| `[decision]` | `DECISION_API_KEY` | the decision models behind `decide` |

Every key is read from the environment first and from the vault second. A blank value counts as absent. Set one in the vault with:

```bash
magi-rs vault set MAGI_API_KEY
magi-rs vault set EMBEDDING_API_KEY
magi-rs vault set DECISION_API_KEY
```

Keys are read once, at startup, after the vault is opened. A `vault set` made from another process shows up on the next start. Keys never go in `magi.toml`.

## Which key a section uses

For the trio (when `[magi].kind` is `openai-compat`), the embedder and the decision models, the choice is made in this order:

1. The section's own key (`MAGI_API_KEY`, `EMBEDDING_API_KEY`, `DECISION_API_KEY`).
2. `OPENAI_API_KEY`, but only if the section's endpoint has the same origin as the main endpoint.
3. Otherwise no key is sent and the request carries no `Authorization` header.

The main endpoint here is the root one: `OPENAI_BASE_URL` if set, else the root `base_url`, else the built-in default. That holds whatever your main provider is, Anthropic included. A section with no `base_url` of its own inherits the root one, so it always has the same origin and shares the key.

Two addresses have the same origin when scheme, host and port match. The port may be implicit (`https` means 443). Capitalization and a trailing dot on the host do not matter. The path does not count. These are different origins: `http` against `https`, `localhost` against `127.0.0.1`, the same host on another port, a host that only shares a prefix or suffix with the other, and any URL that does not parse. In every one of those cases `OPENAI_API_KEY` stays home.

Because the path is ignored, a gateway that routes several services by path is a single origin, and every section pointed at it receives the main key. If you do not want that, declare the section's own key.

Sending nothing is deliberate: a second Ollama on your network that needs no authentication keeps working. An endpoint that does want a key answers 401 or 403, and the message tells you which variable to set.

### The trio on Anthropic

With `[magi].kind = "anthropic"` the seats always talk to Anthropic's API, so there is no other host to leak to and no origin comparison. They use `MAGI_API_KEY` if you set it and `ANTHROPIC_API_KEY` otherwise. Setting `MAGI_API_KEY` lets the trio bill to a different account or quota than the main model. If neither key exists, the seats cannot be built and the error names both variables.

With `kind = "ollama"` the seats carry no key at all.

`MAGI_API_KEY` has to be a key for the provider of `[magi].kind`. The variable carries whatever you put in it, so if you change `kind` and leave an old key behind, that key is sent to the new provider, which will reject it. A `/login` always rebuilds the trio on Anthropic. If the trio already ran on Anthropic, a `MAGI_API_KEY` that is set still wins over the key the login just stored; otherwise the rebuilt trio uses the login key, so a key meant for another provider is never sent to Anthropic.

### A warning about the shape of the key

At startup, if `MAGI_API_KEY` is set and its shape contradicts `[magi].kind`, you get one warning that names `MAGI_API_KEY` and the kind, never the value. It is only a warning and blocks nothing. The only shape magi-rs recognizes is Anthropic's prefix, `sk-ant-`: it warns when an `anthropic` trio has a key without that prefix, and when any other kind has a key with it.

That is the whole check, and it cannot tell two OpenAI-compatible providers apart. A key that starts `sk-or-` on an `openai-compat` trio gives no warning, whether it is for OpenRouter or not, while a plain `sk-...` key on an `anthropic` trio does warn, because it lacks `sk-ant-`. A `/login` repeats the warning for an `anthropic` trio whose own `MAGI_API_KEY` lacks the prefix, since the rebuilt trio keeps using that key.

## When a key is withheld

If `OPENAI_API_KEY` exists, a section has no key of its own and its origin differs from the main one, magi-rs does not send the key. It writes one entry per section to the run log (`.magi/logs/`, level `INFO`) and shows nothing on screen:

```
OPENAI_API_KEY was not sent to http://127.0.0.1:11434: it is not the main endpoint's origin. If that endpoint needs a key, run: magi-rs vault set MAGI_API_KEY
```

The entry names the variable to declare and the origin, with any credentials in the URL hidden. It is quiet on purpose, since an endpoint that needs no key is a valid setup.

If the endpoint does need one, the failure itself shows up:

- Trio. A consult reports `no API key was sent to <origin>; declare MAGI_API_KEY with: magi-rs vault set MAGI_API_KEY`. If a key was sent and refused, the message says so instead: it names `MAGI_API_KEY` and reminds you that the key must be for the provider of `[magi].kind`, or, for an inherited key, it names the variable that was rejected. When all three seats fail, only the first case is reported (no key was sent); a refused key then shows the provider's own error.
- Embedder. The memory status line reads `memory: retrieval failing — the embedding endpoint asked for a key and none was sent; set one with: magi-rs vault set EMBEDDING_API_KEY`. A rejected key of its own, or the rejected inherited `OPENAI_API_KEY`, gets its own wording and points at the same command. Past context stops being used while it fails.
- Decision models. `magi-rs decide` reports `missing_key`.

## Upgrading to 0.24.0

Before 0.24.0 the trio and the embedder always received `OPENAI_API_KEY`, even when their `base_url` pointed somewhere else. Now they receive it only on the main endpoint's origin. Configurations that relied on the old behavior stop sending the key, so a 401 or 403 from the trio or the embedder after upgrading means one of these three cases.

| Case | What it looks like | What to run |
|---|---|---|
| Same machine, different spelling | the main endpoint is `http://localhost:11434` and `[magi]` or `[embedding]` uses `http://127.0.0.1:11434` | `magi-rs vault set MAGI_API_KEY` or `magi-rs vault set EMBEDDING_API_KEY`, or write the same spelling in both places |
| Same host, different port | a proxy next to the daemon, such as a gateway on another port | `magi-rs vault set MAGI_API_KEY` and/or `magi-rs vault set EMBEDDING_API_KEY` |
| Anthropic main model, trio or embedder elsewhere | `provider = "anthropic"`, with an OpenAI-compatible `[magi]` or `[embedding]` on an origin other than the root `base_url` | the same commands; a trio or embedder on the root origin keeps working and needs nothing |

To recognize them, read the message of the 401 or 403 (it names the variable) or look for the `OPENAI_API_KEY was not sent to ...` entry in the day's log. Each entry gives the exact command. Setting the new variable to the same value you used for `OPENAI_API_KEY` restores the old behavior for that section.

A seat of an OpenAI-compatible trio with no key no longer fails at startup; it is built without an `Authorization` header. An existing `magi.toml` needs no change: keys never lived in it.
