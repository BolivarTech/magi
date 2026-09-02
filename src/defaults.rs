// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-06-09

//! Built-in default backend profile (Ollama-first). MANUAL MAINTENANCE: these
//! `:cloud` tags reflect the Ollama catalog at release time and rot as it changes
//! (e.g. `qwen3-max` never existed; `qwen3.6` appeared). Refresh per release; users
//! override via `magi.toml`/env. All default literals live HERE, in one place.

/// Default provider when no `magi.toml`/env is present (RF-1, REQ-A01b).
///
/// **`"ollama"` — the REQ-A01b vocabulary value, not the retired legacy `"openai"`
/// label.** Task 4.1 flips this (it was `"openai"` through v0.11.0-era code, feeding the
/// now-retired `resolve_provider`/`legacy_backend_label` shim and `main.rs`'s
/// `provider_kind == "openai"` string chain): with that chain migrated onto
/// `ProviderKind` in the same task, there is nothing left to normalize onto, so this
/// constant can finally name the REAL default instead of a translation of it.
///
/// Also the single source of truth [`render_default_magi_toml`] emits for `provider =`
/// and the value `MagiConfig`'s "provider is blank" startup notice interpolates —
/// collapsed from the formerly separate `RENDERED_DEFAULT_PROVIDER` constant (Task 4.1),
/// which existed ONLY because this constant used to disagree with it (B3: two constants
/// holding the same string is exactly the kind of accidental-coincidence duplication
/// REQ-A21b already flagged once, for `emb_base_url()`/`DEFAULT_OPENAI_BASE_URL`).
pub const DEFAULT_PROVIDER: &str = "ollama";
/// Default OpenAI-compatible base URL — local Ollama (RF-2).
pub const DEFAULT_OPENAI_BASE_URL: &str = "http://localhost:11434/v1";
/// Default principal model on the openai path (RF-3).
pub const DEFAULT_OPENAI_MODEL: &str = "kimi-k2.6:cloud";
/// Default MAGI trio (openai path only, RF-4). Lineages: Alibaba / OpenAI / DeepSeek.
pub const DEFAULT_MAGI_MELCHIOR: &str = "qwen3.5:397b-cloud";
pub const DEFAULT_MAGI_BALTHASAR: &str = "gpt-oss:120b-cloud";
pub const DEFAULT_MAGI_CASPAR: &str = "deepseek-v4-pro:cloud";
/// Lineage of [`DEFAULT_MAGI_MELCHIOR`] — the independent failure domain its model belongs to.
///
/// Read off the model tag by hand, once: `qwen3.5` is Alibaba's family. It is **declared**, not
/// inferred at runtime (R-R03): a label the project writes down for the models the project ships is
/// a decision, whereas guessing one from an arbitrary user tag would fabricate the value that
/// decides all rotation eligibility.
///
/// It is also the label the guided migration error offers as an example, so what `magi init` writes
/// and what the error suggests cannot drift apart.
pub const DEFAULT_MAGI_MELCHIOR_LINEAGE: &str = "alibaba";
/// Lineage of [`DEFAULT_MAGI_BALTHASAR`] — see [`DEFAULT_MAGI_MELCHIOR_LINEAGE`].
pub const DEFAULT_MAGI_BALTHASAR_LINEAGE: &str = "openai";
/// Lineage of [`DEFAULT_MAGI_CASPAR`] — see [`DEFAULT_MAGI_MELCHIOR_LINEAGE`].
pub const DEFAULT_MAGI_CASPAR_LINEAGE: &str = "deepseek";
/// Default Anthropic model on the opt-in path (RF-5). Was `main.rs::DEFAULT_MODEL`.
pub const DEFAULT_ANTHROPIC_MODEL: &str = "claude-sonnet-4-6";
/// Default embedding model (Ollama-first, local). Single source of truth — also
/// re-exported by `memory::config::d::emb_model` so both resolve identically.
pub const DEFAULT_EMBEDDING_MODEL: &str = "nomic-embed-text-v2-moe:latest";

// ── Lineage rotation (MS3) ────────────────────────────────────────────────────

// These three were `cfg(test)` while nothing in production read them — the honest way to say "no
// caller yet", as opposed to an `#[allow(dead_code)]` or a fabricated caller, which are both lies
// the linter cannot catch. The scaffold below is that first production caller, so the gate is gone.

/// Fallback models a mage may rotate through before its run degrades (REQ-R05).
///
/// **2 — the same value magi-core ships, deliberately not a default of our own** (D-R14). A
/// divergent default would have to be explained in the CHANGELOG and defended on every upgrade.
///
/// The cost is the worst case it implies: with the 90 s ceiling and retry enabled the derived
/// headless `--timeout` becomes `2 attempts x 3 models x 90 s` plus slack, i.e. ~654 s. That is
/// paid **only when something is genuinely hung** — a healthy consult never approaches it — and the
/// escape valve already shipped: `--timeout 300` with a notice naming the computed minimum.
///
/// **`0` is the kill-switch** and must stay reachable: it restores v0.12.0 behaviour exactly.
pub const DEFAULT_MAX_ROTATIONS: u32 = 2;

/// Whether a candidate whose context window could not be measured is refused (REQ-R11).
///
/// **`false`, including on Ollama.** The case the guard bites is the **cold start**: a daemon that
/// has not warmed up answers no probe inside the 5 s ceiling, so with the guard on, the first run
/// anyone makes would find no eligible candidate. That is the worst possible first impression and
/// it is transitory.
pub const DEFAULT_STRICT_CONTEXT_GUARD: bool = false;

/// Whether distinct lineages are required rather than merely encouraged (REQ-R29).
///
/// **`true`.** Lineage diversity is what makes rotation worth anything, so the system demands it and
/// whoever genuinely cannot meet it opts out in one line. It is the correct failure direction: a
/// pool without diversity that starts silently is a safety net the operator believes they have and
/// do not — and they find out when a model falls over, which is the worst moment to find out.
///
/// This key is **exclusive to magi-rs**: magi-core treats the lineage as an opaque string and is
/// agnostic to it, so the value is never passed down.
pub const DEFAULT_ENFORCE_DIVERSITY: bool = true;

/// Fallback pool that `magi init` scaffolds, ordered strongest to weakest (REQ-R27).
///
/// **Every label here is one NO SEAT HAS**, and that is the point rather than a stylistic choice.
/// An entry whose lineage matches a seat covers **only that seat**; an entry with a foreign label
/// covers **all three**. Three foreign labels therefore buy three-way coverage for the same three
/// declared lines that a matching-label pool would spend one per seat.
///
/// The scaffold ships this pool **ACTIVE, not commented**: a commented pool next to a live
/// `max_rotations` tells the operator they have a safety net while they silently fall back to
/// no-rotation behaviour — the same shape of defect as a setting that is declared and not applied.
///
/// **Five entries, matching the trio configuration this project runs alongside**
/// (`magi-ollama.toml`), so an operator moving between the two finds the same depth rather than
/// a shorter list here for no stated reason.
///
/// **What could NOT be carried over, and why it matters more than the count.** Two of that
/// file's five are `deepseek-v4-pro`/`deepseek` and `gpt-oss`/`openai`. Those are foreign to
/// ITS trio (qwen / kimi / glm) and are therefore correct there — but they are two of THIS
/// trio's three seats, model and lineage both. Copying them verbatim would have cut each one's
/// coverage from three seats to one, by the rule stated above, and tripped the duplicate-model
/// notice twice over. The two substitutes are the remaining labels from that same file that no
/// seat here holds, so the pool grows in depth without losing the property that makes depth
/// worth anything.
pub const DEFAULT_SCAFFOLD_POOL: [(&str, &str); 5] = [
    ("glm-5.2:cloud", "zhipu"),
    ("kimi-k2.6:cloud", "moonshot"),
    ("minimax-m3:cloud", "minimax"),
    ("nemotron-3-super:cloud", "nvidia"),
    ("gemma4:cloud", "google"),
];

// ── Headless mode constants ───────────────────────────────────────────────────
//
// The headless numeric caps (`MAX_INPUT_BYTES`, `MAX_JSON_DEPTH`, …) live in the
// lib module `magi_rs::headless::limits` — lib-visible so the `headless` lib
// modules can use them directly, which the bin-only `defaults` module cannot
// provide across the crate split. Reference them via `magi_rs::headless::limits`
// (bin) or `crate::limits` (within `headless`). Overridable via the `[headless]`
// section of `magi.toml`. Origin per the spec, §11.

/// Startup notice shown when no `magi.toml` is present (RF-9). Built by
/// interpolating the default constants (RF-8 DRY) so it tracks any constant edit.
pub fn no_config_notice() -> String {
    format!(
        "No magi.toml — using Ollama defaults ({base}, {model}, \
         Melchior: {mel}, Balthasar: {bal}, Caspar: {cas}). Copy \
         docs/magi.toml.example to customize; `provider` accepts {vocab}.",
        base = DEFAULT_OPENAI_BASE_URL,
        model = DEFAULT_OPENAI_MODEL,
        mel = DEFAULT_MAGI_MELCHIOR,
        bal = DEFAULT_MAGI_BALTHASAR,
        cas = DEFAULT_MAGI_CASPAR,
        // Derived from the vocabulary, never a hand-written fourth copy. This notice reaches
        // the operator who has NOT run `magi init` — precisely the one who never sees the
        // scaffold's improved comment — so naming only `anthropic` here left the openai-compat
        // user, pointing at Groq or OpenRouter, with no surface at all to learn the value
        // from, while a wrong guess is a fatal parse error rather than a warning.
        vocab = magi_rs::magi::kind::ProviderKind::VOCABULARY
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" | "),
    )
}

/// Renders a **complete, self-contained** `magi.toml` from code — every field of
/// [`crate::memory::config::MemoryConfig`] and [`crate::memory::config::EmbeddingConfig`]
/// appears, either as an active essential or as a commented knob showing its default
/// value. Values are derived from the structs' `Default` impls (single source of truth;
/// no hand-copied literals). The binary is self-contained: this function does **not**
/// `include_str!` any external file.
///
/// # Layout
/// - **Active**: `provider`, `[openai]`, `[anthropic]`, `[magi]`, the five `[memory]`
///   essentials (`mode`, `context_budget_tokens`, `distill_enabled`,
///   `evicted_retention_days`, `max_records`), and the `[embedding]` essentials
///   (`model`, `dim` — `base_url` is deliberately NOT among them).
/// - **Commented**: `[embedding].base_url`, whose whole purpose is to be ABSENT so the embedder
///   inherits the root endpoint (REQ-A21); emitting it active pins a value and silently ends
///   that inheritance. Plus all remaining `[memory]`/`[embedding]` fields with inline
///   documentation, ready to uncomment.
pub fn render_default_magi_toml() -> String {
    use crate::memory::config::{EmbeddingConfig, MemoryConfig};
    use std::fmt::Write as FmtWrite;

    let mem = MemoryConfig::default();
    let emb = EmbeddingConfig::default();

    // Format salience_markers as a TOML inline array: ["a", "b", ...]
    let markers: String = mem
        .salience_markers
        .iter()
        .map(|s| format!("\"{}\"", s))
        .collect::<Vec<_>>()
        .join(", ");

    let mut out = String::with_capacity(4096);

    // ── Header ──────────────────────────────────────────────────────────────
    // REQ-A22 (fix round 1, coordinator, 2026-08-02): `--init-config` and
    // `/init-config` are both retired; `magi init` is the only scaffolder that still
    // calls this function. A header naming two retired commands as the source would
    // be exactly the kind of stale reference this round exists to remove.
    writeln!(
        out,
        "# Generated by magi init — built-in Ollama-first defaults."
    )
    .unwrap();
    // The accepted values are named HERE, in the comment block, rather than only as a trailing
    // comment on the key: a value parked at the end of a long line is the part a reader skips.
    //
    // The previous text mentioned `anthropic` alone, which reads as a binary choice and hides
    // `openai-compat` entirely — so an operator pointing at Groq, OpenRouter or any other
    // OpenAI-shaped endpoint had no way to learn the value they needed from the very file
    // `magi init` handed them. `deny_unknown_fields` makes a wrong guess a FATAL parse error
    // rather than a warning, so not knowing costs a process that refuses to start.
    writeln!(
        out,
        "# Edit to customize. `provider` accepts exactly three values:"
    )
    .unwrap();
    writeln!(
        out,
        "#   ollama         local or remote Ollama daemon (this file's default)"
    )
    .unwrap();
    writeln!(
        out,
        "#   openai-compat  any OpenAI-shaped endpoint — OpenAI, Groq, OpenRouter, vLLM, …"
    )
    .unwrap();
    writeln!(
        out,
        "#   anthropic      the Anthropic Messages API (key from the env or the vault)"
    )
    .unwrap();
    writeln!(
        out,
        "# Anything else is a parse error, and a magi.toml that fails to parse is fatal."
    )
    .unwrap();
    // `DEFAULT_PROVIDER` is `"ollama"` (Task 4.1 flipped it) — the REQ-A01b vocabulary
    // value `from_toml_str`'s vocabulary validation actually accepts.
    writeln!(out, "provider = \"{}\"", DEFAULT_PROVIDER).unwrap();
    // `base_url` moved to the ROOT in Task 1.1 (REQ-A21) — it used to live under
    // `[openai]`, which no longer accepts it (`deny_unknown_fields`).
    writeln!(out, "base_url = \"{}\"", DEFAULT_OPENAI_BASE_URL).unwrap();
    writeln!(out).unwrap();

    // ── [openai] ─────────────────────────────────────────────────────────────
    writeln!(out, "[openai]").unwrap();
    writeln!(out, "model    = \"{}\"", DEFAULT_OPENAI_MODEL).unwrap();
    writeln!(out).unwrap();

    // ── [anthropic] (opt-in; model = none means built-in default applies) ────
    writeln!(out, "[anthropic]").unwrap();
    writeln!(out, "model = \"{}\"", DEFAULT_ANTHROPIC_MODEL).unwrap();
    writeln!(out).unwrap();

    // ── [magi] ───────────────────────────────────────────────────────────────
    writeln!(out, "[magi]").unwrap();
    writeln!(out, "melchior_model  = \"{}\"", DEFAULT_MAGI_MELCHIOR).unwrap();
    writeln!(out, "balthasar_model = \"{}\"", DEFAULT_MAGI_BALTHASAR).unwrap();
    writeln!(out, "caspar_model    = \"{}\"", DEFAULT_MAGI_CASPAR).unwrap();
    // A seat that declares a model must declare its lineage (REQ-R02/R22): the scaffold has to
    // satisfy the same rule it teaches, or `magi init` would write a file that does not start.
    writeln!(
        out,
        "# The independent failure domain of each model. Declared, never inferred: rotation only \
         accepts a candidate from a lineage no OTHER seat is holding."
    )
    .unwrap();
    writeln!(
        out,
        "melchior_lineage  = \"{}\"",
        DEFAULT_MAGI_MELCHIOR_LINEAGE
    )
    .unwrap();
    writeln!(
        out,
        "balthasar_lineage = \"{}\"",
        DEFAULT_MAGI_BALTHASAR_LINEAGE
    )
    .unwrap();
    writeln!(
        out,
        "caspar_lineage    = \"{}\"",
        DEFAULT_MAGI_CASPAR_LINEAGE
    )
    .unwrap();
    writeln!(
        out,
        "auto_approve    = false \
         # true = launch MAGI consult automatically (announces in TUI); \
         false = ask before each autonomous launch (default). \
         The explicit /consult TUI command is always user-initiated and never gated."
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "# Rotation. 0 disables it and restores the pre-rotation behaviour exactly."
    )
    .unwrap();
    writeln!(out, "max_rotations   = {DEFAULT_MAX_ROTATIONS}").unwrap();
    writeln!(
        out,
        "# enforce_diversity = {DEFAULT_ENFORCE_DIVERSITY}  \
         # require three distinct lineages; set false if every model you have shares one"
    )
    .unwrap();
    writeln!(
        out,
        "# strict_context_guard = {DEFAULT_STRICT_CONTEXT_GUARD}  \
         # refuse candidates whose context window could not be measured"
    )
    .unwrap();
    writeln!(out).unwrap();

    // ── [memory] active essentials ────────────────────────────────────────────
    writeln!(
        out,
        "# ---------------------------------------------------------------------------"
    )
    .unwrap();
    writeln!(
        out,
        "# [memory] — tiered memory subsystem (absent = built-in defaults apply)."
    )
    .unwrap();
    writeln!(
        out,
        "# Default mode is \"selective\" (embedding-indexed, bounded context)."
    )
    .unwrap();
    writeln!(
        out,
        "# Set mode = \"load_all\" to reproduce v0.6.0 behavior (benchmark control)."
    )
    .unwrap();
    writeln!(out, "[memory]").unwrap();
    writeln!(
        out,
        "mode = \"{}\"                  # selective | load_all  (load_all = v0.6.0 control)",
        mem.mode
    )
    .unwrap();
    writeln!(
        out,
        "context_budget_tokens = {}        # assembled-context token budget",
        mem.context_budget_tokens
    )
    .unwrap();
    writeln!(
        out,
        "distill_enabled = {}              # false = zero LLM egress for distillation",
        mem.distill_enabled
    )
    .unwrap();
    writeln!(
        out,
        "# Retention: -1 = archive forever (never hard-delete), 0 = hard-delete on eviction,"
    )
    .unwrap();
    writeln!(
        out,
        "# N>0 = hard-delete N days after eviction. For truly unlimited storage set BOTH"
    )
    .unwrap();
    writeln!(
        out,
        "# evicted_retention_days = -1 AND max_records = 0 (max_records still prunes the"
    )
    .unwrap();
    writeln!(
        out,
        "# weakest active records when the count exceeds the cap, even at -1)."
    )
    .unwrap();
    writeln!(out, "evicted_retention_days = {}         # -1 archive forever | 0 hard-delete now | N>0 delete after N days", mem.evicted_retention_days).unwrap();
    writeln!(out, "max_records = {}                 # hard ceiling on active records; 0 = unlimited (opt-out the cap)", mem.max_records).unwrap();

    // ── [memory] advanced knobs — commented, default value shown ─────────────
    writeln!(
        out,
        "# --- Advanced [memory] knobs (default values shown; uncomment to override) ---"
    )
    .unwrap();
    writeln!(
        out,
        "# response_headroom_tokens = {}    # tokens reserved for the model reply",
        mem.response_headroom_tokens
    )
    .unwrap();
    writeln!(
        out,
        "# safety_margin_ratio = {}           # fraction of budget held back (heuristic guard)",
        toml_f64(mem.safety_margin_ratio)
    )
    .unwrap();
    writeln!(
        out,
        "# chars_per_token = {}               # token heuristic; es ~3.5, code ~3.0, CJK ~2.0",
        toml_f64(mem.chars_per_token)
    )
    .unwrap();
    writeln!(
        out,
        "# oversized_turn_policy = \"{}\"  # truncate | error",
        mem.oversized_turn_policy
    )
    .unwrap();
    writeln!(
        out,
        "# top_k = {}                          # retrieval candidate count",
        mem.top_k
    )
    .unwrap();
    writeln!(
        out,
        "# weight_similarity = {}              # reranker weight on cosine similarity",
        toml_f64(mem.weight_similarity)
    )
    .unwrap();
    writeln!(
        out,
        "# weight_recency = {}                 # reranker weight on recency",
        toml_f64(mem.weight_recency)
    )
    .unwrap();
    writeln!(
        out,
        "# weight_salience = {}              # reranker weight on salience",
        toml_f64(mem.weight_salience)
    )
    .unwrap();
    writeln!(
        out,
        "# default_salience = {}               # base salience assigned at write",
        toml_f64(mem.default_salience)
    )
    .unwrap();
    writeln!(
        out,
        "# preference_salience = {}            # protected floor for kind=preference",
        toml_f64(mem.preference_salience)
    )
    .unwrap();
    writeln!(
        out,
        "# protect_salience_threshold = {}    # salience at/above which a memory is never evicted",
        toml_f64(mem.protect_salience_threshold)
    )
    .unwrap();
    writeln!(
        out,
        "# decay_half_life_days = {}         # wall-clock recency half-life",
        toml_f64(mem.decay_half_life_days)
    )
    .unwrap();
    writeln!(
        out,
        "# access_saturation_cap = {}         # cap on access-reinforcement contribution",
        mem.access_saturation_cap
    )
    .unwrap();
    writeln!(
        out,
        "# forget_strength_threshold = {}     # strength below which a memory is forgettable",
        toml_f64(mem.forget_strength_threshold)
    )
    .unwrap();
    writeln!(
        out,
        "# supersede_similarity_threshold = {}  # same-subject hard-supersession similarity gate",
        toml_f64(mem.supersede_similarity_threshold)
    )
    .unwrap();
    writeln!(
        out,
        "# distill_every_n_turns = {}        # 0 = on-demand/session-close only",
        mem.distill_every_n_turns
    )
    .unwrap();
    writeln!(
        out,
        "# distill_on_session_close = {}     # run distiller on session close",
        mem.distill_on_session_close
    )
    .unwrap();
    writeln!(
        out,
        "# profile_max_tokens = {}            # always-injected preference profile token bound",
        mem.profile_max_tokens
    )
    .unwrap();
    writeln!(
        out,
        "# seed = {}                         # determinism for retrieval/decay/benchmark",
        mem.seed
    )
    .unwrap();
    writeln!(
        out,
        "# salience_markers = [{}]        # substrings that lift salience at write",
        markers
    )
    .unwrap();
    writeln!(out, "# index = \"{}\"                    # exact (default, deterministic) | ann (build feature)", mem.index).unwrap();
    writeln!(
        out,
        "# distill_max_batch_tokens = {}     # per-run LLM payload cap (privacy)",
        mem.distill_max_batch_tokens
    )
    .unwrap();
    writeln!(
        out,
        "# supersede_max_candidate_pairs = {}  # distiller hard-supersession pair cap per run",
        mem.supersede_max_candidate_pairs
    )
    .unwrap();
    writeln!(
        out,
        "# reembed_batch_size = {}            # lazy re-embed throttle per pass",
        mem.reembed_batch_size
    )
    .unwrap();
    writeln!(
        out,
        "# max_evictions_per_pass = {}       # clock-jump guard on evictions per pass",
        mem.max_evictions_per_pass
    )
    .unwrap();
    writeln!(
        out,
        "# migration_throttle_batch = {}      # lazy migration batch size",
        mem.migration_throttle_batch
    )
    .unwrap();
    writeln!(out).unwrap();

    // ── [embedding] active essentials ────────────────────────────────────────
    writeln!(
        out,
        "# ---------------------------------------------------------------------------"
    )
    .unwrap();
    writeln!(
        out,
        "# [embedding] — embedding provider (OpenAI-compatible, Ollama-first)."
    )
    .unwrap();
    writeln!(out, "# Set model to your installed embedding model.").unwrap();
    writeln!(out, "# Run: ollama pull {}", emb.model).unwrap();
    writeln!(
        out,
        "# The embedding API key is read from OPENAI_API_KEY env only (never in this file)."
    )
    .unwrap();
    writeln!(out, "[embedding]").unwrap();
    // `emb.base_url` is `Option<String>` and `EmbeddingConfig::default()` leaves it
    // `None` (REQ-A21, so it can inherit the root `base_url`) — the display text for
    // the generated file's active essential line comes from
    // `memory::config::d::emb_base_url()`, which wraps the same
    // `DEFAULT_OPENAI_BASE_URL` constant instead of a duplicated literal (B3).
    // COMMENTED, and that is the whole point of the line. `EmbeddingConfig::default()`
    // leaves `base_url` at `None` precisely so the embedder inherits the root endpoint
    // (REQ-A21); emitting it active pinned a value instead, and the two statements —
    // this comment and the line under it — used to contradict each other.
    //
    // The operator-visible cost of getting it wrong: run `magi init`, then point the root
    // `base_url` at a remote host. The agent and the trio move, the embedder silently keeps
    // talking to localhost, and nothing in the file looks wrong, because what it pins is a
    // real and valid endpoint.
    writeln!(
        out,
        "# base_url = \"{}\"   # inherits the root base_url when commented",
        crate::memory::config::d::emb_base_url()
    )
    .unwrap();
    writeln!(
        out,
        "model    = \"{}\"  # CHANGE to your installed embedding model — run: ollama pull <model>",
        emb.model
    )
    .unwrap();
    writeln!(
        out,
        "dim      = {}              # 0 = autodetect the vector size from the first response",
        emb.dim
    )
    .unwrap();

    // ── [embedding] advanced knobs — commented ────────────────────────────────
    writeln!(out, "# --- Advanced [embedding] knobs ---").unwrap();
    writeln!(
        out,
        "# provider = \"{}\"             # embedding provider kind (openai-compatible)",
        emb.provider
    )
    .unwrap();
    writeln!(
        out,
        "# query_prefix = \"{}\"          # prefix applied to query text before embedding",
        emb.query_prefix
    )
    .unwrap();
    writeln!(
        out,
        "# document_prefix = \"{}\"       # prefix applied to stored text before embedding",
        emb.document_prefix
    )
    .unwrap();

    // ── The pool goes LAST IN THE FILE, and that is a TOML rule, not a style choice ──────────
    // Every loose key and every sub-table must precede the first array of tables. A rule phrased
    // as "last in the [magi] block" would let a later addition to [magi] — or a new sub-table —
    // land after the array and parse into the wrong table. "Last in the file" leaves nowhere to
    // get it wrong, which is why [embedding] above is deliberately emitted before this point.
    writeln!(out).unwrap();
    writeln!(
        out,
        "# ---------------------------------------------------------------------------"
    )
    .unwrap();
    writeln!(
        out,
        "# Rotation pool, shared by the three seats, strongest first. Every label here is one no \
         seat holds, so each entry can serve any of the three."
    )
    .unwrap();
    for (model, lineage) in DEFAULT_SCAFFOLD_POOL {
        writeln!(out).unwrap();
        writeln!(out, "[[magi.fallback]]").unwrap();
        writeln!(out, "model   = \"{model}\"").unwrap();
        writeln!(out, "lineage = \"{lineage}\"").unwrap();
    }

    out
}

/// Formats an `f64` for TOML, ensuring a decimal point is always present.
///
/// Rust's `Display` for whole-number floats omits the decimal (e.g. `1.0` →
/// `"1"`, `30.0` → `"30"`), which TOML would parse as integers. This helper
/// appends `.0` when the formatted string contains no `.` or exponent marker,
/// preserving correct TOML float semantics for generated config files.
fn toml_f64(v: f64) -> String {
    let s = format!("{}", v);
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{}.0", s)
    }
}

/// Whether to emit the no-config Ollama-defaults startup notice (RF-9): only when
/// the resolved backend speaks the `[openai]`-transport path (`Ollama` or
/// `OpenAiCompat` — REQ-A01b, they share the same `DEFAULT_OPENAI_*` constants this
/// notice interpolates) AND no `magi.toml` exists. Prevents a misleading "using Ollama
/// defaults" notice under `MAGI_PROVIDER=anthropic` with no file.
///
/// Task 4.1: took `&str` (compared against the legacy `"openai"` label) before the
/// vocabulary unification; now takes [`ProviderKind`] directly — the eighth and last of
/// the `provider_kind == "openai"` sites the migration retires.
#[must_use]
pub fn should_emit_default_notice(
    provider_kind: magi_rs::magi::kind::ProviderKind,
    magi_toml_exists: bool,
) -> bool {
    use magi_rs::magi::kind::ProviderKind;
    matches!(
        provider_kind,
        ProviderKind::Ollama | ProviderKind::OpenAiCompat
    ) && !magi_toml_exists
}

/// [`no_config_notice`] as the startup notice a surface pushes, level included.
///
/// # Returns
///
/// A notice carrying [`no_config_notice`]'s text, at the level RF-9 needs.
///
/// # Why the level lives here and not at the call site
///
/// It is a property of what the text says, and keeping it beside the text is what lets a
/// test read it. Neither surface decides it.
///
/// # Complexity
///
/// [`no_config_notice`]'s.
#[must_use]
pub fn no_config_startup_notice() -> magi_rs::notices::Notice {
    magi_rs::notices::Notice::info(no_config_notice())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_constants_are_the_ollama_first_profile() {
        assert_eq!(DEFAULT_PROVIDER, "ollama");
        assert_eq!(DEFAULT_OPENAI_BASE_URL, "http://localhost:11434/v1");
        assert_eq!(DEFAULT_OPENAI_MODEL, "kimi-k2.6:cloud");
        assert_eq!(DEFAULT_MAGI_MELCHIOR, "qwen3.5:397b-cloud");
        assert_eq!(DEFAULT_MAGI_BALTHASAR, "gpt-oss:120b-cloud");
        assert_eq!(DEFAULT_MAGI_CASPAR, "deepseek-v4-pro:cloud");
        assert_eq!(DEFAULT_ANTHROPIC_MODEL, "claude-sonnet-4-6");
    }

    #[test]
    fn test_should_emit_default_notice_only_for_openai_without_file() {
        use magi_rs::magi::kind::ProviderKind;
        // Notice is for the no-config Ollama default ONLY: [openai]-transport + no magi.toml.
        assert!(should_emit_default_notice(ProviderKind::Ollama, false));
        assert!(should_emit_default_notice(
            ProviderKind::OpenAiCompat,
            false
        ));
        assert!(!should_emit_default_notice(ProviderKind::Ollama, true)); // file present
        assert!(!should_emit_default_notice(ProviderKind::Anthropic, false)); // env opt-in, no file
        assert!(!should_emit_default_notice(ProviderKind::Anthropic, true));
    }

    #[test]
    fn test_no_config_notice_interpolates_all_defaults() {
        // S-9: the notice is built from the constants (DRY), not hardcoded strings.
        let n = no_config_notice();
        assert!(n.contains(DEFAULT_OPENAI_BASE_URL));
        assert!(n.contains(DEFAULT_OPENAI_MODEL));
        assert!(n.contains(DEFAULT_MAGI_MELCHIOR));
        assert!(n.contains(DEFAULT_MAGI_BALTHASAR));
        assert!(n.contains(DEFAULT_MAGI_CASPAR));
    }

    #[test]
    fn test_render_default_magi_toml_interpolates_and_parses() {
        // S-12: rendered from constants (DRY) and valid TOML with provider="ollama".
        // Active [memory] and [embedding] sections must parse into real config values.
        let s = render_default_magi_toml();
        assert!(s.contains(DEFAULT_OPENAI_BASE_URL));
        assert!(s.contains(DEFAULT_OPENAI_MODEL));
        assert!(s.contains(DEFAULT_MAGI_MELCHIOR));
        assert!(s.contains(DEFAULT_MAGI_BALTHASAR));
        assert!(s.contains(DEFAULT_MAGI_CASPAR));
        let parsed = crate::config::MagiConfig::from_toml_str(&s).unwrap();
        assert_eq!(parsed.provider(), Some(DEFAULT_PROVIDER));
        assert_eq!(parsed.base_url(), Some(DEFAULT_OPENAI_BASE_URL));
        assert_eq!(
            parsed.magi().melchior_model.as_deref(),
            Some(DEFAULT_MAGI_MELCHIOR)
        );
        // Active [memory] section must parse into real values (not just commented stubs).
        assert_eq!(parsed.memory().mode, "selective");
        // Active [embedding] section must carry the current default model (DRY).
        assert_eq!(parsed.embedding().model, DEFAULT_EMBEDDING_MODEL);
    }

    /// SC-NEW: `render_default_magi_toml` must include active `[memory]` and
    /// `[embedding]` sections so users can edit the embedding model (the field most
    /// likely to cause a 404) without reading separate docs. Generated TOML must
    /// parse cleanly and the active sections must surface real config values.
    #[test]
    fn test_render_default_magi_toml_includes_memory_and_embedding_sections() {
        let s = render_default_magi_toml();
        // Active section headers must be present
        assert!(
            s.contains("[memory]"),
            "generated toml must contain an active [memory] section"
        );
        assert!(
            s.contains("[embedding]"),
            "generated toml must contain an active [embedding] section"
        );
        // Default embedding model must be shown so users know which model to pull
        assert!(
            s.contains(DEFAULT_EMBEDDING_MODEL),
            "generated toml must show the default embedding model ({DEFAULT_EMBEDDING_MODEL})"
        );
        // Active sections must not break TOML parsing and must parse into real values
        let parsed = crate::config::MagiConfig::from_toml_str(&s)
            .expect("render_default_magi_toml() must produce valid TOML");
        assert_eq!(
            parsed.provider(),
            Some(DEFAULT_PROVIDER),
            "parsed provider must be the value magi_init/render_default_magi_toml emits"
        );
        assert_eq!(
            parsed.memory().mode,
            "selective",
            "active [memory] section must parse mode as 'selective'"
        );
        assert_eq!(
            parsed.embedding().model,
            DEFAULT_EMBEDDING_MODEL,
            "active [embedding] section must parse model as the current default"
        );
    }

    /// Regression guard: every public field of `MemoryConfig` and `EmbeddingConfig`
    /// must appear — active or commented — in the generated TOML so a newly-added
    /// field that is forgotten in `render_default_magi_toml` fails this test
    /// immediately. Also confirms the commented advanced lines are inert (parse
    /// succeeds without them) and that active essentials parse to their defaults.
    #[test]
    fn test_render_default_magi_toml_covers_all_field_names() {
        let s = render_default_magi_toml();

        // ── MemoryConfig fields (all 31 must appear) ─────────────────────────
        let mem_fields = [
            "mode",
            "context_budget_tokens",
            "response_headroom_tokens",
            "safety_margin_ratio",
            "chars_per_token",
            "oversized_turn_policy",
            "top_k",
            "weight_similarity",
            "weight_recency",
            "weight_salience",
            "default_salience",
            "preference_salience",
            "protect_salience_threshold",
            "decay_half_life_days",
            "access_saturation_cap",
            "forget_strength_threshold",
            "evicted_retention_days",
            "max_records",
            "supersede_similarity_threshold",
            "distill_every_n_turns",
            "distill_on_session_close",
            "profile_max_tokens",
            "seed",
            "salience_markers",
            "index",
            "distill_max_batch_tokens",
            "supersede_max_candidate_pairs",
            "distill_enabled",
            "reembed_batch_size",
            "max_evictions_per_pass",
            "migration_throttle_batch",
        ];
        for field in &mem_fields {
            assert!(
                s.contains(field),
                "render_default_magi_toml() is missing MemoryConfig field: {field}"
            );
        }

        // ── EmbeddingConfig fields (all 6 must appear) ───────────────────────
        let emb_fields = [
            "query_prefix",
            "document_prefix",
            "base_url",
            "model",
            "dim",
        ];
        for field in &emb_fields {
            assert!(
                s.contains(field),
                "render_default_magi_toml() is missing EmbeddingConfig field: {field}"
            );
        }

        // The commented advanced lines must be inert — TOML parses correctly
        let parsed = crate::config::MagiConfig::from_toml_str(&s)
            .expect("render_default_magi_toml() must produce valid TOML (commented lines inert)");
        assert_eq!(parsed.memory().mode, "selective");
        assert_eq!(parsed.embedding().model, DEFAULT_EMBEDDING_MODEL);

        // S1 Loop 2 (Caspar): the embedding model was pinned against its constant and the other
        // two were not, so the scaffold could teach a model the project no longer defaults to
        // and nothing would say so. A scaffold that drifts from `defaults.rs` is worse than no
        // scaffold: it is documentation that looks authoritative and is wrong.
        assert_eq!(
            crate::config::resolve_openai_model(&parsed, None),
            DEFAULT_OPENAI_MODEL,
            "the scaffold's [openai].model must track defaults.rs, not a stale literal"
        );
        assert!(
            s.contains(DEFAULT_ANTHROPIC_MODEL),
            "and so must its [anthropic].model"
        );
    }

    /// SC-25: `docs/magi.toml.example` must contain no actual secret material.
    ///
    /// Checks that:
    /// - No TOML field named `api_key` (with underscore) is present — keys live in env vars or
    ///   the vault only (`env > vault`, REQ-V12), never in the config file.
    /// - No `sk-` prefix (Anthropic / OpenAI raw key format) appears anywhere.
    ///
    /// The word "key" in prose (e.g. "API keys NEVER live here") is allowed.
    ///
    /// **The positive control is not ceremony** (S1 Loop 2, Balthasar). Both assertions are
    /// ABSENCE claims, and an absence claim holds trivially against nothing: truncate the example
    /// to an empty file, or replace it with unrelated content, and this test stays green while
    /// the thing it guards has evaporated. So it first establishes that the file IS the artefact
    /// under test, then that the two predicates actually discriminate — a scanner that never
    /// fires is indistinguishable from a file that is clean.
    #[test]
    fn test_config_example_has_no_secret_material() {
        let s = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/docs/magi.toml.example"
        ))
        .unwrap();

        // Control 1: this is the real example, not an empty or unrelated file.
        assert!(
            s.contains("[magi]") && s.contains("melchior_lineage"),
            "precondition: the file scanned below must be the v0.13.0 example itself"
        );

        // Control 2: the predicates fire on material that SHOULD be caught. Without this, the
        // two assertions below prove only that some string does not contain some substring.
        let planted = format!("{s}\napi_key = \"sk-not-a-real-key\"\n");
        assert!(
            planted.to_lowercase().contains("api_key") && planted.contains("sk-"),
            "the scan must detect planted secret material, or it detects nothing at all"
        );

        let low = s.to_lowercase();
        assert!(
            !low.contains("api_key"),
            "docs/magi.toml.example must not contain 'api_key'"
        );
        assert!(
            !s.contains("sk-"),
            "docs/magi.toml.example must not contain 'sk-' key prefix"
        );
    }

    /// SC-R34: `magi init` ships REAL rotation, not decorative.
    ///
    /// A commented pool next to an active `max_rotations` makes the operator believe they have a
    /// safety net while they fall back to no-rotation behaviour — the same shape of defect as a
    /// setting that is declared and never applied.
    #[test]
    fn the_scaffold_ships_an_active_pool_with_lineages_no_seat_has() {
        let scaffold = render_default_magi_toml();
        let cfg = crate::config::MagiConfig::from_toml_str(&scaffold)
            .expect("the scaffold must pass the very validation it teaches");

        let pool = cfg.fallback_pool();
        // DERIVED from the constant, never a literal. This assertion is about the pool being
        // emitted ACTIVE rather than commented — a hardcoded count made it fail for a second,
        // unrelated reason the moment the pool grew, and the message it printed then ("must be
        // ACTIVE, not commented out") described neither the change nor the real problem.
        assert_eq!(
            pool.len(),
            DEFAULT_SCAFFOLD_POOL.len(),
            "the pool must be ACTIVE, not commented out — every declared candidate has to \
             survive the round-trip through the generated file"
        );

        let seat_lineages = [
            DEFAULT_MAGI_MELCHIOR_LINEAGE,
            DEFAULT_MAGI_BALTHASAR_LINEAGE,
            DEFAULT_MAGI_CASPAR_LINEAGE,
        ];
        for entry in pool {
            assert!(
                !seat_lineages.contains(&entry.lineage.as_str()),
                "each pool entry must carry a label NO seat holds, so it covers all three: {entry:?}"
            );
        }

        assert_eq!(
            cfg.effective_max_rotations(),
            DEFAULT_MAX_ROTATIONS,
            "the scaffold must declare rotation, not leave it implicit"
        );
    }

    /// The pool must be the LAST thing in the file. In TOML every loose key and sub-table has to
    /// precede the first array of tables, so anything emitted after it would parse into the pool
    /// entry instead of its own table — silently, which is the part that makes it dangerous.
    #[test]
    fn nothing_is_emitted_after_the_fallback_pool() {
        let scaffold = render_default_magi_toml();
        let first_pool = scaffold
            .find("[[magi.fallback]]")
            .expect("the scaffold must declare a pool");
        let tail = &scaffold[first_pool..];
        for line in tail.lines() {
            let t = line.trim();
            if t.is_empty() || t.starts_with('#') || t.starts_with("[[magi.fallback]]") {
                continue;
            }
            assert!(
                t.starts_with("model") || t.starts_with("lineage"),
                "only pool entries may follow the pool; found: {t}"
            );
        }
    }

    /// The no-config startup notice must name every accepted `provider` value too.
    ///
    /// This is the MORE operator-visible of the two surfaces: it fires for whoever has not run
    /// `magi init`, which is exactly the person who never sees the scaffold comment. Naming
    /// only `anthropic` there left the `openai-compat` user — pointing at Groq, OpenRouter or a
    /// vLLM server — with no surface at all to learn the value from, while a wrong guess is a
    /// fatal parse error rather than a warning.
    #[test]
    fn the_no_config_notice_names_every_accepted_provider_value() {
        let notice = no_config_notice();
        for kind in magi_rs::magi::kind::ProviderKind::VOCABULARY {
            assert!(
                notice.contains(&kind.to_string()),
                "the no-config notice must name `{kind}`; it said: {notice:?}"
            );
        }
    }

    /// RF-9's notice is levelled for the mouth that exists in the state it exists for.
    ///
    /// **Not an assertion that it equals `WARN`.** What the requirement needs is that it
    /// reaches a human, and the predicate the production fallback applies to decide that is
    /// `partition_by_mouth` — so this drives that, and would still pass if RF-9 were one day
    /// judged an `ERROR`. Asserting the level as a literal would pin a decision instead of
    /// the property behind it.
    ///
    /// The state under test is the first run with no `.magi/`: no workspace, therefore no
    /// log directory and no layer, therefore the no-subscriber branch, which keeps only the
    /// screen half of the list and drops the rest on the floor. `INFO` put this notice in the
    /// half that is dropped.
    #[test]
    fn the_no_config_notice_is_levelled_for_the_mouth_a_first_run_has() {
        let (screen, file) = magi_rs::notices::partition_by_mouth(vec![no_config_startup_notice()]);
        assert!(
            file.is_empty(),
            "RF-9's notice went to the half whose only mouth is a log file, and the run it \
             is written for has no log file: {file:?}"
        );
        let text = screen
            .first()
            .map(|n| n.text.clone())
            .unwrap_or_else(|| "<nothing reached the screen half>".to_string());
        assert!(
            text.contains(DEFAULT_MAGI_MELCHIOR),
            "the operator must be told which trio the defaults picked for them, and the \
             `magi init` warning they also get says nothing about it: {text}"
        );
    }

    /// The line above `provider =` must NAME the three accepted values.
    ///
    /// The generated header only ever mentioned `anthropic` ("or set provider = \"anthropic\"
    /// to use Anthropic instead"), which reads as a binary choice and hides `openai-compat`
    /// entirely. An operator pointing at Groq or OpenRouter had no way to learn the value they
    /// needed from the file `magi init` handed them — and `deny_unknown_fields` makes a wrong
    /// guess a FATAL parse error, not a warning, so the cost of not knowing is a process that
    /// refuses to start.
    ///
    /// Asserted on the comment BLOCK, not on a trailing comment: a value at the end of a long
    /// line is the part a reader skips.
    #[test]
    fn the_scaffold_names_every_accepted_provider_value_above_the_key() {
        let scaffold = render_default_magi_toml();
        let lines: Vec<&str> = scaffold.lines().collect();
        let key = lines
            .iter()
            .position(|l| l.trim_start().starts_with("provider ="))
            .expect("the scaffold must declare provider");

        let preceding_comments: String = lines[..key]
            .iter()
            .rev()
            .take_while(|l| l.trim_start().starts_with('#'))
            .copied()
            .collect::<Vec<_>>()
            .join(
                "
",
            );

        for value in ["ollama", "openai-compat", "anthropic"] {
            assert!(
                preceding_comments.contains(value),
                "the comment above `provider =` must name `{value}`; it said:                  {preceding_comments:?}"
            );
        }
    }

    /// The scaffolded pool carries **five** candidates, matching the trio configuration this
    /// project ships alongside (`magi-ollama.toml`) rather than a shorter list of its own.
    ///
    /// The count is the point, and so is what could NOT be copied. That file's five entries
    /// include `deepseek-v4-pro`/`deepseek` and `gpt-oss`/`openai`, which are foreign to ITS
    /// trio but are two of THIS project's three seats — model and lineage both. Copying them
    /// verbatim would have cut each one's coverage from three seats to one and tripped the
    /// duplicate-model notice twice. The five here are the five labels from that same file that
    /// no seat of this trio holds.
    #[test]
    fn the_scaffolded_pool_has_five_candidates() {
        assert_eq!(
            DEFAULT_SCAFFOLD_POOL.len(),
            5,
            "the pool ships five candidates, like the trio configuration this project runs"
        );
    }

    /// The scaffold must leave `[embedding].base_url` **commented**, so the embedder inherits
    /// the root endpoint instead of pinning its own.
    ///
    /// This was a live defect, not a style preference. The generator emitted the key ACTIVE
    /// while the comment three lines above it in this same file explained that
    /// `EmbeddingConfig::default()` leaves it `None` "so it can inherit the root `base_url`" —
    /// the stated intent and the adjacent line disagreeing, which is the shape of defect this
    /// codebase keeps paying for.
    ///
    /// What it cost the operator: `magi init`, then point the root `base_url` at a remote host.
    /// The main agent and the trio move; the embedder silently keeps talking to localhost, and
    /// nothing in the file looks wrong because the value it pins is a real, valid endpoint.
    #[test]
    fn the_scaffold_leaves_the_embedding_endpoint_commented_so_it_inherits() {
        let scaffold = render_default_magi_toml();

        // Located by WHOLE LINE, never by substring. The scaffold also emits a COMMENT
        // mentioning `[embedding]` before the real header, so `split("[embedding]")` lands in
        // the gap between the two and inspects a span holding no keys at all — which reads as
        // a passing test right up until the moment it has to catch something.
        let lines: Vec<&str> = scaffold.lines().collect();
        let header = lines
            .iter()
            .position(|l| l.trim() == "[embedding]")
            .expect("the scaffold must declare an [embedding] section");
        let section: Vec<&str> = lines[header + 1..]
            .iter()
            .take_while(|l| !l.trim_start().starts_with('['))
            .copied()
            .collect();

        for line in &section {
            let t = line.trim();
            assert!(
                !t.starts_with("base_url"),
                "[embedding].base_url must be commented out, not active — an active value \
                 overrides the root endpoint and stops inheriting it: {t}"
            );
        }
        assert!(
            section.iter().any(|l| l.trim().starts_with("# base_url")),
            "the key must still be SHOWN, commented: a knob the operator cannot see is a knob \
             they will not know exists. Section was: {section:?}"
        );
    }
}
