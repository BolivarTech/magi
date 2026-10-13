// Author: Julian Bolivar
// Version: 0.23.0
// Date: 2026-10-12

//! Per-section API key sources: the one env > vault precedence every section shares (REQ-K03).
#![deny(missing_docs)]
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing,
        clippy::string_slice
    )
)]

/// The vault as every surface shares it: passphrase resolution, key discovery and the TUI's
/// `/login` and `/logout` handlers.
///
/// `VaultStore`, the concrete type behind the trait object, is `Send` but deliberately not
/// `Sync` (its mask rotates on every access); the `Mutex` supplies the exclusion.
pub type SharedSecretStore = std::sync::Arc<std::sync::Mutex<dyn crate::vault::SecretStore + Send>>;

/// A secret found by [`env_or_vault`], and where it came from.
#[derive(Clone, PartialEq, Eq)]
pub struct ResolvedSecret {
    /// The trimmed, non-blank secret.
    pub value: String,
    /// `true` when the value came from the consumed environment, `false` from the vault.
    pub from_env: bool,
}

/// Prints the origin but never the value.
impl std::fmt::Debug for ResolvedSecret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedSecret")
            .field("value", &"***")
            .field("from_env", &self.from_env)
            .finish()
    }
}

/// Environment variable and vault entry of the trio's own key (REQ-K01).
pub const MAGI_KEY_ENV: &str = "MAGI_API_KEY";

/// Environment variable and vault entry of the embedder's own key (REQ-K02).
pub const EMBEDDING_KEY_ENV: &str = "EMBEDDING_API_KEY";

/// Environment variable and vault entry of the decision endpoint's own key (REQ-D04).
pub const DECISION_KEY_ENV: &str = "DECISION_API_KEY";

/// The OpenAI-compatible key every OpenAI-compatible section may inherit on the root's origin.
pub const OPENAI_KEY_ENV: &str = "OPENAI_API_KEY";

/// The Anthropic key an Anthropic trio falls back to (REQ-K07).
pub const ANTHROPIC_KEY_ENV: &str = "ANTHROPIC_API_KEY";

/// What a retained entry shows when the section's endpoint has no parsable origin.
const UNPARSABLE_ORIGIN: &str = "(endpoint without a valid origin)";

/// The API keys read (and scrubbed) from the environment at startup; never the passphrase.
///
/// Values are kept as read: blank ones are filtered where they are used ([`env_or_vault`]).
pub struct ApiKeys {
    /// The Anthropic key (`ANTHROPIC_API_KEY`).
    pub anthropic_key: Option<String>,
    /// The OpenAI-compatible key (`OPENAI_API_KEY`), the principal's key when it is not
    /// Anthropic.
    pub openai_key: Option<String>,
    /// The decision endpoint key (`DECISION_API_KEY`).
    pub decision_key: Option<String>,
    /// The trio's own key (`MAGI_API_KEY`).
    pub magi_key: Option<String>,
    /// The embedder's own key (`EMBEDDING_API_KEY`).
    pub embedding_key: Option<String>,
}

/// Resolves a key: the consumed env value if non-blank (trimmed), else the vault entry `name` if
/// non-blank (trimmed).
///
/// Complexity: O(1) plus one vault read. Never fails: a poisoned lock is recovered and an absent
/// entry is `None`. Recovering a poisoned lock is deliberate: the vault is read once at startup,
/// and a panic elsewhere must not leave every section without its key.
///
/// # Arguments
/// * `name` - the vault entry to fall back to.
/// * `env_value` - the value consumed from the environment at startup, if any.
/// * `secret_store` - the open vault, if any.
///
/// # Returns
/// The secret and its origin, or `None` when neither source holds a non-blank value.
///
/// # Examples
/// ```
/// use magi_rs::section_keys::env_or_vault;
///
/// let found = env_or_vault("MAGI_API_KEY", Some(" from-env"), None).expect("env value");
/// assert_eq!(found.value, "from-env");
/// assert!(found.from_env);
/// assert!(env_or_vault("MAGI_API_KEY", Some("  "), None).is_none());
/// ```
pub fn env_or_vault(
    name: &str,
    env_value: Option<&str>,
    secret_store: Option<&SharedSecretStore>,
) -> Option<ResolvedSecret> {
    if let Some(value) = trimmed_non_blank(env_value) {
        return Some(ResolvedSecret {
            value: value.to_string(),
            from_env: true,
        });
    }
    let mut guard = secret_store?.lock().unwrap_or_else(|p| p.into_inner());
    let stored = guard.get(name).ok()?;
    trimmed_non_blank(Some(stored.as_str())).map(|value| ResolvedSecret {
        value: value.to_string(),
        from_env: false,
    })
}

/// The trimmed text, or `None` when it is absent or blank.
fn trimmed_non_blank(text: Option<&str>) -> Option<&str> {
    text.map(str::trim).filter(|t| !t.is_empty())
}

/// Where a section's key came from; the wording of an authentication failure depends on it
/// (REQ-K06).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeySource {
    /// The section's own variable (`MAGI_API_KEY`, `EMBEDDING_API_KEY`, `DECISION_API_KEY`).
    Own,
    /// `OPENAI_API_KEY` on the root endpoint's origin, or `ANTHROPIC_API_KEY` for an Anthropic
    /// trio.
    Inherited,
    /// No key: no `Authorization` header (OpenAI-compatible) or a seat that cannot be built
    /// (Anthropic).
    Absent,
}

/// The key chosen for one section and where it came from. Its `Debug` never prints the value.
#[derive(Clone, PartialEq, Eq)]
pub struct SectionKey {
    value: Option<String>,
    source: KeySource,
}

impl SectionKey {
    /// The key to send, or `None` for no authentication header.
    ///
    /// # Returns
    /// The selected key, if any.
    pub fn value(&self) -> Option<&str> {
        self.value.as_deref()
    }

    /// Where the key came from.
    ///
    /// # Returns
    /// The [`KeySource`] of this section.
    pub fn source(&self) -> KeySource {
        self.source
    }
}

/// Prints the source and whether a key is present, never the key itself.
impl std::fmt::Debug for SectionKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SectionKey")
            .field("value", &self.value.as_ref().map(|_| "***"))
            .field("source", &self.source)
            .finish()
    }
}

impl SectionKey {
    /// Builds a key of a chosen source, for tests that need one without resolving a section.
    ///
    /// # Arguments
    /// * `value` - the key to carry, if any.
    /// * `source` - where the key is said to have come from.
    ///
    /// # Returns
    /// A [`SectionKey`] holding both.
    #[doc(hidden)]
    pub fn for_test(value: Option<String>, source: KeySource) -> SectionKey {
        SectionKey { value, source }
    }

    /// A section without a key.
    fn absent() -> SectionKey {
        SectionKey {
            value: None,
            source: KeySource::Absent,
        }
    }
}

/// A section that did not inherit `OPENAI_API_KEY` because its origin differs from the root's
/// (REQ-K09).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetainedKey {
    variable: &'static str,
    origin_redacted: String,
}

impl RetainedKey {
    /// The variable that would carry the section's own key.
    ///
    /// # Returns
    /// `MAGI_API_KEY`, `EMBEDDING_API_KEY` or `DECISION_API_KEY`.
    pub fn variable(&self) -> &'static str {
        self.variable
    }

    /// The section's origin, redacted.
    ///
    /// # Returns
    /// `scheme://host[:port]` without credentials.
    pub fn origin_redacted(&self) -> &str {
        &self.origin_redacted
    }
}

/// Keys already chosen for every section that can point at its own endpoint (REQ-K01..K03).
///
/// The principal's own key is not here: it keeps its resolution path, which shares
/// [`env_or_vault`]. Holds only owned strings and copyable enums, so it is `Send + Sync` and is
/// shared read-only.
pub struct SectionKeys {
    magi: SectionKey,
    embedding: SectionKey,
    decision: SectionKey,
    retained: Vec<RetainedKey>,
}

/// Everything [`SectionKeys::resolve`] needs, already resolved by the caller.
pub struct KeyInputs<'a> {
    /// The keys consumed (and scrubbed) from the environment at startup.
    pub env: &'a ApiKeys,
    /// The open vault, if any.
    pub secret_store: Option<&'a SharedSecretStore>,
    /// The resolved `[magi].kind`: decides how the trio's key is chosen.
    pub magi_kind: crate::magi::kind::ProviderKind,
    /// The root endpoint `OPENAI_API_KEY` belongs to (`OPENAI_BASE_URL` > root `base_url` >
    /// built-in default), for any principal kind.
    pub root_endpoint: &'a str,
    /// The resolved `[magi]` endpoint, or `None` when it did not resolve.
    pub magi_endpoint: Option<&'a str>,
    /// The resolved `[embedding]` endpoint, or `None` when it did not resolve.
    pub embedding_endpoint: Option<&'a str>,
    /// The resolved `[decision]` endpoint, or `None` when it did not resolve.
    pub decision_endpoint: Option<&'a str>,
}

impl SectionKeys {
    /// Chooses the key of every section (spec §B).
    ///
    /// # Arguments
    /// * `inputs` - consumed environment, vault, trio kind and resolved endpoints.
    ///
    /// # Returns
    /// The chosen keys and the sections whose inheritance was refused.
    ///
    /// Complexity: O(1) per section plus one vault read per key, and O(n) in the length of the
    /// URLs for each origin comparison.
    ///
    /// # Examples
    /// ```
    /// use magi_rs::magi::kind::ProviderKind;
    /// use magi_rs::section_keys::{ApiKeys, KeyInputs, KeySource, SectionKeys};
    ///
    /// let env = ApiKeys {
    ///     anthropic_key: None,
    ///     openai_key: Some("principal-key".into()),
    ///     decision_key: None,
    ///     magi_key: None,
    ///     embedding_key: None,
    /// };
    /// let keys = SectionKeys::resolve(&KeyInputs {
    ///     env: &env,
    ///     secret_store: None,
    ///     magi_kind: ProviderKind::OpenAiCompat,
    ///     root_endpoint: "http://localhost:11434/v1",
    ///     magi_endpoint: Some("http://localhost:11434/v1"),
    ///     embedding_endpoint: Some("http://192.168.0.30:11434/v1"),
    ///     decision_endpoint: None,
    /// });
    /// assert_eq!(keys.magi().source(), KeySource::Inherited); // same origin as the root
    /// assert_eq!(keys.embedding().value(), None); // another origin: no key is sent
    /// assert_eq!(keys.retained()[0].variable(), "EMBEDDING_API_KEY");
    /// ```
    pub fn resolve(inputs: &KeyInputs<'_>) -> SectionKeys {
        let env = inputs.env;
        let store = inputs.secret_store;
        let read = |name: &str, value: &Option<String>| {
            env_or_vault(name, value.as_deref(), store).map(|r| r.value)
        };
        let openai = read(OPENAI_KEY_ENV, &env.openai_key);
        let mut retained = Vec::new();
        // One OpenAI-compatible section, decided by a single `select_api_key` call (REQ-K03): an
        // own key wins; otherwise `OPENAI_API_KEY` only on the root's origin; a refusal by origin
        // is recorded as a `RetainedKey` (REQ-K09). It lives inside `resolve` so that `resolve`
        // stays the only caller of `select_api_key` (the single-source guard checks it).
        let mut openai_section =
            |variable: &'static str, own: Option<String>, endpoint: Option<&str>| -> SectionKey {
                if let Some(own) = own {
                    return SectionKey {
                        value: Some(own),
                        source: KeySource::Own,
                    };
                }
                let Some(endpoint) = endpoint else {
                    return SectionKey::absent();
                };
                match crate::endpoint_key::select_api_key(
                    None,
                    openai.as_deref(),
                    inputs.root_endpoint,
                    endpoint,
                ) {
                    Some(key) => SectionKey {
                        value: Some(key),
                        source: KeySource::Inherited,
                    },
                    None => {
                        if openai.is_some() {
                            let origin = crate::endpoint_key::origin_url(endpoint)
                                .unwrap_or_else(|| UNPARSABLE_ORIGIN.to_string());
                            retained.push(RetainedKey {
                                variable,
                                origin_redacted: crate::redact::redact_url(&origin),
                            });
                        }
                        SectionKey::absent()
                    }
                }
            };
        let magi_own = read(MAGI_KEY_ENV, &env.magi_key);
        let magi = match inputs.magi_kind {
            crate::magi::kind::ProviderKind::OpenAiCompat => {
                openai_section(MAGI_KEY_ENV, magi_own, inputs.magi_endpoint)
            }
            crate::magi::kind::ProviderKind::Anthropic => match magi_own {
                Some(own) => SectionKey {
                    value: Some(own),
                    source: KeySource::Own,
                },
                None => match read(ANTHROPIC_KEY_ENV, &env.anthropic_key) {
                    Some(k) => SectionKey {
                        value: Some(k),
                        source: KeySource::Inherited,
                    },
                    None => SectionKey::absent(),
                },
            },
            crate::magi::kind::ProviderKind::Ollama => SectionKey::absent(),
        };
        let embedding = openai_section(
            EMBEDDING_KEY_ENV,
            read(EMBEDDING_KEY_ENV, &env.embedding_key),
            inputs.embedding_endpoint,
        );
        let decision = openai_section(
            DECISION_KEY_ENV,
            read(DECISION_KEY_ENV, &env.decision_key),
            inputs.decision_endpoint,
        );
        SectionKeys {
            magi,
            embedding,
            decision,
            retained,
        }
    }

    /// The trio's key.
    ///
    /// # Returns
    /// The [`SectionKey`] of `[magi]`.
    pub fn magi(&self) -> &SectionKey {
        &self.magi
    }

    /// The embedder's key.
    ///
    /// # Returns
    /// The [`SectionKey`] of `[embedding]`.
    pub fn embedding(&self) -> &SectionKey {
        &self.embedding
    }

    /// The decision client's key.
    ///
    /// # Returns
    /// The [`SectionKey`] of `[decision]`.
    pub fn decision(&self) -> &SectionKey {
        &self.decision
    }

    /// The sections that did not inherit `OPENAI_API_KEY` because of their origin.
    ///
    /// # Returns
    /// One entry per such section, in the order magi, embedding, decision.
    pub fn retained(&self) -> &[RetainedKey] {
        &self.retained
    }

    /// The run-log entries that say a section's key was withheld (REQ-K09).
    ///
    /// # Returns
    /// One `INFO` notice per retained section, in the order magi, embedding, decision.
    pub fn startup_notices(&self) -> Vec<crate::notices::Notice> {
        self.retained
            .iter()
            .map(|r| {
                crate::notices::Notice::info(format!(
                    "{OPENAI_KEY_ENV} was not sent to {}: it is not the main endpoint's origin. \
                     If that endpoint needs a key, run: magi-rs vault set {}",
                    r.origin_redacted, r.variable
                ))
            })
            .collect()
    }

    /// The keys that only a section's own variable explains, named by that variable, for the
    /// log auditor's exact pass.
    ///
    /// Inherited keys are left out: they are the principal's keys, armed under
    /// `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` already.
    ///
    /// # Returns
    /// One `(name, value)` pair per section whose source is [`KeySource::Own`], in the order
    /// magi, embedding, decision.
    ///
    /// Complexity: O(1).
    pub fn own_secrets(&self) -> Vec<(crate::logging::auditor::SecretName, &str)> {
        use crate::logging::auditor::SecretName;
        [
            (MAGI_KEY_ENV, &self.magi),
            (EMBEDDING_KEY_ENV, &self.embedding),
            (DECISION_KEY_ENV, &self.decision),
        ]
        .into_iter()
        .filter(|(_, key)| key.source == KeySource::Own)
        .filter_map(|(name, key)| key.value.as_deref().map(|v| (SecretName::new(name), v)))
        .collect()
    }
}

/// Prefix of an Anthropic API key, the shape the rebuilt trio expects.
const ANTHROPIC_KEY_PREFIX: &str = "sk-ant-";

/// Key the trio uses after `/login` rebuilds it as Anthropic: the trio's own key wins
/// (REQ-K07); otherwise the key the login just stored. Blank counts as absent.
///
/// # Arguments
/// * `own_magi_key` - the trio's own key (`MAGI_API_KEY` or its vault entry), if any.
/// * `login_key` - the key the login just minted.
///
/// # Returns
/// The key the rebuilt trio authenticates with.
///
/// Complexity: O(n) in the key length (trim).
pub fn post_login_trio_key(own_magi_key: Option<&str>, login_key: String) -> String {
    own_magi_key
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map_or(login_key, str::to_string)
}

/// True when the trio, rebuilt as Anthropic after `/login`, would use an own key that is not
/// Anthropic-shaped (`sk-ant-`): the same shape rule as at startup, evaluated for the rebuilt
/// kind. Never blocks.
///
/// # Arguments
/// * `own_magi_key` - the trio's own key, if any.
///
/// # Returns
/// `true` when a non-blank own key lacks the `sk-ant-` prefix.
///
/// Complexity: O(n) in the key length (trim).
pub fn post_login_shape_mismatch(own_magi_key: Option<&str>) -> bool {
    own_magi_key
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .is_some_and(|key| !key.starts_with(ANTHROPIC_KEY_PREFIX))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vault::{wire, MaskedDek, SecretStore};
    use std::sync::{Arc, Mutex};
    use zeroize::Zeroizing;

    /// An in-memory vault holding `entries` (the pattern of `vault_fixture`, `src/main.rs:10486`).
    fn vault(entries: &[(&str, &str)]) -> SharedSecretStore {
        let conn = rusqlite::Connection::open_in_memory().expect("mem db");
        let dek = MaskedDek::new(Zeroizing::new(vec![3u8; 32])).expect("32B");
        let mut store = wire(Arc::new(Mutex::new(conn)), dek).expect("wire");
        for (k, v) in entries {
            store.set(k, v).expect("set");
        }
        Arc::new(Mutex::new(store)) as SharedSecretStore
    }

    fn pair(r: Option<ResolvedSecret>) -> Option<(String, bool)> {
        r.map(|s| (s.value, s.from_env))
    }

    #[test]
    fn only_own_section_keys_are_armed_by_their_variable_name() {
        let e = env(
            Some("prin-key"),
            None,
            Some("magi-own"),
            Some("embed-own"),
            Some("dec-own"),
        );
        let k = resolve(
            &e,
            ProviderKind::OpenAiCompat,
            Some("http://127.0.0.1:1"),
            Some(ROOT),
            Some(ROOT),
        );
        let armed: Vec<(String, String)> = k
            .own_secrets()
            .into_iter()
            .map(|(n, v)| (n.as_str().to_string(), v.to_string()))
            .collect();
        assert_eq!(
            armed,
            vec![
                ("MAGI_API_KEY".into(), "magi-own".into()),
                ("EMBEDDING_API_KEY".into(), "embed-own".into()),
                ("DECISION_API_KEY".into(), "dec-own".into())
            ]
        );
        let inherited = resolve(
            &env(Some("prin-key"), None, None, None, None),
            ProviderKind::OpenAiCompat,
            Some(ROOT),
            Some(ROOT),
            Some(ROOT),
        );
        assert!(inherited.own_secrets().is_empty());
    }

    #[test]
    fn a_non_blank_env_value_wins_over_the_vault_and_is_trimmed() {
        let ss = vault(&[("MAGI_API_KEY", "from-vault")]);
        assert_eq!(
            pair(env_or_vault("MAGI_API_KEY", Some(" from-env\n"), Some(&ss))),
            Some(("from-env".to_string(), true))
        );
    }

    #[test]
    fn a_blank_env_value_falls_through_to_the_vault() {
        let ss = vault(&[("MAGI_API_KEY", " from-vault\n")]);
        for blank in ["", "   ", "\n"] {
            assert_eq!(
                pair(env_or_vault("MAGI_API_KEY", Some(blank), Some(&ss))),
                Some(("from-vault".to_string(), false)),
                "{blank:?}"
            );
        }
    }

    #[test]
    fn a_blank_vault_entry_is_absent() {
        let ss = vault(&[("EMBEDDING_API_KEY", "   ")]);
        assert_eq!(
            pair(env_or_vault("EMBEDDING_API_KEY", None, Some(&ss))),
            None
        );
    }

    #[test]
    fn without_env_or_vault_there_is_no_key() {
        assert_eq!(pair(env_or_vault("MAGI_API_KEY", None, None)), None);
        assert_eq!(pair(env_or_vault("MAGI_API_KEY", Some(" "), None)), None);
        assert_eq!(
            pair(env_or_vault("MAGI_API_KEY", None, Some(&vault(&[])))),
            None
        );
    }

    #[test]
    fn only_the_named_vault_entry_is_read() {
        let ss = vault(&[("OPENAI_API_KEY", "openai")]);
        assert_eq!(pair(env_or_vault("MAGI_API_KEY", None, Some(&ss))), None);
    }

    #[test]
    fn a_poisoned_vault_lock_is_recovered_not_propagated() {
        let store = vault(&[("MAGI_API_KEY", "vault-key")]); // allow-secret-scan
        let poisoner = store.clone();
        let _ = std::thread::spawn(move || {
            let _guard = poisoner.lock().unwrap();
            panic!("poison the vault lock");
        })
        .join();
        assert!(store.lock().is_err(), "precondition: the lock is poisoned");
        assert_eq!(
            pair(env_or_vault("MAGI_API_KEY", None, Some(&store))),
            Some(("vault-key".to_string(), false))
        );
    }

    #[test]
    fn a_resolved_secret_never_prints_its_value() {
        let s = ResolvedSecret {
            value: "s3cr3t-value".into(),
            from_env: true,
        };
        assert!(!format!("{s:?}").contains("s3cr3t-value"));
    }

    use crate::magi::kind::ProviderKind;

    const ROOT: &str = "http://localhost:11434/v1";

    fn env(
        openai: Option<&str>,
        anthropic: Option<&str>,
        magi: Option<&str>,
        embedding: Option<&str>,
        decision: Option<&str>,
    ) -> ApiKeys {
        let s = |v: Option<&str>| v.map(str::to_string);
        ApiKeys {
            openai_key: s(openai),
            anthropic_key: s(anthropic),
            magi_key: s(magi),
            embedding_key: s(embedding),
            decision_key: s(decision),
        }
    }

    fn resolve(
        env: &ApiKeys,
        kind: ProviderKind,
        magi: Option<&str>,
        embedding: Option<&str>,
        decision: Option<&str>,
    ) -> SectionKeys {
        SectionKeys::resolve(&KeyInputs {
            env,
            secret_store: None,
            magi_kind: kind,
            root_endpoint: ROOT,
            magi_endpoint: magi,
            embedding_endpoint: embedding,
            decision_endpoint: decision,
        })
    }

    fn all(k: &SectionKeys) -> [(Option<&str>, KeySource); 3] {
        [
            (k.magi().value(), k.magi().source()),
            (k.embedding().value(), k.embedding().source()),
            (k.decision().value(), k.decision().source()),
        ]
    }

    #[test]
    fn an_own_key_wins_on_any_origin() {
        let e = env(Some("prin"), None, Some("mk"), Some("ek"), Some("dk"));
        for target in [ROOT, "http://127.0.0.1:9"] {
            let k = resolve(
                &e,
                ProviderKind::OpenAiCompat,
                Some(target),
                Some(target),
                Some(target),
            );
            assert_eq!(
                all(&k),
                [
                    (Some("mk"), KeySource::Own),
                    (Some("ek"), KeySource::Own),
                    (Some("dk"), KeySource::Own)
                ],
                "{target}"
            );
            assert!(k.retained().is_empty());
        }
    }

    #[test]
    fn the_principal_key_is_inherited_on_the_roots_origin() {
        let e = env(Some("prin"), None, None, None, None);
        let k = resolve(
            &e,
            ProviderKind::OpenAiCompat,
            Some("http://localhost:11434"),
            Some(ROOT),
            Some("http://LOCALHOST:11434/x"),
        );
        assert_eq!(all(&k), [(Some("prin"), KeySource::Inherited); 3]);
        assert!(k.retained().is_empty());
    }

    #[test]
    fn another_origin_gets_no_key_and_one_retained_entry_per_section() {
        let e = env(Some("prin"), None, None, None, None);
        let k = resolve(
            &e,
            ProviderKind::OpenAiCompat,
            Some("http://127.0.0.1:11434"),
            Some("http://127.0.0.1:11435/v1"),
            Some("https://openrouter.ai/api/v1"),
        );
        assert_eq!(all(&k), [(None, KeySource::Absent); 3]);
        let got: Vec<(&str, &str)> = k
            .retained()
            .iter()
            .map(|r| (r.variable(), r.origin_redacted()))
            .collect();
        assert_eq!(
            got,
            vec![
                ("MAGI_API_KEY", "http://127.0.0.1:11434"),
                ("EMBEDDING_API_KEY", "http://127.0.0.1:11435"),
                ("DECISION_API_KEY", "https://openrouter.ai")
            ]
        );
    }

    /// RF1 / K-S5: the adversarial origins of `decide`, now for every section.
    #[test]
    fn adversarial_origins_never_inherit_in_any_section() {
        let e = env(Some("prin"), None, None, None, None);
        for target in [
            "http://localhost:11434@evil.com",
            "https://localhost:11434",
            "http://localhost:11435",
            "http://localhost.evil.com:11434",
            "http://evil-localhost:11434",
            "http://evil.com\\@localhost:11434",
            "http://localhost%40evil.com:11434",
            "http://[::ffff:127.0.0.1]:11434",
            "not a url",
        ] {
            let k = resolve(
                &e,
                ProviderKind::OpenAiCompat,
                Some(target),
                Some(target),
                Some(target),
            );
            assert_eq!(all(&k), [(None, KeySource::Absent); 3], "{target}");
        }
    }

    #[test]
    fn equivalent_spellings_inherit_in_every_section() {
        let e = env(Some("prin"), None, None, None, None);
        for target in [
            "http://LOCALHOST:11434",
            "http://localhost.:11434",
            "http://localhost:11434/other/path",
        ] {
            let k = resolve(
                &e,
                ProviderKind::OpenAiCompat,
                Some(target),
                Some(target),
                Some(target),
            );
            assert_eq!(
                all(&k),
                [(Some("prin"), KeySource::Inherited); 3],
                "{target}"
            );
        }
    }

    /// RF2: blank is absent, in both directions, and never produces a retained entry.
    #[test]
    fn a_blank_principal_key_is_absent_and_logs_nothing() {
        let e = env(Some("   "), None, None, None, None);
        let k = resolve(
            &e,
            ProviderKind::OpenAiCompat,
            Some("http://127.0.0.1:1"),
            Some(ROOT),
            None,
        );
        assert_eq!(all(&k), [(None, KeySource::Absent); 3]);
        assert!(k.retained().is_empty());
    }

    #[test]
    fn a_blank_own_key_is_not_an_own_key() {
        let e = env(Some("prin"), None, Some(" "), Some(""), Some("\n"));
        let k = resolve(
            &e,
            ProviderKind::OpenAiCompat,
            Some(ROOT),
            Some(ROOT),
            Some(ROOT),
        );
        assert_eq!(all(&k), [(Some("prin"), KeySource::Inherited); 3]);
    }

    /// RF3 / D1.
    #[test]
    fn an_unresolved_section_endpoint_is_absent_and_logs_nothing() {
        let e = env(Some("prin"), None, None, None, None);
        let k = resolve(&e, ProviderKind::OpenAiCompat, None, None, None);
        assert_eq!(all(&k), [(None, KeySource::Absent); 3]);
        assert!(k.retained().is_empty());
    }

    #[test]
    fn an_own_key_is_read_from_the_vault_when_the_env_has_none() {
        let ss = vault(&[
            ("MAGI_API_KEY", "mk-vault"),
            ("EMBEDDING_API_KEY", "ek-vault"),
        ]);
        let e = env(None, None, None, None, None);
        let k = SectionKeys::resolve(&KeyInputs {
            env: &e,
            secret_store: Some(&ss),
            magi_kind: ProviderKind::OpenAiCompat,
            root_endpoint: ROOT,
            magi_endpoint: Some(ROOT),
            embedding_endpoint: Some(ROOT),
            decision_endpoint: Some(ROOT),
        });
        assert_eq!(
            (k.magi().value(), k.magi().source()),
            (Some("mk-vault"), KeySource::Own)
        );
        assert_eq!(
            (k.embedding().value(), k.embedding().source()),
            (Some("ek-vault"), KeySource::Own)
        );
    }

    /// K-S8 / RF5: an Anthropic trio never compares origins and never logs a retained key.
    #[test]
    fn an_anthropic_trio_never_compares_origins() {
        let without = env(Some("prin"), Some("ak"), None, None, None);
        let k = resolve(
            &without,
            ProviderKind::Anthropic,
            Some("http://127.0.0.1:1"),
            None,
            None,
        );
        assert_eq!(
            (k.magi().value(), k.magi().source()),
            (Some("ak"), KeySource::Inherited)
        );
        assert!(k.retained().is_empty());
        let with = env(Some("prin"), Some("ak"), Some("mk"), None, None);
        let k = resolve(&with, ProviderKind::Anthropic, None, None, None);
        assert_eq!(
            (k.magi().value(), k.magi().source()),
            (Some("mk"), KeySource::Own)
        );
    }

    #[test]
    fn an_ollama_trio_carries_no_key_and_logs_nothing() {
        let e = env(Some("prin"), None, Some("mk"), None, None);
        let k = resolve(
            &e,
            ProviderKind::Ollama,
            Some("http://127.0.0.1:1"),
            None,
            None,
        );
        assert_eq!(
            (k.magi().value(), k.magi().source()),
            (None, KeySource::Absent)
        );
        assert!(k.retained().is_empty());
    }

    #[test]
    fn section_keys_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SectionKeys>();
    }

    #[test]
    fn without_any_key_every_section_is_absent() {
        let e = env(None, None, None, None, None);
        let k = resolve(
            &e,
            ProviderKind::OpenAiCompat,
            Some(ROOT),
            Some(ROOT),
            Some(ROOT),
        );
        assert_eq!(all(&k), [(None, KeySource::Absent); 3]);
    }

    #[test]
    fn a_section_key_never_prints_its_value() {
        let e = env(None, None, Some("s3cr3t-magi"), None, None);
        let k = resolve(&e, ProviderKind::OpenAiCompat, Some(ROOT), None, None);
        assert!(!format!("{:?}", k.magi()).contains("s3cr3t-magi"));
    }

    #[test]
    fn after_login_the_trios_own_key_wins_over_the_login_key() {
        assert_eq!(
            post_login_trio_key(Some("magi-own-key"), "login-key".to_string()), // allow-secret-scan
            "magi-own-key"
        );
    }

    #[test]
    fn after_login_a_non_anthropic_own_key_is_flagged_and_an_anthropic_one_is_not() {
        assert!(post_login_shape_mismatch(Some("sk-or-own"))); // allow-secret-scan
        assert!(!post_login_shape_mismatch(Some("sk-ant-own"))); // allow-secret-scan
        assert!(!post_login_shape_mismatch(None));
        assert!(!post_login_shape_mismatch(Some("  ")));
    }

    #[test]
    fn a_magi_key_of_the_wrong_shape_for_the_trio_kind_warns_once_without_its_value() {
        use crate::notices::Notice;
        for (kind, key, warns) in [
            (ProviderKind::OpenAiCompat, "sk-ant-toy-0001", true), // allow-secret-scan
            (ProviderKind::Ollama, "sk-ant-toy-0001", true),       // allow-secret-scan
            (ProviderKind::Anthropic, "sk-or-toy-0001", true),     // allow-secret-scan
            (ProviderKind::Anthropic, "sk-toy-0001", true),        // allow-secret-scan
            (ProviderKind::OpenAiCompat, "sk-or-toy-0001", false), // allow-secret-scan
            (ProviderKind::Anthropic, "sk-ant-toy-0001", false),   // allow-secret-scan
        ] {
            let e = env(None, None, Some(key), None, None);
            let k = resolve(&e, kind, Some(ROOT), None, None);
            let warn: Vec<Notice> = k
                .startup_notices()
                .into_iter()
                .filter(|n| n.level == tracing::Level::WARN)
                .collect();
            assert_eq!(warn.len(), usize::from(warns), "{kind:?} {key}");
            if let Some(n) = warn.first() {
                assert!(
                    n.text.contains("MAGI_API_KEY")
                        && n.text.contains("[magi].kind")
                        && !n.text.contains(key),
                    "{}",
                    n.text
                );
            }
            if kind != ProviderKind::Ollama {
                assert_eq!(
                    k.magi().value(),
                    Some(key),
                    "never blocks: the key is still used"
                );
            }
        }
    }

    #[test]
    fn no_magi_key_never_warns_about_its_shape() {
        for kind in [
            ProviderKind::OpenAiCompat,
            ProviderKind::Anthropic,
            ProviderKind::Ollama,
        ] {
            let k = resolve(
                &env(
                    Some("sk-ant-toy-0001"), // allow-secret-scan
                    Some("sk-ant-toy-0002"), // allow-secret-scan
                    None,
                    None,
                    None,
                ),
                kind,
                Some(ROOT),
                None,
                None,
            );
            assert!(
                k.startup_notices()
                    .iter()
                    .all(|n| n.level != tracing::Level::WARN),
                "{kind:?}"
            );
        }
    }

    #[test]
    fn after_login_without_an_own_key_the_login_key_is_used() {
        assert_eq!(
            post_login_trio_key(None, "login-key".to_string()),
            "login-key"
        ); // allow-secret-scan
        assert_eq!(
            post_login_trio_key(Some("   "), "login-key".to_string()),
            "login-key"
        ); // allow-secret-scan
    }

    #[test]
    fn a_retained_key_becomes_one_file_only_notice_naming_the_variable_the_origin_and_the_command()
    {
        let e = env(Some("prin-secret"), None, None, None, None);
        let k = resolve(
            &e,
            ProviderKind::OpenAiCompat,
            Some("http://127.0.0.1:11434"),
            Some(ROOT),
            None,
        );
        let (screen, file) = crate::notices::partition_by_mouth(k.startup_notices());
        assert!(
            screen.is_empty(),
            "nothing reaches the TUI or stderr: {screen:?}"
        );
        assert_eq!(file.len(), 1);
        let text = &file[0].text;
        assert!(
            text.contains("OPENAI_API_KEY")
                && text.contains("http://127.0.0.1:11434")
                && text.contains("magi-rs vault set MAGI_API_KEY"),
            "{text}"
        );
        assert!(!text.contains("prin-secret"), "{text}");
    }

    #[test]
    fn three_retained_sections_give_three_notices_in_section_order() {
        let e = env(Some("prin"), None, None, None, None);
        let k = resolve(
            &e,
            ProviderKind::OpenAiCompat,
            Some("http://127.0.0.1:1"),
            Some("http://127.0.0.1:2"),
            Some("http://127.0.0.1:3"),
        );
        let texts: Vec<String> = k.startup_notices().into_iter().map(|n| n.text).collect();
        assert_eq!(texts.len(), 3);
        for (t, var) in texts
            .iter()
            .zip(["MAGI_API_KEY", "EMBEDDING_API_KEY", "DECISION_API_KEY"])
        {
            assert!(t.contains(var), "{t}");
        }
    }

    #[test]
    fn no_retained_key_gives_no_notice() {
        for (e, kind) in [
            (
                env(Some("prin"), None, None, None, None),
                ProviderKind::OpenAiCompat,
            ),
            (
                env(Some("prin"), Some("ak"), None, None, None),
                ProviderKind::Anthropic,
            ),
            (
                env(Some("prin"), None, None, None, None),
                ProviderKind::Ollama,
            ),
            (
                env(Some(" "), None, None, None, None),
                ProviderKind::OpenAiCompat,
            ),
        ] {
            let target =
                if kind == ProviderKind::OpenAiCompat && e.openai_key.as_deref() == Some("prin") {
                    ROOT
                } else {
                    "http://127.0.0.1:1"
                };
            let k = resolve(&e, kind, Some(target), Some(ROOT), Some(ROOT));
            assert!(k.startup_notices().is_empty(), "{kind:?}");
        }
    }

    /// RF4.
    #[test]
    fn a_retained_origin_never_carries_credentials() {
        let e = env(Some("prin"), None, None, None, None);
        let k = resolve(
            &e,
            ProviderKind::OpenAiCompat,
            Some("http://alice:pw-canary@127.0.0.1:11434/v1"), // allow-secret-scan
            None,
            None,
        );
        let text = k
            .startup_notices()
            .into_iter()
            .map(|n| n.text)
            .collect::<String>();
        assert!(
            text.contains("127.0.0.1:11434")
                && !text.contains("pw-canary")
                && !text.contains("alice"),
            "{text}"
        );
    }
}
