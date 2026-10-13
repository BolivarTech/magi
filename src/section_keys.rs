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

/// The vault as every surface shares it (moved from `src/main.rs`).
pub type SharedSecretStore = std::sync::Arc<std::sync::Mutex<dyn crate::vault::SecretStore + Send>>;

/// A secret found by [`env_or_vault`], and where it came from.
#[derive(Clone, PartialEq, Eq)]
pub struct ResolvedSecret {
    /// The trimmed, non-blank secret.
    pub value: String,
    /// `true` when the value came from the consumed environment, `false` from the vault.
    pub from_env: bool,
}

impl std::fmt::Debug for ResolvedSecret {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

/// The consumed env value if non-blank (trimmed), else the vault entry `name` if non-blank (trimmed).
pub fn env_or_vault(
    _name: &str,
    _env_value: Option<&str>,
    _secret_store: Option<&SharedSecretStore>,
) -> Option<ResolvedSecret> {
    todo!()
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
}
