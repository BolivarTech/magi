// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! Test doubles shared by the `review` module tests.

use std::collections::BTreeMap;

use super::error::ReviewError;
use super::source::GitCommand;

/// Answers git by the exact joined argument list; anything else is an error naming it.
pub(crate) struct FakeGit(pub(crate) BTreeMap<String, Result<String, String>>);

impl FakeGit {
    pub(crate) fn from(pairs: &[(&str, Result<&str, &str>)]) -> Self {
        Self(
            pairs
                .iter()
                .map(|(k, v)| {
                    (
                        k.to_string(),
                        (*v).map(str::to_string).map_err(str::to_string),
                    )
                })
                .collect(),
        )
    }
}

impl GitCommand for FakeGit {
    fn run_bytes(&self, args: &[&str]) -> Result<Vec<u8>, ReviewError> {
        self.0
            .get(&args.join(" "))
            .cloned()
            .unwrap_or_else(|| Err(format!("unexpected: {}", args.join(" "))))
            .map(String::into_bytes)
            .map_err(ReviewError::Git)
    }
}
