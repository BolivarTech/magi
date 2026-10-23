// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! Test doubles shared by the `review` module tests.

use std::collections::BTreeMap;

use super::error::ReviewError;
use super::source::{ContentSource, GitCommand};

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

/// An in-memory content source: `new` side and `old` side by path.
pub(crate) struct MapSource {
    new: BTreeMap<String, Vec<u8>>,
    old: BTreeMap<String, Vec<u8>>,
}

impl MapSource {
    pub(crate) fn new(new: &[(&str, &str)], old: &[(&str, &str)]) -> Self {
        let m = |xs: &[(&str, &str)]| {
            xs.iter()
                .map(|(p, t)| (p.to_string(), t.as_bytes().to_vec()))
                .collect()
        };
        Self {
            new: m(new),
            old: m(old),
        }
    }

    pub(crate) fn with_bytes(new: &[(&str, &[u8])]) -> Self {
        Self {
            new: new
                .iter()
                .map(|(p, b)| (p.to_string(), b.to_vec()))
                .collect(),
            old: BTreeMap::new(),
        }
    }
}

impl ContentSource for MapSource {
    fn read(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
        Ok(self.new.get(path).cloned())
    }

    fn read_old(&self, path: &str) -> Result<Option<Vec<u8>>, ReviewError> {
        Ok(self.old.get(path).cloned())
    }
}
