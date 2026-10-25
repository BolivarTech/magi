// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! Building blocks of the `code-review` pipeline.
//!
//! The hunk parser ([`hunk`]), the file context ([`context`]) and the language extractors
//! ([`languages`]) are integrated from Panoptico 1.1.1 (commit `efd387f`), written by the same
//! author and published under the same `MIT OR Apache-2.0` licence. Only the parts the pipeline
//! consumes were brought over, and they were adapted to this crate's rules.
//!
//! # No panics
//!
//! Release builds are compiled with `panic = "abort"`, so a panic on hostile input would end the
//! process with no report. This module therefore denies indexing, string slicing, `unwrap`,
//! `expect`, `panic!` and `unreachable!` in production code; out-of-range reads go through
//! `.get()` and degrade to fewer units instead of aborting.
#![cfg_attr(
    not(test),
    deny(
        clippy::indexing_slicing,
        clippy::string_slice,
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable
    )
)]

pub mod batching;
pub mod clock;
pub mod config;
pub mod consolidate;
pub mod context;
pub mod error;
pub mod filter;
pub mod gate;
pub mod hunk;
pub mod index;
pub mod input;
pub mod languages;
pub mod pipeline;
pub mod report;
pub mod secrets;
pub mod source;
pub mod validate;
pub mod window;

#[cfg(test)]
mod test_support;

#[cfg(test)]
mod tests {
    #[test]
    fn production_code_in_review_has_no_todo_or_unimplemented() {
        // Release is panic=abort: a stub a Green leaves behind aborts with no message.
        fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            for entry in std::fs::read_dir(dir).expect("reads src/review") {
                let p = entry.expect("entry").path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    out.push(p);
                }
            }
        }
        let mut files = Vec::new();
        walk(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("src")
                .join("review"),
            &mut files,
        );
        assert!(files.len() >= 4, "walked {files:?}");
        let needles = [concat!("todo", "!("), concat!("unimplemented", "!(")];
        for f in files.iter().filter(|f| !f.ends_with("test_support.rs")) {
            let text = std::fs::read_to_string(f).expect("reads").replace('\r', "");
            let lines: Vec<&str> = text.lines().collect();
            // Production is everything before the first `#[cfg(test)]` that opens a test
            // module (`mod x {`). A net behind the `cfg_attr(not(test), deny(..))` lints, which
            // enforce: it assumes braced, terminal test modules; a `mod x;` test module or
            // production code below the tests escapes it.
            let cut = lines
                .windows(2)
                .position(|w| {
                    w[0].trim() == "#[cfg(test)]" && w[1].starts_with("mod ") && w[1].ends_with('{')
                })
                .unwrap_or(lines.len());
            // A `#[cfg(test)]` the cut did not recognise fails loudly instead of narrowing the
            // net.
            let has_cfg_test = lines.iter().any(|l| l.trim() == "#[cfg(test)]");
            assert!(
                !has_cfg_test || cut < lines.len(),
                "{}: #[cfg(test)] without a braced test module",
                f.display()
            );
            let production = lines[..cut].join("\n");
            for n in needles {
                assert!(
                    !production.contains(n),
                    "{} has {n} in production code",
                    f.display()
                );
            }
            // An `allow` of a lint the module denies would silence it in production.
            for lint in [
                "indexing_slicing",
                "string_slice",
                "unwrap_used",
                "expect_used",
                "panic",
                "unreachable",
            ] {
                let allow = format!("allow(clippy::{lint}");
                assert!(
                    !production.contains(&allow),
                    "{} allows clippy::{lint} in production code",
                    f.display()
                );
            }
        }
    }
}
