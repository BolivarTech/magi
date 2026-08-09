// Author: Julian Bolivar Version: 1.0.0 Date: 2026-08-03

//! Mode classifier over the PRIMARY provider (REQ-A07c).
//!
//! Lives in the **bin**, not in `src/magi/mode.rs` (lib): [`ProviderClassifier`] needs `Arc<dyn
//! Provider>`, and `agent::provider::Provider` is a type of the binary that the lib cannot see
//! (see the crate split table in `CLAUDE.md`). The pure trait
//! [`magi_rs::magi::mode::ModeClassifier`] that this implements, on the other hand, does live
//! in the lib — it has no I/O, so it does not have the same restriction.

#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(clippy::missing_errors_doc, clippy::missing_panics_doc)]
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented,
        clippy::indexing_slicing,
        clippy::string_slice
    )
)]

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use async_trait::async_trait;
use magi_core::schema::Mode;
use magi_rs::magi::mode::{normalize_label, ModeClassifier};
use magi_rs::magi::CLASSIFY_TIMEOUT_SECS;

use crate::agent::messages::Message;
use crate::agent::provider::Provider;

/// Delimiters for the untrusted content in the classification prompt.
///
/// The content is **delimited** and the prompt declares that what is inside is data to
/// classify, never instructions. It is hygiene: the real containment is [`normalize_label`],
/// because it does not depend on the model behaving well.
const CONTENT_OPEN: &str = "<<<CONTENIDO_A_CLASIFICAR>>>";
/// See [`CONTENT_OPEN`].
const CONTENT_CLOSE: &str = "<<<FIN_CONTENIDO>>>";

/// Key of the COST notice: without `--mode` or `default_mode`, a model call is added. See
/// [`NoticeSink::once`].
const NOTICE_CLASSIFY_COST: &str = "classify.cost";
/// Key of the EXPIRATION notice: the classification expired or failed. Different from the
/// previous one — this one warns that something FAILED, not that something IS GOING TO HAPPEN.
const NOTICE_CLASSIFY_TIMEOUT: &str = "classify.timeout";

/// Emitter of one-time notices, **injectable**.
///
/// **Resolves a real tension between two rules, and therefore is neither a field nor
/// a `static`.**
///
/// The spec demands *"once per process"*: an `AtomicBool` as a field of the classifier
/// satisfies that only if exactly one classifier exists — true today, **not a contract**. But a
/// `static` satisfies the semantics and **breaks B13** («isolated tests, no shared state»): the
/// order of the tests would end up deciding which one sees the notice.
pub trait NoticeSink: Send + Sync {
    /// The sink satisfies both: **one process-level shared instance** in production (for
    /// headless `magi consult`, one process IS one run, so building a [`ProcessNoticeSink`] per
    /// invocation already satisfies "once per process"), **one fresh instance per test** in the
    /// suite. The semantics of "once" live in the sink, not in whoever uses it.
    fn once(&self, key: &'static str, msg: &str);
}

/// Emits `msg` the first time it is called with `key`; subsequent calls are no-ops for that
/// `key`.
#[derive(Default)]
pub struct ProcessNoticeSink {
    /// Production sink: writes to stderr, deduplicating by key.
    seen: Mutex<BTreeSet<&'static str>>,
}

impl NoticeSink for ProcessNoticeSink {
    fn once(&self, key: &'static str, msg: &str) {
        let mut seen = self.seen.lock().unwrap_or_else(PoisonError::into_inner);
        if seen.insert(key) {
            eprintln!("{msg}");
        }
    }
}

/// Keys already emitted, for dedup.
///
/// Classifier over the PRIMARY provider (REQ-A07c).
pub struct ProviderClassifier {
    /// Uses the primary one and not the trio: it is a classification of a label, not a
    /// deliberation; paying for it at the price of three mages would be absurd.
    provider: Arc<dyn Provider>,
    /// The primary provider already resolved (same one that serves the tool loop).
    notices: Arc<dyn NoticeSink>,
}

impl ProviderClassifier {
    /// Emitter of the two one-time notices of this module.
    #[must_use]
    pub fn new(provider: Arc<dyn Provider>, notices: Arc<dyn NoticeSink>) -> Self {
        Self { provider, notices }
    }
}

#[async_trait]
impl ModeClassifier for ProviderClassifier {
    async fn classify(&self, content: &str) -> Option<Mode> {
        self.notices.once(
            NOTICE_CLASSIFY_COST,
            "notice: without `--mode` or `[magi].default_mode`, magi-rs adds a call to the \
             model to infer the lens. Declaring the mode avoids it.",
        );

        let prompt = format!(
            "Classify the delimited content into exactly one of these labels: \
             code-review, design, analysis. Respond with ONLY the label.\n\
             {CONTENT_OPEN}\n{content}\n{CONTENT_CLOSE}"
        );
        let msgs = [Message::user(&prompt)];
        let deadline = Duration::from_secs(CLASSIFY_TIMEOUT_SECS);

        match tokio::time::timeout(deadline, self.provider.send_messages(&msgs, &[], None)).await {
            Ok(Ok(reply)) => normalize_label(&reply.concat_text()),
            Ok(Err(_)) | Err(_) => {
                self.notices.once(
                    NOTICE_CLASSIFY_TIMEOUT,
                    &format!(
                        "notice: mode inference expired ({CLASSIFY_TIMEOUT_SECS}s) or failed; \
                         using `analysis`. On slow providers, declare \
                         `[magi].default_mode`."
                    ),
                );
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Mutex;

    use anyhow::Result;
    use futures::stream::{self, BoxStream, StreamExt};

    use crate::agent::provider::ResponseChunk;
    use crate::tools::Tool;

    use super::*;

    /// Creates a classifier over `provider`, emitting its notices through `notices`.
    struct DelayedProvider {
        /// Double of [`Provider`] that waits `delay` and then responds with `text`. The `sleep`
        /// lives INSIDE `stream_messages`, so the `tokio::time::timeout` from
        /// [`ProviderClassifier::classify`] catches it just as it would catch a real network
        /// latency.
        delay: Duration,
        /// How long it waits before responding.
        text: String,
    }

    #[async_trait]
    impl Provider for DelayedProvider {
        async fn stream_messages(
            &self,
            _messages: &[Message],
            _tools: &[Box<dyn Tool>],
            _system: Option<&str>,
        ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
            tokio::time::sleep(self.delay).await;
            let msg = Message::assistant(&self.text);
            Ok(stream::iter(vec![Ok(ResponseChunk::MessageDone(msg))]).boxed())
        }
    }

    /// What it responds with, once the delay has passed.
    fn slow_provider(delay: Duration) -> Arc<dyn Provider> {
        Arc::new(DelayedProvider {
            delay,
            text: "design".to_string(),
        })
    }

    /// A provider that never responds in time for [`CLASSIFY_TIMEOUT_SECS`].
    fn provider_returning(label: &str) -> Arc<dyn Provider> {
        Arc::new(DelayedProvider {
            delay: Duration::ZERO,
            text: label.to_string(),
        })
    }

    /// A provider that responds `label` immediately.
    #[derive(Default)]
    struct RecordingNoticeSink {
        /// Test sink: accumulates in memory, without touching stderr or global state — which
        /// keeps B13 («isolated tests, no shared state») while the semantics of "once per
        /// process" remain alive in production.
        seen: Mutex<BTreeSet<&'static str>>,
        /// Keys already seen, for dedup (same as `ProcessNoticeSink`).
        messages: Mutex<Vec<String>>,
    }

    impl NoticeSink for RecordingNoticeSink {
        fn once(&self, key: &'static str, msg: &str) {
            let mut seen = self.seen.lock().unwrap_or_else(PoisonError::into_inner);
            if seen.insert(key) {
                self.messages
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(msg.to_string());
            }
        }
    }

    impl RecordingNoticeSink {
        /// The messages that did get emitted, in order.
        fn count_matching(&self, needle: &str) -> usize {
            self.messages
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .iter()
                .filter(|m| m.contains(needle))
                .count()
        }
    }

    /// SC-A07m: a provider slower than [`CLASSIFY_TIMEOUT_SECS`] never makes `classify` inherit
    /// its delay. Paused clock (loop 1 fix round CE, F14): the other four timing tests in this
    /// file already use `start_paused = true`; this one used a real 8-second
    /// `tokio::time::sleep` plus a wall-clock upper bound, reproducing the load-flakiness
    /// pattern `CLAUDE.local.md` documents twice — under concurrent `cargo nextest run`, the
    /// internal `tokio::time::timeout` firing late enough to blow the margin is a plausible
    /// flake, not a hypothetical one.
    ///
    /// **No elapsed-time assertion is needed under a manually-driven clock**, and adding one
    /// back would be the wrong kind of assertion: with the clock paused, "elapsed" is whatever
    /// we choose to `advance()` past, not a measurement of anything the code did. The property
    /// this test actually needs — the provider's delay is never inherited — is asserted
    /// structurally instead, the same way `the_two_notices_fire_once_each` right below proves
    /// it: the advance is `CLASSIFY_TIMEOUT_SECS + 1`, strictly between the classifier's own
    /// ceiling and the `+2` provider delay. If `classify` ever started waiting on the provider's
    /// full delay instead of its own, `handle.await` would have nothing left to resolve it and
    /// the test would hang rather than reach the assertion below.
    #[tokio::test(start_paused = true)]
    async fn a_slow_provider_degrades_every_inference_to_default() {
        let classifier = ProviderClassifier::new(
            slow_provider(Duration::from_secs(CLASSIFY_TIMEOUT_SECS + 2)),
            Arc::new(ProcessNoticeSink::default()),
        );

        let handle = tokio::spawn(async move { classifier.classify("x").await });
        tokio::time::advance(Duration::from_secs(CLASSIFY_TIMEOUT_SECS + 1)).await;
        let inferred = handle.await.expect("the classify task must not panic");

        assert_eq!(
            inferred, None,
            "a provider that does not respond in time fails open"
        );
    }

    /// SC-A07m: on a slow provider, classification degrades to `None` without exceeding its own
    /// ceiling — it never inherits the provider's delay.
    ///
    /// SC-A07n / SC-A07o: the two notices, each one ONCE — with an INJECTED sink.
    #[tokio::test(start_paused = true)]
    async fn the_two_notices_fire_once_each() {
        let sink = Arc::new(RecordingNoticeSink::default());
        let classifier = Arc::new(ProviderClassifier::new(
            slow_provider(Duration::from_secs(30)),
            sink.clone(),
        ));

        for _ in 0..3 {
            let classifier = Arc::clone(&classifier);
            let handle = tokio::spawn(async move { classifier.classify("x").await });
            tokio::time::advance(Duration::from_secs(CLASSIFY_TIMEOUT_SECS + 1)).await;
            let _ = handle.await.expect("the classify task must not panic");
        }

        assert_eq!(
            sink.count_matching("adds a call to the model"),
            1,
            "the COST notice arrives before paying for it, once"
        );
        assert_eq!(
            sink.count_matching("expired"),
            1,
            "the EXPIRY notice is different: it warns that it failed, not that it is \
             about to happen"
        );
    }

    /// PAUSED clock (m2, review Task 2.3): the 30s provider never runs in real time — each
    /// `classify` is launched in its own task and `tokio::time::advance` jumps straight past
    /// the ceiling of [`CLASSIFY_TIMEOUT_SECS`], which is the only thing this test needs to
    /// observe. With a real clock this cost ~18s (3 calls × 6s) on every run of the suite.
    ///
    /// SC-A07o: the COST notice fires even if classification WORKS.
    #[tokio::test]
    async fn the_cost_notice_fires_even_when_classification_succeeds() {
        let sink = Arc::new(RecordingNoticeSink::default());
        let classifier = ProviderClassifier::new(provider_returning("code-review"), sink.clone());

        assert_eq!(
            classifier.classify("x").await,
            Some(Mode::CodeReview),
            "classified correctly"
        );
        assert_eq!(sink.count_matching("adds a call to the model"), 1);
        assert_eq!(
            sink.count_matching("expired"),
            0,
            "there was no expiry to report"
        );
    }

    /// The previous test exercises it with a classification that fails, so it cannot
    /// distinguish the 'cost notice' from the 'failure notice'. This one confirms that the
    /// previous notice is independent of the result: the call is paid for either way, and that
    /// is what it warns about.
    #[tokio::test]
    async fn an_unrecognized_reply_yields_none_without_a_timeout_notice() {
        let sink = Arc::new(RecordingNoticeSink::default());
        let classifier = ProviderClassifier::new(
            provider_returning("el modo apropiado seria code-review"),
            sink.clone(),
        );

        assert_eq!(classifier.classify("x").await, None, "prose is not a label");
        assert_eq!(
            sink.count_matching("expired"),
            0,
            "an unrecognized reply is not an expiry"
        );
    }

    /// Edge case: the provider responds on time but with something that is not a label (prose,
    /// JSON, made-up label). It falls to `None` just like a real failure, but WITHOUT the
    /// expiration notice — the call neither expired nor failed; it simply did not name a valid
    /// mode (REQ-A07j).
    ///
    /// The isolation is real: two tests do not contaminate each other even if they run in any
    /// order (B13).
    #[tokio::test(start_paused = true)]
    async fn two_independent_sinks_do_not_share_state() {
        let a = Arc::new(RecordingNoticeSink::default());
        let b = Arc::new(RecordingNoticeSink::default());

        let handle_a = tokio::spawn(async move {
            ProviderClassifier::new(slow_provider(Duration::from_secs(30)), a.clone())
                .classify("x")
                .await;
            a
        });
        let handle_b = tokio::spawn(async move {
            ProviderClassifier::new(slow_provider(Duration::from_secs(30)), b.clone())
                .classify("x")
                .await;
            b
        });
        tokio::time::advance(Duration::from_secs(CLASSIFY_TIMEOUT_SECS + 1)).await;
        let a = handle_a.await.expect("classifier `a` task must not panic");
        let b = handle_b.await.expect("classifier `b` task must not panic");

        assert_eq!(a.count_matching("expired"), 1);
        assert_eq!(
            b.count_matching("expired"),
            1,
            "a `static` would leave this at 0"
        );
    }
}
