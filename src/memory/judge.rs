// Author: Julian Bolivar
// Version: 0.26.0
// Date: 2026-11-26

//! LLM-backed implementation of [`DistillJudge`].
//!
//! The real implementation calls the configured chat provider with bounded prompts.
//! Non-determinism is isolated here so tests can inject mock providers (R-06).

use async_trait::async_trait;
use std::sync::Arc;

use magi_rs::review::secrets::{outward_prompt_texts, withheld_whole, KEY_BLOCK_WITHHELD};

use crate::agent::messages::Message;
use crate::agent::provider::Provider;
use crate::memory::error::MemoryError;
use crate::memory::profile::DistillJudge;
use crate::memory::store::Memory;

/// Target of this module's run-log entries.
const LOG_TARGET: &str = "magi_rs::memory::judge";

/// [`DistillJudge`] backed by the configured chat provider (REQ-17, agnostic).
///
/// Bounded prompts only; API keys are never logged or serialised (REQ-21). Every memory text
/// goes through [`outward_prompt_texts`] before it is embedded in a prompt: one-line secrets are
/// masked, and a text is withheld whole in two cases. One is a prompt whose memories, joined,
/// hold a private-key block: every memory of it is withheld, since any of them may carry that
/// block's body. The other is a text whose masking still leaves something the egress check
/// flags. The main model is a cloud one by default, and this judge is also the fallback of the
/// `decide` contradiction judge. A placeholder is never judged: a pair with a text withheld
/// whole is kept without a call, and a preference prompt whose texts are all withheld is not
/// sent; each skip writes one run-log entry, with no content.
/// Provider errors are propagated as typed [`MemoryError`] so the distiller's
/// CP2-Z non-fatal handler catches them without panic.
pub struct LlmDistillJudge {
    provider: Arc<dyn Provider>,
}

impl LlmDistillJudge {
    /// Creates a new judge wrapping the given provider.
    pub fn new(provider: Arc<dyn Provider>) -> Self {
        Self { provider }
    }
}

#[async_trait]
impl DistillJudge for LlmDistillJudge {
    async fn summarize_preferences(&self, episodic: &[Memory]) -> Result<Vec<String>, MemoryError> {
        if episodic.is_empty() {
            return Ok(vec![]);
        }

        let texts = episodic.iter().map(|m| m.text.as_str()).collect::<Vec<_>>();
        let texts = outward_prompt_texts(&texts);
        if texts.iter().all(|text| withheld_whole(text)) {
            tracing::event!(
                target: LOG_TARGET,
                tracing::Level::INFO,
                "preference distillation skipped: every memory text was withheld for a secret"
            );
            return Ok(vec![]);
        }
        let texts = texts.join("\n");

        let prompt = format!(
            "From the following user turns, list durable user preferences, one per line, \
             or nothing if none:\n{texts}"
        );

        let response = self
            .provider
            .send_messages(&[Message::user(&prompt)], &[], None)
            .await
            .map_err(|e| MemoryError::Llm(e.to_string()))?;

        let text = response
            .content
            .iter()
            .filter_map(|c| {
                if let crate::agent::messages::Content::Text { text } = c {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        let prefs = text
            .lines()
            .map(|line| {
                // Strip common list prefixes: "- ", "* ", "1. ", "2. " …
                let stripped = line
                    .trim_start_matches(|c: char| c.is_ascii_digit())
                    .trim_start_matches('.')
                    .trim_start_matches(['-', '*'])
                    .trim();
                stripped.to_string()
            })
            .filter(|s| !s.is_empty())
            .collect();

        Ok(prefs)
    }

    async fn contradicts(&self, a: &str, b: &str) -> Result<bool, MemoryError> {
        // One outward text per input text, in order; both are withheld when the pair holds a
        // key block. A missing element (never, by its contract) is withheld, not sent empty.
        let mut pair = outward_prompt_texts(&[a, b]).into_iter();
        let mut next = || {
            pair.next()
                .unwrap_or_else(|| KEY_BLOCK_WITHHELD.to_string())
        };
        let (a, b) = (next(), next());
        if withheld_whole(&a) || withheld_whole(&b) {
            // Nothing left to judge: the conservative answer keeps both memories.
            tracing::event!(
                target: LOG_TARGET,
                tracing::Level::INFO,
                "contradiction check skipped: a memory text was withheld for a secret"
            );
            return Ok(false);
        }
        let prompt = format!(
            "Does statement B contradict or supersede statement A about the same subject? \
             Answer only yes or no.\nA: {a}\nB: {b}"
        );

        let response = self
            .provider
            .send_messages(&[Message::user(&prompt)], &[], None)
            .await
            .map_err(|e| MemoryError::Llm(e.to_string()))?;

        let text = response
            .content
            .iter()
            .filter_map(|c| {
                if let crate::agent::messages::Content::Text { text } = c {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        // F6a: match "yes" as a whole word at the start of the response to avoid
        // false-positive supersession.  `contains("yes")` would match words like
        // "yesterday" or "expressly", turning non-affirmative responses into
        // spurious supersession decisions.  Find the first alphabetic token and
        // check that it is exactly "yes" (case-insensitive).
        let first_word = text
            .split(|c: char| !c.is_alphabetic())
            .find(|s| !s.is_empty());
        Ok(first_word
            .map(|w| w.to_lowercase() == "yes")
            .unwrap_or(false))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use async_trait::async_trait;
    use futures::stream::{self, BoxStream};
    use std::sync::Arc;

    use crate::agent::messages::Message;
    use crate::agent::provider::ResponseChunk;
    use crate::memory::error::MemoryError;
    use crate::memory::store::Memory;
    use crate::memory::MemoryKind;

    // ── CannedProvider ───────────────────────────────────────────────────────

    /// A provider that always returns a fixed canned response from both
    /// `stream_messages` and `send_messages`. Used to test `LlmDistillJudge`
    /// in isolation without HTTP calls (R-06).
    struct CannedProvider(String);

    #[async_trait]
    impl crate::agent::provider::Provider for CannedProvider {
        async fn stream_messages(
            &self,
            _messages: &[Message],
            _tools: &[Box<dyn crate::tools::Tool>],
            _system: Option<&str>,
        ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
            let text = self.0.clone();
            Ok(Box::pin(stream::iter(vec![
                Ok(ResponseChunk::TextDelta(text.clone())),
                Ok(ResponseChunk::MessageDone(Message::assistant(&text))),
            ])))
        }

        async fn send_messages(
            &self,
            _messages: &[Message],
            _tools: &[Box<dyn crate::tools::Tool>],
            _system: Option<&str>,
        ) -> Result<Message> {
            Ok(Message::assistant(&self.0))
        }
    }

    /// A provider that always returns `Err` (for error-path testing).
    struct FailingProvider;

    #[async_trait]
    impl crate::agent::provider::Provider for FailingProvider {
        async fn stream_messages(
            &self,
            _messages: &[Message],
            _tools: &[Box<dyn crate::tools::Tool>],
            _system: Option<&str>,
        ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
            Err(anyhow::anyhow!("provider failed"))
        }

        async fn send_messages(
            &self,
            _messages: &[Message],
            _tools: &[Box<dyn crate::tools::Tool>],
            _system: Option<&str>,
        ) -> Result<Message> {
            Err(anyhow::anyhow!("provider failed"))
        }
    }

    fn make_episodic(text: &str) -> Memory {
        Memory {
            id: "test-id".into(),
            session_id: "s".into(),
            kind: MemoryKind::Episodic,
            text: text.into(),
            embedding: vec![],
            model_id: String::new(),
            dim: 0,
            created_at: 1_000,
            salience: 0.3,
            access_count: 0,
            last_accessed_at: 1_000,
            superseded_by: None,
            evicted_at: None,
            scope: "root".into(),
            distilled_at: None,
        }
    }

    /// `summarize_preferences` parses the provider's response into trimmed,
    /// de-bulleted preference strings, one per line.
    #[tokio::test]
    async fn test_judge_parses_preferences_one_per_line() {
        let provider = Arc::new(CannedProvider("- use rust\n- dark mode".into()));
        let judge = LlmDistillJudge::new(provider);
        let episodic = vec![make_episodic("some user turn")];

        let result = judge.summarize_preferences(&episodic).await.unwrap();
        assert_eq!(
            result,
            vec!["use rust".to_string(), "dark mode".to_string()],
            "summarize_preferences must parse the provider's response into \
             trimmed, de-bulleted preference strings"
        );
    }

    /// `contradicts` returns `true` when the provider says "Yes" (case-insensitive)
    /// and `false` when it says "No".
    #[tokio::test]
    async fn test_judge_contradicts_parses_yes() {
        let yes_judge = LlmDistillJudge::new(Arc::new(CannedProvider("Yes.".into())));
        assert!(
            yes_judge
                .contradicts("statement A", "statement B")
                .await
                .unwrap(),
            "a 'Yes.' response must be parsed as true"
        );

        let no_judge = LlmDistillJudge::new(Arc::new(CannedProvider("No".into())));
        assert!(
            !no_judge
                .contradicts("statement A", "statement B")
                .await
                .unwrap(),
            "a 'No' response must be parsed as false"
        );
    }

    /// F6a: "yes" must be matched as a whole word at the start of the response.
    /// Words that CONTAIN "yes" as a substring (e.g. "Yesterday", "expressly")
    /// must return `false` to avoid false-positive supersession.
    /// Before fix: `contains("yes")` returns `true` for "Yesterday I..." → FAIL.
    #[tokio::test]
    async fn test_judge_contradicts_rejects_yes_as_substring() {
        // "Yesterday" starts with "yes" but is NOT a "yes" answer.
        let judge = LlmDistillJudge::new(Arc::new(CannedProvider(
            "Yesterday I noticed a conflict".into(),
        )));
        assert!(
            !judge
                .contradicts("statement A", "statement B")
                .await
                .unwrap(),
            "F6a: 'Yesterday…' must NOT be parsed as a yes answer (false-positive prevention)"
        );

        // "Yes, statement B..." is a valid affirmative — must still be true.
        let judge2 = LlmDistillJudge::new(Arc::new(CannedProvider(
            "Yes, statement B supersedes A".into(),
        )));
        assert!(
            judge2
                .contradicts("statement A", "statement B")
                .await
                .unwrap(),
            "F6a: 'Yes, …' (comma after yes) must still parse as true"
        );

        // "expressly no" contains "yes" as substring — must be false.
        let judge3 = LlmDistillJudge::new(Arc::new(CannedProvider("expressly no".into())));
        assert!(
            !judge3
                .contradicts("statement A", "statement B")
                .await
                .unwrap(),
            "F6a: 'expressly no' must NOT match (contains 'yes' but is not a yes answer)"
        );
    }

    /// A provider error (e.g. network/connection failure) is propagated as
    /// `MemoryError::Llm`, NOT `MemoryError::Storage` (CP2-Z: non-fatal).
    /// Correct categorisation lets the distiller log "LLM provider error" instead
    /// of the misleading "storage error" when the OpenAI-compatible backend is
    /// unreachable.
    #[tokio::test]
    async fn test_judge_provider_error_returns_memory_error_llm_not_storage() {
        let judge = LlmDistillJudge::new(Arc::new(FailingProvider));
        let episodic = vec![make_episodic("some text")];

        let result = judge.summarize_preferences(&episodic).await;
        assert!(
            result.is_err(),
            "a failing provider must produce Err, not panic"
        );
        assert!(
            matches!(result.unwrap_err(), MemoryError::Llm(_)),
            "a provider/network failure must map to MemoryError::Llm, not Storage"
        );
    }

    /// `contradicts` also maps provider errors to `MemoryError::Llm`.
    #[tokio::test]
    async fn test_judge_contradicts_provider_error_returns_memory_error_llm() {
        let judge = LlmDistillJudge::new(Arc::new(FailingProvider));

        let result = judge.contradicts("A", "B").await;
        assert!(
            result.is_err(),
            "a failing provider must produce Err, not panic"
        );
        assert!(
            matches!(result.unwrap_err(), MemoryError::Llm(_)),
            "a provider/network failure in contradicts must map to MemoryError::Llm, not Storage"
        );
    }

    /// Double of `Provider` that records the text of every prompt it is sent and answers
    /// `reply`, so a test can read what would have reached the main model.
    struct PromptRecorder {
        /// What it answers.
        reply: String,
        /// The concatenated text of every message sent, in order.
        prompts: std::sync::Mutex<Vec<String>>,
    }

    impl PromptRecorder {
        /// A recorder answering `reply`.
        fn answering(reply: &str) -> Arc<Self> {
            Arc::new(Self {
                reply: reply.to_string(),
                prompts: std::sync::Mutex::new(Vec::new()),
            })
        }

        /// How many prompts it received.
        fn calls(&self) -> usize {
            self.prompts
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .len()
        }

        /// The one prompt it received.
        fn only_prompt(&self) -> String {
            let prompts = self
                .prompts
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            assert_eq!(prompts.len(), 1, "one judge call: {prompts:?}");
            prompts.into_iter().next().unwrap_or_default()
        }
    }

    #[async_trait]
    impl crate::agent::provider::Provider for PromptRecorder {
        async fn stream_messages(
            &self,
            messages: &[Message],
            _tools: &[Box<dyn crate::tools::Tool>],
            _system: Option<&str>,
        ) -> Result<BoxStream<'static, Result<ResponseChunk>>> {
            self.prompts
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .extend(messages.iter().map(Message::concat_text));
            let msg = Message::assistant(&self.reply);
            Ok(Box::pin(stream::iter(vec![Ok(
                ResponseChunk::MessageDone(msg),
            )])))
        }
    }

    /// A secret-shaped token for the judge prompts.
    fn token() -> String {
        format!("ghp_{}", "A".repeat(36)) // allow-secret-scan
    }

    /// A private-key block: its header, one body line and its footer.
    fn key_block(body: &str) -> String {
        let d = "-----";
        format!("{d}BEGIN RSA PRIVATE KEY{d}\n{body}\n{d}END RSA PRIVATE KEY{d}")
    }

    /// The contradiction prompt carries both memory texts to the main model, a cloud one by
    /// default: a secret token in either is masked before it is embedded.
    /// Mutation-verified: embedding `a` and `b` raw turns this red.
    #[tokio::test]
    async fn a_secret_token_in_a_memory_pair_never_reaches_the_contradiction_prompt() {
        let recorder = PromptRecorder::answering("no");
        let judge = LlmDistillJudge::new(recorder.clone());
        let a = format!("my key is {}", token());
        let b = format!("the token {} rotated", token());
        judge.contradicts(&a, &b).await.unwrap();
        let prompt = recorder.only_prompt();
        assert!(!prompt.contains(&token()), "{prompt:?}");
        assert!(
            !magi_rs::review::secrets::leaks_secret(&prompt),
            "{prompt:?}"
        );
        assert!(prompt.contains("<redacted:"), "{prompt:?}");
        assert!(prompt.contains("\nA: my key is "), "{prompt:?}");
        assert!(prompt.contains("\nB: the token "), "{prompt:?}");
    }

    /// A memory holding a private-key block is withheld whole, and so is the other memory of
    /// the pair, which may hold part of that block: nothing is left to judge, so no prompt is
    /// sent and the pair is kept (no contradiction).
    #[tokio::test]
    async fn a_private_key_block_in_a_memory_pair_never_reaches_the_contradiction_prompt() {
        let recorder = PromptRecorder::answering("yes");
        let judge = LlmDistillJudge::new(recorder.clone());
        let body = "MIIEowIBAAKCAQEAbodyline";
        let contradicts = judge
            .contradicts("plain statement", &format!("old\n{}", key_block(body)))
            .await
            .unwrap();
        assert!(!contradicts, "a withheld pair is kept");
        assert_eq!(recorder.calls(), 0, "no contradiction prompt is sent");
    }

    /// A key block whose header ends memory A and whose body opens memory B: the header
    /// withholds A, and B, which may carry the body, is withheld with it, so no prompt is sent.
    #[tokio::test]
    async fn a_key_block_split_across_the_memory_pair_never_reaches_the_contradiction_prompt() {
        let recorder = PromptRecorder::answering("yes");
        let judge = LlmDistillJudge::new(recorder.clone());
        let d = "-----";
        let body = "MIIEowIBAAKCAQEAbodyline";
        let contradicts = judge
            .contradicts(
                &format!("old\n{d}BEGIN RSA PRIVATE KEY{d}"),
                &format!("{body}\n{d}END RSA PRIVATE KEY{d}"),
            )
            .await
            .unwrap();
        assert!(!contradicts, "a withheld pair is kept");
        assert_eq!(recorder.calls(), 0, "no contradiction prompt is sent");
    }

    /// The same split across two episodic memories withholds every memory of the preference
    /// prompt, so no prompt is sent and no preference is distilled.
    #[tokio::test]
    async fn a_key_block_split_across_episodic_memories_never_reaches_the_preference_prompt() {
        let recorder = PromptRecorder::answering("- none");
        let judge = LlmDistillJudge::new(recorder.clone());
        let d = "-----";
        let body = "MIIEowIBAAKCAQEAbodyline";
        let episodic = vec![
            make_episodic("I like tabs"),
            make_episodic(&format!("{d}BEGIN RSA PRIVATE KEY{d}")),
            make_episodic(body),
        ];
        let prefs = judge.summarize_preferences(&episodic).await.unwrap();
        assert!(prefs.is_empty(), "{prefs:?}");
        assert_eq!(recorder.calls(), 0, "no preference prompt is sent");
    }

    /// A key body in memory A whose header and footer sit in memory B: the pair holds a block,
    /// so both are withheld and no contradiction prompt is sent.
    #[tokio::test]
    async fn a_key_body_before_its_header_never_reaches_the_contradiction_prompt() {
        let recorder = PromptRecorder::answering("yes");
        let judge = LlmDistillJudge::new(recorder.clone());
        let d = "-----";
        let body = "MIIEowIBAAKCAQEAbodyline";
        let contradicts = judge
            .contradicts(
                body,
                &format!("{d}BEGIN RSA PRIVATE KEY{d}\n{d}END RSA PRIVATE KEY{d}"),
            )
            .await
            .unwrap();
        assert!(!contradicts, "a withheld pair is kept");
        assert_eq!(recorder.calls(), 0, "no contradiction prompt is sent");
    }

    /// The same order across two episodic memories withholds both, so no preference prompt
    /// is sent.
    #[tokio::test]
    async fn a_key_body_before_its_header_never_reaches_the_preference_prompt() {
        let recorder = PromptRecorder::answering("- none");
        let judge = LlmDistillJudge::new(recorder.clone());
        let d = "-----";
        let body = "MIIEowIBAAKCAQEAbodyline";
        let episodic = vec![
            make_episodic(body),
            make_episodic(&format!(
                "{d}BEGIN RSA PRIVATE KEY{d}\n{d}END RSA PRIVATE KEY{d}"
            )),
        ];
        let prefs = judge.summarize_preferences(&episodic).await.unwrap();
        assert!(prefs.is_empty(), "{prefs:?}");
        assert_eq!(recorder.calls(), 0, "no preference prompt is sent");
    }

    /// The preference prompt carries episodic memory texts to the main model too: a token is
    /// masked there as well (a key block withholds every memory, so no prompt is sent; see
    /// the key-block tests above).
    #[tokio::test]
    async fn secrets_in_episodic_memories_never_reach_the_preference_prompt() {
        let recorder = PromptRecorder::answering("- none");
        let judge = LlmDistillJudge::new(recorder.clone());
        let episodic = vec![
            make_episodic(&format!("I use {} daily", token())),
            make_episodic("I like tabs"),
        ];
        judge.summarize_preferences(&episodic).await.unwrap();
        let prompt = recorder.only_prompt();
        assert!(!prompt.contains(&token()), "{prompt:?}");
        assert!(
            !magi_rs::review::secrets::leaks_secret(&prompt),
            "{prompt:?}"
        );
        assert!(prompt.contains("<redacted:"), "{prompt:?}");
        assert!(prompt.contains("I like tabs"), "{prompt:?}");
    }

    /// The run-log lines of `target` emitted while `f` runs.
    fn judge_log_lines<F: std::future::Future<Output = ()>>(f: F) -> Vec<String> {
        crate::agent::decide_classifier::target_lines("magi_rs::memory::judge", f)
    }

    /// Mutation-verified: a pair one of whose texts is withheld whole (here a private-key block
    /// in B, which withholds A with it) is not judged on placeholders: no call, the pair is
    /// kept, and one run-log entry says so without content.
    #[test]
    fn a_pair_with_one_text_withheld_whole_is_kept_without_a_call_and_logged() {
        let recorder = PromptRecorder::answering("yes");
        let body = "MIIEowIBAAKCAQEAbodyline";
        let lines = judge_log_lines({
            let recorder = recorder.clone();
            async move {
                let judge = LlmDistillJudge::new(recorder);
                let contradicts = judge
                    .contradicts("plain statement", &key_block(body))
                    .await
                    .unwrap();
                assert!(!contradicts, "a withheld pair is kept");
            }
        });
        assert_eq!(recorder.calls(), 0, "no contradiction prompt is sent");
        assert_eq!(lines.len(), 1, "{lines:?}");
        let line = lines.first().map(String::as_str).unwrap_or_default();
        assert!(line.contains("contradiction check skipped"), "{line}");
        assert!(
            !line.contains(body) && !line.contains("plain statement"),
            "{line}"
        );
    }

    /// Mutation-verified: when every episodic text is withheld whole, no preference prompt is
    /// sent, the result is the empty one, and one run-log entry says so without content.
    #[test]
    fn episodic_memories_all_withheld_skip_the_preference_call_and_log_it() {
        let recorder = PromptRecorder::answering("- leaked preference");
        let body = "MIIEowIBAAKCAQEAbodyline";
        let lines = judge_log_lines({
            let recorder = recorder.clone();
            async move {
                let judge = LlmDistillJudge::new(recorder);
                let episodic = vec![
                    make_episodic("I like tabs"),
                    make_episodic(&key_block(body)),
                ];
                let prefs = judge.summarize_preferences(&episodic).await.unwrap();
                assert!(prefs.is_empty(), "{prefs:?}");
            }
        });
        assert_eq!(recorder.calls(), 0, "no preference prompt is sent");
        assert_eq!(lines.len(), 1, "{lines:?}");
        let line = lines.first().map(String::as_str).unwrap_or_default();
        assert!(line.contains("preference distillation skipped"), "{line}");
        assert!(
            !line.contains(body) && !line.contains("I like tabs"),
            "{line}"
        );
    }
}
