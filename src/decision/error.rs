// Author: Julian Bolivar
// Version: 0.25.1
// Date: 2026-11-25

//! Error, attempt trace and usage types of a decision (REQ-D08..D10).

use std::collections::BTreeMap;

use crate::decision::protocol::Answer;

/// Why one attempt of the fallback chain failed (REQ-D10 `cause` token).
///
/// Serialize-only on purpose: [`Cause::BadRequest`] and [`Cause::Rejected`] share the token
/// `rejected` (DE-S1 names a 400 skip `rejected`, and the published envelope keeps that closed
/// set), so the token cannot be read back into the variant. A consumer of a serialized trace
/// tells them apart by the failed attempt's `status` field (400 against 413 or 422), never by the
/// token.
// MAGI FIX (S6 loop 4): pinned at compile time by the `AmbiguousIfDeserialize` assertion (S6 loop 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Cause {
    /// HTTP 401 or 403.
    Unauthorized,
    /// HTTP 400: this model refused the request. Unlike [`Cause::Rejected`] the chain moves on
    /// to the next model, because the refusal may belong to the model and not to the request
    /// (a cloud model on an old daemon, a model in an unsupported format, a tiny context
    /// window). It shares the serialized token `rejected` with [`Cause::Rejected`]: in code a
    /// consumer decides by the variant; in a serialized trace, by the failed attempt's `status`
    /// (400), never by the token.
    #[serde(rename = "rejected")]
    BadRequest,
    /// HTTP 413 or 422: the request itself is at fault, so the chain stops.
    Rejected,
    /// HTTP 404 whose body names a missing model.
    ModelNotFound,
    /// HTTP 404 whose body names an account policy.
    AccountPolicy,
    /// HTTP 404 with no recognizable marker.
    NotFound,
    /// HTTP 429.
    RateLimited,
    /// HTTP 5xx.
    #[serde(rename = "http_5xx")]
    Http5xx,
    /// HTTP 3xx (redirects are disabled).
    Redirected,
    /// Connection refused, DNS or TLS failure.
    Transport,
    /// The per-attempt timeout elapsed.
    Timeout,
    /// The response did not validate.
    InvalidResponse,
}

impl Cause {
    /// Whether the chain rotates to the next model after this cause.
    ///
    /// # Returns
    /// `false` for the two causes every model of the endpoint would repeat
    /// ([`Cause::Unauthorized`], [`Cause::Rejected`]), `true` otherwise (a 400 included).
    pub fn rotates(self) -> bool {
        match self {
            Cause::Unauthorized | Cause::Rejected => false,
            Cause::BadRequest
            | Cause::ModelNotFound
            | Cause::AccountPolicy
            | Cause::NotFound
            | Cause::RateLimited
            | Cause::Http5xx
            | Cause::Redirected
            | Cause::Transport
            | Cause::Timeout
            | Cause::InvalidResponse => true,
        }
    }

    /// Whether this cause belongs to the transport class (REQ-D13).
    ///
    /// # Returns
    /// `true` when the endpoint itself failed to answer (transport, timeout, 5xx,
    /// rate limit, redirect), as opposed to a model or request problem.
    pub fn is_transport_class(self) -> bool {
        match self {
            Cause::Transport
            | Cause::Timeout
            | Cause::Http5xx
            | Cause::RateLimited
            | Cause::Redirected => true,
            Cause::Unauthorized
            | Cause::Rejected
            | Cause::BadRequest
            | Cause::ModelNotFound
            | Cause::AccountPolicy
            | Cause::NotFound
            | Cause::InvalidResponse => false,
        }
    }

    /// The serialized token of this cause.
    ///
    /// # Returns
    /// The `cause` value of the envelope, from a closed set.
    pub fn as_str(self) -> &'static str {
        match self {
            Cause::Unauthorized => "unauthorized",
            Cause::Rejected | Cause::BadRequest => "rejected",
            Cause::ModelNotFound => "model_not_found",
            Cause::AccountPolicy => "account_policy",
            Cause::NotFound => "not_found",
            Cause::RateLimited => "rate_limited",
            Cause::Http5xx => "http_5xx",
            Cause::Redirected => "redirected",
            Cause::Transport => "transport",
            Cause::Timeout => "timeout",
            Cause::InvalidResponse => "invalid_response",
        }
    }
}

/// Suffix of the tag of an Ollama model served from Ollama's cloud (`clef:cloud`); such a model
/// needs `ollama signin` on the daemon, a local one does not.
pub(crate) const OLLAMA_CLOUD_SUFFIX: &str = ":cloud";

/// Whether `model` names an Ollama cloud model, by its [`OLLAMA_CLOUD_SUFFIX`].
///
/// It tests the name only: case-sensitive, and blind to the provider. A model of another
/// provider whose name ends in `:cloud` is taken for one, which costs at most one extra refused
/// call on the same endpoint when a chain rotates past its 401, plus a sign-in hint that does not
/// apply; no key is sent anywhere new.
///
/// # Arguments
/// * `model` - a model name of the chain.
///
/// # Returns
/// `true` when the name ends with `:cloud`.
pub(crate) fn is_ollama_cloud_model(model: &str) -> bool {
    model.ends_with(OLLAMA_CLOUD_SUFFIX)
}

/// Hint the decision error messages append after a refused Ollama cloud model.
const CLOUD_SIGNIN_HINT: &str = "; for an Ollama `:cloud` model, run `ollama signin`";
/// What the [`DecisionError::Unauthorized`] message adds after [`CLOUD_SIGNIN_HINT`]: there the
/// hint is the alternative to setting a key.
const CLOUD_SIGNIN_INSTEAD: &str = " instead";

/// The sign-in hint for a chain whose `trace` holds a 401 on an Ollama cloud model, else `""`.
///
/// The [`DecisionError::ChainExhausted`] and [`DecisionError::ModelUnavailable`] messages end
/// with it: a signed-out daemon refuses only `:cloud` models, so a chain can rotate past that
/// refusal and end on another cause. Complexity: O(attempts).
///
/// # Arguments
/// * `trace` - the attempts of the call.
///
/// # Returns
/// [`CLOUD_SIGNIN_HINT`] or the empty string.
fn refused_cloud_hint(trace: &Trace) -> &'static str {
    let refused = trace.attempts.iter().any(|attempt| {
        is_ollama_cloud_model(&attempt.model)
            && matches!(
                attempt.outcome,
                AttemptOutcome::Failed {
                    cause: Cause::Unauthorized,
                    ..
                }
            )
    });
    if refused {
        CLOUD_SIGNIN_HINT
    } else {
        ""
    }
}

/// The message of [`DecisionError::Unauthorized`] (REQ-F14).
///
/// It always names `DECISION_API_KEY`; it names `ollama signin` only when the last attempt of
/// `trace` was an Ollama cloud model, the one case where a signed-out daemon explains the
/// refusal. Complexity: O(1).
///
/// # Arguments
/// * `status` - the HTTP status of the refusal.
/// * `trace` - the attempts of the call.
///
/// # Returns
/// The message, with the `ollama signin` hint or without it.
fn unauthorized_message(status: u16, trace: &Trace) -> String {
    let cloud = trace
        .attempts
        .last()
        .is_some_and(|attempt| is_ollama_cloud_model(&attempt.model));
    format!(
        "the decision endpoint refused the request credentials (HTTP {status}); set DECISION_API_KEY \
         in the environment or with `magi-rs vault set DECISION_API_KEY`{}",
        if cloud {
            format!("{CLOUD_SIGNIN_HINT}{CLOUD_SIGNIN_INSTEAD}")
        } else {
            String::new()
        }
    )
}

/// Token consumption of one attempt or of a whole chain.
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize)]
pub struct Usage {
    /// Input tokens consumed.
    pub input_tokens: u64,
    /// Output tokens produced.
    pub output_tokens: u64,
    /// Cost reported by the provider, if any.
    pub cost: Option<f64>,
}

impl Usage {
    /// Adds `other` into `self`; `cost` stays `None` only if neither side has one.
    ///
    /// # Arguments
    ///
    /// * `other` - The usage to add into this value.
    pub fn add(&mut self, other: &Usage) {
        self.input_tokens = self.input_tokens.saturating_add(other.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(other.output_tokens);
        self.cost = match (self.cost, other.cost) {
            (Some(a), Some(b)) => Some(a + b),
            (a, b) => a.or(b),
        };
    }
}

/// Outcome of one attempt of the chain.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum AttemptOutcome {
    /// The model gave a usable decision.
    Answered,
    /// The attempt failed.
    Failed {
        /// Why it failed.
        cause: Cause,
        /// Text generated by Magi describing the failure.
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
        /// The HTTP status the provider answered with, when it answered (absent for a
        /// transport failure, a timeout or a response that did not validate). It tells apart
        /// what the shared `rejected` token does not: a 400 that rotated from a 413/422 that
        /// stopped the chain.
        // MAGI FIX (S6 loop 13, C2-b1 / C2-c3): a consumer had to parse `detail`'s prose.
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<u16>,
    },
    /// The model answered but below its `min_confidence`.
    LowConfidence {
        /// Lowest Magi confidence among the questions.
        confidence: f64,
        /// The threshold that was not reached.
        min_confidence: f64,
        /// The rejected answers.
        answers: BTreeMap<String, Answer>,
    },
}

/// One attempt of the chain.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Attempt {
    /// Model tried.
    pub model: String,
    /// What happened.
    #[serde(flatten)]
    pub outcome: AttemptOutcome,
    /// Tokens this attempt consumed.
    #[serde(skip)]
    pub usage: Usage,
}

/// The attempts of a decision and their summed usage.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Trace {
    /// Attempts in order.
    pub attempts: Vec<Attempt>,
    /// Sum of the usage of every attempt.
    pub usage: Usage,
}

/// A successful decision.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    /// Model that answered.
    pub model: String,
    /// One answer per question.
    pub answers: BTreeMap<String, Answer>,
    /// The chain walked to get here.
    pub trace: Trace,
}

/// Closed set of `error.kind` values (REQ-D10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    /// The caller's request is invalid or the provider rejected it.
    InvalidInput,
    /// The endpoint needs a key that was not sent or not accepted.
    MissingKey,
    /// An explicitly requested model gave no usable decision.
    ModelUnavailable,
    /// No model of the chain answered, or the client could not be built.
    DecisionUnavailable,
}

/// Why a decision could not be produced.
#[derive(Clone, PartialEq, thiserror::Error)]
pub enum DecisionError {
    /// Local validation failed; no model was called.
    #[error("invalid input: {reason}")]
    InvalidInput {
        /// What is wrong, without quoting values the caller did not choose.
        reason: String,
    },
    /// The provider rejected the request as invalid.
    #[error("the provider rejected the request as invalid (HTTP {status})")]
    Rejected {
        /// HTTP status.
        status: u16,
        /// Provider text; callers redact it before showing it.
        provider_message: String,
        /// Attempts so far.
        trace: Trace,
    },
    /// The endpoint refused the credentials; when the last attempt was an Ollama `:cloud`
    /// model, the message also points a signed-out daemon at `ollama signin` (REQ-F14).
    #[error("{}", unauthorized_message(*status, trace))]
    Unauthorized {
        /// HTTP status.
        status: u16,
        /// Attempts so far.
        trace: Trace,
    },
    /// A vault entry the URL placeholders need does not exist.
    #[error("vault entry {entry} is missing; create it with `magi-rs vault set {entry}`")]
    MissingVaultEntry {
        /// Name of the missing entry.
        entry: &'static str,
    },
    /// An explicitly requested model gave no usable decision; after a 401 on an Ollama `:cloud`
    /// model, the message also names `ollama signin` (REQ-F14).
    #[error("model '{model}' gave no usable decision{}", refused_cloud_hint(trace))]
    ModelUnavailable {
        /// The requested model.
        model: String,
        /// Attempts so far.
        trace: Trace,
    },
    /// Every model of the chain failed or fell below its threshold; after a 401 on an Ollama
    /// `:cloud` model, the message also names `ollama signin` (REQ-F14).
    #[error(
        "no decision model answered: chain exhausted ({} attempts){}",
        trace.attempts.len(),
        refused_cloud_hint(trace)
    )]
    ChainExhausted {
        /// Attempts made.
        trace: Trace,
    },
    /// The client could not be built.
    #[error("the decision client could not be built: {reason}")]
    ClientUnavailable {
        /// Why.
        reason: String,
    },
}

impl std::fmt::Debug for DecisionError {
    /// Prints the variant and its non-textual facts; never provider text (REQ-D11).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecisionError::InvalidInput { .. } => f.write_str("InvalidInput"),
            DecisionError::Rejected { status, trace, .. } => f
                .debug_struct("Rejected")
                .field("status", status)
                .field("attempts", &trace.attempts.len())
                .finish(),
            DecisionError::Unauthorized { status, trace, .. } => f
                .debug_struct("Unauthorized")
                .field("status", status)
                .field("attempts", &trace.attempts.len())
                .finish(),
            DecisionError::MissingVaultEntry { entry } => f
                .debug_struct("MissingVaultEntry")
                .field("entry", entry)
                .finish(),
            DecisionError::ModelUnavailable { model, trace, .. } => f
                .debug_struct("ModelUnavailable")
                .field("model", model)
                .field("attempts", &trace.attempts.len())
                .finish(),
            DecisionError::ChainExhausted { trace } => f
                .debug_struct("ChainExhausted")
                .field("attempts", &trace.attempts.len())
                .finish(),
            DecisionError::ClientUnavailable { .. } => f.write_str("ClientUnavailable"),
        }
    }
}

impl DecisionError {
    /// The `error.kind` this error maps to.
    ///
    /// # Returns
    /// The closed-set kind of the envelope's `error` object.
    pub fn kind(&self) -> ErrorKind {
        match self {
            DecisionError::InvalidInput { .. } => ErrorKind::InvalidInput,
            DecisionError::Rejected { .. } => ErrorKind::InvalidInput,
            DecisionError::Unauthorized { .. } => ErrorKind::MissingKey,
            DecisionError::MissingVaultEntry { .. } => ErrorKind::MissingKey,
            DecisionError::ModelUnavailable { .. } => ErrorKind::ModelUnavailable,
            DecisionError::ChainExhausted { .. } => ErrorKind::DecisionUnavailable,
            DecisionError::ClientUnavailable { .. } => ErrorKind::DecisionUnavailable,
        }
    }

    /// The attempt trace, when any model was called.
    ///
    /// # Returns
    /// The trace, or `None` for errors raised before a model was called.
    pub fn trace(&self) -> Option<&Trace> {
        match self {
            DecisionError::Rejected { trace, .. } => Some(trace),
            DecisionError::Unauthorized { trace, .. } => Some(trace),
            DecisionError::ModelUnavailable { trace, .. } => Some(trace),
            DecisionError::ChainExhausted { trace } => Some(trace),
            DecisionError::InvalidInput { .. }
            | DecisionError::MissingVaultEntry { .. }
            | DecisionError::ClientUnavailable { .. } => None,
        }
    }
}

/// Implemented for every type with `()`, and for every deserializable type with
/// [`Deserializable`]: naming the impl with an inferred parameter is ambiguous, a compile
/// error, exactly when the type deserializes. The negative-bound idiom of the
/// `static_assertions` crate, written out instead of adding a dependency for one assertion.
trait AmbiguousIfDeserialize<Marker> {
    /// The item an assertion names.
    fn item() {}
}

impl<T: ?Sized> AmbiguousIfDeserialize<()> for T {}

/// The marker of the impl only deserializable types have.
struct Deserializable;

impl<T: serde::de::DeserializeOwned> AmbiguousIfDeserialize<Deserializable> for T {}

// [`Cause`] serializes and never deserializes, checked by the compiler (C2 of S6 loop 4, made a
// compile-time check in S6 loop 5): `BadRequest` and `Rejected` both serialize as `rejected`
// (DE-S1 names the 400 skip `rejected`), so the token cannot be read back into the variant, and
// a derived `Deserialize` would turn every 400 into a 413/422. Adding it makes the name below
// ambiguous and the crate stops compiling (E0283), whatever the derive's formatting or place.
// MAGI FIX (S6 loop 5): a static assertion replaces the test that grepped this file's source.
const _: fn() = || {
    fn serializes<T: serde::Serialize>() {}
    serializes::<Cause>();
    <Cause as AmbiguousIfDeserialize<_>>::item();
};

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Cause; 12] = [
        Cause::Unauthorized,
        Cause::Rejected,
        Cause::BadRequest,
        Cause::ModelNotFound,
        Cause::AccountPolicy,
        Cause::NotFound,
        Cause::RateLimited,
        Cause::Http5xx,
        Cause::Redirected,
        Cause::Transport,
        Cause::Timeout,
        Cause::InvalidResponse,
    ];

    #[test]
    fn only_caller_and_credential_errors_do_not_rotate() {
        for c in ALL {
            let expected = !matches!(c, Cause::Unauthorized | Cause::Rejected);
            assert_eq!(c.rotates(), expected, "{c:?}");
        }
    }

    #[test]
    fn the_transport_class_is_the_closed_set_of_req_d13() {
        for c in ALL {
            let expected = matches!(
                c,
                Cause::Transport
                    | Cause::Timeout
                    | Cause::Http5xx
                    | Cause::RateLimited
                    | Cause::Redirected
            );
            assert_eq!(c.is_transport_class(), expected, "{c:?}");
        }
    }

    #[test]
    fn as_str_matches_the_serialized_token() {
        for c in ALL {
            assert_eq!(
                serde_json::to_value(c).expect("ser"),
                serde_json::json!(c.as_str())
            );
        }
        assert_eq!(Cause::Http5xx.as_str(), "http_5xx");
        assert_eq!(Cause::ModelNotFound.as_str(), "model_not_found");
    }

    #[test]
    fn usage_adds_tokens_and_cost_is_none_only_when_nobody_reported_it() {
        let mut u = Usage {
            input_tokens: 1,
            output_tokens: 2,
            cost: None,
        };
        u.add(&Usage {
            input_tokens: 10,
            output_tokens: 20,
            cost: None,
        });
        assert_eq!(
            u,
            Usage {
                input_tokens: 11,
                output_tokens: 22,
                cost: None
            }
        );
        u.add(&Usage {
            input_tokens: 0,
            output_tokens: 0,
            cost: Some(0.25),
        });
        assert_eq!(u.cost, Some(0.25));
        u.add(&Usage {
            input_tokens: 0,
            output_tokens: 0,
            cost: Some(0.5),
        });
        assert_eq!(u.cost, Some(0.75));
    }

    #[test]
    fn every_error_maps_to_its_kind_and_carries_a_trace_when_a_model_was_called() {
        let t = Trace::default();
        let cases: [(DecisionError, ErrorKind, bool); 7] = [
            (
                DecisionError::InvalidInput { reason: "r".into() },
                ErrorKind::InvalidInput,
                false,
            ),
            (
                DecisionError::Rejected {
                    status: 413,
                    provider_message: "m".into(),
                    trace: t.clone(),
                },
                ErrorKind::InvalidInput,
                true,
            ),
            (
                DecisionError::Unauthorized {
                    status: 401,
                    trace: t.clone(),
                },
                ErrorKind::MissingKey,
                true,
            ),
            (
                DecisionError::MissingVaultEntry {
                    entry: "DECISION_BASE_URL_USER",
                },
                ErrorKind::MissingKey,
                false,
            ),
            (
                DecisionError::ModelUnavailable {
                    model: "m".into(),
                    trace: t.clone(),
                },
                ErrorKind::ModelUnavailable,
                true,
            ),
            (
                DecisionError::ChainExhausted { trace: t.clone() },
                ErrorKind::DecisionUnavailable,
                true,
            ),
            (
                DecisionError::ClientUnavailable { reason: "r".into() },
                ErrorKind::DecisionUnavailable,
                false,
            ),
        ];
        for (e, kind, has_trace) in cases {
            assert_eq!(e.kind(), kind, "{e:?}");
            assert_eq!(e.trace().is_some(), has_trace, "{e:?}");
        }
    }

    #[test]
    fn the_unauthorized_message_names_the_key_and_the_vault_command() {
        let msg = DecisionError::Unauthorized {
            status: 403,
            trace: Trace::default(),
        }
        .to_string();
        assert!(
            msg.contains("DECISION_API_KEY") && msg.contains("magi-rs vault set DECISION_API_KEY"),
            "{msg}"
        );
    }

    /// A trace whose only attempt is a 401 on `model`.
    fn refused_on(model: &str) -> Trace {
        Trace {
            attempts: vec![Attempt {
                model: model.into(),
                outcome: AttemptOutcome::Failed {
                    status: Some(401),
                    cause: Cause::Unauthorized,
                    detail: None,
                },
                usage: Usage::default(),
            }],
            usage: Usage::default(),
        }
    }

    /// REQ-F14 / F-S20: the hint names `ollama signin` when the last attempt was a cloud model.
    #[test]
    fn the_unauthorized_message_names_ollama_signin_for_cloud_models() {
        let msg = DecisionError::Unauthorized {
            status: 401,
            trace: refused_on("clef:cloud"),
        }
        .to_string();
        assert!(
            msg.contains("set DECISION_API_KEY")
                && msg.contains("for an Ollama `:cloud` model, run `ollama signin` instead"),
            "{msg}"
        );
    }

    /// REQ-F14 (mutation-verified): the hint follows the one rule of the other messages, a
    /// refused `:cloud` model anywhere in the trace, so a chain that rotated past it to a
    /// refused local model still names `ollama signin`.
    #[test]
    fn the_unauthorized_message_names_ollama_signin_after_an_earlier_refused_cloud_model() {
        let mut trace = refused_on("x:cloud");
        trace.attempts.extend(refused_on("nimble").attempts);
        let msg = DecisionError::Unauthorized { status: 401, trace }.to_string();
        assert!(
            msg.ends_with("for an Ollama `:cloud` model, run `ollama signin` instead"),
            "{msg}"
        );
    }

    /// REQ-F14 / F-S20: on any other model, or with no attempt, only the key is named.
    #[test]
    fn the_unauthorized_message_names_only_the_key_when_the_last_model_is_not_cloud() {
        for trace in [refused_on("nimble"), Trace::default()] {
            let msg = DecisionError::Unauthorized { status: 401, trace }.to_string();
            assert_eq!(
                msg,
                "the decision endpoint refused the request credentials (HTTP 401); set \
                 DECISION_API_KEY in the environment or with \
                 `magi-rs vault set DECISION_API_KEY`"
            );
        }
    }

    /// A trace of a 401 on `cloud`, then a transport failure on `nimble`.
    fn refused_then_failed(cloud: &str) -> Trace {
        let mut trace = refused_on(cloud);
        trace.attempts.push(Attempt {
            model: "nimble".into(),
            outcome: AttemptOutcome::Failed {
                status: None,
                cause: Cause::Transport,
                detail: None,
            },
            usage: Usage::default(),
        });
        trace
    }

    /// REQ-F14: a chain that went past a refused `:cloud` model names `ollama signin`, whichever
    /// attempt came last; so does an explicit model that ended the same way.
    #[test]
    fn an_exhausted_chain_after_a_refused_cloud_model_names_ollama_signin() {
        let exhausted = DecisionError::ChainExhausted {
            trace: refused_then_failed("clef:cloud"),
        }
        .to_string();
        assert_eq!(
            exhausted,
            "no decision model answered: chain exhausted (2 attempts); \
             for an Ollama `:cloud` model, run `ollama signin`"
        );
        let unavailable = DecisionError::ModelUnavailable {
            model: "clef:cloud".into(),
            trace: refused_on("clef:cloud"),
        }
        .to_string();
        assert_eq!(
            unavailable,
            "model 'clef:cloud' gave no usable decision; \
             for an Ollama `:cloud` model, run `ollama signin`"
        );
    }

    /// REQ-F14: no hint without a 401 on a `:cloud` model: a refused local model, or a cloud
    /// model that failed for another cause.
    #[test]
    fn an_exhausted_chain_without_a_refused_cloud_model_has_no_hint() {
        let mut timed_out = refused_on("clef:cloud");
        if let Some(AttemptOutcome::Failed { cause, status, .. }) =
            timed_out.attempts.first_mut().map(|a| &mut a.outcome)
        {
            *cause = Cause::Timeout;
            *status = None;
        }
        for trace in [refused_then_failed("clef-flash"), timed_out] {
            let n = trace.attempts.len();
            let msg = DecisionError::ChainExhausted { trace }.to_string();
            assert_eq!(
                msg,
                format!("no decision model answered: chain exhausted ({n} attempts)")
            );
        }
    }

    #[test]
    fn an_attempt_serializes_flat_with_its_outcome_and_without_usage() {
        let a = Attempt {
            model: "tev1".into(),
            outcome: AttemptOutcome::Failed {
                status: None,
                cause: Cause::Timeout,
                detail: None,
            },
            usage: Usage {
                input_tokens: 9,
                output_tokens: 9,
                cost: None,
            },
        };
        assert_eq!(
            serde_json::to_value(&a).expect("ser"),
            serde_json::json!({"model": "tev1", "outcome": "failed", "cause": "timeout"})
        );
    }

    #[test]
    fn a_bad_request_attempt_serializes_the_rejected_token() {
        let a = Attempt {
            model: "a".into(),
            outcome: AttemptOutcome::Failed {
                status: None,
                cause: Cause::BadRequest,
                detail: None,
            },
            usage: Usage::default(),
        };
        assert_eq!(
            serde_json::to_value(&a).expect("ser"),
            serde_json::json!({"model": "a", "outcome": "failed", "cause": "rejected"})
        );
    }

    /// CP2 MS1 r2 (Melchior): the rejected answers of a low-confidence hop travel with their numbers.
    #[test]
    fn a_low_confidence_attempt_serializes_its_numbers_and_answers() {
        let mut answers = std::collections::BTreeMap::new();
        answers.insert(
            "q".to_string(),
            Answer::Noul {
                noul: 0.6,
                confidence: 0.2,
            },
        );
        let a = Attempt {
            model: "tev1".into(),
            outcome: AttemptOutcome::LowConfidence {
                confidence: 0.2,
                min_confidence: 0.5,
                answers,
            },
            usage: Usage::default(),
        };
        assert_eq!(
            serde_json::to_value(&a).expect("ser"),
            serde_json::json!({"model": "tev1", "outcome": "low_confidence", "confidence": 0.2,
                               "min_confidence": 0.5,
                               "answers": {"q": {"type": "noul", "noul": 0.6, "confidence": 0.2}}})
        );
    }

    /// CP2 MS1 r2 (Caspar): `{:?}` of a rejection never shows what the provider wrote (REQ-D11).
    #[test]
    fn the_debug_of_a_rejection_never_shows_the_provider_text() {
        let e = DecisionError::Rejected {
            status: 413,
            provider_message: "PROVIDER-TEXT-31c7 Bearer sk-x".into(),
            trace: Trace::default(),
        };
        let shown = format!("{e:?}");
        assert!(
            shown.contains("Rejected") && shown.contains("413"),
            "{shown}"
        );
        assert!(!shown.contains("PROVIDER-TEXT-31c7"), "{shown}");

        // CP2 MS1 r3 (Caspar): nor the trace's provider-sourced text (detail, rejected answers, legend).
        let mut answers = std::collections::BTreeMap::new();
        answers.insert(
            "u".to_string(),
            Answer::Score {
                score: 1.0,
                legend: [("0".to_string(), "LEGEND-TEXT-8b2e".to_string())]
                    .into_iter()
                    .collect(),
                probabilities: [("0".to_string(), 1.0)].into_iter().collect(),
                confidence: 1.0,
                provider_confidence: None,
            },
        );
        let e = DecisionError::ChainExhausted {
            trace: Trace {
                attempts: vec![
                    Attempt {
                        model: "a".into(),
                        usage: Usage::default(),
                        outcome: AttemptOutcome::Failed {
                            status: None,
                            cause: Cause::NotFound,
                            detail: Some("DETAIL-TEXT-4c1a".into()),
                        },
                    },
                    Attempt {
                        model: "b".into(),
                        usage: Usage::default(),
                        outcome: AttemptOutcome::LowConfidence {
                            confidence: 0.1,
                            min_confidence: 0.5,
                            answers,
                        },
                    },
                ],
                usage: Usage::default(),
            },
        };
        let shown = format!("{e:?}");
        assert!(
            shown.contains("ChainExhausted") && shown.contains('2'),
            "{shown}"
        );
        for leak in ["LEGEND-TEXT-8b2e", "DETAIL-TEXT-4c1a"] {
            assert!(!shown.contains(leak), "{leak} in {shown}");
        }
    }
}
