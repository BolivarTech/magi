//! Configuration types of the `[decision]` section (REQ-D01, REQ-D21).

/// Value of `[decision].keep_alive`: an Ollama duration (`"10m"`) or a number of
/// seconds (`-1` keeps the model loaded). Sent verbatim when declared (REQ-D01).
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(untagged)]
pub enum KeepAlive {
    /// A duration string such as `"10m"` or `"2h"`.
    Text(String),
    /// A number of seconds; `-1` means indefinitely.
    Seconds(i64),
}
