//! This module implements the BashTool, allowing the agent to execute shell commands.
//! Hardened with strict timeouts and execution sandboxed to the workspace.
//! Now uses a strict whitelist approach for maximum security.

use crate::system::path_guard::PathGuard;
use crate::tools::{Tool, ToolError, ToolResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::process::Command;
use tokio::time::{timeout, Duration};

const DEFAULT_TIMEOUT_MS: u64 = 120_000; // 2 minutes
const MAX_TIMEOUT_MS: u64 = 600_000; // 10 minutes

/// Arguments for the `BashTool`.
#[derive(Debug, Deserialize)]
struct BashArgs {
    /// The command to execute.
    command: String,
    /// Optional timeout in milliseconds.
    timeout: Option<u64>,
}

/// Result of the `BashTool` execution.
#[derive(Debug, Serialize)]
struct BashResult {
    stdout: String,
    stderr: String,
    exit_code: Option<i32>,
    interrupted: bool,
}

/// A tool that executes shell commands.
pub struct BashTool {
    workspace_root: PathBuf,
}

impl BashTool {
    /// Creates a new `BashTool` anchored to the workspace root.
    pub fn new(workspace_root: PathBuf) -> anyhow::Result<Self> {
        let root = workspace_root
            .canonicalize()
            .map_err(|e| anyhow::anyhow!("Invalid workspace root for BashTool: {}", e))?;
        Ok(Self {
            workspace_root: root,
        })
    }
}

/// Strict whitelist of allowed base commands.
/// Anything not in this list (or involving shell-injection tokens) is rejected.
fn is_command_allowed(cmd: &str, workspace_root: &Path) -> bool {
    let allowed_binaries = [
        "ls", "git", "npm", "cargo", "rg", "cat", "echo", "pwd", "grep", "mkdir", "touch", "rm",
        "find", "diff", "node", "python", "pytest",
    ];

    // Security: Broad set of dangerous tokens including sub-shells and variable expansion
    let dangerous_tokens = [
        '|', '&', ';', '>', '<', '`', '$', '(', ')', '{', '}', '\\', '\n', '\0',
    ];
    if cmd.chars().any(|c| dangerous_tokens.contains(&c)) {
        return false;
    }

    // Security: PowerShell stop-parsing token "--%" passes the remainder verbatim
    // to the legacy command line, bypassing PowerShell quoting and re-enabling
    // injection on the Windows code path (W4 / RF-8.2).
    if cmd.contains("--%") {
        return false;
    }

    // Sandbox: every non-flag arg is treated as a path and must resolve inside
    // the workspace. PathGuard rejects absolutes (any form, incl. Windows
    // forward-slash `C:/...`), `..`, and symlink escapes uniformly per platform
    // (replaces the old string heuristics that missed the Windows case). R-6.
    let guard = match PathGuard::new(workspace_root.to_path_buf()) {
        Ok(g) => g,
        Err(_) => return false,
    };

    let mut tokens = cmd.split_whitespace();
    if let Some(base_cmd) = tokens.next() {
        let base_cmd_lower = base_cmd.to_lowercase();

        // Whitelist check
        if !allowed_binaries.iter().any(|&b| base_cmd_lower == b) {
            return false;
        }

        // Argument Hardening: Check ALL arguments for dangerous patterns
        let remaining_tokens: Vec<&str> = tokens.collect();
        for arg in &remaining_tokens {
            let arg_lower = arg.to_lowercase();

            // Non-flag args are treated as paths and must validate inside the
            // workspace via PathGuard. Flags (`-`-prefixed) are skipped — this
            // assumes no whitelisted binary takes a `-`-prefixed path arg;
            // re-check this when extending `allowed_binaries`. By design (spec
            // D1) a non-path token (e.g. a `grep` pattern) is also validated as
            // a workspace-relative path, so a pattern containing `..`/absolute
            // is rejected — intentional, erring strict for the sandbox.
            if !arg.starts_with('-') && guard.validate(Path::new(arg)).is_err() {
                return false;
            }

            // Binary-specific dangerous flags
            match base_cmd_lower.as_str() {
                "git"
                    if arg_lower.contains("exec-path")
                        || arg_lower.contains("config")
                        || arg_lower == "-c" =>
                {
                    return false;
                }
                "cargo" => {
                    // Only allow 'cargo test' and common build commands
                    // Block arbitrary script execution via cargo
                }
                "rm" => {
                    // Block destructive patterns
                    let has_rf = arg_lower == "-rf"
                        || arg_lower == "-fr"
                        || arg_lower == "-r"
                        || arg_lower == "-f";
                    if has_rf {
                        // Scan other args for sensitive paths
                        if remaining_tokens
                            .iter()
                            .any(|&a| a == "/" || a == "/*" || a == ".")
                        {
                            return false;
                        }
                    }
                }
                _ => {}
            }
        }

        // Special case: only 'cargo test|build|check' are allowed.
        // Use .first() to avoid an index-out-of-bounds panic on bare "cargo".
        if base_cmd_lower == "cargo" {
            let sub = remaining_tokens.first().copied();
            if !matches!(sub, Some("test") | Some("build") | Some("check")) {
                return false;
            }
        }

        return true;
    }
    false
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Executes a bash/shell command. STRICTLY WHITELISTED binaries and safe arguments only."
    }

    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to execute."
                },
                "timeout": {
                    "type": "integer",
                    "description": "Optional timeout in milliseconds."
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, args: Value) -> ToolResult<Value> {
        let args: BashArgs =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        // Proactive Whitelist and Argument check
        if !is_command_allowed(&args.command, &self.workspace_root) {
            return Err(ToolError::ExecutionError("Security Violation: Command or arguments are not whitelisted or contain dangerous patterns.".to_string()));
        }

        let timeout_ms = args
            .timeout
            .unwrap_or(DEFAULT_TIMEOUT_MS)
            .min(MAX_TIMEOUT_MS);

        #[cfg(target_os = "windows")]
        let mut cmd = {
            let mut c = Command::new("powershell");
            c.arg("-NoProfile").arg("-Command").arg(&args.command);
            c
        };

        #[cfg(not(target_os = "windows"))]
        let mut cmd = {
            let mut c = Command::new("bash");
            c.arg("-c").arg(&args.command);
            c
        };

        cmd.current_dir(&self.workspace_root)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let child = cmd
            .spawn()
            .map_err(|e| ToolError::ExecutionError(format!("Failed to spawn process: {}", e)))?;

        let exec_future = child.wait_with_output();

        match timeout(Duration::from_millis(timeout_ms), exec_future).await {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();

                let result = BashResult {
                    stdout,
                    stderr,
                    exit_code: output.status.code(),
                    interrupted: false,
                };

                Ok(serde_json::to_value(result)
                    .map_err(|e| ToolError::ExecutionError(e.to_string()))?)
            }
            Ok(Err(e)) => Err(ToolError::ExecutionError(format!(
                "Error reading process output: {}",
                e
            ))),
            Err(_) => {
                let result = BashResult {
                    stdout: String::new(),
                    stderr: "Command timed out and was killed.".to_string(),
                    exit_code: None,
                    interrupted: true,
                };
                Ok(serde_json::to_value(result)
                    .map_err(|e| ToolError::ExecutionError(e.to_string()))?)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    // Validates a command against a throwaway workspace root.
    fn check(cmd: &str) -> bool {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().canonicalize().expect("canonicalize");
        is_command_allowed(cmd, &root)
    }

    #[test]
    fn test_bash_args_sandboxed_via_pathguard() {
        // S-1 (the bug): Windows forward-slash absolute escapes the workspace.
        assert!(
            !check("cat C:/Windows/System32/config/SAM"),
            "S-1 windows absolute must be rejected"
        );
        // S-2: parent-dir traversal.
        assert!(
            !check("cat ../../../etc/passwd"),
            "S-2 traversal must be rejected"
        );
        // S-5: rm targeting outside the workspace.
        assert!(
            !check("rm C:/importante/archivo"),
            "S-5 rm outside must be rejected"
        );
        assert!(
            !check("rm -rf C:/dir"),
            "S-5 rm -rf outside must be rejected"
        );
        // S-3: relative in-workspace path is allowed.
        assert!(
            check("cat archivo.txt"),
            "S-3 relative in-workspace must be allowed"
        );
        // S-4: non-path args are allowed (resolve to workspace-relative).
        assert!(check("echo hola"), "S-4 echo non-path arg must be allowed");
        assert!(check("git log --oneline"), "S-4 git log must be allowed");
        // S-6: rm destructive guard not regressed.
        assert!(!check("rm -rf ."), "S-6 rm -rf . must stay rejected");
        // S-7: no regression on legit commands + C3/W4 intact.
        assert!(check("cargo test"), "cargo test allowed");
        assert!(check("ls"), "ls allowed");
        assert!(check("grep foo bar.txt"), "grep allowed");
        assert!(!check("cargo"), "bare cargo rejected (C3, no panic)");
        assert!(!check("echo --% x"), "--% rejected (W4)");
    }

    #[tokio::test]
    async fn test_bash_tool_execution() {
        let dir = tempdir().expect("Failed to create temp dir");
        let root = dir
            .path()
            .canonicalize()
            .expect("Failed to canonicalize root");

        let tool = BashTool::new(root.clone()).unwrap();

        let args = serde_json::json!({
            "command": "echo 'Hello Rust'",
            "timeout": 5000
        });

        let result = tool.execute(args).await;
        assert!(result.is_ok());

        let result_val = result.unwrap();
        let stdout = result_val["stdout"].as_str().unwrap();
        assert!(stdout.contains("Hello Rust"));
    }

    #[tokio::test]
    async fn test_bash_tool_timeout() {
        let dir = tempdir().expect("Failed to create temp dir");
        let root = dir
            .path()
            .canonicalize()
            .expect("Failed to canonicalize root");

        let tool = BashTool::new(root.clone()).unwrap();

        let args = serde_json::json!({
            "command": "whoami"
        });
        let result = tool.execute(args).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not whitelisted"));
    }

    #[test]
    fn test_whitelist_logic() {
        assert!(check("ls"));
        assert!(check("git status"));
        assert!(check("cargo test"));

        assert!(!check("whoami"), "Common but not whitelisted");
        assert!(!check("sudo apt update"), "Escalation attempt");
        assert!(
            !check("ls | grep test"),
            "Piping is currently disabled for security"
        );
        assert!(!check("echo hello > file.txt"), "Redirection is disabled");
        assert!(!check("rm -rf ."), "Destructive rm on workspace root");
    }

    #[test]
    fn test_cargo_without_subcommand_is_rejected_without_panic() {
        assert!(!check("cargo"), "bare cargo must be rejected");
        assert!(
            !check("cargo "),
            "cargo with trailing space must be rejected"
        );
        assert!(!check("cargo run"), "cargo run must be rejected");
        assert!(
            !check("cargo install ripgrep"),
            "cargo install must be rejected"
        );
        assert!(check("cargo test"), "cargo test must be allowed");
        assert!(check("cargo build"), "cargo build must be allowed");
        assert!(check("cargo check"), "cargo check must be allowed");
    }

    #[test]
    fn test_powershell_stop_parsing_token_is_rejected() {
        assert!(!check("echo --% foo"), "bare --% must be blocked");
        assert!(!check("git log --%"), "--% as last token must be blocked");
        assert!(!check("ls --%bar"), "--% prefix in a token must be blocked");
        assert!(check("git log --oneline"), "ordinary -- flags stay allowed");
    }

    #[test]
    fn test_adversarial_bash_injections() {
        // 1. Sub-shell injection attempts
        assert!(!check("ls $(whoami)"), "Sub-shell $() should be blocked");
        assert!(
            !check("ls `whoami`"),
            "Backtick sub-shell should be blocked"
        );
        assert!(
            !check("echo ${PATH}"),
            "Variable expansion should be blocked"
        );

        // 2. Argument-based injection attempts (common for 'git')
        assert!(
            !check("git --exec-path=/tmp"),
            "Dangerous git flags should be blocked"
        );
        assert!(
            !check("git config --global core.editor 'rm -rf /'"),
            "Dangerous git config should be blocked"
        );

        // 3. Recursive path traversal in arguments
        assert!(
            !check("cat ../../../etc/passwd"),
            "Path traversal in cat arguments should be blocked"
        );
    }
}
