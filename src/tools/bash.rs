//! This module implements the BashTool, allowing the agent to execute shell commands.
//! Hardened with strict timeouts and execution sandboxed to the workspace.
//! Now uses a strict whitelist approach for maximum security.

use crate::system::path_guard::PathGuard;
use crate::tools::{proc_group, Tool, ToolError, ToolResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::process::Command;
use tokio::time::{timeout, Duration};
use tokio_util::sync::CancellationToken;

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

/// Shell metacharacters banned from every command — sub-shells, redirection,
/// variable expansion, escapes, newlines, and nul. This is a hard barrier that
/// blocks injection on both the PowerShell and bash code paths; never widen it.
const DANGEROUS_TOKENS: [char; 14] = [
    '|', '&', ';', '>', '<', '`', '$', '(', ')', '{', '}', '\\', '\n', '\0',
];

/// A tool that executes shell commands.
pub struct BashTool {
    /// Sandbox guard, built once from the canonicalized workspace root and
    /// reused for every command validation — avoids re-canonicalizing the root
    /// on each `is_command_allowed` call.
    guard: PathGuard,
}

impl BashTool {
    /// Creates a new `BashTool` anchored to the workspace root.
    ///
    /// # Errors
    /// Returns an error if the workspace root cannot be canonicalized.
    pub fn new(workspace_root: PathBuf) -> anyhow::Result<Self> {
        let guard = PathGuard::new(workspace_root)
            .map_err(|e| anyhow::anyhow!("Invalid workspace root for BashTool: {}", e))?;
        Ok(Self { guard })
    }
}

/// Strict whitelist of allowed base commands.
/// Anything not in this list (or involving shell-injection tokens) is rejected.
fn is_command_allowed(cmd: &str, guard: &PathGuard) -> bool {
    let allowed_binaries = [
        "ls", "git", "npm", "cargo", "rg", "cat", "echo", "pwd", "grep", "mkdir", "touch", "rm",
        "find", "diff", "node", "python", "pytest",
    ];

    // Security: reject any command carrying a banned shell metacharacter.
    if cmd.chars().any(|c| DANGEROUS_TOKENS.contains(&c)) {
        return false;
    }

    // Security: PowerShell stop-parsing token "--%" passes the remainder verbatim
    // to the legacy command line, bypassing PowerShell quoting and re-enabling
    // injection on the Windows code path (W4 / RF-8.2).
    if cmd.contains("--%") {
        return false;
    }

    // Sandbox: every non-flag arg is treated as a path and must resolve inside
    // the workspace via the caller-provided `guard`. PathGuard rejects absolutes
    // (any form, incl. Windows forward-slash `C:/...`), `..`, and symlink escapes
    // uniformly per platform (replaces the old string heuristics that missed the
    // Windows case). R-6.
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
            // workspace via PathGuard. By design (spec D1) a non-path token
            // (e.g. a `grep` pattern) is also validated as a workspace-relative
            // path, so a pattern containing `..`/absolute is rejected —
            // intentional, erring strict for the sandbox.
            if !arg.starts_with('-') {
                if guard.validate(Path::new(arg)).is_err() {
                    return false;
                }
            } else if let Some((_, value)) = arg.split_once('=') {
                // `--flag=value`: the value can carry a path (e.g. an absolute
                // `--file=/etc/passwd`), which the bare-arg branch above never
                // inspects because the whole token is `-`-prefixed. Validate the
                // value the same way so a `--flag=`-wrapped path cannot escape
                // the sandbox that bare path args are held to. A non-path value
                // (`--format=oneline`) validates as a workspace-relative path
                // and passes; the same strict-sandbox tradeoff as bare tokens
                // applies (D1).
                if !value.is_empty() && guard.validate(Path::new(value)).is_err() {
                    return false;
                }
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
                // `find`'s `-delete` action recursively removes matched entries
                // with no shell metacharacters, so `find . -delete` (or a bare
                // `find -delete`, whose path defaults to `.`) wipes the whole
                // workspace, sidestepping the rm-root guard below. `find` is a
                // search tool here — deletion goes through `rm` (which carries
                // the root guard) — so block `-delete` outright (safe
                // direction). Both the `-delete` and the double-dash `--delete`
                // spelling are rejected: rather than depend on a given `find`
                // build (GNU/BSD/busybox/Windows port) rejecting `--delete` as
                // an unknown predicate, the barrier owns the decision.
                // `-exec`/`-execdir`/`-ok` always need `{}`/`;`/`+` and are
                // already rejected by the metacharacter ban.
                "find" if arg_lower == "-delete" || arg_lower == "--delete" => {
                    return false;
                }
                _ => {}
            }
        }

        // rm workspace-root guard: PathGuard (above) already rejects targets
        // *outside* the workspace but permits the workspace root itself (`.`,
        // `./`, `.//`, a `foo/..` that resolves to root, or an absolute to
        // root). A recursive `rm` on the root would wipe the whole sandbox, so
        // block any destructive `rm` whose target resolves to the root.
        // Accumulating the flag across *all* tokens also closes the split form
        // (`rm -r -f ./`); matching any `-`-flag that carries `r`/`f`/`d` covers
        // the combined (`-rf`), split (`-r -f`), long (`--recursive --force`),
        // and empty-dir (`-d`/`--dir`, so `rm -d .` on an empty root) spellings,
        // erring toward blocking (the safe direction — e.g. the harmless
        // `--preserve-root` is also flagged, but its target is still the root).
        if base_cmd_lower == "rm" {
            let has_destructive = remaining_tokens.iter().any(|&a| {
                let l = a.to_lowercase();
                l.starts_with('-') && (l.contains('r') || l.contains('f') || l.contains('d'))
            });
            if has_destructive {
                let root = guard.workspace_root();
                let targets_root = remaining_tokens.iter().any(|&a| {
                    !a.starts_with('-')
                        && guard
                            .validate(Path::new(a))
                            .map(|p| p.as_path() == root)
                            .unwrap_or(false)
                });
                if targets_root {
                    return false;
                }
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

    async fn execute(&self, args: Value, cancel: &CancellationToken) -> ToolResult<Value> {
        let args: BashArgs =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        // Proactive Whitelist and Argument check — HARD BARRIER, unchanged.
        if !is_command_allowed(&args.command, &self.guard) {
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

        cmd.current_dir(self.guard.workspace_root())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        // REQ-H37 (defense-in-depth): the subprocess inherits ONLY the
        // non-secret allowlist, never `MAGI_PASSPHRASE`/API keys. An in-workspace
        // interpreter (`python`/`node`) that reads its own environment therefore
        // cannot exfiltrate a secret, layering on top of the parent-env scrub in
        // `main.rs`. Applied before `spawn_in_group` so the child is born clean.
        proc_group::scrub_child_env(&mut cmd);

        // Spawn the shell inside a killable group (Job Object on Windows, new
        // process group on POSIX). On a wall-clock cancel (REQ-H36) the whole
        // subprocess TREE is terminated — not just the shell — so a long
        // grandchild (`cargo build`, `python …`) cannot outlive the timeout.
        // The hard barrier above is untouched; only the spawn/kill is wrapped.
        let (child, mut killer) = proc_group::spawn_in_group(&mut cmd)?.into_parts();

        // Race the shell against (a) the tool's own per-command timeout and (b)
        // external run cancellation. `killer` is held out of the wait future so
        // either branch can fire it without a borrow conflict.
        let output_fut = timeout(Duration::from_millis(timeout_ms), child.wait_with_output());
        tokio::pin!(output_fut);

        tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                // Wall-clock `--timeout`: kill the tree now and report an aborted
                // run. (When this future is dropped outright instead of polled —
                // e.g. by `tokio::time::timeout` elapsing at the run layer — the
                // `Drop` backstop on `killer` performs the same tree kill.)
                killer.kill();
                let result = BashResult {
                    stdout: String::new(),
                    stderr: "Command aborted: run wall-clock timeout reached.".to_string(),
                    exit_code: None,
                    interrupted: true,
                };
                serde_json::to_value(result).map_err(|e| ToolError::ExecutionError(e.to_string()))
            }
            res = &mut output_fut => match res {
                Ok(Ok(output)) => {
                    // Clean exit: the group is reaped — disarm the Drop backstop.
                    killer.disarm();
                    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
                    let result = BashResult {
                        stdout,
                        stderr,
                        exit_code: output.status.code(),
                        interrupted: false,
                    };
                    serde_json::to_value(result)
                        .map_err(|e| ToolError::ExecutionError(e.to_string()))
                }
                Ok(Err(e)) => {
                    // I/O error draining the process: state uncertain, kill the tree.
                    killer.kill();
                    Err(ToolError::ExecutionError(format!(
                        "Error reading process output: {}",
                        e
                    )))
                }
                Err(_) => {
                    // Per-command timeout elapsed: kill the TREE (not just the
                    // shell, which `kill_on_drop` alone would leave orphaning
                    // grandchildren).
                    killer.kill();
                    let result = BashResult {
                        stdout: String::new(),
                        stderr: "Command timed out and was killed.".to_string(),
                        exit_code: None,
                        interrupted: true,
                    };
                    serde_json::to_value(result)
                        .map_err(|e| ToolError::ExecutionError(e.to_string()))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// `BashTool` executes shell commands — MUST require approval.
    ///
    /// Fails in RED: `requires_approval` is not a method on the `Tool` trait yet.
    #[tokio::test]
    async fn test_bash_tool_requires_approval() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().canonicalize().expect("canonicalize");
        let tool = BashTool::new(root).unwrap();
        assert!(
            tool.requires_approval(),
            "bash executes shell commands — must always require approval"
        );
    }

    // Validates a command against a throwaway workspace root.
    fn check(cmd: &str) -> bool {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().canonicalize().expect("canonicalize");
        let guard = PathGuard::new(root).expect("guard");
        is_command_allowed(cmd, &guard)
    }

    #[test]
    fn test_rm_root_equivalent_targets_are_rejected() {
        // The literal `.` was already blocked; these lexical equivalents of the
        // workspace root must be too (MAGI S2: `rm -rf ./` slipped the guard).
        assert!(
            !check("rm -rf ./"),
            "rm -rf ./ (root via ./) must be rejected"
        );
        assert!(!check("rm -rf .//"), "rm -rf .// must be rejected");
        assert!(
            !check("rm -r -f ./"),
            "split-flag rm -r -f ./ must be rejected"
        );
        assert!(
            !check("rm --recursive --force ./"),
            "long-form rm --recursive --force ./ must be rejected"
        );
        assert!(
            !check("rm -rf foo/.."),
            "rm -rf foo/.. (resolves to root) must be rejected"
        );
        // No over-blocking: destructive rm of an in-workspace subdir stays legal.
        assert!(check("rm -rf src"), "rm -rf src (subdir) must stay allowed");
        assert!(
            check("rm -r -f ./build"),
            "rm -r -f ./build (subdir via ./) must stay allowed"
        );
        assert!(
            check("rm archivo.txt"),
            "non-recursive rm of a file allowed"
        );
        // `rm -d .` removes an empty workspace root — closed via the `d` flag.
        assert!(
            !check("rm -d ."),
            "rm -d . (empty-root removal) must be rejected"
        );
        assert!(!check("rm -d ./"), "rm -d ./ must be rejected");
    }

    #[test]
    fn test_find_delete_is_rejected() {
        // `find . -delete` recursively wipes the workspace with no shell
        // metacharacters (MAGI S2) — must be blocked regardless of start path.
        assert!(!check("find . -delete"), "find . -delete must be rejected");
        assert!(
            !check("find . --delete"),
            "double-dash find . --delete must also be rejected"
        );
        assert!(
            !check("find -delete"),
            "bare find -delete (path defaults to .) must be rejected"
        );
        assert!(
            !check("find src -delete"),
            "find <subdir> -delete must be rejected"
        );
        assert!(
            !check("find . -name x.rs -delete"),
            "find with -delete anywhere in the expression must be rejected"
        );
        // Non-deleting find stays allowed.
        assert!(
            check("find . -name main.rs"),
            "ordinary find search must stay allowed"
        );
        assert!(check("find . -type f"), "find -type f must stay allowed");
    }

    #[test]
    fn test_flag_value_path_cannot_escape_sandbox() {
        // `--flag=value` where value is a path must be validated too — a
        // `-`-prefixed token used to skip PathGuard entirely (MAGI S2).
        assert!(
            !check("grep --file=C:/Windows/System32/config/SAM foo"),
            "--file=<absolute> must be rejected"
        );
        assert!(
            !check("grep --file=../../etc/passwd foo"),
            "--file=<traversal> must be rejected"
        );
        // Legit non-path flag values still pass (validate as workspace-relative).
        assert!(
            check("git log --format=oneline"),
            "--format=oneline must stay allowed"
        );
        assert!(
            check("git log --max-count=5"),
            "--max-count=5 must stay allowed"
        );
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

        // Generous timeout: this test only verifies `echo` runs, not the timeout
        // path. A tight budget flakes under full-suite CPU contention because a
        // Windows PowerShell cold-start (~1-2 s) can eat most of a small budget.
        let args = serde_json::json!({
            "command": "echo 'Hello Rust'",
            "timeout": 30000
        });

        let result = tool.execute(args, &CancellationToken::new()).await;
        assert!(result.is_ok());

        let result_val = result.unwrap();
        let stdout = result_val["stdout"].as_str().unwrap();
        assert!(stdout.contains("Hello Rust"));
    }

    /// A binary outside `allowed_binaries` (`whoami`) is rejected as a security
    /// violation before any subprocess is spawned. (Formerly misnamed
    /// `test_bash_tool_timeout` — it never exercised the timeout path.)
    #[tokio::test]
    async fn test_non_whitelisted_binary_is_rejected() {
        let dir = tempdir().expect("Failed to create temp dir");
        let root = dir
            .path()
            .canonicalize()
            .expect("Failed to canonicalize root");

        let tool = BashTool::new(root.clone()).unwrap();

        let args = serde_json::json!({
            "command": "whoami"
        });
        let result = tool.execute(args, &CancellationToken::new()).await;
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

    // ── REQ-H36: wall-clock cancel kills the whole subprocess TREE ───────────────

    /// Deterministic subprocess-tree-kill proof for the Windows Job-Object path:
    /// a cancel fired mid-work terminates the grandchild worker, verified by
    /// marker files (START written, DONE never written), with no panic and no
    /// orphaned child. Runs on this host.
    #[cfg(windows)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_bash_cancel_kills_subprocess_tree_windows() {
        use crate::tools::proc_group::test_support::tree_kill_worker;
        cancel_kills_subprocess_tree(&tree_kill_worker()).await;
    }

    /// POSIX process-group analog of the Windows test above (CI-only).
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_bash_cancel_kills_subprocess_tree_unix() {
        use crate::tools::proc_group::test_support::tree_kill_worker;
        cancel_kills_subprocess_tree(&tree_kill_worker()).await;
    }

    /// REQ-H36 (Windows Job Object): a **grandchild** (shell→python→python) dies
    /// when the timeout fires. The direct child spawns a detached grandchild and
    /// lingers; only the Job Object's kill-on-close reaches the grandchild (it has
    /// no job of its own), so DONE never appears. Runs on this host.
    #[cfg(windows)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_bash_cancel_kills_grandchild_windows() {
        use crate::tools::proc_group::test_support::tree_kill_grandchild_worker;
        cancel_kills_subprocess_tree(&tree_kill_grandchild_worker()).await;
    }

    /// POSIX process-group analog of the grandchild-kill test above (CI-only):
    /// `kill(-pgid, SIGKILL)` reaches every descendant in the group.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_bash_cancel_kills_grandchild_unix() {
        use crate::tools::proc_group::test_support::tree_kill_grandchild_worker;
        cancel_kills_subprocess_tree(&tree_kill_grandchild_worker()).await;
    }

    /// REQ-H36 (Windows Job Object) — fork↔assign race window (Melchior's
    /// adversarial ask): the direct child spawns the grandchild **as early as
    /// possible** (the `Popen` is its first substantive action), maximizing the
    /// chance the grandchild exists before the module's immediate-post-spawn Job
    /// assignment completes. If the grandchild is still killed when the timeout
    /// fires (DONE marker never written), the immediate-assign window — the
    /// documented deviation from `CREATE_SUSPENDED` (which would require forbidden
    /// `unsafe`) — is not a practical escape hatch. Runs on this host.
    #[cfg(windows)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_bash_cancel_kills_early_spawned_grandchild_windows() {
        use crate::tools::proc_group::test_support::tree_kill_early_grandchild_worker;
        cancel_kills_subprocess_tree(&tree_kill_early_grandchild_worker()).await;
    }

    /// Shared body: run a real allowlisted long command (`python worker.py`)
    /// through the `bash` tool, fire the cancellation token mid-work, and assert
    /// the whole subprocess tree died — START marker present, DONE marker absent
    /// even after waiting past when a surviving orphan would have written it.
    /// Parameterized on the worker source so a single-child and a grandchild
    /// worker share one proof harness (DRY).
    #[cfg(any(windows, unix))]
    async fn cancel_kills_subprocess_tree(worker_src: &str) {
        use crate::tools::proc_group::test_support::{
            python_available, CANCEL_FIRE_DELAY_MS, POST_KILL_WAIT_MS,
        };

        if !python_available() {
            eprintln!("skipping: python interpreter not found — cannot spawn a real child");
            return;
        }
        let dir = tempdir().expect("tempdir");
        let root = dir.path().canonicalize().expect("canonicalize");
        std::fs::write(root.join("worker.py"), worker_src).expect("write worker");

        let tool = BashTool::new(root.clone()).expect("BashTool");
        let cancel = CancellationToken::new();
        let args = serde_json::json!({ "command": "python worker.py" });

        // Fire cancel after START is surely written (python cold start + margin,
        // generous enough to absorb full-suite CPU contention — see
        // `CANCEL_FIRE_DELAY_MS` rustdoc) but well before the worker's sleep
        // would complete, so the kill genuinely pre-empts live work.
        let cancel_fire = cancel.clone();
        let firer = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(CANCEL_FIRE_DELAY_MS)).await;
            cancel_fire.cancel();
        });

        let result = tool
            .execute(args, &cancel)
            .await
            .expect("execute returns an (aborted) value, never a panic");
        firer.await.expect("cancel task joins");

        assert_eq!(
            result["interrupted"],
            serde_json::json!(true),
            "a cancelled run must report interrupted=true"
        );

        let start = root.join("start.marker");
        let done = root.join("done.marker");
        assert!(
            start.exists(),
            "START marker must exist — the real subprocess actually ran"
        );
        // Wait past when a SURVIVING (orphaned) worker would have written DONE —
        // see `POST_KILL_WAIT_MS` rustdoc for the margin math. If the tree was
        // truly killed, DONE never appears.
        tokio::time::sleep(Duration::from_millis(POST_KILL_WAIT_MS)).await;
        assert!(
            !done.exists(),
            "DONE marker must NOT exist — the subprocess tree was killed, not orphaned"
        );
    }

    // ── REQ-H37: an in-workspace interpreter cannot exfiltrate the magi secrets ──

    /// Names of the three magi-managed secrets that must never reach a tool
    /// subprocess (REQ-H37).
    const EXFIL_SECRET_NAMES: [&str; 3] =
        ["MAGI_PASSPHRASE", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"];

    /// REQ-H37 pen-test: the three secrets are set in the PARENT environment, yet
    /// a tool subprocess spawned through the `bash` tool (`env_clear` + allowlist)
    /// cannot read them — neither from `os.environ` nor (unix) from
    /// `/proc/self/environ`.
    ///
    /// Two layers are asserted:
    /// 1. **Directly** (always): [`tool_child_env`] omits all three secret names
    ///    even while they are live in the parent env — the allowlist excludes
    ///    them by whole-name equality.
    /// 2. **End-to-end** (skips only if `python` is unavailable): a real
    ///    `python probe.py` run through the tool writes an empty `exfil.marker`,
    ///    proving the interpreter saw none of the three.
    ///
    /// Scope note (the `/proc/<ppid>/environ` vector): this test covers the
    /// **child**-environment layer. The complementary parent-scrub layer — which
    /// removes the same three names from the *parent* process env at startup so a
    /// child cannot read them via `/proc/<ppid>/environ` — is REQ-H37's symmetric
    /// `unsetenv`, verified by `test_scrub_removes_passphrase_and_api_keys_from_process_env`
    /// in `main.rs`. A `/proc/<ppid>/environ` probe is deliberately NOT added here:
    /// this test intentionally sets the secrets live in its parent env and cannot
    /// scrub them (`std::env::remove_var` is UB under this multi-thread runtime —
    /// the exact reason the real scrub runs single-threaded pre-runtime), so such a
    /// probe would report the harness's own env, not a magi defect.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[serial_test::serial]
    async fn test_bash_interpreter_cannot_exfiltrate_secrets() {
        use crate::tools::proc_group::test_support::{python_available, EXFIL_PROBE_WORKER};
        use crate::tools::proc_group::tool_child_env;

        // Set all three secrets live in the PARENT environment.
        for name in EXFIL_SECRET_NAMES {
            std::env::set_var(name, "top-secret-value");
        }

        // Layer 1 — direct: the allowlist-filtered child env omits every secret.
        let child_env: std::collections::HashMap<String, String> =
            tool_child_env().into_iter().collect();
        for name in EXFIL_SECRET_NAMES {
            assert!(
                !child_env.contains_key(name),
                "secret `{name}` must be absent from the tool child environment"
            );
        }

        // Layer 2 — end-to-end through the real bash-tool spawn.
        if python_available() {
            let dir = tempdir().expect("tempdir");
            let root = dir.path().canonicalize().expect("canonicalize");
            std::fs::write(root.join("probe.py"), EXFIL_PROBE_WORKER).expect("write probe");

            let tool = BashTool::new(root.clone()).expect("BashTool");
            let result = tool
                .execute(
                    serde_json::json!({ "command": "python probe.py", "timeout": 20000 }),
                    &CancellationToken::new(),
                )
                .await
                .expect("probe executes");
            assert_eq!(
                result["interrupted"],
                serde_json::json!(false),
                "the probe must run to completion, not be interrupted"
            );

            let marker = root.join("exfil.marker");
            assert!(
                marker.exists(),
                "the probe must have run and written exfil.marker"
            );
            let leaked = std::fs::read_to_string(&marker).expect("read exfil.marker");
            assert!(
                leaked.trim().is_empty(),
                "no secret may reach the child; probe reported leaked: {leaked:?}"
            );
        } else {
            eprintln!("skipping end-to-end probe: python interpreter not found");
        }

        // Clean up the parent environment (serialized test — no cross-test leak).
        for name in EXFIL_SECRET_NAMES {
            std::env::remove_var(name);
        }
    }
}
