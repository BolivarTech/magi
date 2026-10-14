# Smoke Certificate

- version: 0.24.2
- commit: 9f08e47
- date: 2026-10-14 (UTC)
- profile: product defaults (no --profile)
- binary: sha256 03b81695f54fb4cc17e3c502c362ee85ff7db7bd5333e2907103e3a888a36ec4 (rebuilt from the commit above)
- real cost: 8 backend run(s) in 680s
- rounds needed: 3
- scope: 43 of 43 scenarios evaluated
- contract coverage: StructuredVerdicts v0.14.3 (6/6 REQ-EA); Vault v0.9.0, Headless v0.10.0, MagiCore MS1-MS3, OperationBudget -- partial
- environment: active memories not measured, 4001 KB database, 2061 KB archived
- result: 127 passed, 0 not passed, 127 total

[PASS] D1 - decide answers one question of each type against local Ollama with exit 0
[PASS] D1 - every answer carries a Magi confidence between 0 and 1
[PASS] D1 - the envelope names the model that answered and its usage
[PASS] D2 - decide answers one question of each type against OpenRouter with exit 0
[PASS] D2 - the envelope reports a numeric cost
[PASS] D2 - the score answer has a legend for every level
[PASS] D3 - a missing primary model rotates to the declared fallback
[PASS] D3 - the chain records the primary as model_not_found
[PASS] D4 - a model excluded by account policy rotates to the next one
[PASS] D4 - the hop names account_policy and repeats no provider text
[PASS] D5 - a hosted endpoint without DECISION_API_KEY exits 2 with missing_key
[PASS] D5 - the message names DECISION_API_KEY
[PASS] D5 - with the same configuration vault ls still runs
[PASS] D6 - a v0.22.0 magi.toml is refused naming the missing [decision] section
[PASS] D6 - an empty magi.toml is refused offering to delete it or run magi-rs init
[PASS] D7 - a dead endpoint exits 1 with decision_unavailable and no answers
[PASS] D7 - a base_url without the API prefix is diagnosed with the URL that was tried
[PASS] D8 - when no model reaches its min_confidence the call exits 1
[PASS] D8 - every hop is reported as low_confidence with its confidence and threshold
[PASS] D9 - the planted credential does not reach stdout
[PASS] D9 - the planted credential does not reach stderr
[PASS] D9 - the planted credential does not reach the run log
[PASS] D10 - -q with questions also in the input exits 2 before any request
[PASS] D10 - -o writes the envelope and leaves stdout empty
[PASS] D10 - --timeout overrides timeout_secs per attempt
[PASS] D10 - an explicit model is tried alone
[PASS] D11 - when the agent calls decide in the default tier the call is approved
[PASS] D12 - a consult with a dead decision endpoint still completes
[PASS] D12 - the run log records one mode fallback with its cause
[PASS] D12 - nothing about the fallback reaches stderr or stdout
[PASS] D13 - the decide call in tool_calls stores the state as <omitted>
[PASS] D13 - the decide call in tool_calls keeps its questions and its result
[PASS] D13 - the state marker reaches neither the decide record nor the decide log lines
[PASS] D14 - the scaffold names clef-flash with its threshold active
[PASS] D14 - magi decide answers the example request with clef-flash
[PASS] K1 - the run log has exactly one entry naming MAGI_API_KEY and http://127.0.0.1:11434
[PASS] K1 - the screen shows nothing about the retained key
[PASS] K2 - a sk-ant- MAGI_API_KEY under openai-compat warns naming MAGI_API_KEY and [magi].kind, without the value
[PASS] K2 - a sk-or- MAGI_API_KEY under anthropic warns the same way
[PASS] K2 - a sk-or- MAGI_API_KEY under openai-compat does not warn
[PASS] K3 - with MAGI_API_KEY the OpenRouter trio answers a consult
[PASS] K3 - without it the consult failure names MAGI_API_KEY
[PASS] S1 run=R1 - exit 0 and stdout parses as JSON
[PASS] S1 run=R1 - schema_version is 1
[PASS] S1 run=R1 - every top-level contract key is present
[PASS] S1 run=R1 - every nested key of applied_caps is present
[PASS] S2 - magi init creates .magi/ in an empty directory
[PASS] S2 - the permissions are restrictive — POSIX bits on Unix, ACL on Windows
[PASS] S2 - a second init refuses and leaves the directory unchanged
[PASS] S2 - query from a nested subdirectory finds the ancestor .magi/
[PASS] S3 - vault set accepts the value from stdin and vault ls lists its name
[PASS] S3 - vault ls prints name and timestamps, never the value
[PASS] S3 - the planted value appears nowhere in stdout, stderr or the run log
[PASS] S3 - no subcommand exists that prints a stored value
[PASS] S3 - vault rm removes it and ls no longer lists it
[PASS] S4 - opening with a wrong passphrase fails with a typed WrongPassphrase, exit 1
[PASS] S4 - reopening with the correct passphrase still finds the accumulated history
[PASS] S5 run=R1 - tool_calls[] records the invocation with ok: true
[PASS] S5 - a prompt asking to read outside the workspace is denied
[PASS] S6 run=R4 - three verdicts are present
[PASS] S6 run=R4 - a consensus was computed
[PASS] S6 run=R4 - the run is not degraded
[PASS] S7 run=R4 - applied_caps satisfies the derived relation between the operation budget, the client timeout and the ceiling
[PASS] S7 run=R4 - the trio is not degraded under a high --timeout
[PASS] S7 run=R4 - ceiling_floored and ceiling_above_sanity report what corresponds
[PASS] S8 run=R4 - the large-payload run completes without truncation
[PASS] S8 run=R4 - usage.input_tokens confirms the size in tokens, above the declared floor
[PASS] S8 run=R4 - the generated payload stayed under the input cap
[PASS] S9 run=R3 - the startup line reports N active with N > 0
[PASS] S9 run=R3 - pending re-embed is 0 — the embedder answered
[PASS] S9 run=R3 - R3's transcript carries turns R2's does not — the assembler loaded them
[PASS] S9 run=R3 - the environment is below the saturation ceiling — otherwise 3 degrades to CANNOT_TEST
[PASS] S9 - with the embedder down, the run completes with a degradation notice
[PASS] S10 run=R6 - the credential appears in neither stdout nor stderr
[PASS] S10 run=R6 - nor in the JSON output
[PASS] S10 run=R6 - nor in the run log
[PASS] S10 run=R6 - every authority emitted anywhere — JSON, stdout, stderr, run log — is redacted; asserted on shape, not on the secret
[PASS] S11 - an unknown field in magi.toml exits 2 naming the field
[PASS] S11 - it cuts before any backend request is issued
[PASS] S11 - a seat declaring a model without its lineage fails naming all three seats
[PASS] S11 - an unknown reasoning value exits 2 naming the key and every accepted value
[PASS] S12 - the run adds no entry to git status --porcelain beyond the certificate
[PASS] S12 - git status --ignored smoke/env/ shows the whole environment on the ignored side
[PASS] S13 - every magi-rs invocation in the published docs names an existing subcommand
[PASS] S13 - every flag in those invocations exists in that subcommand's --help
[PASS] S13 - every magi.toml embedded in those docs parses
[PASS] S13 - every [magi] key the scaffold writes is named in a published guide
[PASS] S13 - the documented decide flags exist
[PASS] S13 - the [decision] example in docs/magi.toml.example parses
[PASS] S13 - the documented /decide forms and the decide tool exist
[PASS] S13 - no help screen of the release binary shows a development id
[PASS] S14 - init -w <dir> scaffolds into <dir> and leaves the current directory untouched
[PASS] S14 - vault -w <dir> ls and vault ls -w <dir> both parse
[PASS] S14 - given twice, the innermost wins
[PASS] S14 - on init it is not global, so repeating it is a clap error
[PASS] S15 - each text-valued variable, exported empty or blank, falls through to the next precedence level
[PASS] S15 - startup succeeds — no vocabulary error, no empty credential short-circuiting the vault lookup
[PASS] S15 - a value that is present and unrecognised is still an error
[PASS] S16 run=R7 - after rotating the stored API key, the DB still opens with the same passphrase
[PASS] S16 run=R7 - the previous history is still there
[PASS] S17 - query --structured-verdicts is a clap parse error — not an accepted no-op, not a runtime exit 2
[PASS] S17 - consult --structured-verdicts --output-format text exits 2
[PASS] S18 run=R4 - with the flag, agents and consensus are both present
[PASS] S18 run=R8 - without it, both are absent
[PASS] S18 run=R4 - agents[] exposes exactly 7 keys
[PASS] S18 run=R4 - findings[] exposes exactly 6 keys
[PASS] S18 run=R4 - consensus exposes exactly 7 keys
[PASS] S19 run=R5 - the consult tool result inside tool_calls[] contains no agents key
[PASS] S19 run=R5 - it contains no consensus key
[PASS] S20 run=R4 - the trio completed against the native wire
[PASS] S20 run=R4 - every transmitted attempt carries the declared completion cap
[PASS] S20 run=R4 - completions are recorded per attempt
[PASS] S20 run=R4 - the published per-mage threshold agrees with the attempt-factor formula
[PASS] S21 run=R4 - every recorded attempt reports a finish this build knows, or an explicit null
[PASS] S21 run=R4 - the rotations report is published and every hop names a known cause and its locality
[PASS] S22 run=R4 - pool_eligibility is present even when empty
[PASS] S22 run=R4 - all three notions of degradation are derivable from published keys
[PASS] S22 run=R4 - degraded is false for a three-verdict run
[PASS] S23 run=R1 - the run id printed on stderr is the one carried in the JSON envelope
[PASS] S23 run=R1 - that same run id appears inside a file in the workspace log directory, so the log survived process exit and belongs to this run
[PASS] S24 run=R1 - the memory-count diagnostic notice never reaches stdout or stderr, the headless run's screen
[PASS] S24 run=R1 - that same notice is written into the day's log file under the workspace's log directory
[PASS] S25 - a consult under reasoning = disabled reports the disabled control on every recorded attempt
[PASS] S25 - under reasoning = disabled no attempt reports a measured reasoning count above zero
[PASS] S25 - no attempt record carries reasoning text, even with the trace opted in
[PASS] S26 run=R4 - with no reasoning key declared every recorded attempt reports the default control
[PASS] S26 run=R4 - every recorded attempt reports its reasoning state in magi-core's own shape
[OUT_OF_SCOPE] - cross-OS linkage and the published crate (REQ-S26, REQ-S27)
