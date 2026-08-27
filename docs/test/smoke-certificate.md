# Smoke Certificate

- version: 0.17.0
- commit: d89c1ee
- date: 2026-08-27 (UTC)
- profile: product defaults (no --profile)
- binary: sha256 3dac3b1638c933fd3744573727a4fce79a37691a0836a54922ce514fe395320d (rebuilt from the commit above)
- real cost: 8 backend run(s) in 171s
- rounds needed: 2
- scope: 22 of 22 scenarios evaluated
- contract coverage: StructuredVerdicts v0.14.3 (6/6 REQ-EA); Vault v0.9.0, Headless v0.10.0, MagiCore MS1-MS3, OperationBudget -- partial
- environment: 18 active memories, 872 KB database, 377 KB archived
- result: 69 passed, 0 not passed, 69 total

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
[PASS] S12 - the run adds no entry to git status --porcelain beyond the certificate
[PASS] S12 - git status --ignored smoke/env/ shows the whole environment on the ignored side
[PASS] S13 - every magi-rs invocation in the published docs names an existing subcommand
[PASS] S13 - every flag in those invocations exists in that subcommand's --help
[PASS] S13 - every magi.toml embedded in those docs parses
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
[OUT_OF_SCOPE] - cross-OS linkage and the published crate (REQ-S26, REQ-S27)
