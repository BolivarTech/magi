# v0.11.0 `magi.toml` fixtures

**Real** magi-rs v0.11.0 configuration files, used by `src/config/migrate.rs` to test the
migration pass to v0.12.0 (REQ-A21c, SC-A21d).

## Why they are real and not hand-written

v0.12.0 breaks **every** v0.11.0 `magi.toml`, and the user's decision (2026-08-01) was not to
provide an escape hatch: the migration message is the **only** defense. If a pattern slips past
the message, the user is left with a binary that will not start and no clean downgrade.

A hand-written fixture proves the message **is emitted**. Only a real one proves it **reaches** —
that it names every incompatibility in that file and that what it proposes actually parses in
v0.12.0.

## Why they are committed and not generated at test time

Generating them during the suite would tie the tests to having v0.11.0 installed. It is not
present in CI, nor in a fresh clone, nor a year from now. A test that cannot run defends nothing,
and this is the only one that defends the migration.

## Provenance

Generated on **2026-08-02** with the published v0.11.0 binary:

```bash
cargo install magi-rs --version 0.11.0 --root <tmp>/magi-v11 --locked
cd <tmp>/genfix && <tmp>/magi-v11/bin/magi-rs.exe --init-config
```

| File | Origin |
|---|---|
| `default.toml` | **Verbatim** output of `magi-rs --init-config` (4769 bytes). Canonical. |
| `with-models.toml` | `default.toml` with the three per-mage models changed to non-built-in values. |
| `full.toml` | `default.toml` with five advanced `[memory]`/`[embedding]` knobs uncommented. |
| `with-credentials.toml` | `default.toml` with `[openai].base_url = "https://user:s3cr3t@host/v1"`, plus a trailing TOML comment on that line carrying the secret-scanner marker (see below). |

The three variants are derived from `default.toml` by adding or changing **only keys the v0.11.0
schema accepts** — verified against `git show v0.11.0:src/config.rs` and
`git show v0.11.0:src/memory/config.rs`. v0.11.0 parses with `deny_unknown_fields`, so a made-up
key would produce a file v0.11.0 itself would have rejected, and the fixture would prove nothing.

## How each one was verified

Each fixture was placed as `.magi/magi.toml` in a temporary workspace and run against the v0.11.0
binary, checking its output for the warning that binary emits when the configuration fails to
parse (`"is invalid and was ignored"`, from `MagiConfig::load`):

```bash
<tmp>/magi-v11/bin/magi-rs.exe init
cp <fixture> .magi/magi.toml
MAGI_PASSPHRASE="…" <tmp>/magi-v11/bin/magi-rs.exe query -i "hi" --timeout 2 2>&1 \
  | grep -ci "is invalid and was ignored"
```

Zero matches means v0.11.0 accepted the file. All four returned zero.

**The method was validated before being trusted.** It was first run against a deliberately
invalid file (`bogus_unknown_key = 1`), which did produce the warning. Without that control,
"the warning did not appear" would be indistinguishable from "the warning never appears," and all
four fixtures would have passed by construction.

## `with-credentials.toml` deliberately contains a secret

It carries `s3cr3t` embedded in the `base_url`. That is exactly what
`tests/no_hardcoded_secrets.rs` looks for, so it needs an explicit exemption in that scanner.

The exemption is **per line, not per directory**: the `allow-secret-scan` marker sits as a TOML
comment at the end of the `base_url` line, and `tests/fixtures` was **added** to the directories
the scanner walks. Excluding the whole directory would have been simpler and worse — fixtures are
exactly the kind of file where someone accidentally pastes a real credential while generating
them, so the surface has to stay watched except on the one line where the secret is deliberate.

The comment **does not change what v0.11.0 parses** — it is a TOML comment — and the file was
re-verified against the v0.11.0 binary **after** adding it, not before.

## Regenerating them

Repeat the commands above. `default.toml` should come out byte-identical as long as the same
published v0.11.0 is used; the variants are re-derived from it with the changes in the table.
