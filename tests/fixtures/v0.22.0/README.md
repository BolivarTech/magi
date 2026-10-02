# v0.22.0 database fixture

`magi-rs-memory.db` was written by the magi-rs 0.22.0 release binary (built from commit `190c19b`) so that
`a_v0_22_0_database_opens_and_keeps_its_records_and_vault_entries` in `src/system/database.rs` can prove a
database from that release still opens with the current build and keeps its records.

Everything in it is toy data:

- passphrase: `correct horse battery staple` (the one `tests/workdir_flag.rs` uses);
- one vault entry, `TOY_SECRET` = `toy-secret-value`;
- one session with the message `fixture message` and the static provider's canned reply.

How it was generated, in an empty directory:

```
magi-rs init -p "correct horse battery staple"
# set provider = "anthropic" in .magi/magi.toml and leave every API key unset (static provider)
printf 'toy-secret-value' | magi-rs vault set TOY_SECRET -p "correct horse battery staple"
echo "fixture message" | magi-rs query -p "correct horse battery staple"
```

The database was copied after the process exited; no `-wal` or `-shm` file was left behind.
