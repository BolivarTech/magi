# Testing

```bash
cargo test                 # all unit + integration tests
cargo nextest run          # same, via nextest (preferred)
cargo test <name>          # single test by substring
cargo test -- --nocapture  # show println! output
```

Full verification (matches the project's per-phase gate):

```bash
cargo nextest run
cargo clippy --all-targets -- -D warnings
cargo fmt --check
cargo build --release
cargo doc --no-deps
cargo audit
cargo deny check licenses
```

## Smoke tests

The suite above exercises units. It cannot tell you whether the binary you
ship works, because a mock always answers and a debug build is not the
artifact anyone installs. The harness under `smoke/` covers that: it builds
the release binary, points it at a real backend and a real encrypted
database, and checks 81 assertions across 26 scenarios.

```bash
python -m smoke --init-env    # once: build the persistent test environment
python -m smoke --smoke-1     # run everything; never writes a certificate
```

It needs a reachable backend, the environment's passphrase in `smoke.toml`,
and the backend credential exported. Full instructions, what each scenario
guards, and what is deliberately out of scope are in
[`smoke/README.md`](https://github.com/BolivarTech/magi/blob/main/smoke/README.md).
That link is absolute because nothing under `smoke/` is part of the published
crate, so a relative one would dangle for anyone reading this on crates.io.

> Some tests use real OS resources: the OAuth callback server binds port `54545`, so avoid two interactive `/login` flows at once. DB/crypto/vault tests are isolated via `tempfile` plus injected connections and passphrase-prompt doubles. No OS keyring is touched anywhere (`system::secrets` was removed in v0.9.0).
