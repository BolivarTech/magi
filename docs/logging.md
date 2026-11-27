# Logging

Magi writes one log file per day under `.magi/logs/`, and everything running in that
workspace writes to the same one. It rolls over at UTC midnight. Past
`compress_after_days` a rotated file becomes `.xz`; past `retain_days` it is deleted.
And if the directory as a whole would exceed `max_total_bytes`, the oldest go first.

Every line passes an auditor before it reaches the disk. A credential that finds its
way into a log line comes out masked, and the masking is announced rather than done
quietly, so you learn that something tried. The pass runs on the writing side of the
layer rather than at the call sites, which is why it also covers text a dependency
composed.

The same auditor covers the JSON envelope `query` and `consult` print. The envelope's
`transcript` carries tool output: what `view` read, what `bash` returned. So a run that
touches a file holding a credential would otherwise put it on stdout and into whatever
your CI keeps. A tool result that is itself a JSON document gets masked field by field
rather than as flat text, so what a consumer parses comes back parseable. One
consequence to know about: if a key inside that document collides with a credential, the
key is masked too and the shape changes.

## What reaches the screen

`ERROR` and `WARN` reach the screen. `INFO` and below go to the day's file and nowhere
else. A startup with nothing degraded and nothing misconfigured therefore prints not one
line before the first prompt; the diagnostics it used to print are in the log, and the
screen is quiet because there is somewhere else for them to be.

A subsystem that starts failing says so once, not once per event, and says so again when
it recovers. The first failure is announced immediately. A defence against flapping that
delayed the first "something broke" would be defending the wrong thing. Every later
transition waits out a 30 second stability window, so an endpoint that flaps six times
in twenty seconds costs you one line while the file still records all twelve events.

An error on the screen names three things: what broke, what it means for this session,
and where to read more. The third is the day's log path, spelled the way your filesystem
spells it, so it pastes into an editor without editing.

## Finding one run in a shared file

Each line carries `run=<id>`. The file belongs to the day, not to an invocation, so
that id is how you tell your run from whatever else was writing at the time:

```bash
grep 'run=12345-a1b2c3d4e5f60789' .magi/logs/magi-2026-08-31.log
```

You get the id three ways. It is a field in the JSON envelope `query` and `consult`
print. It is on a `run: <id>` line on stderr, for a job that reads nothing else. And it
opens the run's own log, naming the subcommand and the workspace. A long event gets
split across several lines and the continuations carry the id too, so filtering returns
the whole thing rather than its first line.

## Configuration

```toml
[logging]
log_dir             = ".magi/logs"
file_filter         = "info"
compress            = true
compress_after_days = 2
retain_days         = 30
max_total_bytes     = 536870912
```

Those six keys and no others. A key the binary does not read is not accepted: write one
and you get a load error naming it, rather than a setting that looks applied and does
nothing. `file_filter` takes a bare level, or a comma-separated list of `target=level`
directives: `magi_rs=debug,warn` turns the agent up and leaves everything else
at `warn`.

The directory can be overridden without touching the file. `--log-dir` wins, then
`MAGI_LOG_DIR`, then the setting above.

## If the log branch stops

A full queue, an oversized event, a writer thread that died: any of these starts the
file branch dropping events. It says so once, on the notice channel, and not per event.
A queue that is full stays full, and an unlatched warning would bury the one problem
under a flood of its own reports. What it will not do is take the agent down. Logging
failing is not a reason to stop answering.
