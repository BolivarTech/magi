# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-08-25
"""Unit tests for the S3 scenario's own shape."""

import unittest

from smoke.outcome import Outcome
from smoke.product import ProductOutput
from smoke.registry import DEFAULT_REGISTRY
from smoke.runs import RunResult
from smoke.scenarios import vault  # noqa: F401 - import registers it
from smoke.tests import support

#: What a well-behaved ``vault ls`` prints: name, created, updated.
_LS_LINE = "%s · 2026-08-25T10:00:00Z · 2026-08-25T10:00:00Z"


class _FakeVault:
    """A product double whose vault behaves, so S3 can be seen to PASS.

    A scenario that only ever reports FAIL against a double is untestable in
    the direction that matters: nothing proves it CAN conclude PASS, so a
    permanent red would look like a working guardian.
    """

    def __init__(self, leak: bool = False) -> None:
        """Create the double.

        Args:
            leak: Whether ``ls`` should echo the value it was handed, which is
                the defect assertion 3 exists to catch.
        """
        self.leak = leak
        self.value = b""
        self.stored = False

    def __call__(self, call: support.Call) -> ProductOutput | None:
        """Answer one invocation.

        Args:
            call: What the fake binary was asked to run.

        Returns:
            ProductOutput: The canned answer, or None for the failed default.
        """
        args = list(call.args)
        if args[:1] != ["vault"]:
            return None
        if "set" in args:
            self.value = (call.stdin or b"").strip()
            self.stored = True
            return self._ok(b"secret '%s' stored" % vault.PROBE_NAME.encode())
        if "rm" in args:
            self.stored = False
            return self._ok(b"secret '%s' removed" % vault.PROBE_NAME.encode())
        if "--help" in args:
            return self._ok(b"Commands:\n  ls\n  set\n  rm\n  passwd\n  diagnose\n")
        if "ls" in args:
            if not self.stored:
                return self._ok(b"(vault empty)")
            line = (_LS_LINE % vault.PROBE_NAME).encode("utf-8")
            if self.leak:
                line += b" " + self.value
            return self._ok(line)
        return None

    @staticmethod
    def _ok(stdout: bytes) -> ProductOutput:
        """Wrap *stdout* in a successful capture.

        Args:
            stdout: What the product printed.

        Returns:
            ProductOutput: Exit 0 with that output.
        """
        return ProductOutput(stdout=stdout, stderr=b"", exit_code=0,
                             command=["magi-rs", "vault"])


def _outcomes(scenario_id: str) -> dict[str, Outcome]:
    """Run one registered scenario and index its outcomes by assertion.

    Args:
        scenario_id: The id to invoke.

    Returns:
        dict[str, Outcome]: What each assertion concluded.
    """
    findings = list(DEFAULT_REGISTRY.get(scenario_id).func(None))
    return {finding.assertion: finding.outcome for finding in findings}


class VaultScenarioTests(unittest.TestCase):
    """S3 is registered standalone and declares its five assertions."""

    def test_s3_is_registered_without_a_run(self) -> None:
        entry = DEFAULT_REGISTRY.get("S3")
        self.assertIsNone(entry.run)
        self.assertFalse(entry.needs_backend)

    def test_the_assertion_texts_are_the_spec_texts(self) -> None:
        self.assertEqual(
            [
                "vault set accepts the value from stdin and vault ls lists its name",
                "vault ls prints name and timestamps, never the value",
                "the planted value appears nowhere in stdout, stderr or the run log",
                "no subcommand exists that prints a stored value",
                "vault rm removes it and ls no longer lists it",
            ],
            list(vault.S3_ASSERTIONS),
        )


class VaultScenarioBodyTests(unittest.TestCase):
    """What S3 concludes, against a product that behaves and one that leaks."""

    def test_a_product_that_does_nothing_still_reports_all_five(self) -> None:
        support.install_fake_runs(self)
        findings = list(DEFAULT_REGISTRY.get("S3").func(None))
        self.assertEqual(list(vault.S3_ASSERTIONS),
                         [finding.assertion for finding in findings])
        self.assertNotIn(Outcome.PASS,
                         {finding.outcome for finding in findings})

    def test_a_well_behaved_vault_passes_every_assertion(self) -> None:
        support.install_fake_runs(self, responder=_FakeVault())
        self.assertEqual({Outcome.PASS}, set(_outcomes("S3").values()))

    def test_a_listing_that_echoes_the_value_fails_the_search(self) -> None:
        """The one assertion that cannot be read off the code. It searches the
        capture for every derived form of the planted value, so a product that
        prints it -- in any encoding -- goes red here and nowhere else.
        """
        support.install_fake_runs(self, responder=_FakeVault(leak=True))
        outcomes = _outcomes("S3")
        leaked = vault.S3_ASSERTIONS[2]
        self.assertEqual(Outcome.FAIL, outcomes[leaked])

    def test_the_value_never_travels_as_an_argument(self) -> None:
        """REQ-V10: the value is read from stdin, never from the command line,
        where any process on the machine can read it while the child lives.
        """
        binary = support.install_fake_runs(self, responder=_FakeVault())
        list(DEFAULT_REGISTRY.get("S3").func(None))
        sets = [call for call in binary.calls if "set" in call.args]
        self.assertTrue(sets, "S3 never ran vault set")
        for call in sets:
            self.assertTrue(call.stdin, "the value must travel on stdin")
            planted = call.stdin.strip().decode("utf-8")
            self.assertNotIn(planted, call.args)

    def test_a_help_listing_a_reveal_subcommand_fails(self) -> None:
        """Assertion 4 is verified against the parser's own output. "I am not
        aware of a get subcommand" is not evidence; a Commands list is.
        """
        class _Revealing(_FakeVault):
            def __call__(self, call: support.Call) -> ProductOutput | None:
                if "--help" in call.args:
                    return self._ok(b"Commands:\n  ls\n  get\n  set\n")
                return super().__call__(call)

        support.install_fake_runs(self, responder=_Revealing())
        outcomes = _outcomes("S3")
        self.assertEqual(Outcome.FAIL, outcomes[vault.S3_ASSERTIONS[3]])


class _FakeDatabase:
    """A product double with an envelope, some history, and a passphrase.

    Attributes:
        counts: The per-table row counts ``vault diagnose`` reports.
        envelope: Whether an envelope exists to fail an unwrap against.
        wrong_passphrase_opens: Whether a wrong passphrase is accepted, which
            is the regression assertion 1 exists to catch.
        destroy_on_wrong: Whether a wrong passphrase wipes the history, which
            is the one REQ-V35 forbids and whose cost is the user's data.
    """

    def __init__(self, counts: tuple[int, ...] = (2, 1, 8, 3, 5),
                 envelope: bool = True,
                 wrong_passphrase_opens: bool = False,
                 destroy_on_wrong: bool = False) -> None:
        """Create the double.

        Args:
            counts: vault, sessions, messages, knowledge, memories.
            envelope: Whether the envelope is present.
            wrong_passphrase_opens: Whether the wrong passphrase is accepted.
            destroy_on_wrong: Whether a refused open empties the tables.
        """
        self.counts = list(counts)
        self.envelope = envelope
        self.wrong_passphrase_opens = wrong_passphrase_opens
        self.destroy_on_wrong = destroy_on_wrong

    def __call__(self, call: support.Call) -> ProductOutput | None:
        """Answer one invocation.

        Args:
            call: What the fake binary was asked to run.

        Returns:
            ProductOutput: The canned answer, or None for the failed default.
        """
        args = list(call.args)
        if args[:1] != ["vault"]:
            return None
        if "diagnose" in args:
            return _capture(self._report().encode("utf-8"), b"", 0)
        given = (call.env or {}).get("MAGI_PASSPHRASE")
        if given == vault.WRONG_PASSPHRASE:
            if self.destroy_on_wrong:
                self.counts = [0] * len(self.counts)
            if not self.wrong_passphrase_opens:
                return _capture(b"", b"error: incorrect passphrase\n", 1)
        return _capture(b"(vault empty)\n", b"", 0)

    def _report(self) -> str:
        """Render what ``vault diagnose`` prints.

        Returns:
            str: The envelope line, the verdict, and the five counts.
        """
        labels = ("vault", "sessions", "messages", "knowledge", "memories")
        lines = ["envelope: %s" % ("present" if self.envelope else "absent"),
                 "fec: ok", "verdict: healthy", "counts:"]
        absent = getattr(self, "missing", ())
        lines += ["  %s: %s" % (label, "missing" if label in absent else count)
                  for label, count in zip(labels, self.counts)]
        return "\n".join(lines) + "\n"


def _capture(stdout: bytes, stderr: bytes, exit_code: int) -> ProductOutput:
    """Build a canned capture.

    Args:
        stdout: What the product printed.
        stderr: What it wrote to the error stream.
        exit_code: The code it exited with.

    Returns:
        ProductOutput: The capture.
    """
    return ProductOutput(stdout=stdout, stderr=stderr, exit_code=exit_code,
                         command=["magi-rs", "vault"])


class WrongPassphraseScenarioTests(unittest.TestCase):
    """S4 is registered standalone and declares its two assertions."""

    def test_s4_is_registered_without_a_run(self) -> None:
        entry = DEFAULT_REGISTRY.get("S4")
        self.assertIsNone(entry.run)
        self.assertFalse(entry.needs_backend)

    def test_the_assertion_texts_are_the_spec_texts(self) -> None:
        self.assertEqual(
            [
                "opening with a wrong passphrase fails with a typed "
                "WrongPassphrase, exit 1",
                "reopening with the correct passphrase still finds the "
                "accumulated history",
            ],
            list(vault.S4_ASSERTIONS),
        )


class WrongPassphraseScenarioBodyTests(unittest.TestCase):
    """The scenario whose failure is total loss of the user's data."""

    def test_a_product_that_does_nothing_still_reports_both(self) -> None:
        support.install_fake_runs(self)
        findings = list(DEFAULT_REGISTRY.get("S4").func(None))
        self.assertEqual(list(vault.S4_ASSERTIONS),
                         [finding.assertion for finding in findings])
        self.assertNotIn(Outcome.PASS,
                         {finding.outcome for finding in findings})

    def test_an_environment_with_history_passes_both(self) -> None:
        support.install_fake_runs(self, responder=_FakeDatabase())
        self.assertEqual({Outcome.PASS}, set(_outcomes("S4").values()))

    def test_an_empty_environment_cannot_test_either_assertion(self) -> None:
        """The trap this scenario is written around. Over a database with no
        envelope and no rows, assertion 2 would pass while checking nothing --
        there is no accumulated history to still be there -- and a guardian
        whose precondition is empty is the failure this harness chases.
        """
        support.install_fake_runs(
            self, responder=_FakeDatabase(counts=(0, 0, 0, 0, 0),
                                          envelope=False))
        self.assertEqual({Outcome.CANNOT_TEST}, set(_outcomes("S4").values()))

    def test_a_wrong_passphrase_that_opens_the_vault_fails(self) -> None:
        support.install_fake_runs(
            self, responder=_FakeDatabase(wrong_passphrase_opens=True))
        outcomes = _outcomes("S4")
        self.assertEqual(Outcome.FAIL, outcomes[vault.S4_ASSERTIONS[0]])

    def test_history_destroyed_by_the_refusal_fails(self) -> None:
        """REQ-V35 itself: a wrong passphrase and a corrupt wrapped key fail
        the same tag check, so wiping on failure turns a typo into total loss.
        """
        support.install_fake_runs(
            self, responder=_FakeDatabase(destroy_on_wrong=True))
        outcomes = _outcomes("S4")
        self.assertEqual(Outcome.FAIL, outcomes[vault.S4_ASSERTIONS[1]])

    def test_the_wrong_passphrase_is_not_the_configured_one(self) -> None:
        """A "wrong" passphrase that happens to be the real one would make
        assertion 1 red for the wrong reason, and assertion 2 vacuous.
        """
        self.assertNotEqual(support.FAKE_PASSPHRASE, vault.WRONG_PASSPHRASE)


class _RotatedDatabase(_FakeDatabase):
    """The database as S16 finds it: after R7 rotated the stored API key.

    Attributes:
        opens: Whether the configured passphrase still opens the database. A
            False here is the defect the durable invariant of section 0.2
            forbids -- rotating a third-party credential re-keying the local
            vault -- and it is what assertion 1 exists to catch.
    """

    def __init__(self, opens: bool = True, missing: tuple = (), **kwargs) -> None:
        """Create the double.

        Args:
            opens: Whether the configured passphrase is still accepted.
            missing: Tables the report renders as ``missing`` rather than a
                number -- what the product does for one it creates lazily.
            **kwargs: Passed through to :class:`_FakeDatabase`.
        """
        super().__init__(**kwargs)
        self.opens = opens
        self.missing = missing

    def __call__(self, call: support.Call) -> ProductOutput | None:
        """Answer one invocation.

        Args:
            call: What the fake binary was asked to run.

        Returns:
            ProductOutput: The canned answer, or None for the failed default.
        """
        args = list(call.args)
        if args[:1] == ["vault"] and "diagnose" not in args and not self.opens:
            return _capture(b"", b"error: incorrect passphrase\n", 1)
        return super().__call__(call)


def _r7_result(timed_out: bool = False, baseline=None) -> RunResult:
    """Build the R7 capture S16 reads.

    R7 exits non-zero on purpose: it queries while the backend credential holds
    the rotation sentinel, so the call it makes cannot authenticate. What S16
    asks about is the database the product kept using around that failure, not
    the answer the backend refused to give.

    Args:
        timed_out: Whether the run exceeded its ceiling.

    Returns:
        RunResult: The capture, or the truncated one.
    """
    return RunResult(run_id="R7", output=_capture(b"", b"error: 401\n", 1),
                     duration_s=1.0, timed_out=timed_out, planted=(),
                     baseline=baseline)


def _s16_outcomes(run: RunResult | None) -> dict[str, Outcome]:
    """Run S16 over one R7 capture and index its outcomes by assertion.

    Args:
        run: What R7 produced, or None when it never executed.

    Returns:
        dict[str, Outcome]: What each assertion concluded.
    """
    findings = list(DEFAULT_REGISTRY.get("S16").func(run))
    return {finding.assertion: finding.outcome for finding in findings}


class CredentialRotationScenarioTests(unittest.TestCase):
    """S16 hangs off R7 and declares its two assertions."""

    def test_s16_is_registered_with_its_declared_run(self) -> None:
        entry = DEFAULT_REGISTRY.get("S16")
        self.assertEqual("R7", entry.run)
        self.assertTrue(entry.needs_backend)

    def test_the_assertion_texts_are_the_spec_texts(self) -> None:
        self.assertEqual(
            [
                "after rotating the stored API key, the DB still opens with "
                "the same passphrase",
                "the previous history is still there",
            ],
            list(vault.S16_ASSERTIONS),
        )


class CredentialRotationScenarioBodyTests(unittest.TestCase):
    """Rotating a third-party credential must not cost the user their data."""

    def test_a_rotation_that_never_ran_cannot_test_either_assertion(self) -> None:
        """With R7 absent nothing was rotated, so a database that opens says
        nothing about surviving a rotation that never happened.
        """
        support.install_fake_runs(self, responder=_RotatedDatabase())
        self.assertEqual({Outcome.CANNOT_TEST},
                         set(_s16_outcomes(None).values()))

    def test_a_timed_out_rotation_is_the_runner_s_to_answer(self) -> None:
        """Same as S10: the branch was unreachable, the declaration is not."""
        self.assertFalse(DEFAULT_REGISTRY.get("S16").inspects_timeouts)

    def test_a_database_that_survived_the_rotation_passes_both(self) -> None:
        support.install_fake_runs(self, responder=_RotatedDatabase())
        run = _r7_result(baseline={"sessions": 1, "messages": 8,
                                   "knowledge": 3, "memories": 5})
        self.assertEqual({Outcome.PASS}, set(_s16_outcomes(run).values()))

    def test_history_the_rotation_destroyed_fails(self) -> None:
        """"The previous history is still there" needs a BEFORE.

        Read once, after the rotation, the assertion passed whenever any table
        was non-zero -- so a rotation that destroyed most of the history was
        green. S4 does the same job correctly with a before/after comparison,
        120 lines away in this same module.
        """
        support.install_fake_runs(self, responder=_RotatedDatabase())
        run = _r7_result(baseline={"sessions": 2, "messages": 40,
                                   "knowledge": 3, "memories": 5})
        outcomes = {f.assertion: f.outcome
                    for f in DEFAULT_REGISTRY.get("S16").func(run)}
        self.assertEqual(Outcome.FAIL, outcomes[vault.S16_ASSERTIONS[1]])

    def test_a_table_nobody_measured_is_not_a_loss(self) -> None:
        """Unknown is not empty.

        ``diagnose_counts`` leaves a table reported ``missing`` out of the
        mapping and says so, and then reading it back with ``.get(table, 0)``
        flattened absent to zero and reported FAIL -- a blocking red -- over a
        table nobody measured. The two halves have to agree.
        """
        support.install_fake_runs(self, responder=_RotatedDatabase(
            counts=(1, 1, 8, 3, 5), missing=("memories",)))
        run = _r7_result(baseline={"sessions": 1, "messages": 8,
                                   "knowledge": 3, "memories": 5})
        outcomes = {f.assertion: f.outcome
                    for f in DEFAULT_REGISTRY.get("S16").func(run)}
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[vault.S16_ASSERTIONS[1]])

    def test_the_vault_table_is_not_part_of_the_history(self) -> None:
        """The rotation writes and removes its own vault entries.

        R7 plants a marker before it rotates and removes it after, and R6's
        authenticated endpoint plants four placeholder entries and removes
        those too. Counting the ``vault`` table therefore reads the rotation's
        own bookkeeping as the user losing data -- which is exactly what a live
        run reported: "vault went from 1 to 0". S4 excludes the table for the
        same reason and says so; the baseline had not.
        """
        support.install_fake_runs(self, responder=_RotatedDatabase())
        run = _r7_result(baseline={"vault": 1, "sessions": 1, "messages": 8,
                                   "knowledge": 3, "memories": 5})
        outcomes = {f.assertion: f.outcome
                    for f in DEFAULT_REGISTRY.get("S16").func(run)}
        self.assertEqual(Outcome.PASS, outcomes[vault.S16_ASSERTIONS[1]])

    def test_history_that_survived_intact_passes(self) -> None:
        support.install_fake_runs(self, responder=_RotatedDatabase())
        run = _r7_result(baseline={"sessions": 1, "messages": 8,
                                   "knowledge": 3, "memories": 5})
        outcomes = {f.assertion: f.outcome
                    for f in DEFAULT_REGISTRY.get("S16").func(run)}
        self.assertEqual(Outcome.PASS, outcomes[vault.S16_ASSERTIONS[1]])

    def test_a_run_with_no_baseline_cannot_test_the_history(self) -> None:
        """A rotation that recorded no baseline leaves nothing to compare."""
        support.install_fake_runs(self, responder=_RotatedDatabase())
        outcomes = _s16_outcomes(_r7_result())
        self.assertEqual(Outcome.CANNOT_TEST, outcomes[vault.S16_ASSERTIONS[1]])

    def test_a_passphrase_the_rotation_invalidated_fails(self) -> None:
        """The durable invariant of section 0.2: rotating a third-party
        credential never invalidates the local encrypted database.
        """
        support.install_fake_runs(self, responder=_RotatedDatabase(opens=False))
        self.assertEqual(Outcome.FAIL,
                         _s16_outcomes(_r7_result())[vault.S16_ASSERTIONS[0]])

    def test_an_empty_environment_cannot_test_the_history(self) -> None:
        """Same trap as S4: over a database with no rows, "the previous
        history is still there" is true of a history that never existed.
        """
        support.install_fake_runs(
            self, responder=_RotatedDatabase(counts=(0, 0, 0, 0, 0)))
        self.assertEqual(Outcome.CANNOT_TEST,
                         _s16_outcomes(_r7_result())[vault.S16_ASSERTIONS[1]])

    def test_a_product_that_does_nothing_still_reports_both(self) -> None:
        support.install_fake_runs(self)
        findings = list(DEFAULT_REGISTRY.get("S16").func(_r7_result()))
        self.assertEqual(list(vault.S16_ASSERTIONS),
                         [finding.assertion for finding in findings])
        self.assertNotIn(Outcome.PASS,
                         {finding.outcome for finding in findings})


if __name__ == "__main__":
    unittest.main()
