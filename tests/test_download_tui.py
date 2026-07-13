from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rich.console import Console


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import vfxdb_downloader as download  # noqa: E402
import vfxdb_tui as tui  # noqa: E402


REVISION = "a" * 40


class ScriptedInput:
    def __init__(self, values: list[str]) -> None:
        self.values = iter(values)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        try:
            return next(self.values)
        except StopIteration as exc:
            raise AssertionError(f"unexpected prompt: {prompt}") from exc


def recording_console(width: int = 100) -> tuple[Console, io.StringIO]:
    stream = io.StringIO()
    console = Console(
        file=stream,
        width=width,
        force_terminal=False,
        color_system=None,
        highlight=False,
        theme=tui.VOLUME_SLATE_THEME,
    )
    return console, stream


def base_options(root: Path) -> download.Options:
    return download.Options(
        data_root=root,
        preset=None,
        percentage=None,
        categories=(),
        max_samples=None,
        include_bad=False,
        revision=None,
        cache_dir=None,
        tui=True,
    )


def fake_plan(*, archives: int = 1):
    units = [
        SimpleNamespace(
            category="Alpha",
            remote_path=f"archives/Alpha/{index}.tar",
            samples=[SimpleNamespace(size_bytes=2048)],
        )
        for index in range(archives)
    ]
    return SimpleNamespace(
        label="preset smoke",
        archives=units,
        all_archive_count=10,
        selected_by_category={"Alpha": archives},
        available_by_category={"Alpha": 10},
        normal_samples_by_category={"Alpha": archives * 120},
        requested_max_samples=None,
    )


class TuiControllerTests(unittest.TestCase):
    def test_category_selection_supports_numbers_ranges_names_and_deduplication(self) -> None:
        categories = ("Alpha", "Beta", "CloudWave", "SurfaceFire")
        self.assertEqual(
            tui.parse_category_selection("1,3-4,beta,1", categories),
            ("Alpha", "CloudWave", "SurfaceFire", "Beta"),
        )
        self.assertEqual(tui.parse_category_selection("all", categories), categories)
        with self.assertRaisesRegex(ValueError, "out of range"):
            tui.parse_category_selection("9", categories)
        with self.assertRaisesRegex(ValueError, "unknown category"):
            tui.parse_category_selection("Missing", categories)

    def test_all_six_mode_choices_build_the_expected_selection(self) -> None:
        catalogs = {name: object() for name in ("Alpha", "Beta", "EnvironmentalFog")}
        cases = (
            (["1"], download.Selection()),
            (["2"], download.Selection(preset="smoke")),
            (["3"], download.Selection(preset="medium")),
            (["4"], download.Selection(preset="full")),
            (["5", "0", "12.5"], download.Selection(percentage=Decimal("12.5"))),
            (
                ["6", "1,3", "0", "250"],
                download.Selection(categories=("Alpha", "EnvironmentalFog"), max_samples=250),
            ),
        )
        for inputs, expected in cases:
            with self.subTest(inputs=inputs):
                console, _stream = recording_console()
                reporter = tui.RichReporter(console, tui.VolumeSlateView(console))
                interaction = tui.RichInteraction(
                    console,
                    tui.PromptPort(console, ScriptedInput(inputs)),
                    reporter,
                )
                self.assertEqual(interaction.choose(catalogs), expected)

    def test_invalid_category_with_rich_brackets_retries_without_markup_error(self) -> None:
        console, stream = recording_console()
        reporter = tui.RichReporter(console, tui.VolumeSlateView(console))
        interaction = tui.RichInteraction(
            console,
            tui.PromptPort(console, ScriptedInput(["6", "[/red]", "1", "100"])),
            reporter,
        )
        selection = interaction.choose({"Alpha": object()})
        self.assertEqual(selection.categories, ("Alpha",))
        self.assertEqual(selection.max_samples, 100)
        self.assertIn("unknown category: [/red]", stream.getvalue())

    def test_non_tty_fails_before_runner_or_filesystem_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "must-not-exist"
            console, stream = recording_console()
            runner = mock.Mock()
            code = tui.launch_tui(
                base_options(destination),
                run_download=runner,
                console=console,
                input_fn=ScriptedInput([]),
                stdin_isatty=False,
                stdout_isatty=True,
            )
            self.assertEqual(code, 2)
            runner.assert_not_called()
            self.assertFalse(destination.exists())
            self.assertIn("requires an interactive stdin and stdout", stream.getvalue())
            self.assertNotIn("\x1b", stream.getvalue())

    def test_cancel_before_preparation_runs_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            console, stream = recording_console()
            runner = mock.Mock()
            code = tui.launch_tui(
                base_options(Path(tmp) / "data"),
                run_download=runner,
                console=console,
                input_fn=ScriptedInput(["", "n", "n"]),
                stdin_isatty=True,
                stdout_isatty=True,
            )
            self.assertEqual(code, 0)
            runner.assert_not_called()
            self.assertIn("CANCELLED", stream.getvalue())

    def test_required_json_only_flow_completes_without_plan_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            console, stream = recording_console()

            def runner(options, *, reporter, interaction, plain_output):
                self.assertFalse(plain_output)
                reporter.emit("resolve", f"pinned dataset revision {REVISION}")
                selection = interaction.choose({"Alpha": object()})
                self.assertEqual(selection, download.Selection())
                return fake_plan(archives=0)

            code = tui.launch_tui(
                base_options(Path(tmp) / "data"),
                run_download=runner,
                console=console,
                input_fn=ScriptedInput(["", "n", "y", "1"]),
                stdin_isatty=True,
                stdout_isatty=True,
            )
            self.assertEqual(code, 0)
            self.assertIn("REQUIRED JSON READY", stream.getvalue())

    def test_plan_rejection_is_success_and_reports_required_json_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            console, stream = recording_console()
            plan = fake_plan()
            remote = {plan.archives[0].remote_path: SimpleNamespace(size=1024)}

            def runner(options, *, reporter, interaction, plain_output):
                reporter.emit("resolve", f"pinned dataset revision {REVISION}")
                self.assertEqual(interaction.choose({"Alpha": object()}).preset, "smoke")
                decision = interaction.confirm_plan(
                    plan,
                    remote,
                    REVISION,
                    options.data_root,
                    Path(tmp) / "cache",
                    options.include_bad,
                )
                self.assertEqual(decision, "quit")
                reporter.emit("cancelled", "required JSON ready")
                return plan

            code = tui.launch_tui(
                base_options(Path(tmp) / "data"),
                run_download=runner,
                console=console,
                input_fn=ScriptedInput(["", "n", "y", "2", "q"]),
                stdin_isatty=True,
                stdout_isatty=True,
            )
            self.assertEqual(code, 0)
            text = stream.getvalue()
            self.assertIn("VDB DOWNLOAD SKIPPED", text)
            self.assertIn("Required JSON", text)

    def test_quit_during_mode_selection_reports_that_required_json_is_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            console, stream = recording_console()

            def runner(_options, *, reporter, interaction, **_kwargs):
                reporter.emit("resolve", f"pinned dataset revision {REVISION}")
                interaction.choose({"Alpha": object()})
                raise AssertionError("quit should leave choose")

            code = tui.launch_tui(
                base_options(Path(tmp) / "data"),
                run_download=runner,
                console=console,
                input_fn=ScriptedInput(["", "n", "y", "q"]),
                stdin_isatty=True,
                stdout_isatty=True,
            )
            self.assertEqual(code, 0)
            self.assertIn("Required JSON", stream.getvalue())
            self.assertIn("ready and retained", stream.getvalue())

    def test_successful_live_flow_tracks_bytes_and_installed_tar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            console, stream = recording_console(width=120)
            plan = fake_plan()
            remote_path = plan.archives[0].remote_path
            remote = {remote_path: SimpleNamespace(size=4096)}

            def runner(options, *, reporter, interaction, plain_output):
                reporter.emit("resolve", f"pinned dataset revision {REVISION}")
                interaction.choose({"Alpha": object()})
                self.assertEqual(
                    interaction.confirm_plan(
                        plan,
                        remote,
                        REVISION,
                        options.data_root,
                        Path(tmp) / "cache",
                        False,
                    ),
                    "download",
                )
                reporter.emit("download", f"[1/1] {remote_path}")
                progress = reporter.make_tqdm_class()(total=4096, initial=0, desc="0.tar")
                progress.update(4096)
                progress.close()
                reporter.emit("verify", f"[1/1] {remote_path}")
                reporter.emit("extract", f"[1/1] {remote_path}")
                reporter.emit("installed", f"[1/1] {remote_path}")
                reporter.emit("done", "installed 1 whole tar")
                return plan

            code = tui.launch_tui(
                base_options(Path(tmp) / "data"),
                run_download=runner,
                console=console,
                input_fn=ScriptedInput(["", "n", "y", "2", "d"]),
                stdin_isatty=True,
                stdout_isatty=True,
            )
            self.assertEqual(code, 0)
            text = stream.getvalue()
            self.assertIn("SELECTION READY", text)
            self.assertIn("1 TAR ARCHIVES VERIFIED", text)
            self.assertIn("CATEGORY TARS", text)
            self.assertIn("Installed this run", text)

    def test_json_item_progress_is_not_labeled_as_network_bytes(self) -> None:
        console, stream = recording_console()
        reporter = tui.RichReporter(console, tui.VolumeSlateView(console))
        reporter.emit("verify", "checked local normal JSON files 100,000/1,019,240")
        self.assertEqual(reporter.work_current, 100_000)
        self.assertEqual(reporter.work_total, 1_019_240)
        self.assertEqual(reporter.transfer_total, 0)
        console.print(reporter.render())
        self.assertIn("VERIFY ITEMS", stream.getvalue())
        self.assertNotIn("KiB", stream.getvalue())

    def test_json_work_replaces_completed_hf_byte_progress(self) -> None:
        console, stream = recording_console()
        reporter = tui.RichReporter(console, tui.VolumeSlateView(console))
        reporter.transfer(2048, 2048, "required.json")
        reporter.emit("verify", "validated 100,000/1,019,240 per-sample JSON files")
        self.assertEqual(reporter.transfer_total, 0)
        self.assertEqual(reporter.work_current, 100_000)
        console.print(reporter.render())
        rendered = stream.getvalue()
        self.assertIn("VERIFY ITEMS", rendered)
        self.assertNotIn("CURRENT FILE", rendered)

    def test_json_install_progress_has_an_exact_final_total(self) -> None:
        console, _stream = recording_console()
        reporter = tui.RichReporter(console, tui.VolumeSlateView(console))
        reporter.emit("json", "installing 1,019,240 per-sample JSON files")
        reporter.emit("json", "installed 1,019,240/1,019,240 per-sample JSON files")
        self.assertEqual(reporter.work_current, 1_019_240)
        self.assertEqual(reporter.work_total, 1_019_240)

    def test_plan_does_not_claim_unverified_state_checkpoint_is_installed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / ".vfxdb/state.json"
            state_path.parent.mkdir(parents=True)
            state_path.write_text(
                json.dumps({"archives": {"archives/Alpha/0.tar": {"tree_digest": "stale"}}}),
                encoding="utf-8",
            )
            console, _stream = recording_console()
            reporter = tui.RichReporter(console, tui.VolumeSlateView(console))
            reporter.configure_plan(fake_plan(), REVISION, root)
            self.assertEqual(reporter.tar_current, 0)
            self.assertEqual(reporter.reused_tar_count, 0)

    def test_tar_and_current_file_have_separate_progress(self) -> None:
        console, stream = recording_console(width=100)
        reporter = tui.RichReporter(console, tui.VolumeSlateView(console))
        plan = fake_plan(archives=2)
        reporter.configure_plan(plan, REVISION, Path("/tmp/data"))
        reporter.emit("download", "[1/2] archives/Alpha/0.tar")
        reporter.transfer(2048, 4096, "0.tar")
        console.print(reporter.render())
        rendered = stream.getvalue()
        self.assertIn("TAR ARCHIVES", rendered)
        self.assertIn("CURRENT FILE", rendered)
        self.assertIn("50%", rendered)

    def test_plan_review_never_implies_that_first_tar_started(self) -> None:
        console, _stream = recording_console(width=100)
        reporter = tui.RichReporter(console, tui.VolumeSlateView(console))
        reporter.configure_plan(fake_plan(archives=2), REVISION, Path("/tmp/data"))
        rail = reporter.phase_rail().plain
        self.assertEqual(rail, "PLAN REVIEW · NO TAR STARTED")
        self.assertNotIn("CURRENT TAR", rail)

    def test_cached_archive_phase_ordinal_matches_completed_tar_count(self) -> None:
        console, _stream = recording_console(width=100)
        reporter = tui.RichReporter(console, tui.VolumeSlateView(console))
        reporter.configure_plan(fake_plan(archives=2), REVISION, Path("/tmp/data"))
        reporter.emit("cache", "[1/2] archives/Alpha/0.tar already installed")
        rail = reporter.phase_rail().plain
        self.assertIn("CURRENT TAR 1/2", rail)
        self.assertNotIn("CURRENT TAR 2/2", rail)

    def test_production_tar_verify_and_extract_events_are_not_json_work(self) -> None:
        console, _stream = recording_console(width=100)
        reporter = tui.RichReporter(console, tui.VolumeSlateView(console))
        reporter.configure_plan(fake_plan(archives=2), REVISION, Path("/tmp/data"))
        reporter.emit(
            "verify",
            "[1/2] archives/Alpha/0.tar validating complete tar",
        )
        self.assertEqual(reporter.work_total, 0)
        self.assertEqual(reporter.message, "Checking Alpha/0")
        reporter.emit(
            "extract",
            "[1/2] archives/Alpha/0.tar installing complete tar",
        )
        self.assertEqual(reporter.work_total, 0)
        self.assertEqual(reporter.message, "Installing Alpha/0")

    def test_full_mode_requires_exact_full_confirmation(self) -> None:
        console, stream = recording_console()
        reporter = tui.RichReporter(console, tui.VolumeSlateView(console))
        plan = fake_plan()
        plan.label = "preset full"
        remote = {plan.archives[0].remote_path: SimpleNamespace(size=1)}
        interaction = tui.RichInteraction(
            console,
            tui.PromptPort(console, ScriptedInput(["full", "FULL"])),
            reporter,
        )
        self.assertEqual(
            interaction.confirm_plan(plan, remote, REVISION, Path("/tmp/data"), Path("/tmp/cache"), False)
            ,
            "download",
        )
        self.assertIn("Choose FULL, b, or q exactly", stream.getvalue())

    def test_plan_can_return_to_selection_without_quitting(self) -> None:
        console, _stream = recording_console()
        reporter = tui.RichReporter(console, tui.VolumeSlateView(console))
        interaction = tui.RichInteraction(
            console,
            tui.PromptPort(console, ScriptedInput(["b", "5", "12.5"])),
            reporter,
        )
        plan = fake_plan()
        remote = {plan.archives[0].remote_path: SimpleNamespace(size=1)}
        self.assertEqual(
            interaction.confirm_plan(
                plan,
                remote,
                REVISION,
                Path("/tmp/data"),
                Path("/tmp/cache"),
                False,
            ),
            "change",
        )
        self.assertEqual(
            interaction.choose({"Alpha": object()}).percentage,
            Decimal("12.5"),
        )

    def test_category_plan_explains_capacity_and_whole_tar_overshoot(self) -> None:
        console, stream = recording_console(width=100)
        plan = fake_plan()
        plan.label = "category max-samples=100"
        plan.requested_max_samples = 100
        plan.normal_samples_by_category["Alpha"] = 120
        remote = {plan.archives[0].remote_path: SimpleNamespace(size=4096)}
        console.print(
            tui.VolumeSlateView(console).plan_table(
                plan,
                remote,
                REVISION,
                Path("/tmp/data"),
                Path("/tmp/cache"),
                False,
            )
        )
        rendered = stream.getvalue()
        self.assertIn("Network upper bound", rendered)
        self.assertIn("VDB install upper bound", rendered)
        self.assertIn("Destination free", rendered)
        self.assertIn("whole-tar", rendered.lower())
        self.assertIn("+20", rendered)

    def test_category_plan_marks_full_category_shortfall(self) -> None:
        console, stream = recording_console(width=100)
        plan = fake_plan()
        plan.label = "category max-samples=1000"
        plan.requested_max_samples = 1000
        plan.normal_samples_by_category["Alpha"] = 800
        plan.selected_by_category["Alpha"] = plan.available_by_category["Alpha"]
        remote = {plan.archives[0].remote_path: SimpleNamespace(size=4096)}
        console.print(
            tui.VolumeSlateView(console).plan_table(
                plan,
                remote,
                REVISION,
                Path("/tmp/data"),
                Path("/tmp/cache"),
                False,
            )
        )
        self.assertIn("FULL · 200 below target", stream.getvalue())

    def test_percentage_plan_shows_ceiling_formula_and_exact_tar_count(self) -> None:
        console, stream = recording_console(width=100)
        plan = fake_plan(archives=2)
        plan.label = "all-category 12.5%"
        plan.all_archive_count = 16
        remote = {archive.remote_path: SimpleNamespace(size=4096) for archive in plan.archives}
        console.print(
            tui.VolumeSlateView(console).plan_table(
                plan,
                remote,
                REVISION,
                Path("/tmp/data"),
                Path("/tmp/cache"),
                False,
            )
        )
        self.assertIn("ceil(16 × 12.5%) = 2 complete tars", stream.getvalue())

    def test_advanced_settings_can_explicitly_include_bad_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            console, _stream = recording_console()
            captured = {}

            def runner(options, *, reporter, interaction, **_kwargs):
                captured["include_bad"] = options.include_bad
                reporter.emit("resolve", f"pinned dataset revision {REVISION}")
                interaction.choose({"Alpha": object()})
                return fake_plan(archives=0)

            code = tui.launch_tui(
                base_options(Path(tmp) / "data"),
                run_download=runner,
                console=console,
                input_fn=ScriptedInput(["", "y", "", "", "y", "y", "1"]),
                stdin_isatty=True,
                stdout_isatty=True,
            )
            self.assertEqual(code, 0)
            self.assertIs(captured["include_bad"], True)

    def test_category_completion_repeats_whole_tar_target_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            console, stream = recording_console(width=100)
            plan = fake_plan()
            plan.label = "category max-samples=100"
            plan.requested_max_samples = 100
            plan.normal_samples_by_category["Alpha"] = 120
            remote_path = plan.archives[0].remote_path
            remote = {remote_path: SimpleNamespace(size=4096)}

            def runner(options, *, reporter, interaction, **_kwargs):
                reporter.emit("resolve", f"pinned dataset revision {REVISION}")
                selection = interaction.choose({"Alpha": object()})
                self.assertEqual(selection.max_samples, 100)
                self.assertEqual(
                    interaction.confirm_plan(
                        plan,
                        remote,
                        REVISION,
                        options.data_root,
                        Path(tmp) / "cache",
                        False,
                    ),
                    "download",
                )
                reporter.emit("download", f"[1/1] {remote_path}")
                reporter.emit("verify", f"[1/1] {remote_path} validating complete tar")
                reporter.emit("extract", f"[1/1] {remote_path} installing complete tar")
                reporter.emit("installed", f"[1/1] {remote_path}")
                reporter.emit("done", "installed 1 whole tar")
                return plan

            code = tui.launch_tui(
                base_options(Path(tmp) / "data"),
                run_download=runner,
                console=console,
                input_fn=ScriptedInput(["", "n", "y", "6", "1", "100", "d"]),
                stdin_isatty=True,
                stdout_isatty=True,
            )
            self.assertEqual(code, 0)
            rendered = stream.getvalue()
            self.assertIn("Per-category target", rendered)
            self.assertIn("+20 (whole-tar rounding)", rendered)

    def test_narrow_rendering_and_long_error_remain_readable(self) -> None:
        for width in (20, 32, 40, 80):
            with self.subTest(width=width):
                console, stream = recording_console(width)
                view = tui.VolumeSlateView(console)
                view.welcome(Path("/a/very/long/destination/path/for/vfxdb"), REVISION, None)
                view.mode_menu()
                view.result(
                    "DOWNLOAD STOPPED",
                    (("Reason", "network timeout while downloading a very long archive path"),),
                    error=True,
                )
                text = stream.getvalue()
                self.assertIn("VfxDB", text)
                self.assertIn("DOWNLOAD STOPPED", text)
                self.assertIn("network", text)
                self.assertIn("timeout", text)

    def test_narrow_live_progress_degrades_to_readable_numbers(self) -> None:
        for width in (20, 32, 40):
            with self.subTest(width=width):
                console, stream = recording_console(width)
                reporter = tui.RichReporter(console, tui.VolumeSlateView(console))
                reporter.configure_plan(fake_plan(archives=2), REVISION, Path("/tmp/data"))
                reporter.emit("download", "[1/2] archives/Alpha/0.tar")
                reporter.transfer(2048, 4096, "0.tar")
                console.print(reporter.render())
                text = stream.getvalue()
                self.assertIn("TARS", text)
                self.assertIn("FILE", text)
                self.assertIn("50%", text)
                self.assertNotIn("CURRENT TAR", text)

    def test_dumb_terminal_uses_periodic_plain_progress_lines(self) -> None:
        stream = io.StringIO()
        with mock.patch.dict(os.environ, {"TERM": "dumb"}):
            console = Console(
                file=stream,
                width=80,
                force_terminal=True,
                color_system=None,
                highlight=False,
                theme=tui.VOLUME_SLATE_THEME,
            )
            reporter = tui.RichReporter(console, tui.VolumeSlateView(console))
            self.assertTrue(reporter.static_mode)
            reporter.start()
            reporter.configure_plan(fake_plan(archives=2), REVISION, Path("/tmp/data"))
            reporter.emit("download", "[1/2] archives/Alpha/0.tar")
            reporter.transfer(2048, 4096, "0.tar")
            reporter.emit("verify", "[1/2] archives/Alpha/0.tar validating complete tar")
            reporter.stop()
        rendered = stream.getvalue()
        self.assertIn("[DOWNLOAD]", rendered)
        self.assertIn("FILE 50%", rendered)
        self.assertIn("[VERIFY]", rendered)
        self.assertNotIn("\x1b", rendered)

    def test_runtime_error_and_keyboard_interrupt_have_distinct_exit_codes(self) -> None:
        cases = ((RuntimeError("disk is full"), 1, "disk is full"), (KeyboardInterrupt(), 130, "STOPPED SAFELY"))
        for raised, expected_code, expected_text in cases:
            with self.subTest(raised=type(raised).__name__), tempfile.TemporaryDirectory() as tmp:
                console, stream = recording_console()

                def runner(_options, *, reporter, **_kwargs):
                    reporter.emit("download", "[3/10] archives/Alpha/2.tar")
                    raise raised

                code = tui.launch_tui(
                    base_options(Path(tmp) / "data"),
                    run_download=runner,
                    console=console,
                    input_fn=ScriptedInput(["", "n", "y"]),
                    stdin_isatty=True,
                    stdout_isatty=True,
                )
                self.assertEqual(code, expected_code)
                self.assertIn(expected_text, stream.getvalue())


if __name__ == "__main__":
    unittest.main()
