from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import vfxdb_tool_bootstrap as bootstrap  # noqa: E402


class ToolBootstrapTests(unittest.TestCase):
    def test_cache_dir_parser_preserves_data_revision_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            expected = Path(tmp).absolute()
            self.assertEqual(
                bootstrap.cache_dir_from_argv(
                    ["/data/vfxdb", "--revision", "data-v2", "--cache-dir", tmp]
                ),
                expected,
            )
            self.assertEqual(
                bootstrap.cache_dir_from_argv([f"--cache-dir={tmp}", "--preset", "smoke"]),
                expected,
            )
            self.assertIsNone(
                bootstrap.cache_dir_from_argv(["/data/vfxdb", "--revision", "data-v2"])
            )

    def test_fetch_requests_only_two_tool_files_at_the_fixed_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "snapshot" / "tools"
            root.mkdir(parents=True)
            for name in ("vfxdb_downloader.py", "vfxdb_tui.py"):
                (root / name).write_text("# fixture\n", encoding="utf-8")
            calls = []

            def fake_download(**kwargs):
                calls.append(kwargs)
                return str(root / Path(kwargs["filename"]).name)

            with mock.patch("huggingface_hub.hf_hub_download", side_effect=fake_download):
                actual = bootstrap.fetch_tool_directory(
                    ["/data/vfxdb", "--revision", "data-v2", "--cache-dir", tmp]
                )

            self.assertEqual(actual, root.absolute())
            self.assertEqual([call["filename"] for call in calls], list(bootstrap.REMOTE_FILES))
            for call in calls:
                self.assertEqual(call["repo_id"], bootstrap.REPO_ID)
                self.assertEqual(call["repo_type"], "dataset")
                self.assertEqual(call["revision"], bootstrap.TOOLS_REVISION)
                self.assertEqual(call["cache_dir"], str(Path(tmp).absolute()))
                self.assertNotIn("local_dir", call)
                self.assertNotIn("allow_patterns", call)

    def test_remote_entry_receives_argv_unchanged_and_returns_its_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "vfxdb_downloader.py").write_text(
                "def main(argv):\n"
                "    assert argv == ['/data/vfxdb', '--revision', 'data-v2']\n"
                "    return 2\n",
                encoding="utf-8",
            )
            (root / "vfxdb_tui.py").write_text("# fixture\n", encoding="utf-8")
            with mock.patch.object(bootstrap, "fetch_tool_directory", return_value=root):
                code = bootstrap.run_remote(
                    "main", ["/data/vfxdb", "--revision", "data-v2"]
                )
            self.assertEqual(code, 2)
            self.assertEqual(
                Path(sys.modules["vfxdb_downloader"].__file__).resolve(),
                (root / "vfxdb_downloader.py").resolve(),
            )
            self.assertFalse((root / "__pycache__").exists())

    def test_all_conventional_exit_codes_are_forwarded(self) -> None:
        for expected in (0, 1, 2, 130):
            with self.subTest(expected=expected), mock.patch.object(
                bootstrap, "fetch_tool_directory", return_value=Path("/snapshot/tools")
            ), mock.patch.object(
                bootstrap, "load_remote_entry", return_value=lambda _argv: expected
            ):
                self.assertEqual(bootstrap.run_remote("main", ["--help"]), expected)

    def test_fetch_failure_is_concise_and_has_no_traceback(self) -> None:
        stderr = io.StringIO()
        with mock.patch.object(
            bootstrap,
            "fetch_tool_directory",
            side_effect=bootstrap.ToolBootstrapError("offline cache miss"),
        ), contextlib.redirect_stderr(stderr):
            code = bootstrap.run_remote("main", ["--help"])
        output = stderr.getvalue()
        self.assertEqual(code, 1)
        self.assertIn(bootstrap.REPO_ID, output)
        self.assertIn(bootstrap.TOOLS_REVISION, output)
        self.assertIn("offline cache miss", output)
        self.assertNotIn("Traceback", output)

    def test_incomplete_snapshot_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            roots = [Path(tmp) / "one", Path(tmp) / "two"]
            for root, name in zip(roots, ("vfxdb_downloader.py", "vfxdb_tui.py")):
                root.mkdir()
                (root / name).write_text("# fixture\n", encoding="utf-8")
            returned = iter(
                (roots[0] / "vfxdb_downloader.py", roots[1] / "vfxdb_tui.py")
            )
            with mock.patch(
                "huggingface_hub.hf_hub_download",
                side_effect=lambda **_kwargs: str(next(returned)),
            ), self.assertRaisesRegex(
                bootstrap.ToolBootstrapError, "incomplete or inconsistent"
            ):
                bootstrap.fetch_tool_directory([])

    def test_socks_proxy_without_socksio_uses_existing_https_proxy(self) -> None:
        environment = {
            "ALL_PROXY": "socks5://127.0.0.1:1080",
            "all_proxy": "socks5://127.0.0.1:1080",
            "HTTPS_PROXY": "http://127.0.0.1:7890",
        }
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "socksio":
                raise ImportError("fixture")
            return real_import(name, *args, **kwargs)

        with mock.patch.dict(bootstrap.os.environ, environment, clear=True), mock.patch(
            "builtins.__import__", side_effect=fake_import
        ):
            bootstrap._prepare_proxy_environment()
            self.assertNotIn("ALL_PROXY", bootstrap.os.environ)
            self.assertNotIn("all_proxy", bootstrap.os.environ)
            self.assertEqual(
                bootstrap.os.environ["HTTPS_PROXY"],
                "http://127.0.0.1:7890",
            )


if __name__ == "__main__":
    unittest.main()
