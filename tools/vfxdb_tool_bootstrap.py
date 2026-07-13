#!/usr/bin/env python3
"""Load the canonical VfxDB downloader from its Hugging Face dataset repo."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence


REPO_ID = "ryogishiki/VfxDB"
REPO_TYPE = "dataset"
TOOLS_REVISION = "447fdc4b4edf6fb59827f31cee7a575d4c9c6617"
REMOTE_FILES = (
    "tools/vfxdb_downloader.py",
    "tools/vfxdb_tui.py",
)


class ToolBootstrapError(RuntimeError):
    """The canonical dataset-local downloader could not be loaded."""


def _prepare_proxy_environment() -> None:
    all_proxy = os.environ.get("ALL_PROXY") or os.environ.get("all_proxy") or ""
    if not all_proxy.lower().startswith("socks"):
        return
    try:
        __import__("socksio")
        return
    except ImportError:
        pass
    if os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"):
        os.environ.pop("ALL_PROXY", None)
        os.environ.pop("all_proxy", None)


def cache_dir_from_argv(argv: Sequence[str]) -> Optional[Path]:
    for index, argument in enumerate(argv):
        if argument == "--cache-dir" and index + 1 < len(argv):
            return Path(argv[index + 1]).expanduser().absolute()
        if argument.startswith("--cache-dir="):
            value = argument.split("=", 1)[1]
            if value:
                return Path(value).expanduser().absolute()
    return None


def fetch_tool_directory(argv: Sequence[str]) -> Path:
    _prepare_proxy_environment()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ToolBootstrapError(
            "huggingface_hub is required; install requirements-core.txt"
        ) from exc

    cache_dir = cache_dir_from_argv(argv)
    paths: list[Path] = []
    try:
        for filename in REMOTE_FILES:
            downloaded = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type=REPO_TYPE,
                revision=TOOLS_REVISION,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
            )
            # Preserve the snapshot path and filename. Hub cache entries are
            # symlinks to content-addressed blobs whose names are hashes.
            paths.append(Path(downloaded).expanduser().absolute())
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        raise ToolBootstrapError(
            f"cannot fetch the dataset-local downloader: {exc}"
        ) from exc

    tool_directory = paths[0].parent
    if any(path.parent != tool_directory or not path.is_file() for path in paths):
        raise ToolBootstrapError(
            "Hugging Face returned an incomplete or inconsistent downloader snapshot"
        )
    return tool_directory


def load_remote_entry(tool_directory: Path, entry_name: str) -> Callable[[Sequence[str]], int]:
    module_path = tool_directory / "vfxdb_downloader.py"
    spec = importlib.util.spec_from_file_location("vfxdb_downloader", module_path)
    if spec is None or spec.loader is None:
        raise ToolBootstrapError(f"cannot import downloader module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(tool_directory))
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    entry = getattr(module, entry_name, None)
    if not callable(entry):
        raise ToolBootstrapError(
            f"dataset-local downloader does not provide the expected {entry_name} entry"
        )
    return entry


def run_remote(entry_name: str, argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if sys.version_info < (3, 10):
        print("error: the VfxDB downloader requires Python 3.10 or newer", file=sys.stderr)
        return 1
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        tool_directory = fetch_tool_directory(arguments)
        entry = load_remote_entry(tool_directory, entry_name)
        return int(entry(arguments))
    except KeyboardInterrupt:
        print("interrupted while loading the VfxDB downloader; rerun to continue", file=sys.stderr)
        return 130
    except ToolBootstrapError as exc:
        print("error: cannot start the VfxDB downloader", file=sys.stderr)
        print(f"  source: https://huggingface.co/datasets/{REPO_ID}", file=sys.stderr)
        print(f"  tool revision: {TOOLS_REVISION}", file=sys.stderr)
        print(f"  reason: {exc}", file=sys.stderr)
        print("  retry: check the network/cache, then rerun the same command", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: dataset-local downloader failed to load: {exc}", file=sys.stderr)
        return 1
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
