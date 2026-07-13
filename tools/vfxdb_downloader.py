#!/usr/bin/env python3
"""Download VfxDB from Hugging Face using whole-sequence tar units.

The normative behavior is documented in ``docs/DOWNLOADER_SPEC.md``.  This
module intentionally keeps dataset membership in the published category
indexes.  Its private state is only an installation checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import errno
import fcntl
import hashlib
import inspect
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Literal, Protocol, Sequence


REPO_ID = "ryogishiki/VfxDB"
REPO_TYPE = "dataset"
BAD_MANIFEST_PATH = "meta/io_bad_vdbs.jsonl"
BAD_MANIFEST_INFO_PATH = "meta/io_bad_vdbs.meta.json"
SAMPLE_JSON_ARCHIVE_PATH = "meta/vfxdb_meta.tar.zst"
STATE_SCHEMA_VERSION = 1
MAX_SEQUENCE_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_SAMPLE_JSON_BYTES = 1024 * 1024
JSON_PROGRESS_INTERVAL = 100_000
MIN_FREE_MARGIN_BYTES = 1024 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
CATEGORY_INDEX_RE = re.compile(r"^([^/]+)/category_index\.json$")
RESERVED_CATEGORY_ENTRIES = {"index", "category_index.json"}


class DownloadError(RuntimeError):
    """Expected user-facing failure."""


class IntegrityError(DownloadError):
    """A downloaded object does not match its published controls."""


class CacheCorruptionError(IntegrityError):
    """A fresh Hugging Face download may repair the cached object."""


class UnsafeArchiveError(DownloadError):
    """An archive contains an unsafe or unsupported member."""


@dataclass(frozen=True, slots=True)
class RemoteFile:
    path: str
    size: int
    sha256: str | None = None
    git_blob_sha1: str | None = None


@dataclass(frozen=True, slots=True)
class Options:
    data_root: Path
    preset: str | None
    percentage: Decimal | None
    categories: tuple[str, ...]
    max_samples: int | None
    include_bad: bool
    revision: str | None
    cache_dir: Path | None
    tui: bool = False

    @property
    def mode(self) -> str:
        if self.preset is not None:
            return "preset"
        if self.percentage is not None:
            return "percentage"
        if self.categories:
            return "category"
        return "metadata-only"


@dataclass(frozen=True, slots=True)
class Selection:
    preset: str | None = None
    percentage: Decimal | None = None
    categories: tuple[str, ...] = ()
    max_samples: int | None = None


class DownloadInteraction(Protocol):
    def choose(self, catalogs: dict[str, CategoryCatalog]) -> Selection: ...

    def confirm_plan(
        self,
        plan: DownloadPlan,
        remote_files: dict[str, RemoteFile],
        revision: str,
        data_root: Path,
        cache_root: Path,
        include_bad: bool,
    ) -> Literal["download", "change", "quit"]: ...


@dataclass(slots=True)
class BadSamples:
    source_relpaths: set[str]
    source_by_archive_member: dict[tuple[str, str], str]
    sources_by_archive: dict[str, set[str]]
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class SampleRecord:
    category: str
    folder: str
    source_relpath: str
    meta_relpath: str
    size_bytes: int
    row_index: int


@dataclass(slots=True)
class ArchiveUnit:
    category: str
    folder: str
    remote_path: str
    samples: list[SampleRecord] = field(default_factory=list)

    def normal_sample_count(self, bad_sources: set[str]) -> int:
        return sum(sample.source_relpath not in bad_sources for sample in self.samples)


@dataclass(slots=True)
class CategoryCatalog:
    category: str
    path: Path
    archives: list[ArchiveUnit]
    samples_by_source: dict[str, SampleRecord]


@dataclass(slots=True)
class DownloadPlan:
    label: str
    archives: list[ArchiveUnit]
    all_archive_count: int
    selected_by_category: dict[str, int]
    available_by_category: dict[str, int]
    normal_samples_by_category: dict[str, int]
    requested_max_samples: int | None = None


class PlainReporter:
    def emit(self, stage: str, message: str) -> None:
        print(f"[{stage}] {message}", flush=True)


@contextmanager
def temporarily_suppress_hf_progress(enabled: bool) -> Iterator[None]:
    """Hide Hub's fallback bar for one call without changing later Hub users."""

    if not enabled:
        yield
        return
    try:
        from huggingface_hub.utils import (
            are_progress_bars_disabled,
            disable_progress_bars,
            enable_progress_bars,
        )
    except ImportError:
        yield
        return

    was_disabled = are_progress_bars_disabled()
    if not was_disabled:
        disable_progress_bars()
    try:
        yield
    finally:
        if not was_disabled:
            enable_progress_bars()


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def percentage_value(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("must be a number greater than 0 and at most 100") from exc
    if not parsed.is_finite() or parsed <= 0 or parsed > 100:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 100")
    return parsed


def preset_value(value: str) -> str:
    lowered = value.strip().lower()
    if lowered not in {"smoke", "medium", "full"}:
        raise argparse.ArgumentTypeError("must be one of: smoke, medium, full")
    return lowered


def parse_args(argv: Sequence[str] | None = None) -> Options:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare all required VfxDB JSON controls, then optionally download "
            "whole sequence tar archives."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data_root", nargs="?", default="data/vdbset", help="Destination VDBSet directory.")
    parser.add_argument("--preset", type=preset_value, metavar="{smoke,medium,full}")
    parser.add_argument(
        "--percentage",
        type=percentage_value,
        metavar="P",
        help="Download P percent of all category tars using balanced whole-tar selection.",
    )
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        metavar="NAME",
        help="Download a category up to --max-samples. Repeat for multiple categories.",
    )
    parser.add_argument(
        "--max-samples",
        type=positive_int,
        metavar="N",
        help="Normal-sample target applied independently to every --category; rounds up by tar.",
    )
    parser.add_argument(
        "--include-bad",
        action="store_true",
        help="Retain IO-bad files from selected tars; archive selection and quotas stay unchanged.",
    )
    parser.add_argument("--revision", default=None, help="Hugging Face branch, tag, or commit.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Hugging Face Hub cache root.",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Open the interactive terminal interface; selection options are chosen inside it.",
    )
    args = parser.parse_args(argv)

    categories: list[str] = []
    for raw in args.category:
        for value in str(raw).split(","):
            category = value.strip()
            if not category:
                parser.error("--category cannot contain an empty name")
            if category not in categories:
                categories.append(category)

    has_preset = args.preset is not None
    has_percentage = args.percentage is not None
    has_category_mode = bool(categories) or args.max_samples is not None
    if args.tui and any((has_preset, has_percentage, has_category_mode)):
        parser.error("--tui chooses the data mode interactively; do not combine it with selection options")
    if sum((has_preset, has_percentage, has_category_mode)) > 1:
        parser.error("--preset, --percentage, and --category/--max-samples are mutually exclusive")
    if bool(categories) != (args.max_samples is not None):
        parser.error("--category and --max-samples must be provided together")

    return Options(
        data_root=Path(args.data_root).expanduser().absolute(),
        preset=args.preset,
        percentage=args.percentage,
        categories=tuple(categories),
        max_samples=args.max_samples,
        include_bad=bool(args.include_bad),
        revision=str(args.revision).strip() if args.revision else None,
        cache_dir=Path(args.cache_dir).expanduser().absolute() if args.cache_dir else None,
        tui=bool(args.tui),
    )


def options_with_selection(options: Options, selection: Selection) -> Options:
    return replace(
        options,
        preset=selection.preset,
        percentage=selection.percentage,
        categories=selection.categories,
        max_samples=selection.max_samples,
    )


def format_bytes(value: int) -> str:
    size = float(max(0, value))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if size < 1024.0 or unit == "PiB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1_file(path: Path, size: int) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_token(value: object, label: str) -> str:
    text = str(value).strip()
    if not TOKEN_RE.fullmatch(text):
        raise DownloadError(f"Invalid {label}: {value!r}")
    return text


def require_json_integer(
    value: object,
    label: str,
    *,
    minimum: int | None = None,
    error_type: type[DownloadError] = DownloadError,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise error_type(f"{label} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise error_type(f"{label} must be at least {minimum}, got {value}")
    return value


def validate_posix_path(
    value: object,
    label: str,
    *,
    exact_parts: int | None = None,
) -> PurePosixPath:
    raw = str(value)
    if not raw or raw != raw.strip() or "\\" in raw or "\x00" in raw:
        raise DownloadError(f"Invalid {label}: {value!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise DownloadError(f"Invalid {label}: {value!r}")
    if exact_parts is not None and len(path.parts) != exact_parts:
        raise DownloadError(f"Invalid {label}: {value!r}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DownloadError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DownloadError(f"Expected a JSON object in {path}")
    return value


def read_utf8_text(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise DownloadError(f"Cannot read {label} {path}: {exc}") from exc


def atomic_write_bytes(path: Path, payload: bytes, *, durable: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    try:
        with temp.open("xb") as handle:
            handle.write(payload)
            if durable:
                handle.flush()
                os.fsync(handle.fileno())
        os.chmod(temp, 0o644)
        os.replace(temp, path)
        if durable:
            fsync_directory(path.parent)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write_bytes(path, encode_json(value))


def encode_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def atomic_copy_file(source: Path, destination: Path, *, durable: bool = True) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.parent / f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
    try:
        with source.open("rb") as input_handle, temp.open("xb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            if durable:
                output_handle.flush()
                os.fsync(output_handle.fileno())
        os.chmod(temp, 0o644)
        os.replace(temp, destination)
        if durable:
            fsync_directory(destination.parent)
    finally:
        if temp.exists():
            temp.unlink()


def fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def remote_fingerprint(remote: RemoteFile) -> str:
    if remote.sha256:
        return f"sha256:{remote.sha256}"
    if remote.git_blob_sha1:
        return f"git:{remote.git_blob_sha1}"
    return f"size:{remote.size}"


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


def _is_permanent_download_error(exc: BaseException) -> bool:
    if isinstance(exc, (KeyboardInterrupt, PermissionError, ValueError)):
        return True
    if isinstance(exc, OSError) and exc.errno in {
        errno.ENOSPC,
        errno.EDQUOT,
        errno.EACCES,
        errno.EROFS,
    }:
        return True
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int) and 400 <= status < 500 and status not in (408, 429):
        return True
    name = type(exc).__name__.lower()
    return any(token in name for token in ("repositorynotfound", "revisionnotfound", "entrynotfound"))


class HubClient:
    """Small testable wrapper around huggingface_hub cache and downloads."""

    def __init__(
        self,
        cache_dir: Path | None,
        reporter: PlainReporter,
        *,
        sleep: Callable[[float], None] = time.sleep,
        attempts: int = 4,
    ) -> None:
        _prepare_proxy_environment()
        # The regular Hub HTTP transport has bounded reads and raises errors
        # that this client can retry. Native Xet downloads can block without
        # returning, which defeats the retry loop and leaves the destination
        # lock held indefinitely. Keep the reliable path as the script
        # default; an explicit caller setting can still opt into Xet.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
        if cache_dir is not None:
            # If a caller explicitly opts into Xet, keep its auxiliary files
            # on the same device and avoid a redundant large chunk cache.
            os.environ.setdefault("HF_XET_CACHE", str(cache_dir / "xet"))
            os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
        from huggingface_hub import HfApi, constants

        # ``huggingface_hub`` caches these values at import time. Synchronize
        # them as well so the downloader remains reliable when embedded in a
        # process that imported the Hub before constructing this client.
        constants.HF_HUB_DISABLE_XET = os.environ["HF_HUB_DISABLE_XET"].strip().upper() in {
            "1",
            "ON",
            "YES",
            "TRUE",
        }
        try:
            timeout = int(os.environ["HF_HUB_DOWNLOAD_TIMEOUT"])
        except ValueError as exc:
            raise DownloadError("HF_HUB_DOWNLOAD_TIMEOUT must be an integer") from exc
        if timeout <= 0:
            raise DownloadError("HF_HUB_DOWNLOAD_TIMEOUT must be greater than zero")
        constants.HF_HUB_DOWNLOAD_TIMEOUT = timeout
        if cache_dir is not None and hasattr(constants, "HF_XET_CACHE"):
            constants.HF_XET_CACHE = str(cache_dir / "xet")
        self._api = HfApi()
        self.cache_dir = cache_dir
        self.reporter = reporter
        self.sleep = sleep
        self.attempts = attempts

    @property
    def cache_root(self) -> Path:
        if self.cache_dir is not None:
            return self.cache_dir
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE).expanduser().absolute()

    def resolve_snapshot(self, requested_revision: str) -> tuple[str, dict[str, RemoteFile]]:
        last_error: BaseException | None = None
        info = None
        for attempt in range(1, self.attempts + 1):
            try:
                info = self._api.dataset_info(
                    REPO_ID,
                    revision=requested_revision,
                    files_metadata=True,
                )
                break
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                last_error = exc
                if _is_permanent_download_error(exc) or attempt >= self.attempts:
                    break
                delay = min(30.0, 1.5 * (2 ** (attempt - 1))) + random.random() * 0.5
                self.reporter.emit("retry", f"cannot reach dataset ({exc}); retrying in {delay:.1f}s")
                self.sleep(delay)
        if info is None:
            assert last_error is not None
            raise DownloadError(
                f"Cannot resolve Hugging Face revision {requested_revision!r}: {last_error}"
            ) from last_error
        revision = str(info.sha or "")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise DownloadError(f"Hugging Face returned an invalid commit: {revision!r}")
        files: dict[str, RemoteFile] = {}
        for sibling in info.siblings or []:
            path = str(sibling.rfilename)
            try:
                size = int(sibling.size) if sibling.size is not None else -1
            except (TypeError, ValueError) as exc:
                raise DownloadError(f"Invalid remote size for {path}: {sibling.size!r}") from exc
            if path in files or size < 0:
                raise DownloadError(f"Invalid duplicate or sizeless remote path: {path}")
            lfs = getattr(sibling, "lfs", None)
            sha256 = str(getattr(lfs, "sha256", "") or "").lower() or None
            blob_id = str(getattr(sibling, "blob_id", "") or "").lower() or None
            git_sha1 = blob_id if sha256 is None else None
            if sha256 is not None and not SHA256_RE.fullmatch(sha256):
                raise DownloadError(f"Invalid remote SHA-256 for {path}")
            if git_sha1 is not None and not SHA1_RE.fullmatch(git_sha1):
                raise DownloadError(f"Invalid remote Git blob ID for {path}")
            if sha256 is None and git_sha1 is None:
                raise DownloadError(f"Remote file has no content identity: {path}")
            files[path] = RemoteFile(path, size, sha256, git_sha1)
        return revision, files

    def download(self, remote_path: str, revision: str, *, force: bool = False) -> Path:
        from huggingface_hub import hf_hub_download

        download_kwargs: dict[str, Any] = {}
        progress_class = getattr(self.reporter, "tqdm_class", None)
        if progress_class is None:
            make_progress_class = getattr(self.reporter, "make_tqdm_class", None)
            if callable(make_progress_class):
                progress_class = make_progress_class()
        supports_custom_progress = "tqdm_class" in inspect.signature(hf_hub_download).parameters
        if progress_class is not None and supports_custom_progress:
            download_kwargs["tqdm_class"] = progress_class
        suppress_fallback_progress = bool(
            getattr(self.reporter, "suppress_hf_progress", False)
            and not (progress_class is not None and supports_custom_progress)
        )

        last_error: BaseException | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                with temporarily_suppress_hf_progress(suppress_fallback_progress):
                    result = hf_hub_download(
                        repo_id=REPO_ID,
                        filename=remote_path,
                        repo_type=REPO_TYPE,
                        revision=revision,
                        cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
                        force_download=force,
                        **download_kwargs,
                    )
                return Path(result)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                last_error = exc
                if _is_permanent_download_error(exc) or attempt >= self.attempts:
                    break
                delay = min(30.0, 1.5 * (2 ** (attempt - 1))) + random.random() * 0.5
                self.reporter.emit("retry", f"{remote_path} failed ({exc}); retrying in {delay:.1f}s")
                self.sleep(delay)
        assert last_error is not None
        raise DownloadError(f"Cannot download {remote_path}: {last_error}") from last_error

    def cached_path(self, remote_path: str, revision: str) -> Path | None:
        from huggingface_hub import try_to_load_from_cache

        try:
            value = try_to_load_from_cache(
                repo_id=REPO_ID,
                filename=remote_path,
                repo_type=REPO_TYPE,
                revision=revision,
                cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
            )
        except Exception:
            return None
        return Path(value) if isinstance(value, str) and Path(value).is_file() else None


def require_remote_file(files: dict[str, RemoteFile], path: str) -> RemoteFile:
    remote = files.get(path)
    if remote is None:
        raise DownloadError(f"Required file is missing from the dataset commit: {path}")
    return remote


def ensure_downloaded_file(path: Path, remote: RemoteFile, label: str) -> None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise DownloadError(f"Cannot access downloaded {label} {path}: {exc}") from exc
    if size != remote.size:
        raise CacheCorruptionError(
            f"{label} size mismatch for {remote.path}: expected {remote.size}, got {size}"
        )
    if remote.sha256 is not None:
        actual = sha256_file(path)
        if actual != remote.sha256:
            raise CacheCorruptionError(
                f"{label} SHA-256 mismatch for {remote.path}: expected {remote.sha256}, got {actual}"
            )
    elif remote.git_blob_sha1 is not None:
        actual = git_blob_sha1_file(path, size)
        if actual != remote.git_blob_sha1:
            raise CacheCorruptionError(
                f"{label} Git blob mismatch for {remote.path}: expected {remote.git_blob_sha1}, got {actual}"
            )


def download_verified(
    hub: HubClient,
    remote: RemoteFile,
    revision: str,
    label: str,
    reporter: PlainReporter,
) -> Path:
    first_error: CacheCorruptionError | None = None
    for force in (False, True):
        if not force and hub.cached_path(remote.path, revision) is not None:
            reporter.emit("cache", f"reusing Hugging Face object for {remote.path}")
        path = hub.download(remote.path, revision, force=force)
        try:
            ensure_downloaded_file(path, remote, label)
            if force:
                reporter.emit("repair", f"replaced corrupt cached {remote.path}")
            return path
        except CacheCorruptionError as exc:
            if force:
                raise
            first_error = exc
            reporter.emit("repair", f"{remote.path} failed integrity; requesting one fresh copy")
    assert first_error is not None
    raise first_error


def cached_remote_is_usable(
    hub: HubClient,
    remote: RemoteFile,
    revision: str,
) -> bool:
    cached = hub.cached_path(remote.path, revision)
    if cached is None:
        return False
    try:
        ensure_downloaded_file(cached, remote, "cached object")
    except DownloadError:
        return False
    return True


def control_dir(data_root: Path, *, create: bool) -> Path:
    if data_root.is_symlink():
        raise DownloadError(f"Destination root must not be a symlink: {data_root}")
    path = data_root / ".vfxdb"
    if path.is_symlink():
        raise DownloadError(f"Downloader control directory must not be a symlink: {path}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if path.exists() and not path.is_dir():
        raise DownloadError(f"Downloader control path is not a directory: {path}")
    return path


def state_path(data_root: Path) -> Path:
    return data_root / ".vfxdb" / "state.json"


def policy_transition_path(data_root: Path) -> Path:
    return data_root / ".vfxdb" / "policy-transition.json"


def begin_policy_transition(
    data_root: Path,
    revision: str,
    include_bad: bool,
) -> None:
    path = policy_transition_path(data_root)
    if path.is_symlink():
        raise DownloadError(f"IO-bad policy transition state must not be a symlink: {path}")
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "revision": revision,
            "include_bad": include_bad,
        },
    )


def finish_policy_transition(data_root: Path) -> None:
    path = policy_transition_path(data_root)
    if not path.exists():
        return
    if path.is_symlink() or not path.is_file():
        raise DownloadError(f"Unsafe IO-bad policy transition state: {path}")
    path.unlink()
    fsync_directory(path.parent)


def load_state(data_root: Path) -> dict[str, Any] | None:
    path = state_path(data_root)
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise DownloadError(f"Downloader state must be a regular file: {path}")
    state = load_json(path)
    schema_version = require_json_integer(
        state.get("schema_version"), f"state schema_version in {path}"
    )
    if schema_version != STATE_SCHEMA_VERSION:
        raise DownloadError(
            f"Unsupported downloader state in {path}; use a new destination for this downloader"
        )
    if state.get("repo_id") != REPO_ID:
        raise DownloadError(f"Downloader state belongs to another repository: {path}")
    if not re.fullmatch(r"[0-9a-f]{40}", str(state.get("revision", ""))):
        raise DownloadError(f"Downloader state has an invalid revision: {path}")
    if not isinstance(state.get("archives", {}), dict):
        raise DownloadError(f"Downloader state has invalid archive checkpoints: {path}")
    return state


def save_state(data_root: Path, state: dict[str, Any]) -> None:
    control_dir(data_root, create=True)
    atomic_write_json(state_path(data_root), state)


def ensure_empty_unmanaged_destination(data_root: Path) -> None:
    public_entries = [entry for entry in data_root.iterdir() if entry.name != ".vfxdb"]
    control = control_dir(data_root, create=True)
    internal_entries = [entry for entry in control.iterdir() if entry.name != "download.lock"]
    if public_entries or internal_entries:
        examples = [str(path) for path in (*public_entries, *internal_entries)[:5]]
        raise DownloadError(
            "Destination contains existing dataset-like files but has no VfxDB downloader state; "
            f"refusing to mix revisions. Use a new empty destination. First entries: {examples}"
        )


@contextmanager
def destination_lock(data_root: Path) -> Iterator[None]:
    control = control_dir(data_root, create=True)
    lock_path = control / "download.lock"
    if lock_path.is_symlink():
        raise DownloadError(f"Downloader lock must not be a symlink: {lock_path}")
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DownloadError(f"Another downloader is already using {data_root}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def ensure_safe_destination(data_root: Path, destination: Path) -> None:
    try:
        relative = destination.absolute().relative_to(data_root.absolute())
    except ValueError as exc:
        raise DownloadError(f"Destination escapes data root: {destination}") from exc
    current = data_root.absolute()
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise DownloadError(f"Refusing symlinked destination directory: {current}")
    if destination.is_symlink():
        raise DownloadError(f"Refusing symlinked destination file: {destination}")


def discover_category_indexes(remote_files: dict[str, RemoteFile]) -> list[tuple[str, RemoteFile]]:
    found: list[tuple[str, RemoteFile]] = []
    for path, remote in remote_files.items():
        match = CATEGORY_INDEX_RE.fullmatch(path)
        if match is None:
            continue
        raw_category = match.group(1)
        category = validate_token(raw_category, "category")
        found.append((category, remote))
    found.sort(key=lambda item: item[0])
    if not found:
        raise DownloadError("The dataset commit has no <Category>/category_index.json files")
    if len({category for category, _ in found}) != len(found):
        raise DownloadError("The dataset commit contains duplicate category indexes")
    return found


def stage_remote_control(
    hub: HubClient,
    remote: RemoteFile,
    revision: str,
    data_root: Path,
    reporter: PlainReporter,
) -> Path:
    staging_root = control_dir(data_root, create=True) / "control-staging" / revision
    if staging_root.is_symlink():
        raise DownloadError(f"Control staging directory must not be a symlink: {staging_root}")
    destination = staging_root / remote.path
    ensure_safe_destination(data_root, destination)
    if destination.exists() and not destination.is_file():
        raise DownloadError(f"Staged control path is not a regular file: {destination}")
    if destination.is_file() and not destination.is_symlink():
        try:
            ensure_downloaded_file(destination, remote, "staged control file")
            reporter.emit("cache", f"reusing local staged control {remote.path}")
            return destination
        except CacheCorruptionError:
            reporter.emit("repair", f"replacing corrupt staged control {remote.path}")
    cached = download_verified(hub, remote, revision, "control file", reporter)
    atomic_copy_file(cached, destination)
    return destination


def publish_staged_control(staged: Path, relative: str, data_root: Path) -> Path:
    destination = data_root / relative
    ensure_safe_destination(data_root, destination)
    atomic_copy_file(staged, destination)
    return destination


def same_file_content(left: Path, right: Path) -> bool:
    try:
        if right.is_symlink() or not right.is_file() or left.stat().st_size != right.stat().st_size:
            return False
    except OSError:
        return False
    return sha256_file(left) == sha256_file(right)


def validate_control_relative(value: object) -> str:
    relative = validate_posix_path(value, "control transaction path", exact_parts=2).as_posix()
    if relative not in {BAD_MANIFEST_PATH, BAD_MANIFEST_INFO_PATH} and not CATEGORY_INDEX_RE.fullmatch(
        relative
    ):
        raise DownloadError(f"Invalid control transaction path: {relative}")
    return relative


def control_transaction_dir(data_root: Path) -> Path:
    return control_dir(data_root, create=True) / "control-transaction"


def recover_control_transaction(data_root: Path) -> None:
    transaction = control_transaction_dir(data_root)
    if not transaction.exists():
        return
    if transaction.is_symlink() or not transaction.is_dir():
        raise DownloadError(f"Unsafe control transaction path: {transaction}")
    journal_path = transaction / "journal.json"
    if not journal_path.is_file() or journal_path.is_symlink():
        # Publication starts only after the durable journal exists.  A
        # journal-less directory therefore contains incomplete backup copies
        # but no public changes.
        shutil.rmtree(transaction)
        fsync_directory(transaction.parent)
        return
    journal = load_json(journal_path)
    status = journal.get("status")
    entries = journal.get("entries")
    if status not in {"publishing", "committed"} or not isinstance(entries, list):
        raise DownloadError(f"Malformed control transaction journal: {journal_path}")
    if status == "publishing":
        for raw_entry in entries:
            if not isinstance(raw_entry, dict):
                raise DownloadError(f"Malformed control transaction entry: {journal_path}")
            relative = validate_control_relative(raw_entry.get("relative", ""))
            destination = data_root / relative
            ensure_safe_destination(data_root, destination)
            existed = raw_entry.get("existed")
            if not isinstance(existed, bool):
                raise DownloadError(f"Malformed control transaction existence flag: {relative}")
            if existed:
                backup = transaction / "backups" / relative
                expected_size = raw_entry.get("backup_size")
                expected_sha = str(raw_entry.get("backup_sha256", ""))
                if (
                    isinstance(expected_size, bool)
                    or not isinstance(expected_size, int)
                    or expected_size < 0
                    or not SHA256_RE.fullmatch(expected_sha)
                    or not backup.is_file()
                    or backup.is_symlink()
                    or backup.stat().st_size != expected_size
                    or sha256_file(backup) != expected_sha
                ):
                    raise IntegrityError(f"Control rollback backup is corrupt: {backup}")
                atomic_copy_file(backup, destination)
            elif destination.exists():
                if destination.is_symlink() or not destination.is_file():
                    raise DownloadError(f"Unsafe new control during rollback: {destination}")
                destination.unlink()
                fsync_directory(destination.parent)
    shutil.rmtree(transaction)
    fsync_directory(transaction.parent)


def publish_control_transaction(
    controls: Sequence[tuple[Path, str]],
    data_root: Path,
) -> None:
    recover_control_transaction(data_root)
    changed: list[tuple[Path, str]] = []
    for staged, raw_relative in controls:
        relative = validate_control_relative(raw_relative)
        if not same_file_content(staged, data_root / relative):
            changed.append((staged, relative))
    if not changed:
        return
    transaction = control_transaction_dir(data_root)
    transaction.mkdir()
    entries: list[dict[str, Any]] = []
    backup_bytes = 0
    backup_files = 0
    new_public_bytes = 0
    replacement_growth = 0
    new_public_files = 0
    largest_publish_temp = 0
    staged_by_relative = {relative: staged for staged, relative in changed}
    try:
        for _staged, relative in changed:
            destination = data_root / relative
            ensure_safe_destination(data_root, destination)
            existed = destination.exists()
            staged_size = staged_by_relative[relative].stat().st_size
            largest_publish_temp = max(largest_publish_temp, staged_size)
            entry: dict[str, Any] = {"relative": relative, "existed": existed}
            if existed:
                if destination.is_symlink() or not destination.is_file():
                    raise DownloadError(f"Unsafe existing control file: {destination}")
                entry["backup_size"] = destination.stat().st_size
                entry["backup_sha256"] = sha256_file(destination)
                backup_bytes += int(entry["backup_size"])
                backup_files += 1
                replacement_growth += max(0, staged_size - int(entry["backup_size"]))
            else:
                new_public_bytes += staged_size
                new_public_files += 1
            entries.append(entry)
        required_free_space(
            transaction,
            backup_bytes + new_public_bytes + replacement_growth + largest_publish_temp,
            backup_files + new_public_files + 1,
        )
        for entry in entries:
            if not entry["existed"]:
                continue
            relative = str(entry["relative"])
            backup = transaction / "backups" / relative
            ensure_safe_destination(data_root, backup)
            atomic_copy_file(data_root / relative, backup)
            if (
                backup.stat().st_size != entry["backup_size"]
                or sha256_file(backup) != entry["backup_sha256"]
            ):
                raise IntegrityError(f"Control changed while its rollback copy was made: {relative}")
        journal = {"schema_version": 1, "status": "publishing", "entries": entries}
        atomic_write_json(transaction / "journal.json", journal)
    except BaseException:
        if transaction.exists():
            shutil.rmtree(transaction)
            fsync_directory(transaction.parent)
        raise

    try:
        for staged, relative in changed:
            publish_staged_control(staged, relative, data_root)
        journal["status"] = "committed"
        atomic_write_json(transaction / "journal.json", journal)
    except BaseException:
        recover_control_transaction(data_root)
        raise
    recover_control_transaction(data_root)


def load_bad_samples(manifest_path: Path, info_path: Path) -> BadSamples:
    info = load_json(info_path)
    schema_version = require_json_integer(
        info.get("schema_version"), f"IO-bad schema_version in {info_path}"
    )
    if schema_version != 1:
        raise DownloadError(f"Unsupported IO-bad schema: {info.get('schema_version')!r}")
    expected_hash = str(info.get("manifest_sha256", "")).strip().lower()
    actual_hash = sha256_file(manifest_path)
    if not SHA256_RE.fullmatch(expected_hash) or actual_hash != expected_hash:
        raise IntegrityError(
            f"IO-bad manifest hash mismatch: expected {expected_hash or '<missing>'}, got {actual_hash}"
        )

    sources: set[str] = set()
    by_member: dict[tuple[str, str], str] = {}
    by_archive: dict[str, set[str]] = {}
    manifest_text = read_utf8_text(manifest_path, "IO-bad manifest")
    for line_number, line in enumerate(manifest_text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DownloadError(f"Malformed IO-bad JSONL at line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise DownloadError(f"Non-object IO-bad row at line {line_number}")
        source = validate_posix_path(
            row.get("source_relpath", ""), "IO-bad source_relpath", exact_parts=3
        )
        category = validate_token(source.parts[0], "IO-bad category")
        folder = validate_token(source.parts[1], "IO-bad sequence")
        if source.suffix.lower() != ".vdb":
            raise DownloadError(f"IO-bad source is not a VDB: {source}")
        archive_path = f"archives/{category}/{folder}.tar"
        member = validate_posix_path(
            row.get("member_path", ""), "IO-bad member_path", exact_parts=1
        )
        sample_key = validate_token(row.get("sample_key", ""), "IO-bad sample_key")
        member_path = member.as_posix()
        if (
            str(row.get("category", "")) != category
            or str(row.get("sequence", "")) != folder
            or str(row.get("archive_path", "")) != archive_path
            or member.suffix.lower() != ".vdb"
            or member.stem != sample_key
        ):
            raise DownloadError(f"IO-bad derived fields disagree at line {line_number}")
        source_text = source.as_posix()
        archive_member = (archive_path, member_path)
        if source_text in sources or archive_member in by_member:
            raise DownloadError(f"Duplicate IO-bad row at line {line_number}: {source_text}")
        sources.add(source_text)
        by_member[archive_member] = source_text
        by_archive.setdefault(archive_path, set()).add(source_text)
    expected_count = require_json_integer(
        info.get("record_count"),
        f"IO-bad record_count in {info_path}",
        minimum=0,
        error_type=IntegrityError,
    )
    if expected_count != len(sources):
        raise IntegrityError(
            f"IO-bad record count mismatch: expected {expected_count}, got {len(sources)}"
        )
    return BadSamples(sources, by_member, by_archive, actual_hash)


def parse_category_catalog(category: str, path: Path) -> CategoryCatalog:
    value = load_json(path)
    if str(value.get("category", "")) != category:
        raise DownloadError(f"Category identity mismatch in {path}")
    rows = value.get("samples")
    if not isinstance(rows, list) or not rows:
        raise DownloadError(f"Category index has no samples: {path}")
    declared_count = require_json_integer(
        value.get("num_samples"), f"num_samples in {path}", minimum=0
    )
    if declared_count != len(rows):
        raise DownloadError(f"Category sample count mismatch in {path}")

    archive_by_folder: dict[str, ArchiveUnit] = {}
    samples_by_source: dict[str, SampleRecord] = {}
    meta_paths: set[str] = set()
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise DownloadError(f"Non-object sample row {row_index + 1} in {path}")
        vdb_path = validate_posix_path(
            row.get("vdb_path", ""), f"{category} vdb_path row {row_index + 1}", exact_parts=2
        )
        if vdb_path.suffix.lower() != ".vdb":
            raise DownloadError(f"Non-VDB sample path in {path}: {vdb_path}")
        folder = validate_token(vdb_path.parts[0], f"{category} sequence")
        if folder.casefold() in RESERVED_CATEGORY_ENTRIES:
            raise DownloadError(
                f"Sequence folder conflicts with mandatory category controls in {path}: {folder}"
            )
        if row.get("folder") not in (None, "") and str(row["folder"]) != folder:
            raise DownloadError(f"folder and vdb_path disagree in {path}: {vdb_path}")
        meta_path = validate_posix_path(
            row.get("meta_path", ""), f"{category} meta_path row {row_index + 1}", exact_parts=2
        )
        if meta_path.parts[0] != "index" or meta_path.suffix.lower() != ".json":
            raise DownloadError(f"Invalid per-sample JSON path in {path}: {meta_path}")
        size_bytes = require_json_integer(
            row.get("size_bytes"),
            f"size_bytes for {vdb_path} in {path}",
            minimum=1,
        )
        source = f"{category}/{vdb_path.as_posix()}"
        full_meta = f"{category}/{meta_path.as_posix()}"
        if source in samples_by_source:
            raise DownloadError(f"Duplicate VDB path in {path}: {source}")
        if full_meta in meta_paths:
            raise DownloadError(f"Duplicate per-sample JSON path in {path}: {full_meta}")
        meta_paths.add(full_meta)
        sample = SampleRecord(category, folder, source, full_meta, size_bytes, row_index)
        samples_by_source[source] = sample
        archive = archive_by_folder.get(folder)
        if archive is None:
            archive = ArchiveUnit(category, folder, f"archives/{category}/{folder}.tar")
            archive_by_folder[folder] = archive
        archive.samples.append(sample)
    return CategoryCatalog(category, path, list(archive_by_folder.values()), samples_by_source)


def expected_sample_json_paths(catalogs: dict[str, CategoryCatalog]) -> set[str]:
    return {
        sample.meta_relpath
        for catalog in catalogs.values()
        for sample in catalog.samples_by_source.values()
    }


def check_bad_manifest_coverage(
    catalogs: dict[str, CategoryCatalog],
    bad_samples: BadSamples,
) -> dict[str, SampleRecord]:
    matched: dict[str, SampleRecord] = {}
    for catalog in catalogs.values():
        for source, sample in catalog.samples_by_source.items():
            if source in bad_samples.source_relpaths:
                matched[source] = sample
    missing = sorted(bad_samples.source_relpaths - set(matched))
    if missing:
        raise IntegrityError(
            f"IO-bad manifest references {len(missing)} samples absent from local category indexes; "
            f"first: {missing[:5]}"
        )
    return matched


def required_free_space(path: Path, bytes_needed: int, inodes_needed: int) -> None:
    required_combined_free_space(((path, bytes_needed, inodes_needed),))


def required_combined_free_space(
    requirements: Sequence[tuple[Path, int, int]],
) -> None:
    grouped: dict[int, dict[str, Any]] = {}
    for path, bytes_needed, inodes_needed in requirements:
        if bytes_needed <= 0 and inodes_needed <= 0:
            continue
        nearest = path
        while not nearest.exists():
            nearest = nearest.parent
        device = nearest.stat().st_dev
        group = grouped.setdefault(
            device,
            {"path": nearest, "bytes": 0, "inodes": 0},
        )
        group["bytes"] += max(0, bytes_needed)
        group["inodes"] += max(0, inodes_needed)

    for group in grouped.values():
        _required_free_space_on_filesystem(
            group["path"], group["bytes"], group["inodes"]
        )


def _required_free_space_on_filesystem(
    nearest: Path,
    bytes_needed: int,
    inodes_needed: int,
) -> None:
    usage = shutil.disk_usage(nearest)
    margin = min(MIN_FREE_MARGIN_BYTES, max(64 * 1024 * 1024, bytes_needed // 20))
    if usage.free < bytes_needed + margin:
        raise DownloadError(
            f"Not enough free space under {nearest}: need about {format_bytes(bytes_needed + margin)}, "
            f"have {format_bytes(usage.free)}"
        )
    try:
        stats = os.statvfs(nearest)
        free_inodes = int(stats.f_favail)
    except (AttributeError, OSError):
        return
    if free_inodes < inodes_needed + 1024:
        raise DownloadError(
            f"Not enough free inodes under {nearest}: need {inodes_needed + 1024:,}, "
            f"have {free_inodes:,}"
        )


@contextmanager
def streamed_zstd_tar(path: Path) -> Iterator[tarfile.TarFile]:
    zstd = shutil.which("zstd")
    if zstd is None:
        raise DownloadError("The zstd command is required to read meta/vfxdb_meta.tar.zst")
    proc = subprocess.Popen(
        [zstd, "-dcqf", "--", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    tf: tarfile.TarFile | None = None
    try:
        tf = tarfile.open(fileobj=proc.stdout, mode="r|", bufsize=1024 * 1024)
        yield tf
        tf.close()
        tf = None
        proc.stdout.read()
        proc.stdout.close()
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        if proc.stderr is not None:
            proc.stderr.close()
        return_code = proc.wait()
        if return_code != 0:
            raise CacheCorruptionError(
                f"zstd failed reading {path} (exit {return_code}): "
                f"{stderr.decode('utf-8', errors='replace').strip()}"
            )
    except BaseException:
        if tf is not None:
            tf.close()
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise


def safe_tar_member_name(value: str) -> str:
    try:
        return validate_posix_path(value, "archive member").as_posix()
    except DownloadError as exc:
        raise UnsafeArchiveError(str(exc)) from exc


def read_limited_member(
    tf: tarfile.TarFile,
    member: tarfile.TarInfo,
    limit: int,
    label: str,
) -> bytes:
    if member.size > limit:
        raise UnsafeArchiveError(f"{label} is too large: {member.name}")
    source = tf.extractfile(member)
    if source is None:
        raise IntegrityError(f"Cannot read {label}: {member.name}")
    payload = source.read(limit + 1)
    if len(payload) != member.size or len(payload) > limit:
        raise IntegrityError(f"Truncated or oversized {label}: {member.name}")
    return payload


def parse_json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"Malformed JSON in {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"Expected a JSON object in {label}")
    return value


@dataclass(frozen=True, slots=True)
class MetadataScan:
    file_count: int
    total_bytes: int
    manifest_payload: bytes


def validate_sample_json_archive(
    archive_path: Path,
    expected_paths: set[str],
    known_categories: set[str],
    reporter: PlainReporter | None = None,
) -> MetadataScan:
    expected_count = len(expected_paths)
    remaining = set(expected_paths)
    file_count = 0
    total_bytes = 0
    manifest: dict[str, Any] | None = None
    manifest_payload = b""
    try:
        with streamed_zstd_tar(archive_path) as tf:
            while True:
                member = tf.next()
                if member is None:
                    break
                try:
                    name = safe_tar_member_name(member.name)
                    if not member.isreg() or (hasattr(member, "issparse") and member.issparse()):
                        raise UnsafeArchiveError(f"Only regular files are allowed: {name}")
                    if name == "meta_manifest.json":
                        if manifest is not None:
                            raise UnsafeArchiveError(
                                "Duplicate meta_manifest.json in sample JSON archive"
                            )
                        manifest_payload = read_limited_member(
                            tf, member, MAX_SEQUENCE_MANIFEST_BYTES, "sample JSON archive manifest"
                        )
                        manifest = parse_json_object(manifest_payload, name)
                        continue
                    path = validate_posix_path(name, "sample JSON archive member", exact_parts=3)
                    if (
                        path.parts[0] not in known_categories
                        or path.parts[1] != "index"
                        or path.suffix.lower() != ".json"
                    ):
                        raise UnsafeArchiveError(f"Unexpected sample JSON path: {name}")
                    if name not in remaining:
                        raise IntegrityError(
                            f"Sample JSON archive contains a duplicate or unreferenced file: {name}"
                        )
                    remaining.remove(name)
                    payload = read_limited_member(tf, member, MAX_SAMPLE_JSON_BYTES, "sample JSON")
                    parse_json_object(payload, name)
                    file_count += 1
                    if reporter is not None and file_count % JSON_PROGRESS_INTERVAL == 0:
                        reporter.emit(
                            "verify",
                            f"validated {file_count:,}/{expected_count:,} per-sample JSON files",
                        )
                    # A million small JSON files consume filesystem blocks, not
                    # just their payload bytes.  Account for a conservative
                    # 4 KiB allocation unit before creating any of them.
                    total_bytes += max(4096, ((len(payload) + 4095) // 4096) * 4096)
                finally:
                    tf.members.clear()
    except (OSError, tarfile.TarError) as exc:
        raise CacheCorruptionError(f"Cannot read sample JSON archive {archive_path}: {exc}") from exc
    if manifest is None:
        raise IntegrityError("meta/vfxdb_meta.tar.zst has no meta_manifest.json")
    if str(manifest.get("repo_id", "")) != REPO_ID:
        raise IntegrityError("Sample JSON archive repository identity mismatch")
    declared_count = require_json_integer(
        manifest.get("file_count"),
        "file_count in meta/vfxdb_meta.tar.zst manifest",
        minimum=0,
        error_type=IntegrityError,
    )
    if declared_count != file_count:
        raise IntegrityError(
            f"Sample JSON archive count mismatch: expected {manifest.get('file_count')}, got {file_count}"
        )
    missing = sorted(remaining)
    if missing:
        raise IntegrityError(
            f"Sample JSON archive is missing {len(missing):,} category-index references; "
            f"first: {missing[:5]}"
        )
    return MetadataScan(file_count, total_bytes, manifest_payload)


def extract_sample_json_archive(
    archive_path: Path,
    data_root: Path,
    reporter: PlainReporter,
    expected_sample_count: int,
) -> None:
    written_samples = 0
    touched_directories: set[Path] = set()
    with streamed_zstd_tar(archive_path) as tf:
        while True:
            member = tf.next()
            if member is None:
                break
            try:
                name = safe_tar_member_name(member.name)
                if not member.isreg() or (hasattr(member, "issparse") and member.issparse()):
                    raise UnsafeArchiveError(f"Only regular files are allowed: {name}")
                payload = read_limited_member(
                    tf,
                    member,
                    MAX_SEQUENCE_MANIFEST_BYTES if name == "meta_manifest.json" else MAX_SAMPLE_JSON_BYTES,
                    "sample JSON archive member",
                )
                destination = data_root / name
                ensure_safe_destination(data_root, destination)
                atomic_write_bytes(destination, payload, durable=False)
                touched_directories.add(destination.parent)
                if name != "meta_manifest.json":
                    written_samples += 1
                    if written_samples % JSON_PROGRESS_INTERVAL == 0:
                        reporter.emit(
                            "json",
                            f"installed {written_samples:,}/{expected_sample_count:,} per-sample JSON files",
                        )
            finally:
                tf.members.clear()
    # Per-file fsync would require roughly two million sync syscalls on the
    # published dataset.  Commit the small set of category directories once
    # after every atomically replaced file is closed.
    for directory in touched_directories:
        fsync_directory(directory)
    reporter.emit(
        "json",
        f"installed {written_samples:,}/{expected_sample_count:,} per-sample JSON files",
    )


def extract_named_sample_jsons(
    archive_path: Path,
    data_root: Path,
    wanted: set[str],
) -> None:
    if not wanted:
        return
    found: set[str] = set()
    with streamed_zstd_tar(archive_path) as tf:
        while True:
            member = tf.next()
            if member is None:
                break
            try:
                name = safe_tar_member_name(member.name)
                if name not in wanted:
                    continue
                if not member.isreg() or (hasattr(member, "issparse") and member.issparse()):
                    raise UnsafeArchiveError(f"Only regular files are allowed: {name}")
                payload = read_limited_member(tf, member, MAX_SAMPLE_JSON_BYTES, "sample JSON")
                parse_json_object(payload, name)
                destination = data_root / name
                ensure_safe_destination(data_root, destination)
                atomic_write_bytes(destination, payload, durable=False)
                found.add(name)
            finally:
                tf.members.clear()
    missing = sorted(wanted - found)
    if missing:
        raise IntegrityError(f"Could not restore {len(missing)} sample JSON files; first: {missing[:5]}")
    for directory in {((data_root / name).parent) for name in found}:
        fsync_directory(directory)


def sample_json_tree_digest(
    data_root: Path,
    wanted: set[str],
    reporter: PlainReporter | None = None,
    progress_label: str = "hashed local per-sample JSON files",
) -> str | None:
    xor_accumulator = 0
    sum_accumulator = 0
    modulus = 1 << 256
    total = len(wanted)
    for position, relative in enumerate(wanted, 1):
        path = data_root / relative
        try:
            stat = path.stat()
        except OSError:
            return None
        if path.is_symlink() or not path.is_file():
            return None
        item = hashlib.sha256(
            relative.encode("utf-8")
            + b"\0"
            + f"{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ctime_ns}".encode("ascii")
        ).digest()
        item_number = int.from_bytes(item, "big")
        xor_accumulator ^= item_number
        sum_accumulator = (sum_accumulator + item_number) % modulus
        if reporter is not None and position % JSON_PROGRESS_INTERVAL == 0:
            reporter.emit("verify", f"{progress_label} {position:,}/{total:,}")
    return hashlib.sha256(
        len(wanted).to_bytes(8, "big")
        + xor_accumulator.to_bytes(32, "big")
        + sum_accumulator.to_bytes(32, "big")
    ).hexdigest()


def install_mandatory_sample_jsons(
    archive_path: Path,
    remote: RemoteFile,
    catalogs: dict[str, CategoryCatalog],
    bad_records: dict[str, SampleRecord],
    data_root: Path,
    state: dict[str, Any],
    reporter: PlainReporter,
) -> tuple[bool, str | None]:
    expected = expected_sample_json_paths(catalogs)
    bad_paths = {record.meta_relpath for record in bad_records.values()}
    normal_paths = expected - bad_paths
    fingerprint = remote_fingerprint(remote)
    checkpoint = state.get("sample_json_archive")
    checkpoint_current = (
        isinstance(checkpoint, dict)
        and checkpoint.get("fingerprint") == fingerprint
        and checkpoint.get("validated") is True
    )
    if checkpoint_current:
        missing_normal: set[str] = set()
        for number, relative in enumerate(normal_paths, 1):
            path = data_root / relative
            if not path.is_file() or path.is_symlink():
                missing_normal.add(relative)
            if number % JSON_PROGRESS_INTERVAL == 0:
                reporter.emit(
                    "verify",
                    f"checked local normal JSON files {number:,}/{len(normal_paths):,}",
                )
        if missing_normal:
            reporter.emit(
                "repair",
                f"restoring {len(missing_normal):,} missing normal per-sample JSON files",
            )
            extract_named_sample_jsons(archive_path, data_root, missing_normal)
        normal_digest = sample_json_tree_digest(data_root, normal_paths, reporter)
        if normal_digest is None:
            raise IntegrityError("Normal per-sample JSON restoration is incomplete")
        if checkpoint.get("normal_tree_digest") != normal_digest:
            reporter.emit(
                "repair",
                "normal per-sample JSON identity changed; restoring published contents",
            )
            extract_named_sample_jsons(archive_path, data_root, set(normal_paths))
            normal_digest = sample_json_tree_digest(data_root, normal_paths, reporter)
            if normal_digest is None:
                raise IntegrityError("Normal per-sample JSON repair is incomplete")

        reporter.emit(
            "cache",
            f"all {len(normal_paths):,} normal per-sample JSON files are local",
        )
        return False, normal_digest

    reporter.emit("verify", "validating every category-index reference against meta/vfxdb_meta.tar.zst")
    scan = validate_sample_json_archive(archive_path, expected, set(catalogs), reporter)
    required_free_space(data_root, scan.total_bytes, scan.file_count)
    reporter.emit("json", f"installing {scan.file_count:,} per-sample JSON files")
    extract_sample_json_archive(archive_path, data_root, reporter, scan.file_count)
    state["sample_json_archive"] = {
        "fingerprint": fingerprint,
        "validated": True,
        "file_count": scan.file_count,
    }
    save_state(data_root, state)
    return True, None


def finalize_sample_json_checkpoint(
    catalogs: dict[str, CategoryCatalog],
    bad_records: dict[str, SampleRecord],
    data_root: Path,
    state: dict[str, Any],
    include_bad: bool,
    known_normal_digest: str | None = None,
    reporter: PlainReporter | None = None,
) -> None:
    expected = expected_sample_json_paths(catalogs)
    bad_paths = {record.meta_relpath for record in bad_records.values()}
    normal_paths = expected - bad_paths
    checkpoint = state.get("sample_json_archive")
    if not isinstance(checkpoint, dict) or checkpoint.get("validated") is not True:
        raise IntegrityError("Per-sample JSON archive has no validated checkpoint")
    normal_digest = known_normal_digest or sample_json_tree_digest(
        data_root, normal_paths, reporter
    )
    if normal_digest is None:
        raise IntegrityError("Normal per-sample JSON installation is incomplete")
    checkpoint["include_bad"] = include_bad
    checkpoint["normal_tree_digest"] = normal_digest
    checkpoint.pop("tree_digest", None)
    if include_bad:
        bad_digest = sample_json_tree_digest(data_root, bad_paths, reporter)
        if bad_digest is None:
            raise IntegrityError("IO-bad per-sample JSON installation is incomplete")
        checkpoint["bad_tree_digest"] = bad_digest
    else:
        checkpoint.pop("bad_tree_digest", None)


def prepare_category_indexes_for_publication(
    catalogs: dict[str, CategoryCatalog],
    bad_records: dict[str, SampleRecord],
    data_root: Path,
    revision: str,
    include_bad: bool,
) -> tuple[dict[str, Path], Path | None]:
    by_category: dict[str, set[int]] = {}
    for record in bad_records.values():
        by_category.setdefault(record.category, set()).add(record.row_index)
    if not by_category:
        return {category: catalog.path for category, catalog in catalogs.items()}, None

    publish_root = control_dir(data_root, create=True) / "category-publish" / revision
    if publish_root.exists():
        if publish_root.is_symlink() or not publish_root.is_dir():
            raise DownloadError(f"Unsafe category-index publication staging: {publish_root}")
        shutil.rmtree(publish_root)

    prepared = {category: catalog.path for category, catalog in catalogs.items()}
    try:
        for category in sorted(by_category):
            row_indexes = by_category[category]
            catalog = catalogs[category]
            value = load_json(catalog.path)
            rows = value["samples"]
            for row_index in row_indexes:
                rows[row_index]["deleted_bad_io_sample"] = not include_bad
            payload = encode_json(value)
            public = data_root / category / "category_index.json"
            ensure_safe_destination(data_root, public)
            if public.exists() and (public.is_symlink() or not public.is_file()):
                raise DownloadError(f"Unsafe public category index: {public}")
            if (
                public.is_file()
                and public.stat().st_size == len(payload)
                and sha256_file(public) == hashlib.sha256(payload).hexdigest()
            ):
                prepared[category] = public
                continue
            destination = publish_root / category / "category_index.json"
            ensure_safe_destination(data_root, destination)
            required_free_space(publish_root, len(payload), 2)
            atomic_write_bytes(destination, payload)
            prepared[category] = destination
    except BaseException:
        if publish_root.exists():
            shutil.rmtree(publish_root)
        raise
    return prepared, publish_root


def cleanup_category_publish_staging(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    if path.is_symlink() or not path.is_dir():
        raise DownloadError(f"Unsafe category-index publication staging: {path}")
    shutil.rmtree(path)
    fsync_directory(path.parent)


def apply_bad_policy_to_files(
    archive_path: Path,
    bad_records: dict[str, SampleRecord],
    data_root: Path,
    state: dict[str, Any],
    include_bad: bool,
    fresh_full_extraction: bool,
    reporter: PlainReporter,
) -> None:
    bad_paths = {record.meta_relpath for record in bad_records.values()}
    if include_bad:
        checkpoint = state.get("sample_json_archive")
        current_digest = sample_json_tree_digest(data_root, bad_paths)
        already_trusted = (
            fresh_full_extraction
            or (
                isinstance(checkpoint, dict)
                and checkpoint.get("include_bad") is True
                and current_digest is not None
                and checkpoint.get("bad_tree_digest") == current_digest
            )
        )
        if not already_trusted:
            reporter.emit(
                "repair",
                f"restoring {len(bad_paths):,} IO-bad per-sample JSON files",
            )
            extract_named_sample_jsons(archive_path, data_root, set(bad_paths))
        return

    for source, record in bad_records.items():
        json_path = data_root / record.meta_relpath
        ensure_safe_destination(data_root, json_path)
        if json_path.exists():
            json_path.unlink()
        vdb_path = data_root / source
        ensure_safe_destination(data_root, vdb_path)
        if vdb_path.exists():
            vdb_path.unlink()


def all_archives(catalogs: dict[str, CategoryCatalog]) -> list[ArchiveUnit]:
    return [archive for category in sorted(catalogs) for archive in catalogs[category].archives]


def balanced_archives(catalogs: dict[str, CategoryCatalog], target: int) -> list[ArchiveUnit]:
    if target <= 0:
        return []
    categories = sorted(catalogs)
    positions = {category: 0 for category in categories}
    selected: list[ArchiveUnit] = []
    while len(selected) < target:
        progressed = False
        for category in categories:
            position = positions[category]
            archives = catalogs[category].archives
            if position >= len(archives):
                continue
            selected.append(archives[position])
            positions[category] = position + 1
            progressed = True
            if len(selected) == target:
                break
        if not progressed:
            break
    if len(selected) != target:
        raise DownloadError(f"Cannot allocate {target} tars from {len(all_archives(catalogs))} available")
    return selected


def summarize_plan(
    label: str,
    selected: list[ArchiveUnit],
    catalogs: dict[str, CategoryCatalog],
    bad_sources: set[str],
    requested_max: int | None = None,
) -> DownloadPlan:
    selected_by_category = {category: 0 for category in sorted(catalogs)}
    available_by_category = {
        category: len(catalogs[category].archives) for category in sorted(catalogs)
    }
    normal_by_category = {category: 0 for category in sorted(catalogs)}
    for archive in selected:
        selected_by_category[archive.category] += 1
        normal_by_category[archive.category] += archive.normal_sample_count(bad_sources)
    return DownloadPlan(
        label,
        selected,
        len(all_archives(catalogs)),
        selected_by_category,
        available_by_category,
        normal_by_category,
        requested_max,
    )


def build_plan(
    options: Options,
    catalogs: dict[str, CategoryCatalog],
    bad_sources: set[str],
) -> DownloadPlan:
    total = len(all_archives(catalogs))
    if options.mode == "metadata-only":
        return summarize_plan("metadata-only", [], catalogs, bad_sources)
    if options.preset == "smoke":
        selected = [
            archive
            for category in sorted(catalogs)
            for archive in catalogs[category].archives[:2]
        ]
        return summarize_plan("preset smoke", selected, catalogs, bad_sources)
    if options.preset == "medium":
        target = (total + 4) // 5
        return summarize_plan(
            "preset medium", balanced_archives(catalogs, target), catalogs, bad_sources
        )
    if options.preset == "full":
        return summarize_plan("preset full", all_archives(catalogs), catalogs, bad_sources)
    if options.percentage is not None:
        target_decimal = (Decimal(total) * options.percentage / Decimal(100)).to_integral_value(
            rounding=ROUND_CEILING
        )
        target = int(target_decimal)
        selected = all_archives(catalogs) if target == total else balanced_archives(catalogs, target)
        return summarize_plan(
            f"all-category {options.percentage}%",
            selected,
            catalogs,
            bad_sources,
        )

    unknown = [category for category in options.categories if category not in catalogs]
    if unknown:
        raise DownloadError(f"Unknown categories: {unknown}. Available: {sorted(catalogs)}")
    assert options.max_samples is not None
    selected: list[ArchiveUnit] = []
    for category in sorted(options.categories):
        cumulative = 0
        for archive in catalogs[category].archives:
            selected.append(archive)
            cumulative += archive.normal_sample_count(bad_sources)
            if cumulative >= options.max_samples:
                break
    return summarize_plan(
        f"category max-samples={options.max_samples}",
        selected,
        catalogs,
        bad_sources,
        requested_max=options.max_samples,
    )


def validate_selected_remote_archives(
    plan: DownloadPlan,
    remote_files: dict[str, RemoteFile],
) -> None:
    missing = [archive.remote_path for archive in plan.archives if archive.remote_path not in remote_files]
    if missing:
        raise DownloadError(f"Selected tar files are missing from the dataset commit: {missing[:5]}")


def print_plan(
    plan: DownloadPlan,
    remote_files: dict[str, RemoteFile],
    revision: str,
    data_root: Path,
    cache_root: Path,
    include_bad: bool,
) -> None:
    remote_bytes = sum(remote_files[archive.remote_path].size for archive in plan.archives)
    print("VfxDB download plan")
    print(f"  Revision    : {revision}")
    print(f"  Destination : {data_root}")
    print(f"  HF cache    : {cache_root}")
    print(f"  Mode        : {plan.label}")
    print(f"  Tars        : {len(plan.archives):,} / {plan.all_archive_count:,}")
    print(f"  Download    : {format_bytes(remote_bytes)} of tar objects")
    print(f"  IO-bad      : {'retain after extraction' if include_bad else 'remove after extraction'}")
    if plan.archives:
        print("  By category :")
        for category in sorted(plan.selected_by_category):
            count = plan.selected_by_category[category]
            if count:
                normal = plan.normal_samples_by_category[category]
                available = plan.available_by_category[category]
                suffix = ""
                if plan.requested_max_samples is not None:
                    suffix = f", target {plan.requested_max_samples:,} (whole-tar upward rounding)"
                print(
                    f"    {category}: {count:,}/{available:,} tars, "
                    f"{normal:,} normal samples{suffix}"
                )


def print_bare_usage(data_root: Path) -> None:
    script = "python tools/download_extract_data.py"
    print()
    print("No VDB data tar was downloaded. Required category indexes and per-sample JSON are ready.")
    print("Choose one data mode when you want VDB files:")
    print(f"  Smoke : {script} {data_root} --preset smoke")
    print(f"  Medium : {script} {data_root} --preset medium")
    print(f"  Full   : {script} {data_root} --preset full")
    print(f"  Percent: {script} {data_root} --percentage 10")
    print(f"  Category: {script} {data_root} --category CloudWave --max-samples 1000")


def read_sequence_manifest(
    tf: tarfile.TarFile,
    member: tarfile.TarInfo,
    archive_path: Path,
) -> dict[str, Any]:
    payload = read_limited_member(tf, member, MAX_SEQUENCE_MANIFEST_BYTES, "sequence manifest")
    return parse_json_object(payload, str(archive_path))


def inspect_sequence_archive(
    archive_path: Path,
    archive: ArchiveUnit,
    bad_samples: BadSamples,
) -> tuple[tarfile.TarFile, dict[str, dict[str, Any]], dict[str, tarfile.TarInfo]]:
    try:
        tf = tarfile.open(archive_path, mode="r:")
        members = tf.getmembers()
    except (OSError, tarfile.TarError) as exc:
        raise CacheCorruptionError(f"Cannot read tar {archive_path}: {exc}") from exc
    try:
        by_name: dict[str, tarfile.TarInfo] = {}
        for member in members:
            name = safe_tar_member_name(member.name)
            if name in by_name:
                raise UnsafeArchiveError(f"Duplicate tar member in {archive_path}: {name}")
            if not member.isreg() or (hasattr(member, "issparse") and member.issparse()):
                raise UnsafeArchiveError(f"Only regular tar members are allowed: {name}")
            by_name[name] = member
        manifest_member = by_name.get("_sequence_manifest.json")
        if manifest_member is None:
            raise IntegrityError(f"Missing _sequence_manifest.json in {archive_path}")
        manifest = read_sequence_manifest(tf, manifest_member, archive_path)
        if (
            manifest.get("package_format") != "webdataset"
            or str(manifest.get("category", "")) != archive.category
            or str(manifest.get("sequence", "")) != archive.folder
        ):
            raise IntegrityError(f"Sequence manifest identity mismatch in {archive_path}")
        rows = manifest.get("samples")
        declared_count = require_json_integer(
            manifest.get("num_samples"),
            f"num_samples in sequence manifest {archive_path}",
            minimum=0,
            error_type=IntegrityError,
        )
        if not isinstance(rows, list) or declared_count != len(rows):
            raise IntegrityError(f"Sequence manifest sample count mismatch in {archive_path}")
        rows_by_source: dict[str, dict[str, Any]] = {}
        described = {"_sequence_manifest.json"}
        bad_in_tar: set[str] = set()
        for row_number, raw in enumerate(rows, 1):
            if not isinstance(raw, dict):
                raise IntegrityError(f"Non-object sequence row {row_number} in {archive_path}")
            row = copy.deepcopy(raw)
            key = validate_token(row.get("key", ""), "sequence member key")
            source = validate_posix_path(
                row.get("source_relpath", ""), "sequence source_relpath", exact_parts=3
            )
            if (
                source.parts[0] != archive.category
                or source.parts[1] != archive.folder
                or source.suffix.lower() != ".vdb"
            ):
                raise IntegrityError(f"Sequence source path disagrees with tar: {source}")
            source_text = source.as_posix()
            member_name = f"{key}.vdb"
            raw_digest = row.get("sha256", "")
            if not isinstance(raw_digest, str):
                raise IntegrityError(f"Invalid SHA-256 for {source_text}")
            digest = raw_digest.lower()
            if digest and not SHA256_RE.fullmatch(digest):
                raise IntegrityError(f"Invalid SHA-256 for {source_text}")
            row["sha256"] = digest
            if source_text in rows_by_source or member_name in described:
                raise IntegrityError(f"Duplicate sequence mapping in {archive_path}: {source_text}")
            if member_name not in by_name:
                raise IntegrityError(f"Missing tar member {member_name} in {archive_path}")
            rows_by_source[source_text] = row
            described.add(member_name)
            expected_bad = bad_samples.source_by_archive_member.get((archive.remote_path, member_name))
            source_bad = source_text in bad_samples.source_relpaths
            if source_bad != (expected_bad is not None) or (expected_bad and expected_bad != source_text):
                raise IntegrityError(f"IO-bad manifest disagrees with tar member {member_name}")
            if source_bad:
                bad_in_tar.add(source_text)
        if set(by_name) != described:
            raise UnsafeArchiveError(
                f"Tar has members absent from its manifest: {sorted(set(by_name) - described)[:5]}"
            )
        expected_sources = {sample.source_relpath for sample in archive.samples}
        if set(rows_by_source) != expected_sources:
            missing = sorted(expected_sources - set(rows_by_source))
            extra = sorted(set(rows_by_source) - expected_sources)
            raise IntegrityError(
                f"Tar/category-index membership mismatch for {archive.remote_path}; "
                f"missing={missing[:3]} extra={extra[:3]}"
            )
        if bad_in_tar != bad_samples.sources_by_archive.get(archive.remote_path, set()):
            raise IntegrityError(f"IO-bad coverage mismatch for {archive.remote_path}")
        for sample in archive.samples:
            row = rows_by_source[sample.source_relpath]
            member = by_name[f"{row['key']}.vdb"]
            if member.size != sample.size_bytes:
                raise IntegrityError(
                    f"Size mismatch for {sample.source_relpath}: index={sample.size_bytes}, tar={member.size}"
                )
        return tf, rows_by_source, by_name
    except BaseException:
        tf.close()
        raise


def copy_tar_member(
    tf: tarfile.TarFile,
    member: tarfile.TarInfo,
    destination: Path,
) -> None:
    source = tf.extractfile(member)
    if source is None:
        raise IntegrityError(f"Cannot read {member.name} from tar")
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with destination.open("xb") as handle:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            handle.write(chunk)
            written += len(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(destination, 0o644)
    if written != member.size:
        raise CacheCorruptionError(
            f"Tar member size mismatch for {member.name}: expected {member.size}, got {written}"
        )


def staging_directory(data_root: Path, archive_path: str) -> Path:
    root = control_dir(data_root, create=True) / "staging"
    if root.is_symlink():
        raise DownloadError(f"Downloader staging directory must not be a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    path = root / hashlib.sha256(archive_path.encode("utf-8")).hexdigest()[:20]
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise DownloadError(f"Unsafe downloader staging path: {path}")
        shutil.rmtree(path)
    path.mkdir()
    return path


def archive_public_directory(data_root: Path, archive: ArchiveUnit) -> Path:
    return data_root / archive.category / archive.folder


def archive_backup_directory(data_root: Path, archive: ArchiveUnit) -> Path:
    backups = control_dir(data_root, create=True) / "backups"
    if backups.is_symlink():
        raise DownloadError(f"Downloader backup directory must not be a symlink: {backups}")
    backups.mkdir(parents=True, exist_ok=True)
    return backups / hashlib.sha256(archive.remote_path.encode("utf-8")).hexdigest()[:20]


def recover_archive_swap(data_root: Path, archive: ArchiveUnit) -> None:
    destination = archive_public_directory(data_root, archive)
    backup = archive_backup_directory(data_root, archive)
    if not backup.exists():
        return
    if backup.is_symlink() or not backup.is_dir():
        raise DownloadError(f"Unsafe archive backup path: {backup}")
    if destination.exists():
        shutil.rmtree(backup)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(backup, destination)
    fsync_directory(destination.parent)


def archive_is_current(
    archive: ArchiveUnit,
    data_root: Path,
    include_bad: bool,
    bad_sources: set[str],
) -> bool:
    for sample in archive.samples:
        path = data_root / sample.source_relpath
        if sample.source_relpath in bad_sources and not include_bad:
            if path.exists():
                return False
            continue
        try:
            if path.is_symlink() or path.stat().st_size != sample.size_bytes:
                return False
        except OSError:
            return False
    return True


def archive_tree_digest(
    archive: ArchiveUnit,
    data_root: Path,
    include_bad: bool,
    bad_sources: set[str],
) -> str | None:
    """Fingerprint local file identities without storing one row per sample."""

    digest = hashlib.sha256()
    for sample in archive.samples:
        digest.update(sample.source_relpath.encode("utf-8"))
        digest.update(b"\0")
        path = data_root / sample.source_relpath
        if sample.source_relpath in bad_sources and not include_bad:
            if path.exists():
                return None
            digest.update(b"absent\0")
            continue
        try:
            stat = path.stat()
        except OSError:
            return None
        if path.is_symlink() or stat.st_size != sample.size_bytes:
            return None
        digest.update(
            f"{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ctime_ns}".encode("ascii")
        )
        digest.update(b"\0")
    return digest.hexdigest()


def remove_bad_vdbs(
    data_root: Path,
    bad_records: dict[str, SampleRecord],
) -> None:
    for source in bad_records:
        path = data_root / source
        ensure_safe_destination(data_root, path)
        if path.exists():
            path.unlink()


def install_whole_archive(
    archive_path: Path,
    archive: ArchiveUnit,
    bad_samples: BadSamples,
    data_root: Path,
    include_bad: bool,
) -> None:
    tf, rows_by_source, by_name = inspect_sequence_archive(archive_path, archive, bad_samples)
    staging = staging_directory(data_root, archive.remote_path)
    try:
        for sample in archive.samples:
            row = rows_by_source[sample.source_relpath]
            member_name = f"{row['key']}.vdb"
            staged = staging / sample.source_relpath
            # The complete tar has already matched its published HF content
            # identity. Legacy sequence manifests contain both empty and stale
            # per-member digests across categories, so member integrity is
            # anchored by that outer identity plus exact membership and size.
            copy_tar_member(tf, by_name[member_name], staged)
        if not include_bad:
            for source in bad_samples.sources_by_archive.get(archive.remote_path, set()):
                staged_bad = staging / source
                if staged_bad.exists():
                    staged_bad.unlink()

        staged_sequence = staging / archive.category / archive.folder
        destination_sequence = archive_public_directory(data_root, archive)
        ensure_safe_destination(data_root, destination_sequence)
        destination_sequence.parent.mkdir(parents=True, exist_ok=True)
        backup = archive_backup_directory(data_root, archive)
        if backup.exists():
            raise DownloadError(f"Unrecovered archive backup blocks publication: {backup}")
        moved_old = False
        if destination_sequence.exists():
            if destination_sequence.is_symlink() or not destination_sequence.is_dir():
                raise DownloadError(f"Unsafe archive destination: {destination_sequence}")
            os.replace(destination_sequence, backup)
            moved_old = True
        try:
            os.replace(staged_sequence, destination_sequence)
            fsync_directory(destination_sequence.parent)
        except BaseException:
            if moved_old and backup.exists() and not destination_sequence.exists():
                os.replace(backup, destination_sequence)
                fsync_directory(destination_sequence.parent)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        tf.close()
        if staging.exists():
            shutil.rmtree(staging)


def install_data_archives(
    plan: DownloadPlan,
    remote_files: dict[str, RemoteFile],
    revision: str,
    hub: HubClient,
    bad_samples: BadSamples,
    bad_records: dict[str, SampleRecord],
    data_root: Path,
    include_bad: bool,
    state: dict[str, Any],
    reporter: PlainReporter,
) -> None:
    archive_state = state.setdefault("archives", {})
    current_digests: dict[str, str] = {}
    needs_install: list[ArchiveUnit] = []
    uncached: list[RemoteFile] = []
    for archive in plan.archives:
        checkpoint = archive_state.get(archive.remote_path)
        current_digest = archive_tree_digest(
            archive, data_root, include_bad, bad_samples.source_relpaths
        )
        remote = remote_files[archive.remote_path]
        already_installed = (
            isinstance(checkpoint, dict)
            and checkpoint.get("fingerprint") == remote_fingerprint(remote)
            and current_digest is not None
            and checkpoint.get("tree_digest") == current_digest
        )
        if already_installed:
            assert current_digest is not None
            current_digests[archive.remote_path] = current_digest
            continue
        needs_install.append(archive)
        if not cached_remote_is_usable(hub, remote, revision):
            uncached.append(remote)
    # Cache blobs persist while complete tar contents are staged and installed.
    # Aggregate both requirements when cache and destination share a device.
    required_combined_free_space(
        (
            (
                hub.cache_root,
                sum(remote.size for remote in uncached),
                len(uncached),
            ),
            (
                data_root,
                sum(
                    sample.size_bytes
                    for archive in needs_install
                    for sample in archive.samples
                ),
                sum(len(archive.samples) for archive in needs_install),
            ),
        )
    )
    needs_install_paths = {archive.remote_path for archive in needs_install}
    for index, archive in enumerate(plan.archives, 1):
        remote = remote_files[archive.remote_path]
        checkpoint = archive_state.get(archive.remote_path)
        if archive.remote_path not in needs_install_paths:
            assert current_digests[archive.remote_path]
            assert isinstance(checkpoint, dict)
            if checkpoint.get("include_bad") is not include_bad:
                checkpoint["include_bad"] = include_bad
                save_state(data_root, state)
            reporter.emit("cache", f"[{index}/{len(plan.archives)}] {archive.remote_path} already installed")
            continue
        reporter.emit("download", f"[{index}/{len(plan.archives)}] {archive.remote_path}")
        cached = download_verified(hub, remote, revision, "sequence tar", reporter)
        reporter.emit(
            "verify",
            f"[{index}/{len(plan.archives)}] {archive.remote_path} validating complete tar",
        )
        reporter.emit(
            "extract",
            f"[{index}/{len(plan.archives)}] {archive.remote_path} installing complete tar",
        )
        install_whole_archive(cached, archive, bad_samples, data_root, include_bad)
        installed_digest = archive_tree_digest(
            archive, data_root, include_bad, bad_samples.source_relpaths
        )
        if installed_digest is None:
            raise IntegrityError(f"Installed tar did not reach a complete state: {archive.remote_path}")
        archive_state[archive.remote_path] = {
            "fingerprint": remote_fingerprint(remote),
            "include_bad": include_bad,
            "tree_digest": installed_digest,
        }
        save_state(data_root, state)
        reporter.emit("installed", f"[{index}/{len(plan.archives)}] {archive.remote_path}")
    if not include_bad:
        remove_bad_vdbs(data_root, bad_records)


def initial_state(revision: str) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "repo_id": REPO_ID,
        "revision": revision,
        "sample_json_archive": {},
        "archives": {},
    }


def run(
    options: Options,
    *,
    hub: HubClient | None = None,
    reporter: PlainReporter | None = None,
    interaction: DownloadInteraction | None = None,
    plain_output: bool = True,
) -> DownloadPlan:
    reporter = reporter or PlainReporter()
    for component in (options.data_root, *options.data_root.parents):
        if component.is_symlink():
            raise DownloadError(f"Destination path must not traverse a symlink: {component}")
    options.data_root.mkdir(parents=True, exist_ok=True)
    if not state_path(options.data_root).exists():
        unmanaged = [
            entry for entry in options.data_root.iterdir() if entry.name != ".vfxdb"
        ]
        if unmanaged:
            raise DownloadError(
                "Destination contains files but has no VfxDB downloader state; refusing to mix "
                f"revisions. Use a new empty destination. First entries: "
                f"{[str(path) for path in unmanaged[:5]]}"
            )
    with destination_lock(options.data_root):
        recover_control_transaction(options.data_root)
        state = load_state(options.data_root)
        if state is None:
            ensure_empty_unmanaged_destination(options.data_root)
        transition = policy_transition_path(options.data_root)
        if transition.exists():
            if transition.is_symlink() or not transition.is_file():
                raise DownloadError(f"Unsafe IO-bad policy transition state: {transition}")
            reporter.emit(
                "resume",
                "an earlier IO-bad policy transition was incomplete; reconciling it now",
            )
        requested_revision = options.revision or (str(state["revision"]) if state else "main")
        hub = hub or HubClient(options.cache_dir, reporter)
        revision, remote_files = hub.resolve_snapshot(requested_revision)
        reporter.emit("resolve", f"pinned dataset revision {revision}")
        if state is not None and revision != state["revision"]:
            raise DownloadError(
                f"Destination is pinned to {state['revision']}, not requested commit {revision}; "
                "use a different destination"
            )
        if state is None:
            state = initial_state(revision)
            save_state(options.data_root, state)

        category_remotes = discover_category_indexes(remote_files)
        bad_manifest_remote = require_remote_file(remote_files, BAD_MANIFEST_PATH)
        bad_info_remote = require_remote_file(remote_files, BAD_MANIFEST_INFO_PATH)
        sample_json_remote = require_remote_file(remote_files, SAMPLE_JSON_ARCHIVE_PATH)
        staged_control_remotes = [
            *(remote for _category, remote in category_remotes),
            bad_manifest_remote,
            bad_info_remote,
        ]
        staging_root = control_dir(options.data_root, create=True) / "control-staging" / revision
        if staging_root.is_symlink():
            raise DownloadError(f"Control staging directory must not be a symlink: {staging_root}")
        controls_needing_stage: list[RemoteFile] = []
        for remote in staged_control_remotes:
            staged = staging_root / remote.path
            ensure_safe_destination(options.data_root, staged)
            if staged.exists() and (staged.is_symlink() or not staged.is_file()):
                raise DownloadError(f"Unsafe staged control path: {staged}")
            if staged.is_file():
                try:
                    ensure_downloaded_file(staged, remote, "staged control file")
                    continue
                except CacheCorruptionError:
                    pass
            controls_needing_stage.append(remote)
        needed_remote_objects = [sample_json_remote, *controls_needing_stage]
        uncached_controls = [
            remote
            for remote in needed_remote_objects
            if not cached_remote_is_usable(hub, remote, revision)
        ]
        staged_bytes = sum(remote.size for remote in controls_needing_stage)
        staged_temp = max((remote.size for remote in controls_needing_stage), default=0)
        required_combined_free_space(
            (
                (
                    hub.cache_root,
                    sum(remote.size for remote in uncached_controls),
                    len(uncached_controls),
                ),
                (
                    options.data_root,
                    staged_bytes + staged_temp,
                    len(controls_needing_stage) * 2 + (1 if controls_needing_stage else 0),
                ),
            )
        )

        reporter.emit("prepare", "preparing every category_index.json")
        staged_category_paths: dict[str, Path] = {}
        for category, remote in category_remotes:
            staged_category_paths[category] = stage_remote_control(
                hub, remote, revision, options.data_root, reporter
            )

        reporter.emit("prepare", "preparing IO-bad controls")
        staged_bad_manifest = stage_remote_control(
            hub, bad_manifest_remote, revision, options.data_root, reporter
        )
        staged_bad_info = stage_remote_control(
            hub, bad_info_remote, revision, options.data_root, reporter
        )
        bad_samples = load_bad_samples(staged_bad_manifest, staged_bad_info)

        catalogs = {
            category: parse_category_catalog(category, staged_category_paths[category])
            for category, _ in category_remotes
        }
        bad_records = check_bad_manifest_coverage(catalogs, bad_samples)

        # A process can die in the narrow window after the old sequence tree
        # was renamed to its backup but before the staged replacement became
        # public.  Recovery is destination-wide, not scoped to today's plan,
        # so even a bare or different-mode rerun repairs that sequence.
        reporter.emit("resume", "checking unfinished whole-tar publications")
        for archive in all_archives(catalogs):
            recover_archive_swap(options.data_root, archive)

        reporter.emit("prepare", "preparing mandatory per-sample JSON archive")
        sample_json_archive = download_verified(
            hub, sample_json_remote, revision, "sample JSON archive", reporter
        )
        fresh_full_extraction, normal_tree_digest = install_mandatory_sample_jsons(
            sample_json_archive,
            sample_json_remote,
            catalogs,
            bad_records,
            options.data_root,
            state,
            reporter,
        )
        begin_policy_transition(options.data_root, revision, options.include_bad)
        publish_category_paths, category_publish_root = prepare_category_indexes_for_publication(
            catalogs,
            bad_records,
            options.data_root,
            revision,
            options.include_bad,
        )

        # Mandatory controls become public only after every category index,
        # IO-bad control, and referenced per-sample JSON has validated.  A
        # control failure therefore cannot downgrade an older usable target.
        controls = [
            (publish_category_paths[category], remote.path)
            for category, remote in category_remotes
        ]
        controls.extend(
            (
                (staged_bad_manifest, BAD_MANIFEST_PATH),
                (staged_bad_info, BAD_MANIFEST_INFO_PATH),
            )
        )
        try:
            publish_control_transaction(controls, options.data_root)
        finally:
            cleanup_category_publish_staging(category_publish_root)
        for category, remote in category_remotes:
            catalogs[category].path = options.data_root / remote.path

        # Policy-specific file changes happen only after the complete control
        # set commits.  A control publication failure therefore leaves the
        # previous bad-file policy untouched.
        apply_bad_policy_to_files(
            sample_json_archive,
            bad_records,
            options.data_root,
            state,
            options.include_bad,
            fresh_full_extraction,
            reporter,
        )
        finalize_sample_json_checkpoint(
            catalogs,
            bad_records,
            options.data_root,
            state,
            options.include_bad,
            normal_tree_digest,
            reporter,
        )

        # Planning deliberately reparses only the installed local category
        # indexes, never the remote listing or staging representation.
        catalogs = {
            category: parse_category_catalog(
                category, options.data_root / category / "category_index.json"
            )
            for category, _ in category_remotes
        }
        save_state(options.data_root, state)
        finish_policy_transition(options.data_root)

        while True:
            selected_options = options
            if interaction is not None:
                selected_options = options_with_selection(options, interaction.choose(catalogs))
            plan = build_plan(selected_options, catalogs, bad_samples.source_relpaths)
            validate_selected_remote_archives(plan, remote_files)
            if plain_output:
                print_plan(
                    plan,
                    remote_files,
                    revision,
                    options.data_root,
                    hub.cache_root,
                    options.include_bad,
                )
            if not plan.archives:
                if plain_output:
                    print_bare_usage(options.data_root)
                return plan

            if interaction is None:
                break
            decision = interaction.confirm_plan(
                plan,
                remote_files,
                revision,
                options.data_root,
                hub.cache_root,
                options.include_bad,
            )
            if decision == "change":
                reporter.emit("plan", "returning to data-mode selection")
                continue
            if decision == "quit":
                reporter.emit(
                    "cancelled",
                    "VDB tar download skipped; required category indexes and sample JSON are ready",
                )
                return plan
            if decision != "download":
                raise DownloadError(f"Interactive plan returned an invalid decision: {decision!r}")
            break

        install_data_archives(
            plan,
            remote_files,
            revision,
            hub,
            bad_samples,
            bad_records,
            options.data_root,
            options.include_bad,
            state,
            reporter,
        )
        save_state(options.data_root, state)
        reporter.emit(
            "done",
            f"installed {len(plan.archives):,} whole tars and "
            f"{sum(plan.normal_samples_by_category.values()):,} normal indexed samples "
            f"at {options.data_root}",
        )
        return plan


def main(argv: Sequence[str] | None = None) -> int:
    try:
        options = parse_args(argv)
        if options.tui:
            try:
                from vfxdb_tui import launch_tui
            except ImportError as exc:
                raise DownloadError(
                    "The interactive interface requires Rich; install requirements-core.txt"
                ) from exc
            return launch_tui(options, run_download=run)
        run(options)
        return 0
    except KeyboardInterrupt:
        print("interrupted; rerun the same command to continue", file=sys.stderr)
        return 130
    except (DownloadError, OSError, tarfile.TarError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def metadata_main(argv: Sequence[str] | None = None) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
