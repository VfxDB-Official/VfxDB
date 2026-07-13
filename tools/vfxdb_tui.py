#!/usr/bin/env python3
"""Rich terminal interface for the VfxDB downloader.

The controller in this module only chooses downloader options and renders
events. Dataset membership, validation, caching, and extraction remain in
``vfxdb_downloader.py``.
"""

from __future__ import annotations

import re
import shutil
import sys
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme


VOLUME_SLATE_THEME = Theme(
    {
        "vfx.title": "bold",
        "vfx.text": "default",
        "vfx.muted": "#6B7472",
        "vfx.ok": "bold #397C70",
        "vfx.current": "bold #9B681F",
        "vfx.error": "bold #B34F3E",
        "vfx.base": "#4A5558",
    }
)

ARCHIVE_EVENT_RE = re.compile(r"^\[(\d+)/(\d+)\]\s+(archives/([^/]+)/[^\s]+)")
COUNT_EVENT_RE = re.compile(r"([\d,]+)/(\d[\d,]*)")
JSON_INSTALL_START_RE = re.compile(r"installing\s+([\d,]+)\s+per-sample JSON files")
JSON_INSTALL_PROGRESS_RE = re.compile(r"installed\s+([\d,]+)\s+archive members")


class UserCancelled(RuntimeError):
    """The user intentionally left the interactive flow."""


def format_bytes(value: int) -> str:
    size = float(max(0, value))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if size < 1024.0 or unit == "PiB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def terminal_supports_unicode(console: Console) -> bool:
    encoding = getattr(console.file, "encoding", None) or ""
    return "utf" in encoding.lower()


def available_space(path: Path) -> int | None:
    nearest = path
    while not nearest.exists() and nearest != nearest.parent:
        nearest = nearest.parent
    try:
        return shutil.disk_usage(nearest).free
    except OSError:
        return None


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def category_abbreviation(name: str) -> str:
    capitals = "".join(character for character in name if character.isupper())
    return (capitals[:2] or name[:2]).upper()


def category_target_result(plan: Any, name: str) -> str:
    target = plan.requested_max_samples
    if target is None:
        return ""
    actual = plan.normal_samples_by_category[name]
    if actual < target:
        return f"FULL · {target - actual:,} below target"
    if actual > target:
        return f"+{actual - target:,} (whole-tar rounding)"
    return "exact target"


class PromptPort:
    def __init__(self, console: Console, input_fn: Callable[[str], str] | None = None) -> None:
        self.console = console
        self.input_fn = input_fn or console.input

    def ask(self, label: str, *, default: str | None = None) -> str:
        prompt = Text()
        prompt.append("› ", style="vfx.current")
        prompt.append(label, style="vfx.text")
        if default is not None:
            prompt.append(f" [{default}]", style="vfx.muted")
        prompt.append(" ")
        self.console.print(prompt, end="")
        try:
            answer = self.input_fn("")
        except EOFError as exc:
            raise UserCancelled("input closed") from exc
        value = answer.strip()
        if value.lower() in {"q", "quit", "exit"}:
            raise UserCancelled("quit")
        return value if value else (default or "")

    def confirm(self, label: str, *, default: bool) -> bool:
        hint = "Y/n" if default else "y/N"
        while True:
            value = self.ask(label, default=hint).lower()
            if value == hint.lower():
                return default
            if value in {"y", "yes"}:
                return True
            if value in {"n", "no"}:
                return False
            self.console.print("Enter y or n.", style="vfx.error")


class VolumeSlateView:
    def __init__(self, console: Console) -> None:
        self.console = console
        self.unicode = terminal_supports_unicode(console)

    @property
    def width(self) -> int:
        return max(1, self.console.size.width)

    def header(self, subtitle: str, revision: str | None = None) -> Group:
        if self.width < 50:
            revision_text = (revision or "unresolved")[:8]
            short_subtitle = {
                "Volume Slate / required files": "SETUP",
                "Review exact local-index plan": "PLAN REVIEW",
                "Choose data mode": "DATA MODE",
            }.get(subtitle, subtitle.upper())
            status = Text(short_subtitle, style="vfx.muted", overflow="fold")
            revision_line = Text(f"REV {revision_text}", style="vfx.muted")
            return Group(
                Text("VfxDB DOWNLOAD", style="vfx.title"),
                status,
                revision_line,
                Rule(style="vfx.base"),
            )
        line = Table.grid(expand=True)
        line.add_column(ratio=1)
        line.add_column(justify="right")
        line.add_row(
            Text("VfxDB  DATASET DOWNLOADER", style="vfx.title"),
            Text(f"REV {(revision or 'unresolved')[:8]}", style="vfx.muted"),
        )
        return Group(line, Text(subtitle.upper(), style="vfx.muted"), Rule(style="vfx.base"))

    def key_values(self, details: Sequence[tuple[str, str]]) -> Any:
        if self.width < 60:
            lines: list[Text] = []
            for key, value in details:
                lines.append(Text(key.upper(), style="vfx.muted"))
                lines.append(Text(f"  {value}", style="vfx.text", overflow="fold"))
            return Group(*lines)
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="vfx.muted", no_wrap=True)
        grid.add_column(style="vfx.text", overflow="fold")
        for key, value in details:
            grid.add_row(Text(key, style="vfx.muted"), Text(value, style="vfx.text"))
        return grid

    def welcome(self, destination: Path, revision: str | None, cache_dir: Path | None) -> None:
        self.console.print(self.header("Volume Slate / required files", revision))
        self.console.print(
            self.key_values(
                (
                    ("Destination", str(destination)),
                    ("HF cache", str(cache_dir) if cache_dir else "Hugging Face default"),
                    ("Always prepared", "all category_index.json + all <Category>/index/*.json"),
                )
            )
        )
        self.console.print(Text("q quits safely at any prompt.", style="vfx.muted"))
        self.console.print()

    def mode_menu(self) -> None:
        rows = (
            ("1", "Required JSON only", "No VDB tar archives"),
            ("2", "Smoke", "First 2 complete tars in every category"),
            ("3", "Medium", "20% of total tar count, balanced by category"),
            ("4", "Full", "Every published tar"),
            ("5", "Percentage", "Percent of total tars, balanced by category"),
            ("6", "Selected categories", "Same usable-sample target per category"),
        )
        if self.width < 60:
            for key, title, behavior in rows:
                line = Text()
                line.append(f"{key}  ", style="vfx.current")
                line.append(title, style="vfx.title")
                self.console.print(line)
                self.console.print(Text(f"   {behavior}", style="vfx.muted", overflow="fold"))
            return
        table = Table(box=box.SIMPLE_HEAD, expand=self.width >= 80, show_edge=False)
        table.add_column("KEY", style="vfx.current", width=4)
        table.add_column("DOWNLOAD MODE", style="vfx.title")
        table.add_column("BEHAVIOR", style="vfx.muted")
        for row in rows:
            table.add_row(*row)
        self.console.print(table)

    def category_menu(self, categories: Sequence[str]) -> None:
        columns = 1 if self.width < 70 else 2 if self.width < 120 else 3
        table = Table.grid(expand=True, padding=(0, 2))
        for _ in range(columns):
            table.add_column()
        cells = [Text(f"{index:>2}  {name}", style="vfx.text") for index, name in enumerate(categories, 1)]
        for start in range(0, len(cells), columns):
            row = cells[start : start + columns]
            row.extend(Text("") for _ in range(columns - len(row)))
            table.add_row(*row)
        self.console.print(table)

    def plan_table(
        self,
        plan: Any,
        remote_files: dict[str, Any],
        revision: str,
        data_root: Path,
        cache_root: Path,
        include_bad: bool,
    ) -> Group:
        total_bytes = sum(remote_files[archive.remote_path].size for archive in plan.archives)
        extracted_bytes = sum(
            sample.size_bytes for archive in plan.archives for sample in archive.samples
        )
        total_samples = sum(plan.normal_samples_by_category.values())
        data_free = available_space(data_root)
        cache_free = available_space(cache_root)
        if plan.label == "preset smoke":
            rule = "first 2 complete tars in every category"
        elif plan.label == "preset medium":
            rule = (
                f"ceil({plan.all_archive_count:,} × 20%) = {len(plan.archives):,} complete tars, "
                "balanced in fixed category order"
            )
        elif plan.label == "preset full":
            rule = "every published tar"
        elif plan.requested_max_samples is not None:
            rule = (
                f"{plan.requested_max_samples:,} usable samples per selected category; "
                "the final complete tar rounds upward"
            )
        else:
            percentage = plan.label.removeprefix("all-category ")
            rule = (
                f"ceil({plan.all_archive_count:,} × {percentage}) = {len(plan.archives):,} "
                "complete tars, balanced by category"
            )
        details: list[tuple[str, str]] = [
            ("Mode", str(plan.label)),
            ("Selection rule", rule),
            ("Destination", str(data_root)),
            ("Tar archives", f"{len(plan.archives):,} / {plan.all_archive_count:,}"),
            ("Usable VDB samples", f"{total_samples:,}"),
            ("Network upper bound", f"{format_bytes(total_bytes)} (HF cache may reduce this)"),
            ("VDB install upper bound", format_bytes(extracted_bytes)),
            ("Destination free", format_bytes(data_free) if data_free is not None else "unknown"),
            ("HF cache", str(cache_root)),
            ("HF cache free", format_bytes(cache_free) if cache_free is not None else "unknown"),
        ]
        if include_bad:
            details.append(("Advanced override", "known IO-bad samples will also be retained"))
        summary = self.key_values(details)

        category = Table(box=box.SIMPLE_HEAD, expand=True, show_edge=False)
        category.add_column("CATEGORY", style="vfx.text")
        category.add_column("TARS", justify="right", style="vfx.current")
        category.add_column("USABLE VDB", justify="right", style="vfx.text")
        if plan.requested_max_samples is not None:
            category.add_column("TARGET RESULT", justify="right", style="vfx.muted")
        for name in sorted(plan.selected_by_category):
            count = plan.selected_by_category[name]
            if count:
                row = [
                    Text(name, style="vfx.text"),
                    Text(f"{count:,}/{plan.available_by_category[name]:,}", style="vfx.current"),
                    Text(f"{plan.normal_samples_by_category[name]:,}", style="vfx.text"),
                ]
                if plan.requested_max_samples is not None:
                    row.append(Text(category_target_result(plan, name), style="vfx.muted"))
                category.add_row(*row)
        parts: list[Any] = [self.header("Review exact local-index plan", revision), summary]
        if self.width >= 70:
            parts.extend((Text(""), category))
        else:
            compact = Text("CATEGORY PLAN\n", style="vfx.muted")
            for name in sorted(plan.selected_by_category):
                count = plan.selected_by_category[name]
                if count:
                    compact.append(
                        f"{name}: {count}/{plan.available_by_category[name]} tars · "
                        f"{plan.normal_samples_by_category[name]:,} usable VDB",
                        style="vfx.text",
                    )
                    if plan.requested_max_samples is not None:
                        compact.append(
                            f" · {category_target_result(plan, name)}",
                            style="vfx.muted",
                        )
                    compact.append("\n")
            parts.extend((Text(""), compact))
        return Group(*parts)

    def category_tar_strip(self, planned: dict[str, int], completed: dict[str, int]) -> Group:
        if not planned:
            return Group(Text(""))
        empty, full = (".", "#") if not self.unicode else ("░", "█")
        cells: list[Text] = []
        for name in sorted(planned):
            total = max(1, planned[name])
            done = min(total, completed.get(name, 0))
            ratio = min(1.0, done / total)
            filled = min(4, int(ratio * 4))
            style = "vfx.ok" if ratio >= 1 else "vfx.current" if ratio > 0 else "vfx.muted"
            cell = Text()
            cell.append(f"{category_abbreviation(name)} {done}/{total} ", style="vfx.text")
            cell.append(full * filled + empty * (4 - filled), style=style)
            cells.append(cell)
        per_line = max(1, (self.width - 2) // 18)
        lines: list[Text] = [Text("CATEGORY TARS", style="vfx.muted")]
        for start in range(0, len(cells), per_line):
            line = Text("  ")
            for index, cell in enumerate(cells[start : start + per_line]):
                if index:
                    line.append("   ")
                line.append_text(cell)
            lines.append(line)
        return Group(*lines)

    def result(self, title: str, details: Sequence[tuple[str, str]], *, error: bool = False) -> None:
        style = "vfx.error" if error else "vfx.ok"
        self.console.print()
        self.console.print(Text(title, style=f"bold {style}"))
        self.console.print(self.key_values(details))


class RichReporter:
    """Translate downloader events and Hugging Face byte updates into one Live view."""

    suppress_hf_progress = True

    def __init__(self, console: Console, view: VolumeSlateView) -> None:
        self.console = console
        self.view = view
        self.live: Live | None = None
        self.static_mode = bool(console.is_dumb_terminal)
        self.stage = "prepare"
        self.message = "Waiting to prepare required files"
        self.recent: deque[tuple[str, str]] = deque(maxlen=6)
        self.started = 0.0
        self._timer_started = False
        self.plan: Any | None = None
        self.revision: str | None = None
        self.data_root: Path | None = None
        self.tar_current = 0
        self.tar_total = 0
        self.current_remote: str | None = None
        self.last_event_was_archive = False
        self.planned_by_category: dict[str, int] = {}
        self.completed_by_category: dict[str, int] = {}
        self.transfer_current = 0
        self.transfer_total = 0
        self.transfer_name = ""
        self.transfer_started = 0.0
        self.transfer_initial = 0
        self.transfer_rate = 0.0
        self.work_current = 0
        self.work_total = 0
        self.work_name = ""
        self._completed_paths: set[str] = set()
        self._completion_kind: dict[str, str] = {}
        self.reused_tar_count = 0
        self.installed_tar_count = 0
        self.required_json_ready = False
        self._last_transfer_refresh = 0.0
        self._static_transfer_bucket = -1

    def start(self) -> None:
        if not self._timer_started:
            self.started = time.monotonic()
            self._timer_started = True
        if self.static_mode:
            return
        if self.live is None:
            self.live = Live(
                self.render(),
                console=self.console,
                refresh_per_second=8,
                transient=False,
            )
            self.live.start()

    def stop(self) -> None:
        if self.live is not None:
            self.live.stop()
            self.live = None

    @contextmanager
    def paused(self, *, restart: bool = True) -> Iterator[None]:
        running = self.live is not None
        if running:
            self.stop()
        try:
            yield
        finally:
            if running and restart:
                self.start()

    def refresh(self, *, force: bool = True) -> None:
        if self.live is not None:
            self.live.update(self.render(), refresh=force)
        elif self.static_mode and self._timer_started and force:
            self.console.print(self.static_status())

    def configure_plan(self, plan: Any, revision: str, data_root: Path) -> None:
        self.plan = plan
        self.revision = revision
        self.data_root = data_root
        self.tar_total = len(plan.archives)
        self.tar_current = 0
        self._completed_paths = set()
        self._completion_kind = {}
        self.reused_tar_count = 0
        self.installed_tar_count = 0
        self.transfer_current = 0
        self.transfer_total = 0
        self.transfer_name = ""
        self.transfer_started = 0.0
        self.transfer_initial = 0
        self.transfer_rate = 0.0
        self.work_current = 0
        self.work_total = 0
        self.work_name = ""
        self.last_event_was_archive = False
        self._static_transfer_bucket = -1
        self.planned_by_category = {
            name: count for name, count in plan.selected_by_category.items() if count
        }
        self.completed_by_category = {name: 0 for name in self.planned_by_category}
        self.stage = "plan"
        self.message = "Exact plan generated from local category indexes"
        if self.live is not None:
            self.refresh()

    def emit(self, stage: str, message: str) -> None:
        if self.transfer_total and stage != "retry":
            self.transfer_current = 0
            self.transfer_total = 0
            self.transfer_name = ""
            self.transfer_rate = 0.0
        self.stage = stage
        archive = ARCHIVE_EVENT_RE.match(message)
        self.last_event_was_archive = archive is not None
        if stage == "resolve":
            revision = re.search(r"\b[0-9a-f]{40}\b", message)
            if revision:
                self.revision = revision.group(0)
        if archive:
            index, total, remote_path, category = archive.groups()
            self.tar_total = int(total)
            self.current_remote = remote_path
            if stage == "download":
                self.transfer_current = 0
                self.transfer_total = 0
                self.transfer_name = remote_path
                self.transfer_started = 0.0
                self.transfer_rate = 0.0
            if stage in {"cache", "installed"}:
                if remote_path not in self._completed_paths:
                    self._completed_paths.add(remote_path)
                    if category in self.completed_by_category:
                        self.completed_by_category[category] += 1
                    kind = "reused" if stage == "cache" else "installed"
                    self._completion_kind[remote_path] = kind
                    if kind == "reused":
                        self.reused_tar_count += 1
                    else:
                        self.installed_tar_count += 1
                self.tar_current = len(self._completed_paths)
            else:
                self.tar_current = len(self._completed_paths)
        count = COUNT_EVENT_RE.search(message)
        json_start = JSON_INSTALL_START_RE.search(message)
        json_progress = JSON_INSTALL_PROGRESS_RE.search(message)
        if json_start:
            self.work_name = "install JSON"
            self.work_current = 0
            self.work_total = int(json_start.group(1).replace(",", ""))
        elif json_progress:
            self.work_name = "install JSON"
            self.work_current = int(json_progress.group(1).replace(",", ""))
        elif count and not archive and "json" in message.lower():
            self.work_name = stage
            self.work_current = int(count.group(1).replace(",", ""))
            self.work_total = int(count.group(2).replace(",", ""))
        elif stage not in {"json", "retry"}:
            self.work_current = 0
            self.work_total = 0
            self.work_name = ""
        self.message = self.friendly_message(stage, message, archive)
        if stage not in {"verify"} or archive:
            self.recent.append((stage, self.message))
        self.refresh()

    def transfer(self, current: int, total: int | None, description: str) -> None:
        now = time.monotonic()
        description = description or self.current_remote or "current file"
        normalized_total = max(0, total or 0)
        if (
            description != self.transfer_name
            or normalized_total != self.transfer_total
            or current < self.transfer_current
            or not self.transfer_started
        ):
            self.transfer_started = now
            self.transfer_initial = max(0, current)
            self.transfer_rate = 0.0
        self.transfer_current = max(0, current)
        self.transfer_total = normalized_total
        self.transfer_name = description
        elapsed = now - self.transfer_started
        if elapsed > 0:
            self.transfer_rate = max(0.0, (self.transfer_current - self.transfer_initial) / elapsed)
        complete = bool(self.transfer_total and self.transfer_current >= self.transfer_total)
        static_update = False
        if self.static_mode and self.transfer_total:
            bucket = min(10, int(self.transfer_current * 10 / self.transfer_total))
            if bucket != self._static_transfer_bucket:
                self._static_transfer_bucket = bucket
                static_update = True
        if complete or static_update or now - self._last_transfer_refresh >= 0.125:
            self._last_transfer_refresh = now
            self.refresh(force=complete or static_update)

    def static_status(self) -> Text:
        line = Text()
        line.append(f"[{self.stage.upper()}] ", style="vfx.muted")
        has_metric = False
        if self.transfer_total:
            percent = min(100, int(self.transfer_current * 100 / self.transfer_total))
            line.append(
                f"FILE {percent}% · {format_bytes(self.transfer_current)} / "
                f"{format_bytes(self.transfer_total)}",
                style="vfx.current",
            )
            has_metric = True
        elif self.work_total:
            line.append(
                f"JSON {self.work_current:,}/{self.work_total:,}",
                style="vfx.current",
            )
            has_metric = True
        elif self.tar_total:
            line.append(f"TARS {self.tar_current:,}/{self.tar_total:,}", style="vfx.current")
            has_metric = True
        else:
            line.append(self.message, style="vfx.text")
        if has_metric:
            line.append(" · ", style="vfx.muted")
            line.append(self.message, style="vfx.text")
        return line

    def friendly_message(
        self,
        stage: str,
        message: str,
        archive_match: re.Match[str] | None,
    ) -> str:
        if archive_match:
            _index, _total, remote_path, category = archive_match.groups()
            sequence = Path(remote_path).stem
            action = {
                "download": "Downloading",
                "cache": "Using installed",
                "verify": "Checking",
                "extract": "Installing",
                "installed": "Installed",
            }.get(stage, stage.capitalize())
            return f"{action} {category}/{sequence}"
        if stage == "resolve":
            return "Dataset revision pinned"
        if stage == "prepare":
            if "category_index" in message:
                return "Preparing category indexes"
            if "per-sample JSON" in message:
                return "Preparing per-sample JSON"
            return "Preparing published dataset controls"
        if stage == "resume":
            return "Checking unfinished local installations"
        if stage == "verify":
            if "json" in message.lower() and (count := COUNT_EVENT_RE.search(message)):
                return f"Checking per-sample JSON {count.group(0)}"
            return "Checking required files"
        if stage == "json":
            if json_progress := JSON_INSTALL_PROGRESS_RE.search(message):
                suffix = f"/{self.work_total:,}" if self.work_total else ""
                return f"Installing per-sample JSON {json_progress.group(1)}{suffix}"
            if count := COUNT_EVENT_RE.search(message):
                return f"Installing per-sample JSON {count.group(0)}"
            return "Installing per-sample JSON"
        if stage == "cache":
            return "Using a verified cached required file"
        if stage == "repair":
            return "Repairing an incomplete local required file"
        if stage == "retry":
            return f"Retrying after a temporary error: {message}"
        if stage == "plan":
            return "Returning to data-mode selection"
        if stage == "cancelled":
            return "Required JSON is ready; VDB tar download skipped"
        if stage == "done":
            return "Selected VDB data is ready"
        return message

    def make_tqdm_class(self) -> type:
        reporter = self

        class ReporterTqdm:
            def __init__(self, *, total=None, initial=0, desc="", **_kwargs):
                self.total = total
                self.n = initial
                self.desc = desc
                reporter.transfer(self.n, self.total, self.desc)

            def __enter__(self):
                return self

            def __exit__(self, _exc_type, _exc, _tb):
                self.close()

            def update(self, amount=1):
                self.n += amount
                reporter.transfer(self.n, self.total, self.desc)
                return True

            def close(self):
                reporter.transfer(self.n, self.total, self.desc)

        return ReporterTqdm

    def phase_rail(self) -> Text:
        if self.plan is None:
            stages = (
                ("REQUIRED JSON", {"resolve", "prepare", "resume", "json", "repair", "verify", "cache"}),
                ("PLAN", {"plan"}),
            )
            current_index = 1 if self.stage == "plan" else 0
            prefix = ""
        elif self.stage in {"plan", "cancelled"}:
            status = "PLAN REVIEW · NO TAR STARTED" if self.stage == "plan" else "NO TAR STARTED"
            return Text(status, style="vfx.muted")
        else:
            stages = (
                ("GET", {"download", "cache", "retry"}),
                ("CHECK", {"verify"}),
                ("INSTALL", {"extract", "installed"}),
            )
            current_index = 0
            for index, (_label, stage_names) in enumerate(stages):
                if self.stage in stage_names:
                    current_index = index
                    break
            if self.stage == "done":
                current_index = len(stages)
            completed_event = self.stage == "installed" or (
                self.stage == "cache" and self.last_event_was_archive
            )
            current_tar = self.tar_current if completed_event else self.tar_current + 1
            prefix = f"CURRENT TAR {min(current_tar, self.tar_total):,}/{self.tar_total:,}   "
        labels = [label for label, _ in stages]
        rail = Text()
        rail.append(prefix, style="vfx.muted")
        for index, label in enumerate(labels):
            if index < current_index:
                marker, style = ("✓" if self.view.unicode else "+"), "vfx.ok"
            elif index == current_index and current_index < len(stages):
                marker, style = ("●" if self.view.unicode else ">"), "vfx.current"
            else:
                marker, style = ("○" if self.view.unicode else "."), "vfx.muted"
            rail.append(f"{marker} {label}", style=style)
            if index != len(labels) - 1:
                rail.append("   " if self.view.width >= 70 else " ", style="vfx.base")
        return rail

    def transfer_status(self) -> Text:
        status = Text()
        status.append("CURRENT FILE  ", style="vfx.muted")
        status.append(
            f"{format_bytes(self.transfer_current)} / {format_bytes(self.transfer_total)}",
            style="vfx.text",
        )
        if self.transfer_rate > 0:
            status.append(f"  {format_bytes(int(self.transfer_rate))}/s", style="vfx.muted")
            remaining = max(0, self.transfer_total - self.transfer_current)
            status.append(f"  ETA {format_duration(remaining / self.transfer_rate)}", style="vfx.muted")
        return status

    def compact_progress(self) -> Group:
        lines: list[Text] = []
        if self.tar_total:
            lines.append(Text(f"TARS  {self.tar_current:,}/{self.tar_total:,}", style="vfx.current"))
        if self.transfer_total:
            percent = min(100, int(self.transfer_current * 100 / self.transfer_total))
            lines.append(Text(f"FILE  {percent}%", style="vfx.current"))
            lines.append(
                Text(
                    f"{format_bytes(self.transfer_current)} / {format_bytes(self.transfer_total)}",
                    style="vfx.text",
                    overflow="fold",
                )
            )
        elif self.work_total:
            percent = min(100, int(self.work_current * 100 / self.work_total))
            lines.append(
                Text(
                    f"{self.work_name.upper()}  {self.work_current:,}/{self.work_total:,}  {percent}%",
                    style="vfx.current",
                    overflow="fold",
                )
            )
        elif not self.tar_total:
            lines.append(Text("WORKING", style="vfx.current"))
        return Group(*lines)

    def render(self) -> Group:
        revision = self.revision or "unresolved"
        parts: list[Any] = [self.view.header(self.stage, revision)]
        if self.view.width >= 60:
            parts.append(self.phase_rail())
        if self.plan is not None and self.view.width >= 100:
            parts.append(self.view.category_tar_strip(self.planned_by_category, self.completed_by_category))
        if self.view.width < 60:
            parts.append(self.compact_progress())
        else:
            progress = Progress(
                TextColumn("{task.description}", style="vfx.current"),
                BarColumn(bar_width=None, complete_style="#397C70", finished_style="#397C70"),
                TaskProgressColumn(),
                expand=True,
            )
            if self.tar_total:
                progress.add_task("TAR ARCHIVES", total=self.tar_total, completed=self.tar_current)
            if self.transfer_total:
                progress.add_task(
                    "CURRENT FILE",
                    total=self.transfer_total,
                    completed=self.transfer_current,
                )
            elif self.work_total:
                progress.add_task(
                    f"{self.work_name.upper()} ITEMS",
                    total=self.work_total,
                    completed=self.work_current,
                )
            elif not self.tar_total:
                progress.add_task("REQUIRED FILES", total=None)
            parts.append(progress)
        parts.append(Text(self.message, style="vfx.text", overflow="fold"))
        if self.transfer_total and self.view.width >= 60:
            parts.append(self.transfer_status())
        if self.recent and self.view.width >= 90:
            recent = Table.grid(padding=(0, 1))
            recent.add_column(style="vfx.muted", width=10)
            recent.add_column(style="vfx.text", overflow="ellipsis")
            for stage, message in self.recent:
                recent.add_row(
                    Text(stage.upper(), style="vfx.muted"),
                    Text(message, style="vfx.text", overflow="ellipsis"),
                )
            parts.extend((Rule("RECENT", style="vfx.base"), recent))
        footer = (
            "Ctrl-C safe stop; rerun to continue."
            if self.view.width < 60
            else "Ctrl-C stops safely; rerun the same selection to continue."
        )
        parts.append(Text(footer, style="vfx.muted", overflow="fold"))
        return Group(*parts)


def parse_category_selection(raw: str, categories: Sequence[str]) -> tuple[str, ...]:
    lookup = {name.lower(): name for name in categories}
    if raw.strip().lower() == "all":
        return tuple(categories)
    selected: list[str] = []
    for token in (part.strip() for part in raw.split(",")):
        if not token:
            continue
        if "-" in token and all(part.strip().isdigit() for part in token.split("-", 1)):
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                start, end = end, start
            indexes = range(start, end + 1)
            names = [categories[index - 1] for index in indexes if 1 <= index <= len(categories)]
            if len(names) != abs(end - start) + 1:
                raise ValueError("category number is out of range")
        elif token.isdigit():
            index = int(token)
            if not 1 <= index <= len(categories):
                raise ValueError("category number is out of range")
            names = [categories[index - 1]]
        else:
            name = lookup.get(token.lower())
            if name is None:
                raise ValueError(f"unknown category: {token}")
            names = [name]
        for name in names:
            if name not in selected:
                selected.append(name)
    if not selected:
        raise ValueError("select at least one category")
    return tuple(selected)


class RichInteraction:
    def __init__(self, console: Console, prompt: PromptPort, reporter: RichReporter) -> None:
        self.console = console
        self.prompt = prompt
        self.reporter = reporter
        self.view = reporter.view
        self.cancelled = False

    def choose(self, catalogs: dict[str, Any]):
        from vfxdb_downloader import Selection

        self.reporter.required_json_ready = True
        categories = tuple(sorted(catalogs))
        with self.reporter.paused(restart=False):
            self.console.print()
            self.console.print(self.view.header("Choose data mode", self.reporter.revision))
            self.view.mode_menu()
            self.console.print(Text("Choose q at any prompt to leave required JSON in place.", style="vfx.muted"))
            while True:
                choice = self.prompt.ask("Mode", default="2")
                if choice in {"1", "2", "3", "4", "5", "6"}:
                    break
                self.console.print("Choose a number from 1 to 6.", style="vfx.error")
            if choice == "1":
                return Selection()
            if choice in {"2", "3", "4"}:
                return Selection(preset={"2": "smoke", "3": "medium", "4": "full"}[choice])
            if choice == "5":
                while True:
                    raw = self.prompt.ask("Percentage across all categories")
                    try:
                        percentage = Decimal(raw)
                    except InvalidOperation:
                        percentage = Decimal(0)
                    if percentage.is_finite() and Decimal(0) < percentage <= Decimal(100):
                        return Selection(percentage=percentage)
                    self.console.print("Percentage must be greater than 0 and at most 100.", style="vfx.error")

            self.view.category_menu(categories)
            while True:
                raw = self.prompt.ask("Categories (numbers, ranges, names, or all)")
                try:
                    selected = parse_category_selection(raw, categories)
                    break
                except ValueError as exc:
                    self.console.print(Text(str(exc), style="vfx.error"))
            while True:
                raw_max = self.prompt.ask("Usable-sample target per selected category")
                try:
                    maximum = int(raw_max)
                except ValueError:
                    maximum = 0
                if maximum > 0:
                    return Selection(categories=selected, max_samples=maximum)
                self.console.print("Maximum samples must be a positive integer.", style="vfx.error")

    def confirm_plan(
        self,
        plan: Any,
        remote_files: dict[str, Any],
        revision: str,
        data_root: Path,
        cache_root: Path,
        include_bad: bool,
    ) -> str:
        # Keep the exact review as one stable screen. Stopping Live before
        # configuring avoids printing a redundant intermediate plan frame.
        self.reporter.stop()
        self.reporter.configure_plan(plan, revision, data_root)
        self.console.print()
        self.console.print(
            self.view.plan_table(
                plan,
                remote_files,
                revision,
                data_root,
                cache_root,
                include_bad,
            )
        )
        while True:
            try:
                if plan.label == "preset full":
                    answer = self.prompt.ask(
                        "Type FULL to download, b to change selection, or q to quit"
                    )
                    if answer == "FULL":
                        if self.reporter._timer_started:
                            self.reporter.start()
                        return "download"
                else:
                    answer = self.prompt.ask(
                        "Action: d download, b change selection, q quit",
                        default="b",
                    ).lower()
                    if answer in {"d", "download", "y", "yes"}:
                        if self.reporter._timer_started:
                            self.reporter.start()
                        return "download"
                if answer.lower() in {"b", "back", "change", "n", "no"}:
                    return "change"
                expected = "FULL, b, or q" if plan.label == "preset full" else "d, b, or q"
                self.console.print(f"Choose {expected} exactly.", style="vfx.error")
            except UserCancelled:
                self.cancelled = True
                return "quit"


def _is_tty(value: bool | Callable[[], bool] | None, stream: Any) -> bool:
    if value is None:
        return bool(stream.isatty())
    return bool(value() if callable(value) else value)


def launch_tui(
    base_options: Any,
    *,
    run_download: Callable[..., Any],
    console: Console | None = None,
    input_fn: Callable[[str], str] | None = None,
    stdin_isatty: bool | Callable[[], bool] | None = None,
    stdout_isatty: bool | Callable[[], bool] | None = None,
) -> int:
    """Run the interactive controller; return a conventional process exit code."""
    console = console or Console(theme=VOLUME_SLATE_THEME, highlight=False)
    if not (_is_tty(stdin_isatty, sys.stdin) and _is_tty(stdout_isatty, sys.stdout)):
        console.print(
            "TUI requires an interactive stdin and stdout. Use the normal CLI in pipes or jobs.",
            style="vfx.error",
        )
        return 2

    view = VolumeSlateView(console)
    prompt = PromptPort(console, input_fn)
    reporter = RichReporter(console, view)
    interaction = RichInteraction(console, prompt, reporter)
    configured = base_options
    try:
        view.welcome(base_options.data_root, base_options.revision, base_options.cache_dir)
        destination = Path(
            prompt.ask("Destination", default=str(base_options.data_root))
        ).expanduser().absolute()
        revision = base_options.revision
        cache_dir = base_options.cache_dir
        include_bad = bool(base_options.include_bad)
        has_custom_advanced = bool(revision or cache_dir or include_bad)
        if prompt.confirm("Open advanced settings?", default=has_custom_advanced):
            revision_text = prompt.ask("Revision", default=revision or "main")
            revision = None if revision_text == "main" else revision_text
            cache_default = str(cache_dir) if cache_dir else "default"
            cache_text = prompt.ask(
                "HF cache directory ('default' uses Hugging Face default)",
                default=cache_default,
            )
            cache_dir = None if cache_text.lower() == "default" else Path(cache_text).expanduser().absolute()
            include_bad = prompt.confirm(
                "Include known IO-bad samples? (advanced diagnostic use only)",
                default=include_bad,
            )
        configured = replace(
            base_options,
            data_root=destination,
            revision=revision,
            cache_dir=cache_dir,
            include_bad=include_bad,
        )
        if not prompt.confirm(
            "Prepare every category index and per-sample JSON now?",
            default=True,
        ):
            raise UserCancelled("preparation cancelled")

        reporter.start()
        plan = run_download(
            configured,
            reporter=reporter,
            interaction=interaction,
            plain_output=False,
        )
        reporter.stop()
        elapsed = time.monotonic() - reporter.started
        if interaction.cancelled:
            view.result(
                "VDB DOWNLOAD SKIPPED",
                (
                    ("Required JSON", "ready"),
                    ("VDB tar archives", "none downloaded after plan review"),
                    ("Continue", "rerun --tui and choose the same destination"),
                ),
            )
            return 0
        if plan.archives:
            completion_details: list[tuple[str, str]] = [
                ("Destination", str(configured.data_root)),
                ("Installed this run", f"{reporter.installed_tar_count:,}"),
                ("Reused local", f"{reporter.reused_tar_count:,}"),
                ("Usable VDB samples", f"{sum(plan.normal_samples_by_category.values()):,}"),
            ]
            if plan.requested_max_samples is not None:
                target_results = "; ".join(
                    f"{name}: {plan.normal_samples_by_category[name]:,} "
                    f"({category_target_result(plan, name)})"
                    for name in sorted(plan.selected_by_category)
                    if plan.selected_by_category[name]
                )
                completion_details.append(("Per-category target", target_results))
            completion_details.append(("Elapsed", f"{elapsed:.1f} s"))
            view.result(
                f"SELECTION READY — {len(plan.archives):,} TAR ARCHIVES VERIFIED",
                completion_details,
            )
        else:
            view.result(
                "REQUIRED JSON READY — NO VDB TAR DOWNLOADED",
                (
                    ("Destination", str(configured.data_root)),
                    ("Prepared", "all category indexes and per-sample JSON"),
                ),
            )
        return 0
    except UserCancelled:
        reporter.stop()
        details: list[tuple[str, str]] = [
            ("VDB tar archives", "no new tar download was started"),
        ]
        if reporter.required_json_ready:
            details.insert(0, ("Required JSON", "ready and retained"))
        else:
            details.insert(0, ("Required JSON", "preparation was not started"))
        view.result("CANCELLED", details)
        return 0
    except KeyboardInterrupt:
        reporter.stop()
        view.result(
            "STOPPED SAFELY",
            (
                ("Destination", str(configured.data_root)),
                ("Continue", "rerun the same TUI selection; completed work is retained"),
            ),
            error=True,
        )
        return 130
    except Exception as exc:
        reporter.stop()
        current = reporter.current_remote or reporter.message
        view.result(
            "DOWNLOAD STOPPED",
            (
                ("Reason", str(exc)),
                ("Stage", reporter.stage),
                ("Current", current),
                ("Destination", str(configured.data_root)),
                ("Continue", "fix the reason, then rerun the same selection"),
            ),
            error=True,
        )
        return 1
