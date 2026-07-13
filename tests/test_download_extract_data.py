from __future__ import annotations

import contextlib
import copy
import errno
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "vfxdb_downloader.py"
SPEC = importlib.util.spec_from_file_location("vfxdb_downloader_under_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
download = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = download
SPEC.loader.exec_module(download)


REVISION = "a" * 40


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def add_bytes(tf: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tf.addfile(info, io.BytesIO(payload))


@dataclass(frozen=True)
class FixtureSample:
    category: str
    folder: str
    key: str
    payload: bytes
    meta_relpath: str
    source_relpath: str
    row: dict[str, object]


def make_fixture_sample(
    category: str,
    folder: str,
    ordinal: int,
    *,
    seq: int,
) -> FixtureSample:
    key = f"{category.lower()}_{folder}_{ordinal}".replace("-", "_")
    filename = f"{key}.vdb"
    source_relpath = f"{category}/{folder}/{filename}"
    meta_relpath = f"{category}/index/{key}.json"
    payload = f"payload:{source_relpath}".encode("utf-8")
    row: dict[str, object] = {
        "id": key,
        "base_id": key,
        "seq": seq,
        "folder": folder,
        "vdb_path": f"{folder}/{filename}",
        "meta_path": f"index/{key}.json",
        "size_bytes": len(payload),
        "mtime": 1,
        "future_extension": {"keep": [1, {"verbatim": True}]},
    }
    return FixtureSample(
        category,
        folder,
        key,
        payload,
        meta_relpath,
        source_relpath,
        row,
    )


def write_sequence_tar(
    path: Path,
    category: str,
    folder: str,
    samples: list[FixtureSample],
    *,
    sha_overrides: dict[str, str] | None = None,
    member_type_overrides: dict[str, bytes] | None = None,
    extras: list[tuple[str, bytes, bytes]] | None = None,
) -> None:
    sha_overrides = sha_overrides or {}
    member_type_overrides = member_type_overrides or {}
    manifest = {
        "package_format": "webdataset",
        "category": category,
        "sequence": folder,
        "num_samples": len(samples),
        "samples": [
            {
                "key": sample.key,
                "source_relpath": sample.source_relpath,
                "source_filename": Path(sample.source_relpath).name,
                "sha256": sha_overrides.get(
                    sample.source_relpath,
                    hashlib.sha256(sample.payload).hexdigest(),
                ),
            }
            for sample in samples
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tf:
        add_bytes(tf, "_sequence_manifest.json", json.dumps(manifest).encode("utf-8"))
        for sample in samples:
            name = f"{sample.key}.vdb"
            member_type = member_type_overrides.get(name, tarfile.REGTYPE)
            if member_type == tarfile.REGTYPE:
                add_bytes(tf, name, sample.payload)
            else:
                info = tarfile.TarInfo(name)
                info.type = member_type
                if member_type in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                    info.linkname = "../outside"
                tf.addfile(info)
        for name, member_type, payload in extras or []:
            if member_type == tarfile.REGTYPE:
                add_bytes(tf, name, payload)
            else:
                info = tarfile.TarInfo(name)
                info.type = member_type
                tf.addfile(info)


def write_sample_json_archive(
    path: Path,
    samples: list[FixtureSample],
    *,
    omit: set[str] | None = None,
) -> None:
    omit = omit or set()
    zstd = shutil.which("zstd")
    if zstd is None:
        raise RuntimeError("Tests require zstd, as does the downloader")
    tar_path = path.parent / "sample-jsons.tar"
    members = [sample for sample in samples if sample.meta_relpath not in omit]
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tf:
        for sample in members:
            payload = {
                "id": sample.row["id"],
                "bbox_min": [0, 0, 0],
                "bbox_max": [1, 1, 1],
                "voxel_count": 1,
            }
            add_bytes(tf, sample.meta_relpath, json.dumps(payload).encode("utf-8"))
        manifest = {
            "repo_id": download.REPO_ID,
            "file_count": len(members),
            "categories": sorted({sample.category for sample in members}),
        }
        add_bytes(tf, "meta_manifest.json", json.dumps(manifest).encode("utf-8"))
    subprocess.run(
        [zstd, "-q", "-f", str(tar_path), "-o", str(path)],
        check=True,
    )
    tar_path.unlink()


def write_bad_controls(root: Path, bad_samples: list[FixtureSample]) -> None:
    records = [
        {
            "category": bad.category,
            "sequence": bad.folder,
            "source_relpath": bad.source_relpath,
            "archive_path": f"archives/{bad.category}/{bad.folder}.tar",
            "sample_key": bad.key,
            "member_path": f"{bad.key}.vdb",
            "reason": "exception",
            "return_code": 2,
        }
        for bad in bad_samples
    ]
    payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    manifest = root / download.BAD_MANIFEST_PATH
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(payload, encoding="utf-8")
    write_json(
        root / download.BAD_MANIFEST_INFO_PATH,
        {
            "schema_version": 1,
            "record_count": len(records),
            "manifest_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        },
    )


class FixtureHub:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.remote_files: dict[str, download.RemoteFile] = {}
        self.resolve_calls: list[str] = []
        self.download_calls: list[tuple[str, str, bool]] = []
        self.before_archive: object | None = None
        self.refresh_remote_files()

    @property
    def cache_root(self) -> Path:
        return self.root

    def refresh_remote_files(self) -> None:
        self.remote_files = {}
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(self.root).as_posix()
            self.remote_files[relative] = download.RemoteFile(
                relative,
                path.stat().st_size,
                download.sha256_file(path),
            )

    def resolve_snapshot(self, requested_revision: str):
        self.resolve_calls.append(requested_revision)
        return REVISION, self.remote_files

    def download(self, remote_path: str, revision: str, *, force: bool = False) -> Path:
        self.download_calls.append((remote_path, revision, force))
        if remote_path.startswith("archives/") and self.before_archive is not None:
            callback = self.before_archive
            callback()  # type: ignore[operator]
        return self.root / remote_path

    def cached_path(self, remote_path: str, revision: str) -> Path | None:
        path = self.root / remote_path
        return path if path.is_file() else None


class RepositoryFixture:
    ALPHA_FOLDERS = ["2", "10", "1", "3", "4", "5", "6", "7", "8", "9", "11", "12"]
    BETA_FOLDERS = [f"b{i}" for i in range(8)]
    FOG_FOLDERS = [f"fog{i:02d}" for i in range(11)]

    def __init__(self, root: Path) -> None:
        self.root = root / "remote"
        self.root.mkdir(parents=True)
        self.samples_by_archive: dict[str, list[FixtureSample]] = {}
        self.samples: list[FixtureSample] = []
        self.original_indexes: dict[str, dict[str, object]] = {}
        self._make_category("Alpha", self.ALPHA_FOLDERS, self._alpha_counts)
        self._make_category("Beta", self.BETA_FOLDERS, lambda _: 1)
        self._make_category("EnvironmentalFog", self.FOG_FOLDERS, lambda _: 2, fog=True)
        self.bad = self.samples_by_archive["archives/Alpha/2.tar"][0]
        self.bad_only = self.samples_by_archive["archives/Alpha/10.tar"][0]
        self.bads = (self.bad, self.bad_only)
        write_bad_controls(self.root, list(self.bads))
        write_sample_json_archive(self.root / download.SAMPLE_JSON_ARCHIVE_PATH, self.samples)
        self.hub = FixtureHub(self.root)

    @staticmethod
    def _alpha_counts(index: int) -> int:
        # The second tar is deliberately IO-bad-only.  It must still be
        # selected in fixed order, but contributes zero to --max-samples.
        return {0: 3, 1: 1, 2: 3}.get(index, 1)

    def _make_category(
        self,
        category: str,
        folders: list[str],
        count_for_index,
        *,
        fog: bool = False,
    ) -> None:
        category_samples: list[FixtureSample] = []
        for archive_index, folder in enumerate(folders):
            archive_samples = [
                make_fixture_sample(
                    category,
                    folder,
                    sample_index,
                    seq=0 if fog else sample_index,
                )
                for sample_index in range(count_for_index(archive_index))
            ]
            category_samples.extend(archive_samples)
            remote_path = f"archives/{category}/{folder}.tar"
            self.samples_by_archive[remote_path] = archive_samples
            write_sequence_tar(
                self.root / remote_path,
                category,
                folder,
                archive_samples,
            )
        value: dict[str, object] = {
            "category": category,
            "num_samples": len(category_samples),
            "description": {"preserve": [category, 7]},
            "samples": [copy.deepcopy(sample.row) for sample in category_samples],
        }
        self.original_indexes[category] = copy.deepcopy(value)
        write_json(self.root / category / "category_index.json", value)
        self.samples.extend(category_samples)

    def catalogs(self) -> dict[str, download.CategoryCatalog]:
        return {
            category: download.parse_category_catalog(
                category, self.root / category / "category_index.json"
            )
            for category in sorted(self.original_indexes)
        }

    def bad_samples(self) -> download.BadSamples:
        return download.load_bad_samples(
            self.root / download.BAD_MANIFEST_PATH,
            self.root / download.BAD_MANIFEST_INFO_PATH,
        )

    def replace_sample_json_archive(self, *, omit: set[str]) -> None:
        write_sample_json_archive(
            self.root / download.SAMPLE_JSON_ARCHIVE_PATH,
            self.samples,
            omit=omit,
        )
        self.hub.refresh_remote_files()


def options(
    root: Path,
    *,
    preset: str | None = None,
    percentage: Decimal | None = None,
    categories: tuple[str, ...] = (),
    max_samples: int | None = None,
    include_bad: bool = False,
) -> download.Options:
    return download.Options(
        data_root=root,
        preset=preset,
        percentage=percentage,
        categories=categories,
        max_samples=max_samples,
        include_bad=include_bad,
        revision=None,
        cache_dir=None,
    )


class CollectingReporter:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def emit(self, stage: str, message: str) -> None:
        self.events.append((stage, message))


class CliContractTests(unittest.TestCase):
    def test_bare_and_all_three_modes_parse(self) -> None:
        destination = Path(tempfile.gettempdir()) / "vfxdb-cli"
        bare = download.parse_args([str(destination)])
        self.assertEqual(bare.mode, "metadata-only")
        self.assertEqual(bare.data_root, destination.absolute())

        cache_dir = destination.parent / "hf-cache"
        smoke = download.parse_args(
            [
                str(destination),
                "--preset",
                "SMOKE",
                "--include-bad",
                "--revision",
                "release-v1",
                "--cache-dir",
                str(cache_dir),
            ]
        )
        self.assertEqual(smoke.preset, "smoke")
        self.assertEqual(smoke.mode, "preset")
        self.assertTrue(smoke.include_bad)
        self.assertEqual(smoke.revision, "release-v1")
        self.assertEqual(smoke.cache_dir, cache_dir.absolute())

        percent = download.parse_args([str(destination), "--percentage", "12.5"])
        self.assertEqual(percent.percentage, Decimal("12.5"))
        self.assertEqual(percent.mode, "percentage")

        category = download.parse_args(
            [
                str(destination),
                "--category",
                "Alpha,Beta",
                "--category",
                "Alpha",
                "--max-samples",
                "7",
            ]
        )
        self.assertEqual(category.categories, ("Alpha", "Beta"))
        self.assertEqual(category.max_samples, 7)
        self.assertEqual(category.mode, "category")

        tui = download.parse_args(
            [
                str(destination),
                "--tui",
                "--include-bad",
                "--revision",
                "release-v1",
                "--cache-dir",
                str(cache_dir),
            ]
        )
        self.assertTrue(tui.tui)
        self.assertTrue(tui.include_bad)
        self.assertEqual(tui.mode, "metadata-only")

    def test_scope_modes_are_strictly_mutually_exclusive(self) -> None:
        invalid = (
            ["out", "--preset", "smoke", "--percentage", "20"],
            ["out", "--preset", "medium", "--category", "Alpha", "--max-samples", "1"],
            ["out", "--percentage", "20", "--category", "Alpha", "--max-samples", "1"],
            ["out", "--category", "Alpha"],
            ["out", "--max-samples", "1"],
            ["out", "--preset", "unknown"],
            ["out", "--tui", "--preset", "smoke"],
            ["out", "--tui", "--percentage", "10"],
            ["out", "--tui", "--category", "Alpha", "--max-samples", "1"],
        )
        for argv in invalid:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    download.parse_args(argv)
                self.assertEqual(raised.exception.code, 2)

    def test_percentage_max_and_retired_flags_fail_closed(self) -> None:
        invalid = (
            ["out", "--percentage", "0"],
            ["out", "--percentage", "-1"],
            ["out", "--percentage", "100.1"],
            ["out", "--percentage", "nan"],
            ["out", "--percentage", "Infinity"],
            ["out", "--category", "Alpha", "--max-samples", "0"],
            ["out", "--limit", "4"],
            ["out", "--metadata"],
            ["out", "--all"],
        )
        for argv in invalid:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    download.parse_args(argv)
                self.assertEqual(raised.exception.code, 2)


class PlanContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = RepositoryFixture(Path(self.temporary.name))
        self.catalogs = self.fixture.catalogs()
        self.bad = self.fixture.bad_samples()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_smoke_medium_full_and_global_percentage(self) -> None:
        smoke = download.build_plan(
            options(Path("/unused"), preset="smoke"), self.catalogs, self.bad.source_relpaths
        )
        self.assertEqual(
            smoke.selected_by_category,
            {"Alpha": 2, "Beta": 2, "EnvironmentalFog": 2},
        )
        self.assertEqual(len(smoke.archives), 6)
        self.assertEqual(
            [item.folder for item in smoke.archives if item.category == "Alpha"],
            self.fixture.ALPHA_FOLDERS[:2],
        )

        medium = download.build_plan(
            options(Path("/unused"), preset="medium"), self.catalogs, self.bad.source_relpaths
        )
        self.assertEqual(len(medium.archives), 7)  # ceil(31 * .20)
        self.assertEqual(
            medium.selected_by_category,
            {"Alpha": 3, "Beta": 2, "EnvironmentalFog": 2},
        )

        half = download.build_plan(
            options(Path("/unused"), percentage=Decimal("50")),
            self.catalogs,
            self.bad.source_relpaths,
        )
        self.assertEqual(len(half.archives), 16)
        self.assertEqual(
            half.selected_by_category,
            {"Alpha": 6, "Beta": 5, "EnvironmentalFog": 5},
        )

        almost_all = download.build_plan(
            options(Path("/unused"), percentage=Decimal("95")),
            self.catalogs,
            self.bad.source_relpaths,
        )
        self.assertEqual(len(almost_all.archives), 30)
        self.assertEqual(
            almost_all.selected_by_category,
            {"Alpha": 11, "Beta": 8, "EnvironmentalFog": 11},
        )

        full = download.build_plan(
            options(Path("/unused"), preset="full"), self.catalogs, self.bad.source_relpaths
        )
        self.assertEqual(len(full.archives), 31)
        self.assertEqual(
            full.selected_by_category,
            {"Alpha": 12, "Beta": 8, "EnvironmentalFog": 11},
        )

    def test_category_max_samples_rounds_up_by_whole_tar_per_category(self) -> None:
        normal = download.build_plan(
            options(
                Path("/unused"),
                categories=("Alpha", "Beta"),
                max_samples=4,
            ),
            self.catalogs,
            self.bad.source_relpaths,
        )
        self.assertEqual(normal.selected_by_category["Alpha"], 3)
        self.assertEqual(normal.normal_samples_by_category["Alpha"], 5)
        self.assertEqual(normal.selected_by_category["Beta"], 4)
        self.assertEqual(normal.normal_samples_by_category["Beta"], 4)
        self.assertEqual(
            [item.folder for item in normal.archives if item.category == "Alpha"],
            ["2", "10", "1"],
        )

        above_total = download.build_plan(
            options(Path("/unused"), categories=("Beta",), max_samples=999),
            self.catalogs,
            self.bad.source_relpaths,
        )
        self.assertEqual(above_total.selected_by_category["Beta"], 8)

    def test_include_bad_does_not_change_plan_and_fog_uses_folder_not_seq(self) -> None:
        default_options = options(
            Path("/unused"), categories=("Alpha",), max_samples=4, include_bad=False
        )
        retained_options = options(
            Path("/unused"), categories=("Alpha",), max_samples=4, include_bad=True
        )
        default = download.build_plan(default_options, self.catalogs, self.bad.source_relpaths)
        retained = download.build_plan(retained_options, self.catalogs, self.bad.source_relpaths)
        self.assertEqual(
            [item.remote_path for item in default.archives],
            [item.remote_path for item in retained.archives],
        )

        fog_rows = self.fixture.original_indexes["EnvironmentalFog"]["samples"]
        self.assertTrue(all(row["seq"] == 0 for row in fog_rows))
        self.assertEqual(len(self.catalogs["EnvironmentalFog"].archives), 11)
        fog = download.build_plan(
            options(Path("/unused"), categories=("EnvironmentalFog",), max_samples=3),
            self.catalogs,
            self.bad.source_relpaths,
        )
        self.assertEqual(fog.selected_by_category["EnvironmentalFog"], 2)
        self.assertEqual(fog.normal_samples_by_category["EnvironmentalFog"], 4)

    def test_medium_matches_the_published_9310_tar_allocation(self) -> None:
        """Lock the production-scale water-fill result, not only the toy fixture."""

        published_counts = {
            "BuoyantExplosion": 2000,
            "CloudWave": 88,
            "EnvironmentalFog": 200,
            "IsotropicBurst": 1999,
            "LiquidSplash": 924,
            "RingBlast": 1999,
            "RisingFlame": 199,
            "SmokePlume": 199,
            "SurfaceFire": 200,
            "ViscousFlow": 999,
            "VortexColumn": 503,
        }
        catalogs = {
            category: download.CategoryCatalog(
                category,
                Path(f"/{category}/category_index.json"),
                [
                    download.ArchiveUnit(
                        category,
                        str(index),
                        f"archives/{category}/{index}.tar",
                    )
                    for index in range(count)
                ],
                {},
            )
            for category, count in published_counts.items()
        }

        plan = download.build_plan(
            options(Path("/unused"), preset="medium"), catalogs, set()
        )
        percentage_plan = download.build_plan(
            options(Path("/unused"), percentage=Decimal("20")), catalogs, set()
        )
        repeated_plan = download.build_plan(
            options(Path("/unused"), preset="medium"), catalogs, set()
        )

        self.assertEqual(plan.all_archive_count, 9310)
        self.assertEqual(len(plan.archives), 1862)
        self.assertEqual(
            plan.selected_by_category,
            {
                "BuoyantExplosion": 178,
                "CloudWave": 88,
                "EnvironmentalFog": 178,
                "IsotropicBurst": 178,
                "LiquidSplash": 178,
                "RingBlast": 177,
                "RisingFlame": 177,
                "SmokePlume": 177,
                "SurfaceFire": 177,
                "ViscousFlow": 177,
                "VortexColumn": 177,
            },
        )
        paths = [archive.remote_path for archive in plan.archives]
        self.assertEqual(paths, [archive.remote_path for archive in percentage_plan.archives])
        self.assertEqual(paths, [archive.remote_path for archive in repeated_plan.archives])

    def test_fractional_and_full_percentages_have_exact_global_targets(self) -> None:
        fractional = download.build_plan(
            options(Path("/unused"), percentage=Decimal("0.1")),
            self.catalogs,
            self.bad.source_relpaths,
        )
        full = download.build_plan(
            options(Path("/unused"), percentage=Decimal("100")),
            self.catalogs,
            self.bad.source_relpaths,
        )

        self.assertEqual(len(fractional.archives), 1)  # ceil(31 * 0.1%)
        self.assertEqual(len(full.archives), 31)
        self.assertEqual(
            [archive.remote_path for archive in full.archives],
            [archive.remote_path for archive in download.all_archives(self.catalogs)],
        )


class SchemaAndSpaceTests(unittest.TestCase):
    def test_malformed_scalar_and_invalid_utf8_raise_contextual_download_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            malformed = root / "category_index.json"
            write_json(
                malformed,
                {
                    "category": "Smoke",
                    "num_samples": [],
                    "samples": [{"vdb_path": "0/x.vdb"}],
                },
            )
            with self.assertRaisesRegex(download.DownloadError, "num_samples"):
                download.parse_category_catalog("Smoke", malformed)

            invalid_utf8 = root / "invalid.json"
            invalid_utf8.write_bytes(b"{\xff}")
            with self.assertRaisesRegex(download.DownloadError, "Cannot read JSON"):
                download.load_json(invalid_utf8)

            reserved = root / "reserved.json"
            sample = make_fixture_sample("Smoke", "index", 0, seq=0)
            write_json(
                reserved,
                {
                    "category": "Smoke",
                    "num_samples": 1,
                    "samples": [sample.row],
                },
            )
            with self.assertRaisesRegex(download.DownloadError, "conflicts"):
                download.parse_category_catalog("Smoke", reserved)

    def test_space_requirements_on_one_device_are_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(download, "_required_free_space_on_filesystem") as check:
                download.required_combined_free_space(
                    ((root, 100, 2), (root / "not-created", 250, 3))
                )
            check.assert_called_once_with(root, 350, 5)

    def test_corrupt_cached_tar_is_budgeted_as_a_fresh_download_before_install(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cached = root / "cached-corrupt.tar"
            cached.write_bytes(b"x" * 100)
            sample = download.SampleRecord(
                "Smoke", "0", "Smoke/0/a.vdb", "Smoke/index/a.json", 4, 0
            )
            archive = download.ArchiveUnit(
                "Smoke", "0", "archives/Smoke/0.tar", [sample]
            )
            plan = download.DownloadPlan(
                "test",
                [archive],
                1,
                {"Smoke": 1},
                {"Smoke": 1},
                {"Smoke": 1},
            )
            remote = download.RemoteFile(
                archive.remote_path,
                100,
                hashlib.sha256(b"g" * 100).hexdigest(),
            )

            class CorruptCacheHub:
                cache_root = root

                def cached_path(self, _path, _revision):
                    return cached

            captured: list[tuple[Path, int, int]] = []

            def capture(requirements):
                captured.extend(requirements)

            with mock.patch.object(
                download, "required_combined_free_space", side_effect=capture
            ), mock.patch.object(
                download, "download_verified", side_effect=RuntimeError("stop after preflight")
            ):
                with self.assertRaisesRegex(RuntimeError, "stop after preflight"):
                    download.install_data_archives(
                        plan,
                        {archive.remote_path: remote},
                        REVISION,
                        CorruptCacheHub(),
                        download.BadSamples(set(), {}, {}, ""),
                        {},
                        root / "data",
                        False,
                        download.initial_state(REVISION),
                        CollectingReporter(),
                    )

            self.assertEqual(captured[0][1], 100)
            self.assertEqual(captured[1][1], 4)


class EndToEndContractTests(unittest.TestCase):
    def test_interactive_plan_cancel_keeps_required_json_and_downloads_no_vdb_tar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = RepositoryFixture(root)
            data_root = root / "data"
            reporter = CollectingReporter()

            class CancelAfterPlan:
                def __init__(self) -> None:
                    self.plan = None

                def choose(self, catalogs):
                    self.asserted_categories = sorted(catalogs)
                    return download.Selection(preset="smoke")

                def confirm_plan(
                    self,
                    plan,
                    _remote_files,
                    revision,
                    destination,
                    _cache_root,
                    include_bad,
                ):
                    self.plan = plan
                    self.revision = revision
                    self.destination = destination
                    self.include_bad = include_bad
                    return "quit"

            interaction = CancelAfterPlan()
            plan = download.run(
                options(data_root),
                hub=fixture.hub,
                reporter=reporter,
                interaction=interaction,
                plain_output=False,
            )

            self.assertEqual(interaction.asserted_categories, ["Alpha", "Beta", "EnvironmentalFog"])
            self.assertIs(interaction.plan, plan)
            self.assertEqual(interaction.revision, REVISION)
            self.assertEqual(interaction.destination, data_root)
            self.assertFalse(interaction.include_bad)
            self.assertEqual(len(plan.archives), 6)
            self.assertFalse(
                any(path.startswith("archives/") for path, _, _ in fixture.hub.download_calls)
            )
            self.assertTrue((data_root / "Alpha/category_index.json").is_file())
            self.assertTrue(any((data_root / "Alpha/index").glob("*.json")))
            self.assertTrue(any(stage == "cancelled" for stage, _ in reporter.events))
            self.assertTrue(
                any(
                    stage == "json" and "per-sample JSON files" in message and "/" in message
                    for stage, message in reporter.events
                )
            )

    def test_interactive_plan_can_change_selection_before_any_tar_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = RepositoryFixture(root)
            data_root = root / "data"

            class ChangeThenDownload:
                def __init__(self) -> None:
                    self.choose_count = 0
                    self.plans = []

                def choose(self, _catalogs):
                    self.choose_count += 1
                    if self.choose_count == 1:
                        return download.Selection(percentage=Decimal("5"))
                    return download.Selection(preset="smoke")

                def confirm_plan(self, plan, *_args):
                    self.plans.append(plan)
                    return "change" if len(self.plans) == 1 else "download"

            interaction = ChangeThenDownload()
            plan = download.run(
                options(data_root),
                hub=fixture.hub,
                reporter=CollectingReporter(),
                interaction=interaction,
                plain_output=False,
            )

            self.assertEqual(interaction.choose_count, 2)
            self.assertEqual(len(interaction.plans), 2)
            self.assertEqual(plan.label, "preset smoke")
            downloaded_archives = [
                path for path, _revision, _force in fixture.hub.download_calls if path.startswith("archives/")
            ]
            self.assertEqual(len(downloaded_archives), 6)

    def test_clean_fixture_runs_every_selection_mode_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = RepositoryFixture(root)
            cases = (
                (
                    "smoke",
                    lambda data_root: options(data_root, preset="smoke"),
                    {"Alpha": 2, "Beta": 2, "EnvironmentalFog": 2},
                ),
                (
                    "medium",
                    lambda data_root: options(data_root, preset="medium"),
                    {"Alpha": 3, "Beta": 2, "EnvironmentalFog": 2},
                ),
                (
                    "full",
                    lambda data_root: options(data_root, preset="full"),
                    {"Alpha": 12, "Beta": 8, "EnvironmentalFog": 11},
                ),
                (
                    "percentage",
                    lambda data_root: options(data_root, percentage=Decimal("50")),
                    {"Alpha": 6, "Beta": 5, "EnvironmentalFog": 5},
                ),
                (
                    "category",
                    lambda data_root: options(
                        data_root,
                        categories=("Alpha", "Beta"),
                        max_samples=4,
                    ),
                    {"Alpha": 3, "Beta": 4, "EnvironmentalFog": 0},
                ),
            )
            for label, make_options, expected_counts in cases:
                with self.subTest(mode=label):
                    data_root = root / f"data-{label}"
                    calls_before = len(fixture.hub.download_calls)
                    with contextlib.redirect_stdout(io.StringIO()):
                        plan = download.run(make_options(data_root), hub=fixture.hub)
                    calls = fixture.hub.download_calls[calls_before:]
                    archive_calls = [path for path, _, _ in calls if path.startswith("archives/")]
                    self.assertEqual(len(archive_calls), len(plan.archives))
                    self.assertEqual(len(archive_calls), len(set(archive_calls)))
                    self.assertEqual(plan.selected_by_category, expected_counts)

                    selected = {archive.remote_path for archive in plan.archives}
                    for archive_path, samples in fixture.samples_by_archive.items():
                        for sample in samples:
                            expected_vdb = archive_path in selected and sample not in fixture.bads
                            self.assertEqual(
                                (data_root / sample.source_relpath).is_file(),
                                expected_vdb,
                                f"{label}: unexpected VDB state for {sample.source_relpath}",
                            )
                            self.assertEqual(
                                (data_root / sample.meta_relpath).is_file(),
                                sample not in fixture.bads,
                                f"{label}: per-sample JSON state for {sample.meta_relpath}",
                            )

    def test_bare_run_installs_all_required_json_before_any_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = RepositoryFixture(root)
            data_root = root / "data"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                plan = download.run(options(data_root), hub=fixture.hub)

            self.assertEqual(plan.label, "metadata-only")
            self.assertEqual(plan.archives, [])
            self.assertFalse(any(path.startswith("archives/") for path, _, _ in fixture.hub.download_calls))
            self.assertFalse((data_root / "dataset_index.json").exists())
            self.assertTrue((data_root / download.BAD_MANIFEST_PATH).is_file())
            self.assertTrue((data_root / download.BAD_MANIFEST_INFO_PATH).is_file())
            for category in fixture.original_indexes:
                self.assertTrue((data_root / category / "category_index.json").is_file())
            for sample in fixture.samples:
                expected = sample not in fixture.bads
                self.assertEqual((data_root / sample.meta_relpath).is_file(), expected)
            self.assertEqual(list(data_root.rglob("*.vdb")), [])
            rendered = output.getvalue()
            for text in ("No VDB data tar was downloaded", "Smoke", "Medium", "Full", "--percentage", "--max-samples"):
                self.assertIn(text, rendered)

    def test_dataset_index_is_ignored_even_when_remote_contents_are_poisoned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = RepositoryFixture(root)
            write_json(
                fixture.root / "dataset_index.json",
                {"categories": ["WrongCategory"], "archives": ["do-not-read.tar"]},
            )
            fixture.hub.refresh_remote_files()
            data_root = root / "data"
            with contextlib.redirect_stdout(io.StringIO()):
                plan = download.run(
                    options(data_root, categories=("Beta",), max_samples=1),
                    hub=fixture.hub,
                )

            self.assertEqual(plan.selected_by_category["Beta"], 1)
            self.assertFalse((data_root / "dataset_index.json").exists())
            self.assertNotIn(
                "dataset_index.json",
                [path for path, _revision, _force in fixture.hub.download_calls],
            )

    def test_unknown_category_fails_after_controls_but_before_any_data_tar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = RepositoryFixture(root)
            data_root = root / "data"
            with self.assertRaisesRegex(download.DownloadError, "Unknown categories"):
                with contextlib.redirect_stdout(io.StringIO()):
                    download.run(
                        options(
                            data_root,
                            categories=("DoesNotExist",),
                            max_samples=1,
                        ),
                        hub=fixture.hub,
                    )

            self.assertTrue((data_root / "Alpha" / "category_index.json").is_file())
            self.assertFalse(
                any(path.startswith("archives/") for path, _, _ in fixture.hub.download_calls)
            )

    def test_every_local_json_exists_before_the_first_tar_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = RepositoryFixture(root)
            data_root = root / "data"
            checks = 0

            def assert_prepared() -> None:
                nonlocal checks
                checks += 1
                for category in fixture.original_indexes:
                    self.assertTrue((data_root / category / "category_index.json").is_file())
                for sample in fixture.samples:
                    self.assertEqual(
                        (data_root / sample.meta_relpath).is_file(),
                        sample not in fixture.bads,
                    )
                self.assertTrue((data_root / download.BAD_MANIFEST_PATH).is_file())

            fixture.hub.before_archive = assert_prepared
            with contextlib.redirect_stdout(io.StringIO()):
                download.run(
                    options(data_root, categories=("Alpha",), max_samples=1),
                    hub=fixture.hub,
                )
            self.assertEqual(checks, 1)
            first_archive = next(
                index
                for index, (path, _, _) in enumerate(fixture.hub.download_calls)
                if path.startswith("archives/")
            )
            controls = [path for path, _, _ in fixture.hub.download_calls[:first_archive]]
            for category in fixture.original_indexes:
                self.assertIn(f"{category}/category_index.json", controls)
            self.assertIn(download.SAMPLE_JSON_ARCHIVE_PATH, controls)

    def test_default_bad_cleanup_marker_and_original_fields_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = RepositoryFixture(root)
            data_root = root / "data"
            cached_index_bytes = (fixture.root / "Alpha/category_index.json").read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                plan = download.run(
                    options(data_root, categories=("Alpha",), max_samples=1),
                    hub=fixture.hub,
                )

            self.assertEqual([archive.folder for archive in plan.archives], ["2"])
            first_tar_samples = fixture.samples_by_archive["archives/Alpha/2.tar"]
            self.assertEqual(len(first_tar_samples), 3)
            for sample in first_tar_samples:
                should_exist = sample not in fixture.bads
                self.assertEqual((data_root / sample.source_relpath).is_file(), should_exist)
                self.assertEqual((data_root / sample.meta_relpath).is_file(), should_exist)

            local = download.load_json(data_root / "Alpha/category_index.json")
            original = fixture.original_indexes["Alpha"]
            self.assertEqual(local["num_samples"], original["num_samples"])
            self.assertEqual(local["description"], original["description"])
            for local_row, original_row in zip(local["samples"], original["samples"], strict=True):
                comparison = copy.deepcopy(local_row)
                marker = comparison.pop("deleted_bad_io_sample", None)
                self.assertEqual(comparison, original_row)
                if original_row["vdb_path"] in {
                    bad.row["vdb_path"] for bad in fixture.bads
                }:
                    self.assertIs(marker, True)
                else:
                    self.assertIsNone(marker)
            self.assertEqual((fixture.root / "Alpha/category_index.json").read_bytes(), cached_index_bytes)

    def test_switching_bad_policy_restores_then_removes_files_and_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = RepositoryFixture(root)
            data_root = root / "data"
            with contextlib.redirect_stdout(io.StringIO()):
                default_plan = download.run(
                    options(data_root, categories=("Alpha",), max_samples=4),
                    hub=fixture.hub,
                )
                retained_plan = download.run(
                    options(
                        data_root,
                        categories=("Alpha",),
                        max_samples=4,
                        include_bad=True,
                    ),
                    hub=fixture.hub,
                )
            self.assertEqual(
                [archive.remote_path for archive in default_plan.archives],
                [archive.remote_path for archive in retained_plan.archives],
            )
            retained_index = download.load_json(data_root / "Alpha/category_index.json")
            for bad in fixture.bads:
                self.assertTrue((data_root / bad.source_relpath).is_file())
                self.assertTrue((data_root / bad.meta_relpath).is_file())
                self.assertEqual(
                    next(
                        row["deleted_bad_io_sample"]
                        for row in retained_index["samples"]
                        if row["vdb_path"] == bad.row["vdb_path"]
                    ),
                    False,
                )

            with contextlib.redirect_stdout(io.StringIO()):
                default_again = download.run(
                    options(data_root, categories=("Alpha",), max_samples=4),
                    hub=fixture.hub,
                )
            self.assertEqual(
                [archive.remote_path for archive in default_plan.archives],
                [archive.remote_path for archive in default_again.archives],
            )
            default_index = download.load_json(data_root / "Alpha/category_index.json")
            for bad in fixture.bads:
                self.assertFalse((data_root / bad.source_relpath).exists())
                self.assertFalse((data_root / bad.meta_relpath).exists())
                self.assertEqual(
                    next(
                        row["deleted_bad_io_sample"]
                        for row in default_index["samples"]
                        if row["vdb_path"] == bad.row["vdb_path"]
                    ),
                    True,
                )

    def test_resume_skips_installed_tars_and_same_size_corruption_is_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = RepositoryFixture(root)
            data_root = root / "data"
            run_options = options(data_root, categories=("Alpha",), max_samples=4)
            with contextlib.redirect_stdout(io.StringIO()):
                download.run(run_options, hub=fixture.hub)
            archive_calls = [
                call for call in fixture.hub.download_calls if call[0].startswith("archives/")
            ]
            self.assertEqual(len(archive_calls), 3)
            good = fixture.samples_by_archive["archives/Alpha/2.tar"][1]
            good_path = data_root / good.source_relpath
            original_mtime = good_path.stat().st_mtime_ns

            with contextlib.redirect_stdout(io.StringIO()):
                download.run(run_options, hub=fixture.hub)
            self.assertEqual(
                [call for call in fixture.hub.download_calls if call[0].startswith("archives/")],
                archive_calls,
            )
            self.assertEqual(good_path.stat().st_mtime_ns, original_mtime)

            good_path.write_bytes(b"x" * len(good.payload))
            with contextlib.redirect_stdout(io.StringIO()):
                download.run(run_options, hub=fixture.hub)
            self.assertEqual(good_path.read_bytes(), good.payload)

    def test_same_size_sample_json_corruption_is_repaired_without_data_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = RepositoryFixture(root)
            data_root = root / "data"
            reporter = CollectingReporter()
            with contextlib.redirect_stdout(io.StringIO()):
                download.run(options(data_root), hub=fixture.hub, reporter=reporter)

            sample = next(sample for sample in fixture.samples if sample not in fixture.bads)
            meta_path = data_root / sample.meta_relpath
            original = meta_path.read_bytes()
            original_stat = meta_path.stat()
            replacement = bytearray(original)
            replacement[len(replacement) // 2] ^= 1
            meta_path.write_bytes(bytes(replacement))
            os.utime(
                meta_path,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )

            calls_before = len(fixture.hub.download_calls)
            reporter.events.clear()
            with contextlib.redirect_stdout(io.StringIO()):
                download.run(options(data_root), hub=fixture.hub, reporter=reporter)

            self.assertEqual(meta_path.read_bytes(), original)
            later_calls = fixture.hub.download_calls[calls_before:]
            self.assertFalse(any(path.startswith("archives/") for path, _, _ in later_calls))
            self.assertTrue(any(stage == "repair" for stage, _ in reporter.events))

    def test_policy_switch_cannot_hide_normal_json_or_vdb_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = RepositoryFixture(root)
            data_root = root / "data"
            run_options = options(data_root, categories=("Beta",), max_samples=1)
            with contextlib.redirect_stdout(io.StringIO()):
                download.run(run_options, hub=fixture.hub)

            normal_json_sample = next(
                sample for sample in fixture.samples if sample not in fixture.bads
            )
            json_path = data_root / normal_json_sample.meta_relpath
            json_original = json_path.read_bytes()
            json_path.write_bytes(b"x" * len(json_original))

            beta_sample = fixture.samples_by_archive["archives/Beta/b0.tar"][0]
            vdb_path = data_root / beta_sample.source_relpath
            vdb_path.write_bytes(b"x" * len(beta_sample.payload))

            switched = options(
                data_root,
                categories=("Beta",),
                max_samples=1,
                include_bad=True,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                download.run(switched, hub=fixture.hub)

            self.assertEqual(json_path.read_bytes(), json_original)
            self.assertEqual(vdb_path.read_bytes(), beta_sample.payload)

    def test_metadata_validation_failure_preserves_existing_public_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = RepositoryFixture(root)
            data_root = root / "data"
            with contextlib.redirect_stdout(io.StringIO()):
                download.run(options(data_root), hub=fixture.hub)

            public_controls = {
                relative: (data_root / relative).read_bytes()
                for relative in (
                    *(f"{category}/category_index.json" for category in fixture.original_indexes),
                    download.BAD_MANIFEST_PATH,
                    download.BAD_MANIFEST_INFO_PATH,
                )
            }
            normal = next(sample for sample in fixture.samples if sample not in fixture.bads)
            normal_json = (data_root / normal.meta_relpath).read_bytes()
            fixture.replace_sample_json_archive(omit={normal.meta_relpath})

            calls_before = len(fixture.hub.download_calls)
            with self.assertRaisesRegex(download.IntegrityError, "missing 1"):
                with contextlib.redirect_stdout(io.StringIO()):
                    download.run(
                        options(data_root, preset="smoke", include_bad=True),
                        hub=fixture.hub,
                    )

            for relative, previous in public_controls.items():
                self.assertEqual((data_root / relative).read_bytes(), previous)
            self.assertEqual((data_root / normal.meta_relpath).read_bytes(), normal_json)
            later_calls = fixture.hub.download_calls[calls_before:]
            self.assertFalse(any(path.startswith("archives/") for path, _, _ in later_calls))

    def test_control_publication_failure_rolls_back_every_category_and_bad_file_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = RepositoryFixture(root)
            beta_bad = fixture.samples_by_archive["archives/Beta/b0.tar"][0]
            fixture.bads = (*fixture.bads, beta_bad)
            write_bad_controls(fixture.root, list(fixture.bads))
            fixture.hub.refresh_remote_files()
            data_root = root / "data"
            with contextlib.redirect_stdout(io.StringIO()):
                download.run(options(data_root), hub=fixture.hub)

            control_paths = (
                *(f"{category}/category_index.json" for category in fixture.original_indexes),
                download.BAD_MANIFEST_PATH,
                download.BAD_MANIFEST_INFO_PATH,
            )
            previous_controls = {
                relative: (data_root / relative).read_bytes() for relative in control_paths
            }
            real_publish = download.publish_staged_control
            publication_count = 0

            def fail_second_publication(staged, relative, destination):
                nonlocal publication_count
                publication_count += 1
                if publication_count == 2:
                    raise OSError("injected control publication failure")
                return real_publish(staged, relative, destination)

            with mock.patch.object(
                download,
                "publish_staged_control",
                side_effect=fail_second_publication,
            ):
                with self.assertRaisesRegex(OSError, "injected control publication failure"):
                    with contextlib.redirect_stdout(io.StringIO()):
                        download.run(
                            options(data_root, include_bad=True),
                            hub=fixture.hub,
                        )

            self.assertGreaterEqual(publication_count, 2)
            for relative, previous in previous_controls.items():
                self.assertEqual((data_root / relative).read_bytes(), previous)
            for bad in fixture.bads:
                self.assertFalse((data_root / bad.meta_relpath).exists())
                self.assertFalse((data_root / bad.source_relpath).exists())
            self.assertFalse(download.control_transaction_dir(data_root).exists())

            with contextlib.redirect_stdout(io.StringIO()):
                download.run(options(data_root, include_bad=True), hub=fixture.hub)
            for bad in fixture.bads:
                self.assertTrue((data_root / bad.meta_relpath).is_file())
            for category in ("Alpha", "Beta"):
                index = download.load_json(data_root / category / "category_index.json")
                marked = [
                    row["deleted_bad_io_sample"]
                    for row in index["samples"]
                    if f"{category}/{row['vdb_path']}"
                    in {bad.source_relpath for bad in fixture.bads}
                ]
                self.assertTrue(marked)
                self.assertTrue(all(marker is False for marker in marked))

    def test_failed_post_control_policy_cleanup_blocks_training_until_rerun_converges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = RepositoryFixture(root)
            data_root = root / "data"
            include_options = options(
                data_root,
                categories=("Alpha",),
                max_samples=4,
                include_bad=True,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                download.run(include_options, hub=fixture.hub)
            self.assertTrue(all((data_root / bad.source_relpath).is_file() for bad in fixture.bads))

            with mock.patch.object(
                download,
                "apply_bad_policy_to_files",
                side_effect=OSError("injected cleanup failure after controls committed"),
            ):
                with self.assertRaisesRegex(OSError, "injected cleanup failure"):
                    with contextlib.redirect_stdout(io.StringIO()):
                        download.run(
                            options(
                                data_root,
                                categories=("Alpha",),
                                max_samples=4,
                            ),
                            hub=fixture.hub,
                        )

            transition = download.policy_transition_path(data_root)
            self.assertTrue(transition.is_file())
            from tile_dataloader import _collect_samples

            with self.assertRaisesRegex(RuntimeError, "policy transition is incomplete"):
                _collect_samples(str(data_root), categories=["Alpha"], require_meta=False)

            with contextlib.redirect_stdout(io.StringIO()):
                download.run(
                    options(data_root, categories=("Alpha",), max_samples=4),
                    hub=fixture.hub,
                )
            self.assertFalse(transition.exists())
            for bad in fixture.bads:
                self.assertFalse((data_root / bad.source_relpath).exists())
                self.assertFalse((data_root / bad.meta_relpath).exists())

    def test_bare_include_bad_restores_json_but_never_fetches_bad_vdb_tars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = RepositoryFixture(root)
            data_root = root / "data"
            with contextlib.redirect_stdout(io.StringIO()):
                plan = download.run(
                    options(data_root, include_bad=True),
                    hub=fixture.hub,
                )

            self.assertEqual(plan.archives, [])
            self.assertFalse(
                any(path.startswith("archives/") for path, _, _ in fixture.hub.download_calls)
            )
            index = download.load_json(data_root / "Alpha/category_index.json")
            for bad in fixture.bads:
                self.assertTrue((data_root / bad.meta_relpath).is_file())
                self.assertFalse((data_root / bad.source_relpath).exists())
                marker = next(
                    row["deleted_bad_io_sample"]
                    for row in index["samples"]
                    if row["vdb_path"] == bad.row["vdb_path"]
                )
                self.assertIs(marker, False)

    def test_training_loader_enumerates_only_installed_normal_samples_with_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = RepositoryFixture(root)
            data_root = root / "data"
            with contextlib.redirect_stdout(io.StringIO()):
                plan = download.run(options(data_root, preset="smoke"), hub=fixture.hub)

            from tile_dataloader import _collect_samples

            categories = sorted(fixture.original_indexes)
            with contextlib.redirect_stdout(io.StringIO()):
                records, _ = _collect_samples(
                    str(data_root), categories=categories, require_meta=True
                )
            selected = {archive.remote_path for archive in plan.archives}
            expected_sources = {
                sample.source_relpath
                for archive_path, samples in fixture.samples_by_archive.items()
                if archive_path in selected
                for sample in samples
                if sample not in fixture.bads
            }
            actual_sources = {
                Path(record.abs_data_path).relative_to(data_root).as_posix()
                for record in records
            }
            self.assertEqual(actual_sources, expected_sources)
            self.assertTrue(all(record.abs_meta_path is not None for record in records))

    def test_vdb_return_meta_loads_json_and_rejects_invalid_content(self) -> None:
        from tile_dataloader import MetadataLoadError, SampleRecord, VolumeMultiDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta_path = root / "sample.json"
            write_json(meta_path, {"bbox_min": [0, 0, 0], "name": "sample"})
            record = SampleRecord(
                id="sample",
                category="Alpha",
                data_path="0/sample.vdb",
                meta_path="index/sample.json",
                abs_data_path=str(root / "sample.vdb"),
                abs_meta_path=str(meta_path),
                abs_cat_dir=str(root),
            )
            dataset = object.__new__(VolumeMultiDataset)
            dataset.samples = [record]
            dataset.return_meta = True

            with mock.patch.object(
                VolumeMultiDataset,
                "vdb_ext_get",
                return_value={"category": "Alpha"},
            ):
                sample = dataset._get_core(0)
            self.assertEqual(sample["meta"]["name"], "sample")

            meta_path.write_text("[]\n", encoding="utf-8")
            with mock.patch.object(
                VolumeMultiDataset,
                "vdb_ext_get",
                return_value={"category": "Alpha"},
            ), self.assertRaisesRegex(MetadataLoadError, "must contain an object"):
                dataset._get_core(0)

    def test_return_meta_requires_json_and_does_not_silently_skip_samples(self) -> None:
        from tile_dataloader import MetadataLoadError, build_dataset_splits

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vdb_path = root / "Alpha/0/sample.vdb"
            vdb_path.parent.mkdir(parents=True)
            vdb_path.write_bytes(b"fixture")
            write_json(
                root / "Alpha/category_index.json",
                {
                    "samples": [
                        {
                            "id": "sample",
                            "vdb_path": "0/sample.vdb",
                            "meta_path": "index/sample.json",
                        }
                    ]
                },
            )

            with self.assertRaisesRegex(MetadataLoadError, "sample JSON is missing"):
                build_dataset_splits(
                    str(root),
                    categories=["Alpha"],
                    return_meta=True,
                    balance_train=False,
                )

    def test_metadata_failure_is_not_hidden_by_getitem_fallback(self) -> None:
        from tile_dataloader import MetadataLoadError, VolumeMultiDataset

        dataset = object.__new__(VolumeMultiDataset)
        dataset.samples = [object(), object()]
        dataset._get_core = mock.Mock(
            side_effect=MetadataLoadError("broken requested JSON")
        )
        with self.assertRaisesRegex(MetadataLoadError, "broken requested JSON"):
            dataset[0]
        dataset._get_core.assert_called_once_with(0)

    def test_metadata_batch_collation_accepts_different_json_keys(self) -> None:
        import torch
        from tile_dataloader import collate_volume_batch

        batch = [
            {"volume": torch.zeros(1, 2, 2, 2), "category_id": 0, "meta": {"a": 1}},
            {"volume": torch.ones(1, 2, 2, 2), "category_id": 1, "meta": {"b": 2}},
        ]
        collated = collate_volume_batch(batch)
        self.assertEqual(tuple(collated["volume"].shape), (2, 1, 2, 2, 2))
        self.assertEqual(collated["meta"], [{"a": 1}, {"b": 2}])

    def test_frame_bboxes_are_unioned_for_sequence_sampling(self) -> None:
        import numpy as np
        import tile_dataloader as loader

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            for frame, bmin, bmax in (
                (1, [-2, 0, -1], [3, 4, 5]),
                (2, [-7, -1, 0], [8, 2, 9]),
            ):
                meta = root / f"index/sample__n{frame:04d}.json"
                write_json(meta, {"bbox_min": bmin, "bbox_max": bmax})
                records.append(
                    loader.SampleRecord(
                        id=f"sample-{frame}",
                        category="Alpha",
                        data_path=f"0/sample__n{frame:04d}.vdb",
                        meta_path=f"index/{meta.name}",
                        abs_data_path=str(root / f"0/sample__n{frame:04d}.vdb"),
                        abs_meta_path=str(meta),
                        abs_cat_dir=str(root),
                    )
                )
            with mock.patch.object(loader, "HAS_VDB_EXT", True):
                dataset = loader.VolumeMultiDataset(
                    records,
                    {"Alpha": 0},
                    prev_k=1,
                    prev_bbox_mode="seq",
                    transform_included=False,
                )
            bmin, bmax = dataset._seq_bbox_minmax["Alpha/0"]
            np.testing.assert_array_equal(bmin, [-7, -1, -1])
            np.testing.assert_array_equal(bmax, [8, 4, 9])

    def test_destination_symlinks_are_rejected_before_hugging_face_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = RepositoryFixture(root)
            target = root / "real-root"
            target.mkdir()
            direct_link = root / "data-link"
            direct_link.symlink_to(target, target_is_directory=True)
            ancestor_link = root / "parent-link"
            ancestor_link.symlink_to(target, target_is_directory=True)

            for data_root in (direct_link, ancestor_link / "nested"):
                with self.subTest(data_root=data_root):
                    fixture.hub.resolve_calls.clear()
                    with self.assertRaisesRegex(download.DownloadError, "symlink"):
                        download.run(options(data_root), hub=fixture.hub)
                    self.assertEqual(fixture.hub.resolve_calls, [])
            self.assertFalse((target / ".vfxdb").exists())
            self.assertFalse((target / "nested").exists())

    def test_unmanaged_nonempty_destination_is_rejected_without_mutation_or_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = RepositoryFixture(root)
            data_root = root / "old-data"
            old_file = data_root / "Smoke" / "0" / "old.vdb"
            old_file.parent.mkdir(parents=True)
            old_file.write_bytes(b"old")

            with self.assertRaisesRegex(download.DownloadError, "no VfxDB downloader state"):
                download.run(options(data_root), hub=fixture.hub)

            self.assertEqual(fixture.hub.resolve_calls, [])
            self.assertEqual(old_file.read_bytes(), b"old")
            self.assertFalse((data_root / ".vfxdb").exists())

    def test_revision_pin_and_destination_lock_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = RepositoryFixture(root)
            data_root = root / "data"
            with contextlib.redirect_stdout(io.StringIO()):
                download.run(options(data_root), hub=fixture.hub)

            with mock.patch.object(
                fixture.hub,
                "resolve_snapshot",
                return_value=("b" * 40, fixture.hub.remote_files),
            ):
                with self.assertRaisesRegex(download.DownloadError, "pinned"):
                    download.run(options(data_root), hub=fixture.hub)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = RepositoryFixture(root)
            data_root = root / "locked"
            data_root.mkdir()
            with download.destination_lock(data_root):
                with self.assertRaisesRegex(download.DownloadError, "Another downloader"):
                    download.run(options(data_root), hub=fixture.hub)
            self.assertEqual(fixture.hub.resolve_calls, [])

    def test_bare_rerun_recovers_an_interrupted_archive_swap_from_another_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = RepositoryFixture(root)
            data_root = root / "data"
            with contextlib.redirect_stdout(io.StringIO()):
                plan = download.run(
                    options(data_root, categories=("Alpha",), max_samples=1),
                    hub=fixture.hub,
                )
            archive = plan.archives[0]
            destination = download.archive_public_directory(data_root, archive)
            backup = download.archive_backup_directory(data_root, archive)
            os.replace(destination, backup)
            archive_calls_before = len(
                [call for call in fixture.hub.download_calls if call[0].startswith("archives/")]
            )

            with contextlib.redirect_stdout(io.StringIO()):
                bare = download.run(options(data_root), hub=fixture.hub)

            self.assertEqual(bare.archives, [])
            normal = next(
                sample
                for sample in fixture.samples_by_archive[archive.remote_path]
                if sample not in fixture.bads
            )
            self.assertEqual((data_root / normal.source_relpath).read_bytes(), normal.payload)
            self.assertFalse(backup.exists())
            archive_calls_after = len(
                [call for call in fixture.hub.download_calls if call[0].startswith("archives/")]
            )
            self.assertEqual(archive_calls_after, archive_calls_before)

    def test_noop_rerun_does_not_require_space_for_evicted_hf_tar_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = RepositoryFixture(root)
            data_root = root / "data"
            run_options = options(data_root, categories=("Beta",), max_samples=1)
            with contextlib.redirect_stdout(io.StringIO()):
                download.run(run_options, hub=fixture.hub)

            def reject_nonzero_space(requirements):
                for _path, bytes_needed, _inodes_needed in requirements:
                    if bytes_needed:
                        raise AssertionError(f"unexpected space reservation: {bytes_needed}")

            def evict_only_data_tars(remote_path, _revision):
                if remote_path.startswith("archives/"):
                    return None
                return fixture.root / remote_path

            with mock.patch.object(
                fixture.hub, "cached_path", side_effect=evict_only_data_tars
            ), mock.patch.object(
                download, "required_combined_free_space", side_effect=reject_nonzero_space
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    download.run(run_options, hub=fixture.hub)

    def test_missing_mandatory_sample_json_fails_before_data_tar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = RepositoryFixture(root)
            missing = next(sample for sample in fixture.samples if sample not in fixture.bads)
            fixture.replace_sample_json_archive(omit={missing.meta_relpath})
            with self.assertRaisesRegex(download.IntegrityError, "missing 1"):
                with contextlib.redirect_stdout(io.StringIO()):
                    download.run(
                        options(root / "data", preset="smoke"),
                        hub=fixture.hub,
                    )
            self.assertFalse(
                any(path.startswith("archives/") for path, _, _ in fixture.hub.download_calls)
            )

    def test_missing_io_bad_sample_json_also_fails_before_default_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = RepositoryFixture(root)
            fixture.replace_sample_json_archive(omit={fixture.bad.meta_relpath})
            with self.assertRaisesRegex(download.IntegrityError, "missing 1"):
                with contextlib.redirect_stdout(io.StringIO()):
                    download.run(
                        options(root / "data", preset="smoke"),
                        hub=fixture.hub,
                    )
            self.assertFalse(
                any(path.startswith("archives/") for path, _, _ in fixture.hub.download_calls)
            )


class TarIntegrityTests(unittest.TestCase):
    def _archive(self, root: Path) -> tuple[FixtureSample, download.ArchiveUnit, Path]:
        sample = make_fixture_sample("Smoke", "0", 0, seq=0)
        archive = download.ArchiveUnit(
            "Smoke",
            "0",
            "archives/Smoke/0.tar",
            [
                download.SampleRecord(
                    "Smoke",
                    "0",
                    sample.source_relpath,
                    sample.meta_relpath,
                    len(sample.payload),
                    0,
                )
            ],
        )
        path = root / "0.tar"
        return sample, archive, path

    @staticmethod
    def no_bad() -> download.BadSamples:
        return download.BadSamples(set(), {}, {}, hashlib.sha256(b"").hexdigest())

    def test_malformed_member_sha_never_publishes_member(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample, archive, path = self._archive(root)
            write_sequence_tar(
                path,
                "Smoke",
                "0",
                [sample],
                sha_overrides={sample.source_relpath: "not-a-sha"},
            )
            data_root = root / "data"
            data_root.mkdir()
            with self.assertRaises(download.IntegrityError):
                download.install_whole_archive(
                    path, archive, self.no_bad(), data_root, include_bad=False
                )
            self.assertFalse((data_root / sample.source_relpath).exists())

    def test_legacy_empty_or_stale_member_sha_uses_verified_tar_and_size(self) -> None:
        for published_digest in ("", "0" * 64):
            with self.subTest(published_digest=published_digest), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                sample = make_fixture_sample("LiquidSplash", "1", 0, seq=0)
                archive = download.ArchiveUnit(
                    "LiquidSplash",
                    "1",
                    "archives/LiquidSplash/1.tar",
                    [
                        download.SampleRecord(
                            "LiquidSplash",
                            "1",
                            sample.source_relpath,
                            sample.meta_relpath,
                            len(sample.payload),
                            0,
                        )
                    ],
                )
                path = root / "1.tar"
                write_sequence_tar(
                    path,
                    "LiquidSplash",
                    "1",
                    [sample],
                    sha_overrides={sample.source_relpath: published_digest},
                )
                data_root = root / "data"
                data_root.mkdir()

                download.install_whole_archive(
                    path, archive, self.no_bad(), data_root, include_bad=False
                )

                self.assertEqual((data_root / sample.source_relpath).read_bytes(), sample.payload)

    def test_links_traversal_duplicates_and_unmanifested_members_are_rejected(self) -> None:
        unsafe_cases = (
            ("link", {"smoke_0_0.vdb": tarfile.SYMTYPE}, None),
            ("traversal", None, [("../escape.vdb", tarfile.REGTYPE, b"x")]),
            ("extra", None, [("extra.vdb", tarfile.REGTYPE, b"x")]),
            ("duplicate", None, [("smoke_0_0.vdb", tarfile.REGTYPE, b"x")]),
        )
        for label, types, extras in unsafe_cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                sample, archive, path = self._archive(root)
                write_sequence_tar(
                    path,
                    "Smoke",
                    "0",
                    [sample],
                    member_type_overrides=types,
                    extras=extras,
                )
                with self.assertRaises(download.DownloadError):
                    download.inspect_sequence_archive(path, archive, self.no_bad())
                self.assertFalse((root / "escape.vdb").exists())

    def test_failed_directory_swap_restores_the_complete_previous_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample, archive, path = self._archive(root)
            write_sequence_tar(path, "Smoke", "0", [sample])
            data_root = root / "data"
            old_sequence = data_root / "Smoke" / "0"
            old_sequence.mkdir(parents=True)
            old_payload = b"previous-complete-version"
            old_file = old_sequence / Path(sample.source_relpath).name
            old_file.write_bytes(old_payload)
            sentinel = old_sequence / "old-only.txt"
            sentinel.write_text("keep", encoding="utf-8")

            real_replace = os.replace
            replace_count = 0

            def fail_publication(source, destination):
                nonlocal replace_count
                replace_count += 1
                if replace_count == 2:
                    raise OSError("injected publication failure")
                return real_replace(source, destination)

            with mock.patch.object(download.os, "replace", side_effect=fail_publication):
                with self.assertRaisesRegex(OSError, "injected publication failure"):
                    download.install_whole_archive(
                        path, archive, self.no_bad(), data_root, include_bad=False
                    )

            self.assertEqual(old_file.read_bytes(), old_payload)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            self.assertFalse(download.archive_backup_directory(data_root, archive).exists())

    def test_enospc_during_staging_never_replaces_the_previous_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample, archive, path = self._archive(root)
            write_sequence_tar(path, "Smoke", "0", [sample])
            data_root = root / "data"
            old_sequence = data_root / "Smoke" / "0"
            old_sequence.mkdir(parents=True)
            old_file = old_sequence / Path(sample.source_relpath).name
            old_file.write_bytes(b"old-complete")

            def fail_with_enospc(_tf, _member, destination):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"partial")
                raise OSError(errno.ENOSPC, "injected disk full")

            with mock.patch.object(
                download,
                "copy_tar_member",
                side_effect=fail_with_enospc,
            ):
                with self.assertRaisesRegex(OSError, "injected disk full"):
                    download.install_whole_archive(
                        path,
                        archive,
                        self.no_bad(),
                        data_root,
                        include_bad=False,
                    )

            self.assertEqual(old_file.read_bytes(), b"old-complete")
            self.assertFalse(download.archive_backup_directory(data_root, archive).exists())

    def test_recovery_restores_backup_left_between_atomic_renames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample, archive, path = self._archive(root)
            write_sequence_tar(path, "Smoke", "0", [sample])
            data_root = root / "data"
            data_root.mkdir()
            download.install_whole_archive(
                path, archive, self.no_bad(), data_root, include_bad=False
            )
            destination = download.archive_public_directory(data_root, archive)
            backup = download.archive_backup_directory(data_root, archive)
            os.replace(destination, backup)
            self.assertFalse(destination.exists())
            self.assertTrue(backup.exists())

            download.recover_archive_swap(data_root, archive)

            self.assertEqual((data_root / sample.source_relpath).read_bytes(), sample.payload)
            self.assertFalse(backup.exists())


class MetadataArchiveSafetyTests(unittest.TestCase):
    def test_local_json_digest_reports_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wanted = {"Alpha/index/a.json", "Alpha/index/b.json"}
            for relative in wanted:
                write_json(root / relative, {})
            reporter = CollectingReporter()
            with mock.patch.object(download, "JSON_PROGRESS_INTERVAL", 1):
                digest = download.sample_json_tree_digest(root, wanted, reporter)

            self.assertIsNotNone(digest)
            self.assertEqual(
                reporter.events,
                [
                    ("verify", "hashed local per-sample JSON files 1/2"),
                    ("verify", "hashed local per-sample JSON files 2/2"),
                ],
            )

    def test_metadata_validation_reports_progress_without_mutating_expected_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tar_path = root / "metadata.tar"
            compressed = root / "metadata.tar.zst"
            expected = {"Alpha/index/a.json", "Alpha/index/b.json"}
            with tarfile.open(tar_path, "w") as tf:
                for name in sorted(expected):
                    add_bytes(tf, name, b"{}")
                add_bytes(
                    tf,
                    "meta_manifest.json",
                    json.dumps(
                        {
                            "repo_id": download.REPO_ID,
                            "file_count": len(expected),
                            "categories": ["Alpha"],
                        }
                    ).encode("utf-8"),
                )
            subprocess.run(
                ["zstd", "-q", "-f", str(tar_path), "-o", str(compressed)],
                check=True,
            )
            reporter = CollectingReporter()
            with mock.patch.object(download, "JSON_PROGRESS_INTERVAL", 1):
                scan = download.validate_sample_json_archive(
                    compressed,
                    expected,
                    {"Alpha"},
                    reporter,
                )

            self.assertEqual(scan.file_count, 2)
            self.assertEqual(expected, {"Alpha/index/a.json", "Alpha/index/b.json"})
            self.assertEqual(
                reporter.events,
                [
                    ("verify", "validated 1/2 per-sample JSON files"),
                    ("verify", "validated 2/2 per-sample JSON files"),
                ],
            )

    def test_unsafe_duplicate_and_malformed_metadata_members_are_rejected(self) -> None:
        cases = ("traversal", "link", "duplicate", "malformed")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                tar_path = root / "metadata.tar"
                compressed = root / "metadata.tar.zst"
                expected_name = "Alpha/index/a.json"
                with tarfile.open(tar_path, "w") as tf:
                    if case == "traversal":
                        add_bytes(tf, "../escape.json", b"{}")
                    elif case == "link":
                        info = tarfile.TarInfo(expected_name)
                        info.type = tarfile.SYMTYPE
                        info.linkname = "../outside"
                        tf.addfile(info)
                    elif case == "duplicate":
                        add_bytes(tf, expected_name, b"{}")
                        add_bytes(tf, expected_name, b"{}")
                    else:
                        add_bytes(tf, expected_name, b"{")
                    add_bytes(
                        tf,
                        "meta_manifest.json",
                        json.dumps(
                            {
                                "repo_id": download.REPO_ID,
                                "file_count": 1,
                                "categories": ["Alpha"],
                            }
                        ).encode("utf-8"),
                    )
                subprocess.run(
                    ["zstd", "-q", "-f", str(tar_path), "-o", str(compressed)],
                    check=True,
                )

                with self.assertRaises(download.DownloadError):
                    download.validate_sample_json_archive(
                        compressed,
                        {expected_name},
                        {"Alpha"},
                    )
                self.assertFalse((root / "escape.json").exists())


class HubRetryTests(unittest.TestCase):
    def test_reporter_progress_class_is_forwarded_when_hub_supports_it(self) -> None:
        class ProgressClass:
            pass

        class Reporter(CollectingReporter):
            tqdm_class = ProgressClass

        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "archive.tar"
            result.write_bytes(b"archive")
            reporter = Reporter()
            client = self.client(reporter, mock.Mock())
            captured = {}

            def fake_download(*, tqdm_class=None, **_kwargs):
                captured["tqdm_class"] = tqdm_class
                return str(result)

            with mock.patch("huggingface_hub.hf_hub_download", new=fake_download):
                actual = client.download("archives/Smoke/0.tar", REVISION)
            self.assertEqual(actual, result)
            self.assertIs(captured["tqdm_class"], ProgressClass)

    def test_fallback_progress_suppression_is_restored_after_each_hub_call(self) -> None:
        class Reporter(CollectingReporter):
            suppress_hf_progress = True

        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "archive.tar"
            result.write_bytes(b"archive")
            client = self.client(Reporter(), mock.Mock())

            # No explicit tqdm_class parameter models older huggingface_hub.
            def fake_download(**_kwargs):
                return str(result)

            disabled = mock.Mock()
            enabled = mock.Mock()
            with mock.patch("huggingface_hub.hf_hub_download", new=fake_download), mock.patch(
                "huggingface_hub.utils.are_progress_bars_disabled", return_value=False
            ), mock.patch(
                "huggingface_hub.utils.disable_progress_bars", disabled
            ), mock.patch(
                "huggingface_hub.utils.enable_progress_bars", enabled
            ):
                self.assertEqual(client.download("archives/Smoke/0.tar", REVISION), result)

            disabled.assert_called_once_with()
            enabled.assert_called_once_with()

    def test_defaults_to_bounded_http_and_scopes_optional_xet_cache(self) -> None:
        from huggingface_hub import constants

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            download.os.environ, {}, clear=False
        ), mock.patch.object(
            constants, "HF_HUB_DISABLE_XET", False
        ), mock.patch.object(
            constants, "HF_HUB_DOWNLOAD_TIMEOUT", 10
        ):
            download.os.environ.pop("HF_HUB_DISABLE_XET", None)
            download.os.environ.pop("HF_HUB_DOWNLOAD_TIMEOUT", None)
            download.os.environ.pop("HF_XET_CACHE", None)
            download.os.environ.pop("HF_XET_CHUNK_CACHE_SIZE_BYTES", None)
            cache = Path(tmp) / "hub"
            client = download.HubClient(cache, CollectingReporter())

            self.assertEqual(client.cache_root, cache)
            self.assertEqual(download.os.environ["HF_HUB_DISABLE_XET"], "1")
            self.assertEqual(download.os.environ["HF_HUB_DOWNLOAD_TIMEOUT"], "60")
            self.assertIs(constants.HF_HUB_DISABLE_XET, True)
            self.assertEqual(constants.HF_HUB_DOWNLOAD_TIMEOUT, 60)
            self.assertEqual(download.os.environ["HF_XET_CACHE"], str(cache / "xet"))
            self.assertEqual(download.os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"], "0")

    def test_explicit_transport_environment_is_preserved(self) -> None:
        from huggingface_hub import constants

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            download.os.environ,
            {
                "HF_HUB_DISABLE_XET": "0",
                "HF_HUB_DOWNLOAD_TIMEOUT": "17",
                "HF_XET_CACHE": "/caller/xet",
                "HF_XET_CHUNK_CACHE_SIZE_BYTES": "1234",
            },
            clear=False,
        ), mock.patch.object(
            constants, "HF_HUB_DISABLE_XET", True
        ), mock.patch.object(
            constants, "HF_HUB_DOWNLOAD_TIMEOUT", 60
        ):
            download.HubClient(Path(tmp) / "hub", CollectingReporter())
            self.assertEqual(download.os.environ["HF_HUB_DISABLE_XET"], "0")
            self.assertEqual(download.os.environ["HF_HUB_DOWNLOAD_TIMEOUT"], "17")
            self.assertEqual(download.os.environ["HF_XET_CACHE"], "/caller/xet")
            self.assertEqual(download.os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"], "1234")
            self.assertIs(constants.HF_HUB_DISABLE_XET, False)
            self.assertEqual(constants.HF_HUB_DOWNLOAD_TIMEOUT, 17)

    @staticmethod
    def client(reporter: CollectingReporter, sleep: mock.Mock) -> download.HubClient:
        client = object.__new__(download.HubClient)
        client.cache_dir = None
        client.reporter = reporter
        client.sleep = sleep
        client.attempts = 4
        return client

    def test_transient_downloads_retry_but_permanent_404_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "archive.tar"
            result.write_bytes(b"archive")
            reporter = CollectingReporter()
            sleep = mock.Mock()
            client = self.client(reporter, sleep)
            effects = [TimeoutError("timeout"), ConnectionError("reset"), str(result)]
            with mock.patch(
                "huggingface_hub.hf_hub_download", side_effect=effects
            ) as hf_download, mock.patch.object(download.random, "random", return_value=0.0):
                actual = client.download("archives/Smoke/0.tar", REVISION)
            self.assertEqual(actual, result)
            self.assertEqual(hf_download.call_count, 3)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [1.5, 3.0])

        class PermanentHTTPError(RuntimeError):
            def __init__(self) -> None:
                super().__init__("not found")
                self.response = SimpleNamespace(status_code=404)

        reporter = CollectingReporter()
        sleep = mock.Mock()
        client = self.client(reporter, sleep)
        with mock.patch(
            "huggingface_hub.hf_hub_download", side_effect=PermanentHTTPError()
        ) as hf_download:
            with self.assertRaisesRegex(download.DownloadError, "Cannot download"):
                client.download("archives/Smoke/missing.tar", REVISION)
        self.assertEqual(hf_download.call_count, 1)
        sleep.assert_not_called()

    def test_corrupt_cached_object_is_force_downloaded_once_and_reverified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corrupt = root / "corrupt.tar"
            repaired = root / "repaired.tar"
            corrupt.write_bytes(b"xxxx")
            repaired.write_bytes(b"good")
            remote = download.RemoteFile(
                "archives/Smoke/0.tar",
                4,
                hashlib.sha256(b"good").hexdigest(),
            )

            class RepairHub:
                def __init__(self) -> None:
                    self.calls: list[bool] = []

                def cached_path(self, _path, _revision):
                    return corrupt

                def download(self, _path, _revision, *, force=False):
                    self.calls.append(force)
                    return repaired if force else corrupt

            hub = RepairHub()
            reporter = CollectingReporter()
            actual = download.download_verified(
                hub, remote, REVISION, "test tar", reporter
            )

            self.assertEqual(actual, repaired)
            self.assertEqual(hub.calls, [False, True])
            self.assertTrue(any(stage == "repair" for stage, _ in reporter.events))


if __name__ == "__main__":
    unittest.main()
