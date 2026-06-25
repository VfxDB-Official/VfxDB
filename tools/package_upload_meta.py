#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_REPO_ID = "ryogishiki/VfxDB"
DEFAULT_REMOTE_PATH = "meta/vfxdb_meta.tar.zst"
SKIP_CATEGORY_PREFIXES = ("_",)
SKIP_CATEGORY_SUBSTRINGS = ("rename",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package VfxDB meta JSON files into one compressed tar and optionally upload it to Hugging Face."
    )
    parser.add_argument("--data-root", required=True, help="Local VDBSet root containing dataset_index.json.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help=f"Hugging Face dataset repo id. Default: {DEFAULT_REPO_ID}")
    parser.add_argument(
        "--path-in-repo",
        default=DEFAULT_REMOTE_PATH,
        help=f"Remote path for the tar archive. Default: {DEFAULT_REMOTE_PATH}",
    )
    parser.add_argument(
        "--work-dir",
        default="/tmp/vfxdb_meta_pack",
        help="Directory for the generated archive, manifest, and file list.",
    )
    parser.add_argument("--archive-name", default="vfxdb_meta.tar.zst", help="Local archive filename under --work-dir.")
    parser.add_argument("--compression-level", type=int, default=19, help="zstd compression level.")
    parser.add_argument("--categories", nargs="*", default=None, help="Optional category allowlist.")
    parser.add_argument("--strict-missing", action="store_true", help="Fail if category_index references missing meta files.")
    parser.add_argument("--skip-upload", action="store_true", help="Only build the archive; do not call hf upload.")
    parser.add_argument("--dry-run", action="store_true", help="Scan and print the plan without writing archive or uploading.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing local archive in --work-dir.",
    )
    parser.add_argument(
        "--proxy",
        default=os.environ.get("VFXDB_HF_PROXY", ""),
        help="Optional HTTP(S) proxy for hf upload, e.g. http://127.0.0.1:7890.",
    )
    parser.add_argument("--revision", default="", help="Optional HF branch/revision to upload to.")
    parser.add_argument("--commit-message", default="Add VfxDB metadata archive")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def should_skip_category(category: str) -> bool:
    lowered = category.lower()
    if any(category.startswith(prefix) for prefix in SKIP_CATEGORY_PREFIXES):
        return True
    return any(part in lowered for part in SKIP_CATEGORY_SUBSTRINGS)


def dataset_categories(data_root: Path, allowlist: list[str] | None) -> tuple[list[str], list[dict]]:
    index_path = data_root / "dataset_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"dataset_index.json not found: {index_path}")

    dataset_index = load_json(index_path)
    raw_categories = dataset_index.get("categories", [])
    if not isinstance(raw_categories, list):
        raise RuntimeError(f"dataset_index.json has invalid categories: {type(raw_categories).__name__}")

    allow = set(allowlist or raw_categories)
    categories: list[str] = []
    skipped: list[dict] = []
    for category in raw_categories:
        category = str(category)
        if category not in allow:
            skipped.append({"category": category, "reason": "not_in_allowlist"})
            continue
        if should_skip_category(category):
            skipped.append({"category": category, "reason": "temporary_or_rename_category"})
            continue
        if not (data_root / category).is_dir():
            skipped.append({"category": category, "reason": "category_dir_missing"})
            continue
        categories.append(category)
    return categories, skipped


def collect_meta_files(data_root: Path, categories: list[str], strict_missing: bool) -> tuple[list[str], dict]:
    rel_paths: set[str] = set()
    stats: dict[str, dict] = {}

    for category in categories:
        category_dir = data_root / category
        index_path = category_dir / "category_index.json"
        item = {
            "samples": 0,
            "meta_refs": 0,
            "meta_existing": 0,
            "meta_missing": 0,
            "included_files": 0,
        }
        if not index_path.is_file():
            item["missing_category_index"] = True
            stats[category] = item
            continue

        category_index = load_json(index_path)
        samples = category_index.get("samples", [])
        if not isinstance(samples, list):
            raise RuntimeError(f"{index_path} has invalid samples: {type(samples).__name__}")

        item["samples"] = len(samples)
        missing: list[str] = []
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            meta_path = sample.get("meta_path")
            if not meta_path:
                continue
            meta_path = str(meta_path).replace("\\", "/").lstrip("/")
            item["meta_refs"] += 1
            abs_meta = category_dir / meta_path
            if abs_meta.is_file():
                item["meta_existing"] += 1
                rel_paths.add(f"{category}/{meta_path}")
            else:
                item["meta_missing"] += 1
                missing.append(f"{category}/{meta_path}")

        item["included_files"] = sum(1 for rel in rel_paths if rel.startswith(f"{category}/"))
        if strict_missing and missing:
            preview = "\n".join(missing[:20])
            raise RuntimeError(
                f"{category} has {len(missing)} missing meta files referenced by category_index.json.\n"
                f"First missing paths:\n{preview}"
            )
        stats[category] = item

    return sorted(rel_paths), stats


def write_plan(work_dir: Path, rel_paths: list[str], manifest: dict) -> tuple[Path, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    file_list = work_dir / "meta_files.txt"
    manifest_path = work_dir / "meta_manifest.json"
    file_list.write_text("\n".join(rel_paths) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return file_list, manifest_path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_archive(data_root: Path, work_dir: Path, archive_name: str, file_list: Path, overwrite: bool, level: int) -> Path:
    archive = work_dir / archive_name
    if archive.exists() and not overwrite:
        raise FileExistsError(f"archive already exists: {archive}. Pass --overwrite to replace it.")
    if archive.suffixes[-2:] != [".tar", ".zst"] and archive.suffix != ".tgz":
        raise RuntimeError("archive name must end with .tar.zst or .tgz")

    if archive.suffix == ".tgz":
        cmd = [
            "tar",
            "-czf",
            str(archive),
            "-C",
            str(data_root),
            "--files-from",
            str(file_list),
            "-C",
            str(work_dir),
            "meta_manifest.json",
        ]
    else:
        zstd = shutil.which("zstd")
        if zstd is None:
            raise RuntimeError("zstd not found. Install zstd or use --archive-name vfxdb_meta.tgz")
        level = max(1, min(int(level), 22))
        cmd = [
            "tar",
            "-I",
            f"{zstd} -T0 -{level}",
            "-cf",
            str(archive),
            "-C",
            str(data_root),
            "--files-from",
            str(file_list),
            "-C",
            str(work_dir),
            "meta_manifest.json",
        ]
    print("[tar]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return archive


def upload_archive(repo_id: str, archive: Path, path_in_repo: str, commit_message: str, revision: str, proxy: str) -> None:
    cmd = [
        "hf",
        "upload",
        repo_id,
        str(archive),
        path_in_repo,
        "--repo-type",
        "dataset",
        "--commit-message",
        commit_message,
    ]
    if revision:
        cmd.extend(["--revision", revision])

    env = os.environ.copy()
    if proxy:
        env["http_proxy"] = proxy
        env["https_proxy"] = proxy
        env["HTTP_PROXY"] = proxy
        env["HTTPS_PROXY"] = proxy
        env["all_proxy"] = proxy
        env["ALL_PROXY"] = proxy

    print("[hf]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()

    categories, skipped_categories = dataset_categories(data_root, args.categories)
    rel_paths, stats = collect_meta_files(data_root, categories, strict_missing=bool(args.strict_missing))
    if not rel_paths:
        raise RuntimeError("No existing meta files found to package.")

    manifest = {
        "created_at_unix": int(time.time()),
        "data_root": str(data_root),
        "repo_id": args.repo_id,
        "path_in_repo": args.path_in_repo,
        "archive_name": args.archive_name,
        "category_count": len(categories),
        "file_count": len(rel_paths),
        "categories": categories,
        "skipped_categories": skipped_categories,
        "stats": stats,
        "layout": "Archive paths are relative to the VDBSet root, e.g. Category/index/file.json.",
    }

    total_missing = sum(int(v.get("meta_missing", 0)) for v in stats.values())
    print(f"[scan] categories={len(categories)} files={len(rel_paths)} missing_refs={total_missing}")
    if skipped_categories:
        print("[scan] skipped:", ", ".join(f"{x['category']}:{x['reason']}" for x in skipped_categories))

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    file_list, manifest_path = write_plan(work_dir, rel_paths, manifest)
    archive = build_archive(data_root, work_dir, args.archive_name, file_list, args.overwrite, args.compression_level)
    digest = sha256_file(archive)

    manifest["archive_sha256"] = digest
    manifest["archive_size_bytes"] = archive.stat().st_size
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[archive] {archive}")
    print(f"[archive] sha256={digest}")
    print(f"[manifest] {manifest_path}")

    if args.skip_upload:
        print("[upload] skipped")
        return 0

    upload_archive(args.repo_id, archive, args.path_in_repo, args.commit_message, args.revision, args.proxy)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
