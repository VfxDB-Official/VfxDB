#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath


DEFAULT_REPO_ID = "ryogishiki/VfxDB"
DEFAULT_CATEGORY = "CloudWave"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download VfxDB index files and selected VDB tar archives from Hugging Face, then extract them into a VDBSet root."
    )
    parser.add_argument("--data-root", required=True, help="Local VDBSet root to populate.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help=f"Hugging Face dataset repo id. Default: {DEFAULT_REPO_ID}")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help=f"Categories to download. Default: {DEFAULT_CATEGORY}, unless --folders is provided.",
    )
    parser.add_argument(
        "--folders",
        nargs="*",
        default=None,
        help="Explicit category folders to download, e.g. SurfaceFire:24 VortexColumn:360.",
    )
    parser.add_argument("--all-categories", action="store_true", help="Use every category listed in dataset_index.json.")
    parser.add_argument(
        "--max-samples-per-category",
        type=int,
        default=1000,
        help="Select enough folder archives to cover this many indexed samples per category. Default: 1000.",
    )
    parser.add_argument(
        "--download-dir",
        default="/tmp/vfxdb_data_download",
        help="Directory used by hf download before extraction.",
    )
    parser.add_argument("--revision", default="", help="Optional HF branch/revision to download from.")
    parser.add_argument(
        "--proxy",
        default=os.environ.get("VFXDB_HF_PROXY", ""),
        help="Optional HTTP(S) proxy for hf download, e.g. http://127.0.0.1:7890.",
    )
    parser.add_argument("--skip-meta", action="store_true", help="Do not download/extract the metadata archive.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected remote files without downloading or extracting.")
    parser.add_argument("--plan-only", action="store_true", help="Download indexes and print selected archives without downloading data tar files.")
    return parser.parse_args()


def proxy_env(proxy: str) -> dict[str, str]:
    env = os.environ.copy()
    if proxy:
        env["http_proxy"] = proxy
        env["https_proxy"] = proxy
        env["HTTP_PROXY"] = proxy
        env["HTTPS_PROXY"] = proxy
        env["all_proxy"] = proxy
        env["ALL_PROXY"] = proxy
    return env


def run_hf_download(
    repo_id: str,
    files: list[str],
    download_dir: Path,
    revision: str,
    proxy: str,
    dry_run: bool = False,
    chunk_size: int = 64,
) -> None:
    if not files:
        return
    download_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(files), chunk_size):
        chunk = files[start : start + chunk_size]
        cmd = [
            "hf",
            "download",
            repo_id,
            *chunk,
            "--repo-type",
            "dataset",
            "--local-dir",
            str(download_dir),
        ]
        if revision:
            cmd.extend(["--revision", revision])
        if dry_run:
            cmd.append("--dry-run")
        print("[hf]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, env=proxy_env(proxy))


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def copy_index_files(download_dir: Path, data_root: Path, categories: list[str]) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(download_dir / "dataset_index.json", data_root / "dataset_index.json")
    for category in categories:
        src = download_dir / category / "category_index.json"
        dst_dir = data_root / category
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / "category_index.json")


def sample_folder(sample: dict) -> str:
    folder = sample.get("folder")
    if folder is not None:
        return str(folder).strip("/")
    for key in ("vdb_path", "npz_path", "numpy_path"):
        rel = sample.get(key)
        if rel:
            return PurePosixPath(str(rel).replace("\\", "/")).parts[0]
    raise RuntimeError(f"sample has no folder or volume path: {sample}")


def parse_folder_specs(specs: list[str] | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for spec in specs or []:
        if ":" not in spec:
            raise RuntimeError(f"bad --folders spec '{spec}', expected Category:folder")
        category, folder = spec.split(":", 1)
        category = category.strip()
        folder = folder.strip().strip("/")
        if not category or not folder:
            raise RuntimeError(f"bad --folders spec '{spec}', expected Category:folder")
        out.setdefault(category, [])
        if folder not in out[category]:
            out[category].append(folder)
    return out


def selected_archive_paths(
    download_dir: Path,
    categories: list[str],
    max_samples_per_category: int,
    folder_specs: dict[str, list[str]] | None = None,
) -> tuple[list[str], dict[str, int]]:
    remote_paths: list[str] = []
    selected_counts: dict[str, int] = {}
    seen: set[str] = set()
    limit = max(0, int(max_samples_per_category))
    folder_specs = folder_specs or {}

    for category in categories:
        index_path = download_dir / category / "category_index.json"
        category_index = load_json(index_path)
        samples = category_index.get("samples", [])
        if not isinstance(samples, list) or not samples:
            raise RuntimeError(f"{index_path} has no samples")

        forced_folders = folder_specs.get(category)
        if forced_folders:
            forced = set(forced_folders)
            selected = [s for s in samples if isinstance(s, dict) and sample_folder(s) in forced]
            if not selected:
                raise RuntimeError(f"{index_path} has no samples in folders={forced_folders}")
        else:
            selected = samples if limit <= 0 else samples[:limit]

        selected_counts[category] = len(selected)
        for sample in selected:
            if not isinstance(sample, dict):
                continue
            folder = sample_folder(sample)
            remote = f"archives/{category}/{folder}.tar"
            if remote not in seen:
                seen.add(remote)
                remote_paths.append(remote)

    return remote_paths, selected_counts


def list_tar_members(archive: Path) -> list[str]:
    proc = subprocess.run(["tar", "-tf", str(archive)], check=True, text=True, stdout=subprocess.PIPE)
    members = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not members:
        raise RuntimeError(f"archive has no members: {archive}")
    return members


def validate_member_path(member: str) -> None:
    posix = PurePosixPath(member)
    if posix.is_absolute():
        raise RuntimeError(f"refusing absolute archive path: {member}")
    if any(part == ".." for part in posix.parts):
        raise RuntimeError(f"refusing parent traversal archive path: {member}")


def extract_archive(archive: Path, data_root: Path, category: str) -> None:
    members = list_tar_members(archive)
    for member in members:
        validate_member_path(member)

    category_prefix = f"{category}/"
    extract_root = data_root if all(m.startswith(category_prefix) for m in members) else data_root / category
    extract_root.mkdir(parents=True, exist_ok=True)
    cmd = ["tar", "-xf", str(archive), "-C", str(extract_root)]
    print("[tar]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def extract_archives(download_dir: Path, data_root: Path, archive_paths: list[str]) -> None:
    for remote in archive_paths:
        parts = PurePosixPath(remote).parts
        if len(parts) != 3 or parts[0] != "archives" or not parts[2].endswith(".tar"):
            raise RuntimeError(f"unexpected archive path: {remote}")
        category = parts[1]
        archive = download_dir / remote
        if not archive.is_file():
            raise FileNotFoundError(f"downloaded archive not found: {archive}")
        extract_archive(archive, data_root, category)


def run_meta_download(data_root: Path, download_dir: Path, repo_id: str, revision: str, proxy: str) -> None:
    script = Path(__file__).with_name("download_extract_meta.py")
    cmd = [
        sys.executable,
        str(script),
        "--data-root",
        str(data_root),
        "--repo-id",
        repo_id,
        "--download-dir",
        str(download_dir / "_meta"),
    ]
    if revision:
        cmd.extend(["--revision", revision])
    if proxy:
        cmd.extend(["--proxy", proxy])
    print("[meta]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    download_dir = Path(args.download_dir).expanduser().resolve()

    run_hf_download(
        repo_id=args.repo_id,
        files=["dataset_index.json"],
        download_dir=download_dir,
        revision=args.revision,
        proxy=args.proxy,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print("[dry-run] category index and archive selection require dataset_index.json locally; rerun without --dry-run.")
        return 0

    dataset_index = load_json(download_dir / "dataset_index.json")
    all_categories = [str(c) for c in dataset_index.get("categories", []) if not str(c).startswith("_")]
    folder_specs = parse_folder_specs(args.folders)
    if args.all_categories:
        categories = all_categories
    elif args.categories is not None:
        categories = [str(c) for c in args.categories]
    elif folder_specs:
        categories = list(folder_specs.keys())
    else:
        categories = [DEFAULT_CATEGORY]

    unknown = [c for c in categories if c not in all_categories]
    if unknown:
        raise RuntimeError(f"unknown categories: {unknown}. Available: {all_categories}")
    unknown_folder_cats = [c for c in folder_specs if c not in categories]
    if unknown_folder_cats:
        raise RuntimeError(f"--folders categories not selected: {unknown_folder_cats}")

    index_files = [f"{category}/category_index.json" for category in categories]
    run_hf_download(
        repo_id=args.repo_id,
        files=index_files,
        download_dir=download_dir,
        revision=args.revision,
        proxy=args.proxy,
    )
    copy_index_files(download_dir, data_root, categories)

    archive_paths, selected_counts = selected_archive_paths(
        download_dir,
        categories,
        args.max_samples_per_category,
        folder_specs=folder_specs,
    )
    print(f"[select] categories={categories}")
    print(f"[select] selected_indexed_samples={selected_counts}")
    print(f"[select] archive_count={len(archive_paths)}")
    for remote in archive_paths[:20]:
        print(f"  {remote}")
    if len(archive_paths) > 20:
        print(f"  ... {len(archive_paths) - 20} more")

    if args.plan_only:
        print("[plan-only] skipped archive download, extraction, and metadata download")
        return 0

    run_hf_download(
        repo_id=args.repo_id,
        files=archive_paths,
        download_dir=download_dir,
        revision=args.revision,
        proxy=args.proxy,
    )
    extract_archives(download_dir, data_root, archive_paths)

    if not args.skip_meta:
        run_meta_download(data_root, download_dir, args.repo_id, args.revision, args.proxy)

    print(f"[done] data_root={data_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
