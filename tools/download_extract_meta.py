#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path, PurePosixPath


DEFAULT_REPO_ID = "ryogishiki/VfxDB"
DEFAULT_REMOTE_PATH = "meta/vfxdb_meta.tar.zst"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract the VfxDB metadata archive into a local VDBSet root."
    )
    parser.add_argument("--data-root", required=True, help="Local VDBSet root to extract into.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help=f"Hugging Face dataset repo id. Default: {DEFAULT_REPO_ID}")
    parser.add_argument(
        "--path-in-repo",
        default=DEFAULT_REMOTE_PATH,
        help=f"Archive path inside the dataset repo. Default: {DEFAULT_REMOTE_PATH}",
    )
    parser.add_argument(
        "--download-dir",
        default="/tmp/vfxdb_meta_download",
        help="Directory used by hf download when --archive is not provided.",
    )
    parser.add_argument("--archive", default="", help="Use an existing local archive instead of downloading from HF.")
    parser.add_argument("--revision", default="", help="Optional HF branch/revision to download from.")
    parser.add_argument(
        "--proxy",
        default=os.environ.get("VFXDB_HF_PROXY", ""),
        help="Optional HTTP(S) proxy for hf download, e.g. http://127.0.0.1:7890.",
    )
    parser.add_argument("--list-only", action="store_true", help="Validate and list archive contents without extracting.")
    parser.add_argument("--allow-rename-paths", action="store_true", help="Allow paths containing rename/_rename.")
    return parser.parse_args()


def proxy_env(proxy: str) -> dict[str, str]:
    env = os.environ.copy()
    if proxy:
        env["http_proxy"] = proxy
        env["https_proxy"] = proxy
        env["HTTP_PROXY"] = proxy
        env["HTTPS_PROXY"] = proxy
    return env


def download_archive(repo_id: str, path_in_repo: str, download_dir: Path, revision: str, proxy: str) -> Path:
    cmd = [
        "hf",
        "download",
        repo_id,
        "--repo-type",
        "dataset",
        "--include",
        path_in_repo,
        "--local-dir",
        str(download_dir),
    ]
    if revision:
        cmd.extend(["--revision", revision])
    print("[hf]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=proxy_env(proxy))

    archive = download_dir / path_in_repo
    if not archive.is_file():
        raise FileNotFoundError(f"downloaded archive not found: {archive}")
    return archive


def tar_filter_args(archive: Path) -> list[str]:
    name = archive.name
    if name.endswith(".tar.zst"):
        zstd = shutil.which("zstd")
        if zstd is None:
            raise RuntimeError("zstd not found. Install zstd to extract .tar.zst archives.")
        return ["-I", zstd]
    if name.endswith(".tgz") or name.endswith(".tar.gz"):
        return ["-z"]
    if name.endswith(".tar"):
        return []
    raise RuntimeError(f"unsupported archive extension: {archive}")


def list_members(archive: Path) -> list[str]:
    cmd = ["tar", *tar_filter_args(archive), "-tf", str(archive)]
    print("[tar]", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)
    members = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not members:
        raise RuntimeError(f"archive has no members: {archive}")
    return members


def validate_member_path(member: str, allow_rename_paths: bool) -> None:
    posix = PurePosixPath(member)
    if posix.is_absolute():
        raise RuntimeError(f"refusing absolute archive path: {member}")
    if any(part == ".." for part in posix.parts):
        raise RuntimeError(f"refusing parent traversal archive path: {member}")
    lowered = member.lower()
    if not allow_rename_paths and ("rename" in lowered or "_rename" in lowered):
        raise RuntimeError(f"refusing temporary/rename archive path: {member}")


def validate_members(members: list[str], allow_rename_paths: bool) -> None:
    for member in members:
        validate_member_path(member, allow_rename_paths)


def extract_archive(archive: Path, data_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    cmd = ["tar", *tar_filter_args(archive), "-xf", str(archive), "-C", str(data_root)]
    print("[tar]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()

    if args.archive:
        archive = Path(args.archive).expanduser().resolve()
        if not archive.is_file():
            raise FileNotFoundError(f"archive not found: {archive}")
    else:
        archive = download_archive(
            repo_id=args.repo_id,
            path_in_repo=args.path_in_repo,
            download_dir=Path(args.download_dir).expanduser().resolve(),
            revision=args.revision,
            proxy=args.proxy,
        )

    members = list_members(archive)
    validate_members(members, allow_rename_paths=bool(args.allow_rename_paths))
    print(f"[archive] {archive}")
    print(f"[archive] members={len(members)}")
    print("[archive] first members:")
    for member in members[:10]:
        print(f"  {member}")

    if args.list_only:
        print("[extract] skipped")
        return 0

    extract_archive(archive, data_root)
    print(f"[extract] done: {data_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
