#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


def _blob(size: int, category_id: int, seq_id: int, frame_id: int) -> np.ndarray:
    grid = np.linspace(-1.0, 1.0, int(size), dtype=np.float32)
    z, y, x = np.meshgrid(grid, grid, grid, indexing="ij")
    angle = 0.45 * frame_id + 0.25 * seq_id + 0.8 * category_id
    cx = 0.35 * math.sin(angle)
    cy = 0.35 * math.cos(angle * 0.7)
    cz = -0.25 + 0.12 * frame_id
    sigma = 0.28 + 0.03 * category_id
    vol = np.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2.0 * sigma * sigma))
    if category_id % 2 == 1:
        vol = vol + 0.35 * np.exp(-((x + cx * 0.7) ** 2 + (y - 0.2) ** 2 + (z + 0.1) ** 2) / 0.18)
    return np.clip(vol, 0.0, 1.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny VDBSet-like NPZ dataset.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-categories", type=int, default=2)
    parser.add_argument("--num-sequences", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--size", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    categories = [f"class{i:03d}" for i in range(int(args.num_categories))]
    with (root / "dataset_index.json").open("w", encoding="utf-8") as handle:
        json.dump({"version": "dummy", "resolution": int(args.size), "categories": categories}, handle, indent=2)

    for cat_id, cat in enumerate(categories):
        cat_dir = root / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        samples = []
        for seq_id in range(int(args.num_sequences)):
            seq_dir = cat_dir / f"seq{seq_id:03d}"
            seq_dir.mkdir(parents=True, exist_ok=True)
            for frame_id in range(int(args.num_frames)):
                sid = f"{cat}_s{seq_id:03d}__n{frame_id:04d}"
                vol = _blob(int(args.size), cat_id, seq_id, frame_id)
                occ = (vol > 1e-5).astype(np.uint8)
                npz_rel = f"seq{seq_id:03d}/{sid}.npz"
                meta_rel = f"seq{seq_id:03d}/{sid}.json"
                np.savez_compressed(
                    cat_dir / npz_rel,
                    vol=vol,
                    occ=occ,
                    leaf_base=np.asarray([max(1, int(args.size) // 4)], dtype=np.int32),
                    lvl_sizes=np.asarray([max(1, int(args.size) // 4), max(1, int(args.size) // 2)], dtype=np.int32),
                )
                meta = {
                    "id": sid,
                    "category": cat,
                    "bbox_min": [0, 0, 0],
                    "bbox_max": [int(args.size) - 1, int(args.size) - 1, int(args.size) - 1],
                    "seq_bbox_min": [0, 0, 0],
                    "seq_bbox_max": [int(args.size) - 1, int(args.size) - 1, int(args.size) - 1],
                }
                with (cat_dir / meta_rel).open("w", encoding="utf-8") as handle:
                    json.dump(meta, handle, indent=2)
                samples.append({"id": sid, "npz_path": npz_rel, "meta_path": meta_rel})

        with (cat_dir / "category_index.json").open("w", encoding="utf-8") as handle:
            json.dump({"category": cat, "samples": samples}, handle, indent=2)

    print(f"Dummy dataset written to: {root}")


if __name__ == "__main__":
    main()
