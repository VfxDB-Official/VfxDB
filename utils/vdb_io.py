# utils/vdb_io.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, List, Tuple
import os
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import openvdb as vdb
    HAS_OPENVDB = True
except Exception:
    vdb = None
    HAS_OPENVDB = False


def upsample_volume(vol: np.ndarray, target_size: int = 128) -> np.ndarray:
    assert vol.ndim == 3
    t = torch.from_numpy(vol.astype(np.float32))[None, None]
    with torch.no_grad():
        t_up = F.interpolate(t, size=(target_size, target_size, target_size), mode="trilinear", align_corners=False)
    return t_up[0, 0].cpu().numpy()


def upsample_mask(mask: np.ndarray, target_size: int = 128, mode: str = "nearest") -> np.ndarray:
    assert mask.ndim == 3
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    with torch.no_grad():
        if mode == "trilinear":
            t_up = F.interpolate(t, size=(target_size, target_size, target_size), mode="trilinear", align_corners=False)
        else:
            t_up = F.interpolate(t, size=(target_size, target_size, target_size), mode="nearest")
    return t_up[0, 0].cpu().numpy()


def save_volume_as_vdb(
    vol: np.ndarray,
    out_path: str,
    threshold: float = 1e-5,
    grid_name: str = "density",
    dense_write: bool = False,
):
    """
    vol: dense (D,H,W) float32
    """
    if not HAS_OPENVDB:
        return

    vol = vol.astype(np.float32)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if dense_write:
        grid = vdb.FloatGrid()
        grid.name = grid_name
        grid.background = 0.0
        grid.copyFromArray(vol)
        vdb.write(out_path, grids=[grid])
        return

    grid = vdb.FloatGrid()
    grid.name = grid_name
    grid.background = 0.0

    mask = np.abs(vol) > float(threshold)
    if not mask.any():
        vdb.write(out_path, grids=[grid])
        return

    coords = np.argwhere(mask)  # (N,3) in (z,y,x)
    vals = vol[mask]

    acc = grid.getAccessor()
    for (z, y, x), v in zip(coords, vals):
        acc.setValueOn((int(x), int(y), int(z)), float(v))

    try:
        grid.prune()
    except Exception:
        pass

    # mat_grid_gt = np.array([
    #     [0.2, 0.0, 0.0, -16.7],
    #     [0.0, 0.2, 0.0,  0.1],
    #     [0.0, 0.0, 0.2, -18.5],
    #     [0.0, 0.0, 0.0,  1.0]
    # ])

    # # 2. 定义 GT 的 Object 旋转矩阵 (X轴旋转 90度)
    # mat_obj_rot = np.array([
    #     [1.0,  0.0,  0.0, 0.0],
    #     [0.0,  0.0, -1.0, 0.0],
    #     [0.0,  1.0,  0.0, 0.0],
    #     [0.0,  0.0,  0.0, 1.0]
    # ])

    # # 3. 计算最终矩阵 (Blender 坐标系下)
    # final_mat_np = mat_obj_rot

    # # 4. 【关键步骤】转置矩阵以适配 OpenVDB 的格式
    # # OpenVDB 期望位移在最后一行 (Row-major)，而我们计算在最后一列。
    # final_mat_list = final_mat_np.T.tolist()

    # 5. 创建并应用变换
    # grid.transform = vdb.createLinearTransform(final_mat_list)

    vdb.write(out_path, grids=[grid])


def render_vdb_to_jpg(
    vdb_path: str,
    jpg_path: str,
    quality: int = 85,
    res: str = "512x512",
):
    """
    Uses external `vdb_render` to render vdb -> ppm -> jpg.
    """
    work_dir = os.path.dirname(vdb_path) or "."
    vdb_name = os.path.basename(vdb_path)

    ppm_path = os.path.splitext(jpg_path)[0] + ".ppm"
    ppm_name = os.path.basename(ppm_path)
    jpg_name = os.path.basename(jpg_path)

    cmd = [
        "vdb_render",
        vdb_name,
        ppm_name,
        "-res", res,
        "-samples", "1",
        "-translate", "256,256,256",
        "-lookat", "0,0,0",
        "-v",
        "-scatter", "2.0,2.0,2.0",
        "-absorb", "0.18,0.10,0.06",
        "-gain", "1.1",
    ]
    subprocess.run(cmd, check=True, cwd=work_dir)

    ppm_full = os.path.join(work_dir, ppm_name)
    jpg_full = os.path.join(work_dir, jpg_name)

    with Image.open(ppm_full) as img:
        img = img.convert("RGB")
        img.save(jpg_full, "JPEG", quality=int(quality))

    try:
        os.remove(ppm_full)
    except Exception:
        pass


def try_open_rgb(path: str) -> Optional[Image.Image]:
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:
        return None


# ============================================================================
# VDB helpers (from eval_mc_one_stage.py)
# ============================================================================

def maybe_write_vdb_and_render(
    vol_export: np.ndarray,
    vdb_path: str,
    jpg_path: str,
    vdb_thr: float,
    dense_write: bool,
    no_vdb: bool,
    no_vdb_render: bool,
) -> Optional[Image.Image]:
    """
    Write volume as VDB and optionally render to image.

    Only writes if OpenVDB is available and volume is non-empty.

    Args:
        vol_export: Volume to export
        vdb_path: Path to save VDB file
        jpg_path: Path to save rendered JPG
        vdb_thr: Threshold for non-empty voxels
        dense_write: Write as dense or sparse
        no_vdb: Skip VDB writing entirely
        no_vdb_render: Skip rendering

    Returns:
        PIL Image of rendered VDB, or None if not produced
    """
    if no_vdb or (not HAS_OPENVDB):
        return None
    v = vol_export
    vmax = float(np.max(v))
    nnz = int(np.count_nonzero(v > float(vdb_thr)))
    if nnz == 0:
        print(
            f"[VDB][SKIP] empty grid: vmax={vmax:.6g} nnz={nnz} thr={float(vdb_thr):.6g} | "
            f"{os.path.basename(vdb_path)}"
        )
        return None
    save_volume_as_vdb(vol_export, vdb_path, threshold=float(vdb_thr), dense_write=bool(dense_write))
    if no_vdb_render:
        return None
    try:
        render_vdb_to_jpg(vdb_path, jpg_path)
    except Exception as e:
        print("[VDB][WARN] vdb_render failed, skip:", repr(e))
        return None
    return try_open_rgb(jpg_path)


def maybe_add_vdb_render(
    items: List[Tuple[str, Image.Image]],
    vol_export: np.ndarray,
    vdb_path: str,
    jpg_path: str,
    vdb_thr: float,
    dense_write: bool,
    no_vdb: bool,
    no_vdb_render: bool,
    label: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Write VDB, render it, and add to items list if successful.

    Args:
        items: List to add rendered image to
        vol_export: Volume to export
        vdb_path: Path for VDB file
        jpg_path: Path for rendered JPG
        vdb_thr: Threshold for non-empty voxels
        dense_write: Write as dense or sparse
        no_vdb: Skip VDB writing
        no_vdb_render: Skip rendering
        label: Label for rendered image in items list

    Returns:
        Tuple of (vdb_filename, render_filename) or (None, None)
    """
    vdb_im = maybe_write_vdb_and_render(
        vol_export,
        vdb_path,
        jpg_path,
        vdb_thr=vdb_thr,
        dense_write=bool(dense_write),
        no_vdb=bool(no_vdb),
        no_vdb_render=bool(no_vdb_render),
    )
    vdb_render = None
    if vdb_im is not None:
        items.insert(0, (label, vdb_im))
        vdb_render = os.path.basename(jpg_path)

    vdb_name = None
    if not bool(no_vdb):
        vdb_name = os.path.basename(vdb_path)

    return vdb_name, vdb_render
