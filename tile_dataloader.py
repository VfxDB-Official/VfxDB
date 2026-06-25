#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VDB/NPZ dataset for diffusion training.

核心修复：
1) train/val/test split 改为 “按序列（同类别 + 同序列文件夹）分组切分”，避免 sequence leakage。
2) temporal prev 索引同样按该 seq_key 构建，避免 prev 串到别的序列/类别。
3) 修复 max_train/max_val/max_test 条件判断 bug（or -> and）。
4) __getitem__ 不再缓存首个样本的后缀，避免混数据类型时读错。

假设目录结构类似：
root/
  Smoke/
    seqA/
      hash__n000.npz
      hash__n001.npz
    seqB/
      ...
  Fire/
    seqC/
      ...

如果你实际是：
root/Smoke/npz/*.npz（没有序列文件夹），那“同序列文件夹”会退化成整类一个序列，
这时你应该改成用 hashid 作为 seq_key（见下方 _seqkey_from_record 的注释）。
"""

import os
import time
import re
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable, Any, Union
from collections import defaultdict
from utils.tools import hard_log
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
try:
    import openvdb as vdb
    HAS_OPENVDB = True
except ImportError:
    vdb = None
    HAS_OPENVDB = False

try:
    import vdb_ext
    HAS_VDB_EXT = True
    print(f"using vdb_ext of :{vdb_ext.__file__}")
except ImportError:
    vdb_ext = None
    HAS_VDB_EXT = False

# ===================== 数据记录结构 =====================

@dataclass
class SampleRecord:
    id: str
    category: str

    data_path: str
    meta_path: Optional[str]

    abs_data_path: str
    abs_meta_path: Optional[str]

    # NEW: 记录该类别的根目录，用于计算“序列文件夹相对路径”
    abs_cat_dir: str


# ===================== 小工具 =====================

def _npz_get(z: Any, key: str, default=None):
    try:
        return z[key]
    except Exception:
        return default

def _as_int3(x):
    a = np.asarray(x, dtype=np.int64).reshape(-1)
    if a.size != 3:
        raise ValueError(f"bbox must have 3 elems, got {a}")
    return a

def _read_seq_bbox_from_meta(meta_json: dict):
    """
    兼容不同 key 命名：优先找 seq/sequence bbox；找不到就退化用 bbox_min/max
    （如果你的 meta 里就是序列 bbox 存在 bbox_min/max，那也 OK）
    """
    cand = [
        ("seq_bbox_min", "seq_bbox_max"),
        ("sequence_bbox_min", "sequence_bbox_max"),
        ("seq_bmin", "seq_bmax"),
        ("bbox_min", "bbox_max"),
    ]
    for kmin, kmax in cand:
        if kmin in meta_json and kmax in meta_json:
            return _as_int3(meta_json[kmin]), _as_int3(meta_json[kmax])
    return None


# 解析 hashid__nXXX（支持 .vdb/.npz/.npy 等）
_RE_SID_FRAME = re.compile(r"^(?P<sid>.+?)__n(?P<f>\d+)$", re.IGNORECASE)

def _parse_sid_and_frame_from_filename(path: str) -> Tuple[str, int]:
    base = os.path.basename(path)
    name, _ext = os.path.splitext(base)
    m = _RE_SID_FRAME.search(name)
    if m is None:
        return name, 0
    return m.group("sid"), int(m.group("f"))


def _seqkey_from_record(rec: SampleRecord) -> Tuple[str, int]:
    """
    你要求的“同类别 + 同序列文件夹算同一个序列”的 seq_key 定义。

    seq_folder_rel = 相对 category 根目录的父目录路径（例如 seqA / shots/seqA 等）
    seq_key = f"{category}/{seq_folder_rel}"

    frame: 从文件名解析 __nXXX

    ⚠️ 如果你的数据没有序列文件夹（全部在 category/npz 下），那 seq_folder_rel 会变成 'npz'，
       导致整类只有一个 seq_key。这时建议改成：
       seq_key = f"{category}/{sid}"  # 用 hashid 作为序列 key
    """
    sid, frame = _parse_sid_and_frame_from_filename(rec.abs_data_path)

    parent_dir = os.path.dirname(rec.abs_data_path)
    try:
        seq_folder_rel = os.path.relpath(parent_dir, rec.abs_cat_dir)
    except Exception:
        seq_folder_rel = parent_dir  # fallback: abs path

    seq_folder_rel = seq_folder_rel.replace("\\", "/")  # Windows 兼容

    # 如果你担心 'npz' / 'vdb' 这种平铺目录导致整类变成一个序列，可以在这里自动 fallback：
    if seq_folder_rel in (".", "npz", "vdb", "npy", "numpy", "data"):
        seq_key = f"{rec.category}/{sid}"
    else:
        seq_key = f"{rec.category}/{seq_folder_rel}"

    return seq_key, frame


# ===================== OpenVDB 读入（可选） =====================

def normal_to_cube(vol: torch.Tensor) -> torch.Tensor:
    """把 (1,D,H,W) 居中填充到 S x S x S, S = max(D,H,W), 返回 (1,S,S,S)"""
    if vol.dim() >= 3:
        D, H, W = vol.shape[-3:]
    else:
        raise ValueError(f"Invalid volume shape: {vol.shape}")
    S = int(max(D, H, W))
    cube = torch.zeros((S, S, S), dtype=vol.dtype, device=vol.device)

    ox = (S - D) // 2
    oy = (S - H) // 2
    oz = (S - W) // 2

    cube[ox:ox + D, oy:oy + H, oz:oz + W] = vol
    return cube[None, ...]


def cube_rescale(vol: torch.Tensor, size: int = 32) -> torch.Tensor:
    """使用 PyTorch 三线性插值缩放到 (size, size, size)"""
    with torch.no_grad():
        t_res = F.interpolate(
            vol.unsqueeze(0),
            size=(size, size, size),
            mode="trilinear",
            align_corners=False,
        )
    return t_res.squeeze(0)

def _cap_by_seq_groups(samples: List[SampleRecord], max_n: Optional[int], seed: int) -> List[SampleRecord]:
    """按 seq_key 成组截断：不允许把一个序列砍成碎片。"""
    if max_n is None or max_n <= 0 or len(samples) <= max_n:
        return samples

    from collections import defaultdict
    by_seq = defaultdict(list)
    for r in samples:
        seq_key, _ = _seqkey_from_record(r)
        by_seq[seq_key].append(r)

    groups = list(by_seq.values())
    rng = random.Random(seed)
    rng.shuffle(groups)

    out: List[SampleRecord] = []
    for g in groups:
        if len(out) + len(g) > max_n:
            break
        out.extend(g)

    rng.shuffle(out)
    return out

# ===================== 主 Dataset =====================

def _balance_by_seq_groups_equal_per_class(
    samples: List[SampleRecord],
    seed: int,
    mode: str = "oversample",                 # "oversample" | "undersample"
    target_seq_groups_per_class: Optional[int] = None,
    max_seq_groups_per_class: Optional[int] = None,
    shuffle_within_group: bool = False,       # True: 组内帧也打散；False: 保持原顺序
) -> List[SampleRecord]:
    """
    以“序列组”为单位做 class balance：不会把一个序列砍成碎片。
    - 每个类别先拆成若干 seq_group（同 category + 同 seq_key）
    - 再让每个类别拥有相同数量的 seq_group（过采样/欠采样）
    - 最后把选中的 seq_group 展开回 sample 列表
    """
    if not samples:
        return samples

    mode = str(mode).lower()
    assert mode in ("oversample", "undersample")

    rng = random.Random(seed)

    # cat -> seq_key -> list[SampleRecord]
    by_cat_seq: Dict[str, Dict[str, List[SampleRecord]]] = defaultdict(lambda: defaultdict(list))
    for r in samples:
        seq_key, _ = _seqkey_from_record(r)
        by_cat_seq[r.category][seq_key].append(r)

    cats = sorted(by_cat_seq.keys())

    # 每个 cat 的 seq_groups 列表（每个元素是一整个序列的所有帧）
    cat_groups: Dict[str, List[List[SampleRecord]]] = {}
    for c in cats:
        groups = list(by_cat_seq[c].values())
        # 组内按 frame 排序，保证 temporal 一致
        for g in groups:
            g.sort(key=lambda rec: _seqkey_from_record(rec)[1])
            if shuffle_within_group:
                rng.shuffle(g)
        rng.shuffle(groups)
        cat_groups[c] = groups

    group_counts = {c: len(cat_groups[c]) for c in cats}
    if target_seq_groups_per_class is None:
        target = max(group_counts.values()) if mode == "oversample" else min(group_counts.values())
    else:
        target = int(target_seq_groups_per_class)

    if max_seq_groups_per_class is not None:
        target = min(target, int(max_seq_groups_per_class))
    target = max(1, target)

    out: List[SampleRecord] = []
    for c in cats:
        groups = cat_groups[c]
        if not groups:
            continue

        if mode == "undersample":
            picked = groups[: min(len(groups), target)]
        else:
            if len(groups) >= target:
                picked = groups[:target]
            else:
                picked = list(groups)
                need = target - len(groups)
                picked.extend([rng.choice(groups) for _ in range(need)])

        # 展开：整个序列组全拿出来
        for g in picked:
            out.extend(g)

    rng.shuffle(out)
    return out

class VolumeMultiDataset(Dataset):
    def __init__(
        self,
        samples: List[SampleRecord],
        cat_to_idx: Dict[str, int],
        return_meta: bool = False,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        return_lvl_maps: bool = False,
        occ_eps_fallback: float = 1e-5,
        vol_size: int = 32,

        # temporal
        prev_k: int = 0,
        prev_policy: str = "nearest",
        return_prev_occ: bool = False,
        prev_bbox_mode: str = "seq",   # "seq" | "frame"
        transform_included: Union[bool, str] = "auto",
    ) -> None:
        super().__init__()
        self.samples = samples
        self.cat_to_idx = cat_to_idx
        self.return_meta = return_meta
        self.transform = transform
        self.prev_k = int(prev_k)
        self.return_lvl_maps = return_lvl_maps
        self.occ_eps_fallback = float(occ_eps_fallback)
        self.vol_size = int(vol_size)
        if isinstance(transform_included, str):
            m = transform_included.lower().strip()
            if m not in ("auto", "force_on", "force_off"):
                print(f"[WARN] bad transform_included={transform_included}, fallback to auto")
                m = "auto"
            self.transform_included_mode = m
        else:
            self.transform_included_mode = "force_on" if bool(transform_included) else "force_off"
        self.prev_bbox_mode = str(prev_bbox_mode).lower().strip()
        if self.prev_bbox_mode not in ("seq", "frame"):
            raise ValueError(f"prev_bbox_mode must be 'seq' or 'frame', got {prev_bbox_mode}")
        if self.prev_k > 0 and self.prev_bbox_mode == "frame":
            print(f"[WARN] prev_bbox_mode='frame' may lead to inconsistent bbox for prev samples")
            # self.prev_bbox_mode = "seq"
        elif self.prev_k == 0 and self.prev_bbox_mode == "seq":
            print(f"[WARN] prev_k=0 makes prev_bbox_mode='seq' meaningless")
            # self.prev_bbox_mode = "frame"

        # 兼容旧逻辑：你其他路径如果还读 self.transform_included，这里给一个明确 bool
        self.transform_included = (self.transform_included_mode == "force_on")
        self._seq_bbox_minmax: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._seq_need_include: Dict[str, bool] = {}
        self.prev_k = int(prev_k)
        self.prev_policy = str(prev_policy).lower()
        self.return_prev_occ = bool(return_prev_occ)

        if self.prev_policy not in ("nearest", "clamp0", "zero"):
            print(f"[WARN] unknown prev_policy={self.prev_policy}, fallback to nearest")
            self.prev_policy = "nearest"

        # ---- build seq index on *this split* ----
        from collections import defaultdict
        tmp = defaultdict(list)  # seq_key -> list[(frame, idx)]

        for i, rec in enumerate(self.samples):
            seq_key, f = _seqkey_from_record(rec)
            tmp[seq_key].append((int(f), int(i)))

        self._seq_to_frames: Dict[str, List[int]] = {}
        self._seqframe_to_index: Dict[Tuple[str, int], int] = {}

        for seq_key, lst in tmp.items():
            lst.sort(key=lambda x: x[0])
            frames = [f for f, _i in lst]
            self._seq_to_frames[seq_key] = frames
            for f, _i in lst:
                self._seqframe_to_index.setdefault((seq_key, int(f)), int(_i))

        for rec in self.samples:
            seq_key, _ = _seqkey_from_record(rec)
            if seq_key in self._seq_bbox_minmax:
                continue

            bminmax = None
            if rec.abs_meta_path is not None and os.path.exists(rec.abs_meta_path):
                try:
                    with open(rec.abs_meta_path, "r", encoding="utf-8") as f:
                        mj = json.load(f)
                    bminmax = _read_seq_bbox_from_meta(mj)
                except Exception:
                    bminmax = None

            if bminmax is None:
                # 没有 meta / meta 没 bbox：让 vdb_ext_get fallback 到 file bbox（bbox_min/max 传 None）
                continue

            seq_bmin, seq_bmax = bminmax
            self._seq_bbox_minmax[seq_key] = (seq_bmin, seq_bmax)

            # 方案二自动判定规则：任意维度 < 0 就认为需要 include
            self._seq_need_include[seq_key] = bool((seq_bmin < 0).any())
    def __len__(self) -> int:
        return len(self.samples)

    # ---------- core read (no temporal) ----------

    def _get_core(self, idx: int) -> Dict[str, Any]:
        rec = self.samples[idx]
        ext = os.path.splitext(rec.abs_data_path)[1].lower()

        if ext in (".npz", ".npy"):
            return self.np_get(idx)
        if ext == ".vdb":
            vdb_ex_get_batch = self.vdb_ext_get(idx)
            # vdb_get_batch = self.vdb_get(idx)
            # diff = vdb_get_batch["volume"] - vdb_ex_get_batch["volume"]
            # print(f"diff.abs().std = {diff.abs().std():.6f}, mean = {diff.abs().mean():.6f}")
            # if diff.abs().std() > 1e-3 :
            #     print(f"vdb_get_batch volume: {vdb_get_batch['volume'].abs().std():.6f}, mean = {vdb_get_batch['volume'].abs().mean():.6f}")
            #     print(f"vdb_ex_get_batch volume: {vdb_ex_get_batch['volume'].abs().std():.6f}, mean = {vdb_ex_get_batch['volume'].abs().mean():.6f}")
            #     print(f"transform_included: {self.transform_included}, pad_to_cube: {getattr(self, 'pad_to_cube', True)}")
            #     print(f"[WARN] vdb_get_batch diff.abs().std > 1e-3: {diff.abs().std():.6f} in path {rec.abs_data_path}")
            return vdb_ex_get_batch
        raise ValueError(f"unknown datatype ext={ext}, path={rec.abs_data_path}")

    def np_get(self, idx: int) -> Dict[str, Any]:
        rec = self.samples[idx]

        if not os.path.exists(rec.abs_data_path):
            raise FileNotFoundError(f"npz not found: {rec.abs_data_path}")

        z = np.load(rec.abs_data_path, allow_pickle=False)

        vol = z if isinstance(z, np.ndarray) else _npz_get(z, "vol", None)
        if vol is None:
            raise KeyError(f"npz missing key 'vol': {rec.abs_data_path}")
        vol = np.asarray(vol, dtype=np.float32)
        if vol.ndim != 3:
            raise ValueError(f"Expected vol 3D, got {vol.shape} at {rec.abs_data_path}")

        T = int(vol.shape[0])
        if vol.shape[1] != T or vol.shape[2] != T:
            raise ValueError(f"vol must be cubic, got {vol.shape} at {rec.abs_data_path}")

        occ = _npz_get(z, "occ", None)
        if occ is None:
            occ = (np.abs(vol) > self.occ_eps_fallback).astype(np.uint8)
        else:
            occ = np.asarray(occ).astype(np.uint8)
            if occ.shape != vol.shape:
                raise ValueError(f"occ shape mismatch: occ={occ.shape}, vol={vol.shape} at {rec.abs_data_path}")

        leaf_base = _npz_get(z, "leaf_base", None)
        if leaf_base is None:
            leaf_base = np.asarray([T // 8], dtype=np.int32)
        leaf_base_i = int(np.asarray(leaf_base).reshape(-1)[0])

        lvl_sizes = _npz_get(z, "lvl_sizes", None)
        if lvl_sizes is None:
            lvl_sizes = np.asarray([leaf_base_i, 2 * leaf_base_i, 4 * leaf_base_i], dtype=np.int32)
        lvl_sizes = np.asarray(lvl_sizes).astype(np.int32).reshape(-1)

        vol_t = torch.from_numpy(vol)[None, ...]  # [1,T,T,T]
        occ_t = torch.from_numpy(occ.astype(np.float32))[None, ...]

        if self.transform is not None:
            vol_t = self.transform(vol_t)

        cat_id = self.cat_to_idx[rec.category]
        sample: Dict[str, Any] = {
            "volume": vol_t,
            "occ": occ_t,
            "leaf_base": leaf_base_i,
            "lvl_sizes": torch.from_numpy(lvl_sizes.astype(np.int64)),
            "id": rec.id,
            "category": rec.category,
            "category_id": int(cat_id),
        }

        if self.return_lvl_maps:
            lvl_maps: Dict[str, torch.Tensor] = {}
            if isinstance(z, np.ndarray):
                vt = vol_t.unsqueeze(1)
                ot = occ_t.unsqueeze(1)
                for bs in lvl_sizes.tolist():
                    if bs <= 0 or (T % bs) != 0:
                        continue
                    m = F.avg_pool3d(vt, kernel_size=bs, stride=bs)
                    o = F.max_pool3d(ot, kernel_size=bs, stride=bs)
                    lvl_maps[f"lvl{bs}_mean"] = m[0, 0].detach()
                    lvl_maps[f"lvl{bs}_occ"] = o[0, 0].detach()
            else:
                for bs in lvl_sizes.tolist():
                    k_mean = f"lvl{bs}_mean"
                    k_occ = f"lvl{bs}_occ"
                    v_mean = _npz_get(z, k_mean, None)
                    v_occ = _npz_get(z, k_occ, None)
                    if v_mean is not None:
                        lvl_maps[k_mean] = torch.from_numpy(np.asarray(v_mean, dtype=np.float32))
                    if v_occ is not None:
                        lvl_maps[k_occ] = torch.from_numpy(np.asarray(v_occ, dtype=np.uint8).astype(np.float32))
            sample["lvl_maps"] = lvl_maps

        if self.return_meta and rec.abs_meta_path is not None and os.path.exists(rec.abs_meta_path):
            try:
                with open(rec.abs_meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = None
            sample["meta"] = meta

        return sample

    def vdb_ext_get(self, idx: int):
        if not HAS_VDB_EXT:
            raise RuntimeError("vdb_ext is required for .vdb datasets. Use .npz data or install vdb_ext.")
        rec = self.samples[idx]
        vdb_path = rec.abs_data_path
        if not os.path.exists(vdb_path):
            raise FileNotFoundError(f"vdb not found: {vdb_path}")

        seq_key, _ = _seqkey_from_record(rec)

        # --- resolve per-seq include mode ---
        mode = getattr(self, "transform_included_mode", "force_off")
        if mode == "force_on":
            use_include = True
        elif mode == "force_off":
            use_include = False
        else:
            use_include = bool(getattr(self, "_seq_need_include", {}).get(seq_key, False))

        # --- resolve per-seq bbox ---
        bbox_min_arg = None
        bbox_max_arg = None
        bpair = getattr(self, "_seq_bbox_minmax", {}).get(seq_key, None)

        if self.prev_bbox_mode == "frame":
            bbox_min_arg = None
            bbox_max_arg = None
        elif self.prev_bbox_mode == "seq":
            if bpair is not None:
                bmin, bmax = bpair

                # 关键：同一序列 bbox 一致
                bmin_eff = bmin.copy()
                bmax_eff = bmax.copy()

                if use_include:
                    # 你原本 C++ 只做 min(bmin,0)，这里我建议更稳健：把 0 真正纳入 bbox
                    # 如果你确定 bbox_max 永远>=0，也可以只保留 bmin 的那句
                    bmin_eff = np.minimum(bmin_eff, np.array((0, 0, 0), dtype=np.int64))
                    bmax_eff = np.maximum(bmax_eff, np.array((0, 0, 0), dtype=np.int64))

                bbox_min_arg = [int(bmin_eff[0]), int(bmin_eff[1]), int(bmin_eff[2])]
                bbox_max_arg = [int(bmax_eff[0]), int(bmax_eff[1]), int(bmax_eff[2])]
        else:
            raise ValueError(f"unknown prev_bbox_mode={self.prev_bbox_mode}")
        # --- call C++ sampler (bbox_min/max 走序列 union bbox) ---
        if getattr(self, "debug_vdb_ext_get", False):
            pid = os.getpid()
            msg = f"[pid={pid}] t={time.time():.3f} idx={idx} path={vdb_path}, bbox_min={bbox_min_arg}, bbox_max={bbox_max_arg}\n"
            hard_log(msg, "dataloader_logger.txt")
        arr, tt_ms = vdb_ext.sample_hybrid(
            vdb_path,
            grid_name="density",
            out=self.vol_size,
            mode="trilinear",
            backend="tree",
            align_corners=False,
            delayed=True,
            pad_to_cube=getattr(self, "pad_to_cube", True),
            # 方案二：每个 seq 单独决定
            transform_included=use_include,
            # 方案二：每个 seq 传入统一 bbox
            bbox_min=bbox_min_arg,
            bbox_max=bbox_max_arg,
            roi_read=False,
            dense_ratio=12.0,
            max_tbb_threads=0,
        )


        x = torch.from_numpy(arr)
        x5 = x.unsqueeze(0)
        vol_t = x5
        occ_t = (torch.abs(vol_t) > self.occ_eps_fallback).to(torch.uint8)
        T = vol_t.shape[-1]
        leaf_base = np.asarray([T // 8], dtype=np.int32)
        cat_id = self.cat_to_idx[rec.category]
        sample: Dict[str, Any] = {
            "volume": vol_t,                         # CPU tensor
            "occ": occ_t.to(torch.float32),          # CPU tensor
            "leaf_base": int(np.asarray(leaf_base).reshape(-1)[0]),
            "lvl_sizes": torch.tensor([T // 8, T // 4, T // 2], dtype=torch.long),
            "id": rec.id,
            "category": rec.category,
            "category_id": int(cat_id),
            "grid_name": "density fake",                      # 方便 debug
        }
        return sample

    def vdb_get(self, idx: int) -> Dict[str, Any]:
        if not HAS_OPENVDB:
            raise RuntimeError("openvdb is required for .vdb datasets. Use .npz data or install openvdb.")
        rec = self.samples[idx]
        vdb_path = rec.abs_data_path

        if not os.path.exists(vdb_path):
            raise FileNotFoundError(f"vdb not found: {vdb_path}")

        # ---------------------------------------------------------------------
        # (可选) meta json：如果它只是“做存在性校验”，建议关掉以省一次磁盘 I/O
        # ---------------------------------------------------------------------
        if getattr(self, "check_meta_json", False):
            meta = rec.abs_meta_path
            try:
                with open(meta, "r", encoding="utf-8") as f:
                    meta_json = json.load(f)
            except Exception:
                raise FileNotFoundError(f"failed to read vdb meta: {meta}")

            if meta_json.get("bbox_max", None) is None:
                raise KeyError(f"vdb missing key 'bbox_max': {vdb_path}")

        # ---------------------------------------------------------------------
        # 读 grid：优先读 density，失败再 fallback 到第一个 grid
        # ---------------------------------------------------------------------
        grid_name = getattr(self, "grid_name", None) or "density"
        try:
            grid = vdb.read(vdb_path, grid_name)
            gname = grid_name
        except Exception:
            grids = vdb.readAllGridMetadata(vdb_path)
            if not grids:
                raise RuntimeError(f"no grids in {vdb_path}")
            gname = grids[0].name
            try:
                grid = vdb.read(vdb_path, gname)
            except Exception:
                raise RuntimeError(f"failed to read vdb grid: {vdb_path} (tried {grid_name} and {gname})")

        # bbox
        bbox_min = np.array(grid["file_bbox_min"], dtype=np.int64)
        bbox_max = np.array(grid["file_bbox_max"], dtype=np.int64)

        if self.transform_included:
            bbox_min = np.minimum(bbox_min, np.array((0, 0, 0), dtype=np.int64))

        dims = bbox_max - bbox_min + (1, 1, 1)  # inclusive

        try:
            grid.background = 0
        except Exception:
            pass

        # ---------------------------------------------------------------------
        # 1) copyToArray -> numpy dense (X,Y,Z) = (i,j,k)
        # ---------------------------------------------------------------------
        vol = np.zeros((int(dims[0]), int(dims[1]), int(dims[2])), dtype=np.float32)
        grid.copyToArray(vol, ijk=bbox_min.tolist())

        # ---------------------------------------------------------------------
        # 2) CPU trilinear fast: 不对大体素 permute
        #    - x5: (1,1,X,Y,Z) 作为 (N,C,D,H,W) 处理
        #    - 先插值到 (out,out,out)
        #    - 再对“小 tensor” permute，把轴变成你原先语义 (D=Z,H=Y,W=X)
        # ---------------------------------------------------------------------
        out = int(self.vol_size)
        x = torch.from_numpy(vol)                     # (X,Y,Z) CPU, zero-copy view
        x5 = x.unsqueeze(0).unsqueeze(0)              # (1,1,X,Y,Z)

        # (可选) 保持比例：先 pad 成 cube 再插值（如果你原来 normal_to_cube 是这个含义）
        pad_to_cube = getattr(self, "pad_to_cube", True)
        if pad_to_cube:
            X, Y, Z = x.shape
            M = int(max(X, Y, Z))
            pd = M - X
            ph = M - Y
            pw = M - Z
            # F.pad 顺序: (W_left, W_right, H_left, H_right, D_left, D_right)
            pad = (
                pw // 2, pw - pw // 2,
                ph // 2, ph - ph // 2,
                pd // 2, pd - pd // 2,
            )
            x5 = F.pad(x5, pad, mode="constant", value=0.0)

        # CPU trilinear 到 out^3
        y5 = F.interpolate(x5, size=(out, out, out), mode="trilinear", align_corners=False)  # (1,1,out,out,out) D=X

        # 小 permute：把 (D=X,H=Y,W=Z) 变成 (D=Z,H=Y,W=X)，等价于你以前大 permute 的最终语义
        # y5 = y5.permute(0, 1, 4, 3, 2).contiguous()   # (1,1,out,out,out) D=Z

        vol_t = y5[0]                                  # (1,out,out,out)
        T = int(vol_t.shape[1])

        occ_t = (torch.abs(vol_t) > self.occ_eps_fallback).to(torch.uint8)

        leaf_base = np.asarray([T // 8], dtype=np.int32)
        cat_id = self.cat_to_idx[rec.category]

        sample: Dict[str, Any] = {
            "volume": vol_t,                         # CPU tensor
            "occ": occ_t.to(torch.float32),          # CPU tensor
            "leaf_base": int(np.asarray(leaf_base).reshape(-1)[0]),
            "lvl_sizes": torch.tensor([T // 8, T // 4, T // 2], dtype=torch.long),
            "id": rec.id,
            "category": rec.category,
            "category_id": int(cat_id),
            "grid_name": gname,                      # 方便 debug
        }
        if vdb_path == "/data5/sjw/VDBSet/explosion_burning/734/0iuriy2ulwpgc4q3__n0001.vdb":
            print(f"pad_to_cube: {pad_to_cube}, transform_included: {self.transform_included}")
        return sample

    def vdb_get_old(self, idx: int) -> Dict[str, Any]:
        rec = self.samples[idx]
        vdb_path = rec.abs_data_path


        if not os.path.exists(vdb_path):
            raise FileNotFoundError(f"vdb not found: {vdb_path}")
        meta = rec.abs_meta_path
        meta_json = None
        try:
            with open(meta, "r", encoding="utf-8") as f:
                meta_json = json.load(f)
        except Exception:
            meta_json = None
            raise FileNotFoundError(f"failed to read vdb meta: {meta}")

        valid = meta_json.get("bbox_max", None)
        if valid is None:
            raise KeyError(f"vdb missing key 'bbox_max': {vdb_path}")
        grids = vdb.readAllGridMetadata(vdb_path)
        if not grids:
            raise RuntimeError(f"no grids in {vdb_path}")

        try:
            grid = vdb.read(vdb_path, grids[0].name)
        except Exception:
            # 尽量不要递归随机跳（会造成难以复现）；这里直接抛错更干净
            raise RuntimeError(f"failed to read vdb grid: {vdb_path}")

        bbox_min = np.array(grid["file_bbox_min"], dtype=np.int64)
        bbox_max = np.array(grid["file_bbox_max"], dtype=np.int64)

        if self.transform_included:
            bbox_min = np.minimum(bbox_min, np.array((0, 0, 0), dtype=np.int64))
        dims = bbox_max - bbox_min + (1, 1, 1)

        try:
            grid.background = 0
        except Exception:
            pass
        vol = np.zeros((int(dims[0]), int(dims[1]), int(dims[2])), dtype=np.float32)
        grid.copyToArray(vol, ijk=bbox_min.tolist())

        vol_t = torch.from_numpy(vol)[None, ...]
        vol_t = cube_rescale(normal_to_cube(vol_t), size=self.vol_size)
        T = int(vol_t.shape[1])

        occ_t = (torch.abs(vol_t) > self.occ_eps_fallback).to(torch.uint8)

        leaf_base = np.asarray([T // 8], dtype=np.int32)
        cat_id = self.cat_to_idx[rec.category]

        sample: Dict[str, Any] = {
            "volume": vol_t,
            "occ": occ_t.to(torch.float32),
            "leaf_base": int(np.asarray(leaf_base).reshape(-1)[0]),
            "lvl_sizes": torch.tensor([T // 8, T // 4, T // 2], dtype=torch.long),
            "id": rec.id,
            "category": rec.category,
            "category_id": int(cat_id),
        }
        return sample

    # ---------- temporal attach ----------

    def _get_seq_and_frame_by_idx(self, idx: int) -> Tuple[str, int]:
        return _seqkey_from_record(self.samples[idx])

    def _find_prev_index(self, idx: int) -> Tuple[Optional[int], int, int]:
        """
        返回 (prev_idx or None, prev_valid(0/1), prev_dt)
        prev_dt = 当前帧号 - 实际使用的prev帧号（若prev不存在则0）
        """
        if self.prev_k <= 0:
            return None, 0, 0

        seq_key, f = self._get_seq_and_frame_by_idx(idx)
        target = int(f) - int(self.prev_k)

        hit = self._seqframe_to_index.get((seq_key, target), None)
        if hit is not None:
            return int(hit), 1, int(self.prev_k)

        if self.prev_policy == "zero":
            return None, 0, 0

        frames = self._seq_to_frames.get(seq_key, [])
        if not frames:
            return None, 0, 0

        if self.prev_policy == "clamp0":
            f0 = int(frames[0])
            prev_idx = self._seqframe_to_index.get((seq_key, f0), None)
            if prev_idx is None:
                return None, 0, 0
            return int(prev_idx), 1, int(f) - int(f0)

        # nearest-backfill: largest frame < f
        prev_f = None
        for ff in reversed(frames):
            if int(ff) < int(f):
                prev_f = int(ff)
                break
        if prev_f is None:
            return None, 0, 0

        prev_idx = self._seqframe_to_index.get((seq_key, prev_f), None)
        if prev_idx is None:
            return None, 0, 0
        return int(prev_idx), 1, int(f) - int(prev_f)

    def _attach_prev_fields(self, sample: Dict[str, Any], idx: int) -> Dict[str, Any]:
        if self.prev_k <= 0:
            return sample

        prev_idx, prev_valid, prev_dt = self._find_prev_index(idx)

        vol = sample["volume"]
        if prev_idx is None:
            prev_vol = torch.zeros_like(vol)
            prev_occ = torch.zeros_like(sample["occ"]) if ("occ" in sample) else None
        else:
            prev_s = self._get_core(prev_idx)  # 注意：不再依赖 self.datasub
            prev_vol = prev_s["volume"]
            prev_occ = prev_s.get("occ", None)

        sample["prev_volume"] = prev_vol
        sample["prev_valid"] = torch.tensor([prev_valid], dtype=torch.float32)
        sample["prev_dt"] = torch.tensor([prev_dt], dtype=torch.float32)

        if self.return_prev_occ:
            if prev_occ is None:
                prev_occ = torch.zeros_like(sample["occ"])
            sample["prev_occ"] = prev_occ

        return sample

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            sample = self._get_core(idx)
            sample = self._attach_prev_fields(sample, idx)
        except Exception as e:
            print(f"[WARN] failed to get sample {idx}. {e}, skip.")
            return self.__getitem__((idx + 1) % len(self))

        return sample


# ===================== 索引构建工具 =====================

def _load_dataset_index(root_dir: str) -> Dict:
    idx_path = os.path.join(root_dir, "dataset_index.json")
    if os.path.exists(idx_path):
        with open(idx_path, "r", encoding="utf-8") as f:
            return json.load(f)

    cats = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d not in ("__pycache__",)
    ]
    return {
        "version": "unknown",
        "source_version": "unknown",
        "resolution": None,
        "categories": sorted(cats),
    }


def _collect_samples(
    root_dir: str,
    categories: Optional[List[str]] = None,
) -> Tuple[List[SampleRecord], Dict[str, int]]:

    ds_index = _load_dataset_index(root_dir)
    all_cats = ds_index.get("categories", [])

    if categories is None:
        categories = all_cats
    else:
        categories = [c for c in categories if c in all_cats]

    samples: List[SampleRecord] = []

    for cat in categories:
        cat_dir = os.path.join(root_dir, cat)
        if not os.path.isdir(cat_dir):
            continue

        cat_index_path = os.path.join(cat_dir, "category_index.json")
        if not os.path.exists(cat_index_path):
            print(f"[WARN] category_index.json not found for category={cat}, skip.")
            continue

        try:
            with open(cat_index_path, "r", encoding="utf-8") as f:
                cat_index = json.load(f)
        except Exception as e:
            print(f"[WARN] failed to load {cat_index_path}: {e}, skip category.")
            continue

        sample_list = cat_index.get("samples", [])
        if not isinstance(sample_list, list) or len(sample_list) == 0:
            print(f"[WARN] no valid samples in {cat_index_path}, skip category.")
            continue

        path_type_list = ["npz_path", "vdb_path", "numpy_path"]
        sample_path_type = None
        for path_type in path_type_list:
            if sample_list[0].get(path_type, None) is not None:
                sample_path_type = path_type
                break
        if sample_path_type is None:
            print(f"[WARN] no valid path type found in {cat_index_path}, skip category.")
            continue

        for s in sample_list:
            sid = str(s.get("id"))
            vol_rel = s.get(sample_path_type, None)
            meta_rel = s.get("meta_path")

            if vol_rel is None or meta_rel is None:
                continue

            abs_vol = os.path.join(cat_dir, vol_rel)
            abs_meta = os.path.join(cat_dir, meta_rel) if meta_rel else None
            if not os.path.exists(abs_vol) or not os.path.isfile(abs_meta):
                continue

            samples.append(
                SampleRecord(
                    id=sid,
                    category=cat,
                    data_path=vol_rel,
                    meta_path=meta_rel,
                    abs_data_path=abs_vol,
                    abs_meta_path=abs_meta,
                    abs_cat_dir=cat_dir,
                )
            )

    if not samples:
        raise RuntimeError(f"No samples found under {root_dir} with categories={categories}")

    cats_found = sorted({r.category for r in samples})
    cat_to_idx: Dict[str, int] = {c: i for i, c in enumerate(cats_found)}
    return samples, cat_to_idx


# ===================== 按序列分组切分（关键改动） =====================

def _alloc_counts(G: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    """
    给定组数 G，按比例分配 train/val/test 的组数。
    尽量保证总和为 G，且在 G 足够时 val/test 不至于全为 0。
    """
    if G <= 0:
        return 0, 0, 0

    # 基础分配
    n_train = int(G * train_ratio)
    n_val = int(G * val_ratio)
    n_test = G - n_train - n_val

    # 微调：如果 test_ratio>0 但 n_test==0 且 G>=2，挪一个出来
    if test_ratio > 0 and n_test == 0 and G >= 2:
        if n_train > 0:
            n_train -= 1
            n_test += 1
        else:
            n_val = max(0, n_val - 1)
            n_test += 1

    # 如果 val_ratio>0 但 n_val==0 且 G>=3，再挪一个出来
    if val_ratio > 0 and n_val == 0 and G >= 3:
        if n_train > 1:
            n_train -= 1
            n_val += 1
        elif n_test > 0:
            n_test -= 1
            n_val += 1

    # clamp 修正
    n_train = max(0, min(n_train, G))
    n_val = max(0, min(n_val, G - n_train))
    n_test = G - n_train - n_val
    return n_train, n_val, n_test


def _split_by_sequence_folder(
    samples: List[SampleRecord],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[SampleRecord], List[SampleRecord], List[SampleRecord]]:
    """
    按 “同类别 + 同序列文件夹(seq_key)” 分组切分。
    做法：每个 category 内，先把样本按 seq_key 分成若干组（组=一个序列文件夹），再按比例切分组。
    """
    from collections import defaultdict

    # cat -> seq_key -> list[SampleRecord]
    by_cat_seq: Dict[str, Dict[str, List[SampleRecord]]] = defaultdict(lambda: defaultdict(list))
    for r in samples:
        seq_key, _f = _seqkey_from_record(r)
        by_cat_seq[r.category][seq_key].append(r)

    rng = random.Random(seed)

    train_out: List[SampleRecord] = []
    val_out: List[SampleRecord] = []
    test_out: List[SampleRecord] = []

    for cat, seq_map in by_cat_seq.items():
        groups = list(seq_map.values())  # list of list[records]
        rng.shuffle(groups)

        G = len(groups)
        n_tr_g, n_va_g, n_te_g = _alloc_counts(G, train_ratio, val_ratio, test_ratio)

        tr_groups = groups[:n_tr_g]
        va_groups = groups[n_tr_g:n_tr_g + n_va_g]
        te_groups = groups[n_tr_g + n_va_g:]

        for g in tr_groups:
            train_out.extend(g)
        for g in va_groups:
            val_out.extend(g)
        for g in te_groups:
            test_out.extend(g)

    # 最后再整体打散（不改变组归属，只改变样本顺序）
    rng.shuffle(train_out)
    rng.shuffle(val_out)
    rng.shuffle(test_out)
    return train_out, val_out, test_out


# ===================== balance（保持你原逻辑） =====================

from collections import defaultdict

def _balance_samples_equal_per_class(
    samples: List[SampleRecord],
    seed: int,
    mode: str = "oversample",
    target_per_class: Optional[int] = None,
    max_per_class: Optional[int] = None,
) -> List[SampleRecord]:
    if not samples:
        return samples

    by_cat: Dict[str, List[SampleRecord]] = defaultdict(list)
    for s in samples:
        by_cat[s.category].append(s)

    cats = sorted(by_cat.keys())
    rng = random.Random(seed)
    for c in cats:
        rng.shuffle(by_cat[c])

    counts = {c: len(by_cat[c]) for c in cats}
    if target_per_class is None:
        target = max(counts.values()) if mode == "oversample" else min(counts.values())
    else:
        target = int(target_per_class)

    if max_per_class is not None:
        target = min(target, int(max_per_class))
    target = max(1, target)

    out: List[SampleRecord] = []
    for c in cats:
        lst = by_cat[c]
        if not lst:
            continue
        if mode == "undersample":
            out.extend(lst[: min(len(lst), target)])
        else:
            if len(lst) >= target:
                out.extend(lst[:target])
            else:
                out.extend(lst)
                need = target - len(lst)
                out.extend([rng.choice(lst) for _ in range(need)])

    rng.shuffle(out)
    return out


# ===================== 构建 train/val/test split =====================

def build_dataset_splits(
    root_dir: str,
    categories: Optional[List[str]] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
    vol_size: int = 32,
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
    max_test: Optional[int] = None,
    return_meta: bool = False,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    return_lvl_maps: bool = False,
    balance_train: bool = True,
    balance_mode: str = "oversample",
    balance_target_per_class: Optional[int] = None,
    balance_max_per_class: Optional[int] = None,
    prev_k: int = 0,
    prev_bbox_mode: str = "seq",
    transform_included: Union[bool, str] = "auto",
) -> Dict[str, VolumeMultiDataset]:

    samples, cat_to_idx = _collect_samples(root_dir, categories)

    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("Sum of ratios must be > 0")
    train_ratio /= ratio_sum
    val_ratio /= ratio_sum
    test_ratio /= ratio_sum

    # ★关键：按序列文件夹分组切分
    train_samples, val_samples, test_samples = _split_by_sequence_folder(
        samples=samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    # 修复：max_* 判断必须用 and
    train_samples = _cap_by_seq_groups(train_samples, max_train, seed=seed + 101)
    val_samples   = _cap_by_seq_groups(val_samples,   max_val,   seed=seed + 102)
    test_samples  = _cap_by_seq_groups(test_samples,  max_test,  seed=seed + 103)

    if balance_train:
        # 建议：balance 用“序列组数量”作为目标，而不是“帧数量”
        # 如果你传了 max_train，并且你想让最终总帧数 ≈ max_train，
        # 我们可以根据“平均每组多少帧”估一个 target_groups。
        if max_train is not None and max_train > 0:
            # 估计每个类别目标的 seq_group 数量
            # 先统计当前 train_samples 里每个 cat 的组，以及每组帧数
            from collections import defaultdict
            by_cat_seq = defaultdict(lambda: defaultdict(list))
            for r in train_samples:
                seq_key, _ = _seqkey_from_record(r)
                by_cat_seq[r.category][seq_key].append(r)

            cats_present = sorted(by_cat_seq.keys())
            C = max(1, len(cats_present))

            # 用全体组的平均帧数估计：target_groups_per_class
            all_group_sizes = []
            for c in cats_present:
                for g in by_cat_seq[c].values():
                    all_group_sizes.append(len(g))
            avg_group = max(1.0, sum(all_group_sizes) / max(1, len(all_group_sizes)))

            # 目标：总帧数≈max_train => 总组数≈max_train/avg_group
            total_target_groups = max(1, int(max_train / avg_group))
            target_groups_per_class = max(1, int(total_target_groups / C))

            train_samples = _balance_by_seq_groups_equal_per_class(
                train_samples,
                seed=seed + 1337,
                mode=str(balance_mode),  # "oversample"/"undersample"
                target_seq_groups_per_class=target_groups_per_class,
                max_seq_groups_per_class=balance_max_per_class,  # 你也可以复用这个字段
                shuffle_within_group=False,
            )
        else:
            # 不指定 max_train：直接对齐每类的 seq_group 数
            train_samples = _balance_by_seq_groups_equal_per_class(
                train_samples,
                seed=seed + 1337,
                mode=str(balance_mode),
                target_seq_groups_per_class=balance_target_per_class,  # 这里建议你传“每类序列数”，不是帧数
                max_seq_groups_per_class=balance_max_per_class,
                shuffle_within_group=False,
            )
    ds_train = VolumeMultiDataset(
        samples=train_samples,
        cat_to_idx=cat_to_idx,
        vol_size=vol_size,
        return_meta=return_meta,
        transform=transform,
        return_lvl_maps=return_lvl_maps,
        prev_k=prev_k,
        prev_bbox_mode=prev_bbox_mode,
        transform_included=transform_included,
    )
    ds_val = VolumeMultiDataset(
        samples=val_samples,
        cat_to_idx=cat_to_idx,
        vol_size=vol_size,
        return_meta=return_meta,
        transform=transform,
        return_lvl_maps=return_lvl_maps,
        prev_k=prev_k,
        prev_bbox_mode=prev_bbox_mode,
        transform_included=transform_included,
    )
    ds_test = VolumeMultiDataset(
        samples=test_samples,
        cat_to_idx=cat_to_idx,
        vol_size=vol_size,
        return_meta=return_meta,
        transform=transform,
        return_lvl_maps=return_lvl_maps,
        prev_k=prev_k,
        prev_bbox_mode=prev_bbox_mode,
        transform_included=transform_included,
    )

    return {"train": ds_train, "val": ds_val, "test": ds_test}


# ===================== 小测试 =====================
def _debug_check_prev(ds, num_checks: int = 30, seed: int = 0):
    import random, os
    rng = random.Random(seed)

    print("\n[CHECK] dataset size =", len(ds))
    print("[CHECK] prev_k =", ds.prev_k, "prev_policy =", ds.prev_policy)

    for t in range(num_checks):
        idx = rng.randrange(0, len(ds))
        rec = ds.samples[idx]
        curr_path = rec.abs_data_path
        curr_dir = os.path.dirname(curr_path)

        curr_seq_key, curr_frame = ds._get_seq_and_frame_by_idx(idx)

        frames = ds._seq_to_frames.get(curr_seq_key, [])
        frames_min = min(frames) if frames else None
        frames_max = max(frames) if frames else None

        target = curr_frame - ds.prev_k if ds.prev_k > 0 else None
        target_in = (target in set(frames)) if (frames and target is not None) else False

        prev_idx, prev_valid, prev_dt = ds._find_prev_index(idx) if ds.prev_k > 0 else (None, 0, 0)

        print("\n" + "-" * 80)
        print(f"[CHECK#{t}] idx={idx}")
        print(f"  CURR: cat={rec.category} seq_key={curr_seq_key} frame={curr_frame}")
        print(f"        path={curr_path}")
        print(f"  SEQ(frames in ds): count={len(frames)} min={frames_min} max={frames_max}")
        if ds.prev_k > 0:
            print(f"  TARGET: frame={target}  in_ds={target_in}  (prev_k={ds.prev_k})")

        if prev_idx is None or prev_valid == 0:
            print(f"  PREV: NONE (prev_valid={prev_valid}, prev_dt={prev_dt})")
            # 解释 nearest 会选谁
            if frames:
                prev_candidates = [ff for ff in frames if ff < curr_frame]
                if prev_candidates:
                    print(f"  NOTE: nearest candidate in-ds would be {max(prev_candidates)}")
                else:
                    print("  NOTE: no earlier frame exists in-ds for this seq_key")
            continue

        prev_rec = ds.samples[prev_idx]
        prev_path = prev_rec.abs_data_path
        prev_dir = os.path.dirname(prev_path)
        prev_seq_key, prev_frame = ds._get_seq_and_frame_by_idx(prev_idx)

        print(f"  PREV: idx={prev_idx} cat={prev_rec.category} seq_key={prev_seq_key} frame={prev_frame}")
        print(f"        path={prev_path}")
        print(f"        prev_dt={prev_dt} (should be curr-prev = {curr_frame - prev_frame})")

        # 关键断言：同类、同 seq 文件夹
        assert prev_rec.category == rec.category, "category mismatch"
        assert prev_seq_key == curr_seq_key, "seq_key mismatch"
        assert prev_dir == curr_dir, "seq folder mismatch"
        assert prev_frame < curr_frame, "prev_frame not < curr_frame"
        print("  [OK] prev is within same category & same seq folder.")

if __name__ == "__main__":
    ROOT_DIR = "/data5/sjw/VDBSet/"

    splits = build_dataset_splits(
        root_dir=ROOT_DIR,
        categories=None,
        seed=42,
        max_train=5000,  # 多取点更容易看到跨帧情况
        max_val=50,
        max_test=50,
        return_meta=False,
        transform=None,
        return_lvl_maps=True,
        prev_k=2,                 # 你想检查前几帧，就改这里
        transform_included=False,
    )

    for k, ds in splits.items():
        print(k, len(ds))

    # 只检查 train（也可以 val/test 各跑一遍）
    _debug_check_prev(splits["train"], num_checks=10, seed=0)
