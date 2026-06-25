# one_stages/factory.py
from __future__ import annotations

import os
import sys
from typing import Dict, Any

import torch

from utils.diffusion_utils import DiffusionConfig
from utils.ckpt import EMA
from utils.cond_utils import infer_class_names

from one_stages.features import OneStageFeatureExtractor
from one_stages.model_adapter import build_default_model_adapter

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tile_dataloader import build_dataset_splits  # noqa: E402
from models.vfx_model import UNet3DModel  # noqa: E402


def _dataset_has_prev(sample: dict) -> bool:
    # 兼容多种prev字段命名
    if not isinstance(sample, dict):
        return False
    for k in ("prev_volume", "volume_prev", "prev", "prev_vol", "prev_volume_raw"):
        if k in sample:
            return True
    return False


def build_everything(cfg):
    # 统一构建：dataset / model / optimizer / diffusion / feature extractor
    raw = cfg.raw
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset拆分
    categories = raw.get("categories", None)
    max_train = int(raw.get("max_train_samples", 0))
    max_train = None if max_train <= 0 else max_train
    max_val = int(raw.get("max_val_samples", 0))
    max_val = None if max_val <= 0 else max_val
    max_test = int(raw.get("max_test_samples", 0))
    max_test = None if max_test <= 0 else max_test

    splits = build_dataset_splits(
        root_dir=str(raw["data_root"]),
        categories=categories,
        train_ratio=float(raw.get("train_ratio", 0.8)),
        val_ratio=float(raw.get("val_ratio", 0.1)),
        test_ratio=float(raw.get("test_ratio", 0.1)),
        vol_size=int(raw.get("volume_size", 32)),
        seed=int(raw.get("seed", 42)),
        max_train=max_train,
        max_val=max_val,
        max_test=max_test,
        return_meta=bool(raw.get("return_meta", False)),
        transform=None,
        prev_k = int(raw.get("prev_k", 0)),
        prev_bbox_mode = str(raw.get("prev_bbox_mode", "seq")),
        transform_included=bool(raw.get("transform_included", False)),
        metadata_mode=str(raw.get("metadata_mode", "required")),
    )
    ds = splits["train"]

    # class map (CFG相关)
    num_classes = int(raw.get("num_classes", 0))
    class_names = None
    class_to_id = None
    if num_classes != 0:
        if num_classes < 0:
            cat_to_idx = getattr(ds, "cat_to_idx", None)
            if not isinstance(cat_to_idx, dict) or not cat_to_idx:
                raise RuntimeError("auto num_classes failed: dataset has no cat_to_idx")
            class_names_auto = sorted(cat_to_idx.keys(), key=lambda k: cat_to_idx[k])
            C = len(class_names_auto)
            if C <= 1:
                raw["num_classes"] = 0
                num_classes = 0
            else:
                raw["num_classes"] = C
                num_classes = C
                if raw.get("class_names", None) is None:
                    raw["class_names"] = class_names_auto
                if not str(raw.get("cond_key", "")).strip():
                    raw["cond_key"] = "category_id"

        if num_classes > 0:
            class_names = infer_class_names(type("A", (object,), raw)(), ds)
            if class_names is not None:
                if len(class_names) != int(raw.get("num_classes", 0)):
                    raise RuntimeError(
                        f"[cond] class_names length mismatch: {len(class_names)} != {int(raw.get('num_classes',0))}"
                    )
                class_to_id = {str(n): i for i, n in enumerate(class_names)}

    # temporal开关 + in_channels切换
    temporal_enable = bool(raw.get("temporal_enable", False))
    in_channels = 2 if temporal_enable else 1

    if temporal_enable:
        # one-time sanity check (loads 1 sample)
        try:
            s0 = ds[0]
        except Exception as e:
            raise RuntimeError(f"[temporal] enabled but failed to read ds[0]: {e}")
        if not _dataset_has_prev(s0):
            raise RuntimeError(
                "[temporal] temporal_enable=true, but dataset sample has no prev volume.\n"
                "Please modify your dataset __getitem__ to return one of:\n"
                "  - 'prev_volume' (recommended)\n"
                "  - 'volume_prev'\n"
                "Or set temporal_enable=false.\n"
            )
        print("[temporal] enabled: expecting dataset to provide prev_volume/volume_prev; model in_channels=2")

    # model构建（可选occ头）
    use_occupancy = not bool(raw.get("no_occupancy", False))
    out_channels = 2 if use_occupancy else 1

    model = UNet3DModel(
        in_channels=in_channels,
        base_channels=int(raw.get("base_channels", 48)),
        channel_mults=tuple(raw.get("channel_mults", [1, 2, 4, 4])),
        time_emb_dim=int(raw.get("time_emb_dim", 256)),
        out_channels=out_channels,
        num_classes=int(raw.get("num_classes", 0)),
        sample_size=int(raw.get("volume_size", 32)) # [建议新增] 给 Config 记录一下尺寸
    ).to(device)

    # optimizer / ema
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(raw["lr"]),
        weight_decay=float(raw.get("weight_decay", 0.0)),
    )
    ema = EMA(model, decay=float(raw.get("ema_decay", 0.9999)))

    # diffusion配置
    diff = DiffusionConfig.build(
        int(raw["timesteps"]),
        device,
        float(raw["beta_start"]),
        float(raw["beta_end"]),
    )

    # batch特征抽取（含class cond / temporal prev）
    feat_extractor = OneStageFeatureExtractor(raw, class_to_id)
    model_adapter = build_default_model_adapter(use_occupancy=use_occupancy)

    return dict(
        device=device,
        ds=ds,
        splits=splits,
        model=model,
        opt=opt,
        ema=ema,
        diff=diff,
        feat_extractor=feat_extractor,
        model_adapter=model_adapter,
        use_occupancy=use_occupancy,
        out_channels=out_channels,
        class_names=class_names,
    )
