# one_stages/features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch

from utils.ops import normalize_vol_to_m11
from utils.cond_utils import get_class_id_from_batch


@dataclass
class OneStageFeatures:
    # 训练用的批量特征（主值 + occ + 可选类别/时序）
    x_m11: torch.Tensor
    occ_gt: Optional[torch.Tensor]
    y: Optional[torch.Tensor]

    # --- NEW: temporal prev-frame conditioning ---
    prev_x_m11: Optional[torch.Tensor]  # [B,1,R,R,R] in [-1,1]
    prev_occ_gt: Optional[torch.Tensor]  # optional, may be None


class OneStageFeatureExtractor:
    def __init__(self, cfg_raw: dict, class_to_id: Optional[dict]):
        self.cfg = cfg_raw
        self.class_to_id = class_to_id

    def _to_5d_m11(self, vol: torch.Tensor) -> torch.Tensor:
        # 统一为[B,1,R,R,R]并归一化到[-1,1]
        vol = vol.float()
        if vol.ndim == 4:
            vol = vol[:, None]
        if vol.ndim != 5:
            raise ValueError(f"expect 4D/5D volume, got shape={tuple(vol.shape)}")
        vol = torch.clamp(vol, min=0.0)
        x_m11, _ = normalize_vol_to_m11(vol)
        return x_m11

    def _occ_from_raw(self, vol_raw: torch.Tensor) -> torch.Tensor:
        # 根据阈值生成占据mask
        thr = float(self.cfg.get("occ_threshold", 1e-6))
        return (vol_raw > thr).float()

    def __call__(self, batch: Dict[str, Any], device: torch.device) -> OneStageFeatures:
        # 从dataloader批次提取训练需要的特征
        if "volume" not in batch:
            raise KeyError("batch missing 'volume'")

        # 当前帧主密度
        vol = batch["volume"].to(device, non_blocking=True).float()
        x_m11 = self._to_5d_m11(vol)

        occ_gt = None
        if not bool(self.cfg.get("no_occupancy", False)):
            occ_gt = self._occ_from_raw(vol if vol.ndim == 5 else vol[:, None])

        # class cond (CFG)
        y = None
        num_classes = int(self.cfg.get("num_classes", 0))
        if num_classes > 0:
            B = int(x_m11.shape[0])
            y_cpu = get_class_id_from_batch(
                batch=batch,
                B=B,
                class_to_id=self.class_to_id,
                cond_key=str(self.cfg.get("cond_key", "category_id")),
            )
            y = y_cpu.to(device, non_blocking=True).long()

            drop_p = float(self.cfg.get("cond_drop_prob", 0.0))
            if drop_p > 0:
                uncond_id = num_classes
                m = (torch.rand((B,), device=device) < drop_p)
                if m.any():
                    y = y.clone()
                    y[m] = uncond_id

        # --- temporal prev-frame conditioning ---
        temporal_enable = bool(self.cfg.get("temporal_enable", False))
        prev_x_m11 = None
        prev_occ_gt = None

        if temporal_enable:
            # 兼容多种prev字段命名
            prev_key = None
            for k in ("prev_volume", "volume_prev", "prev", "prev_vol", "prev_volume_raw"):
                if k in batch:
                    prev_key = k
                    break

            missing_mode = str(self.cfg.get("prev_missing_mode", "error")).lower()
            # missing_mode: "error" | "self" | "zero"
            if prev_key is None:
                if missing_mode == "self":
                    prev_x_m11 = x_m11
                    prev_occ_gt = occ_gt
                elif missing_mode == "zero":
                    prev_x_m11 = torch.zeros_like(x_m11)
                    prev_occ_gt = torch.zeros_like(occ_gt) if occ_gt is not None else None
                else:
                    raise KeyError(
                        "[temporal] enabled but batch has no prev volume key. "
                        "Need one of: prev_volume / volume_prev / prev_vol. "
                        "Or set prev_missing_mode=self|zero."
                    )
            else:
                prev_vol = batch[prev_key].to(device, non_blocking=True).float()
                prev_x_m11 = self._to_5d_m11(prev_vol)
                if not bool(self.cfg.get("no_occupancy", False)):
                    prev_occ_gt = self._occ_from_raw(prev_vol if prev_vol.ndim == 5 else prev_vol[:, None])

        return OneStageFeatures(
            x_m11=x_m11,
            occ_gt=occ_gt,
            y=y,
            prev_x_m11=prev_x_m11,
            prev_occ_gt=prev_occ_gt,
        )
