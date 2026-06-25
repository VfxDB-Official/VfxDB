import torch
import torch.nn.functional as F
import os
import numpy as np
import math
import re


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Apply sigmoid activation to numpy array."""
    x = x.astype(np.float32)
    return 1.0 / (1.0 + np.exp(-x))


def log1p_m11_inverse_torch(y_m11: torch.Tensor, scale: float = 0.02) -> torch.Tensor:
    """Inverse log1p transformation from [-1, 1] range to recovered values."""
    if scale <= 0:
        raise ValueError("log1p_scale must be > 0")
    y01 = ((y_m11 + 1.0) * 0.5).clamp(0.0, 1.0)
    denom = math.log1p(1.0 / float(scale))
    v01 = float(scale) * torch.expm1(y01 * float(denom))
    v01 = v01.clamp(0.0, 1.0)
    x_m11 = (v01 * 2.0 - 1.0).clamp(-1.0, 1.0)
    return x_m11


def safe_tag(s: str | None, maxlen: int = 48) -> str:
    """Convert string to safe tag for filename (alphanumeric and underscore only)."""
    if not s:
        return "uncond"
    s = str(s).strip()
    if not s:
        return "uncond"
    s = re.sub(r"[^0-9a-zA-Z\-_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        return "uncond"
    return s[:maxlen]


def upsample_nearest(x: torch.Tensor, target: int) -> torch.Tensor:
    if int(x.shape[-1]) == int(target):
        return x
    return F.interpolate(x, size=(target, target, target), mode="nearest")


def normalize_vol_to_m11(vol: torch.Tensor, eps: float = 1e-6):
    vol = torch.clamp(vol, min=0.0)
    B = vol.shape[0]
    vmax = vol.view(B, -1).max(dim=1).values.clamp(min=eps).view(B, 1, 1, 1, 1)
    vol01 = vol / vmax
    return vol01 * 2.0 - 1.0, vmax


def coords_to_lin(coords_3: torch.Tensor, R: int) -> torch.Tensor:
    return coords_3[:, 0] * (R * R) + coords_3[:, 1] * R + coords_3[:, 2]


def normalize01_per_sample(vol: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # vol: [B,1,D,H,W], clamp >=0 then /max
    vol = torch.clamp(vol, min=0.0)
    B = vol.shape[0]
    flat = vol.view(B, -1)
    mx = flat.max(dim=1).values.clamp(min=eps).view(B, 1, 1, 1, 1)
    return (vol / mx).clamp(0.0, 1.0)

def avg_pool_to_leaf_base(vol01: torch.Tensor, leaf_base: int) -> torch.Tensor:
    k = int(leaf_base)
    if k <= 1:
        return vol01
    return F.avg_pool3d(vol01, kernel_size=k, stride=k)