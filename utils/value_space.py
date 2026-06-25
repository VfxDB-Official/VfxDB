from __future__ import annotations

import math

import torch


def log1p_m11_forward(
    x_m11: torch.Tensor,
    *,
    scale: float,
    dequant_delta: float = 0.0,
    training: bool = True,
) -> torch.Tensor:
    if scale <= 0:
        raise ValueError("log1p scale must be > 0")
    v01 = ((x_m11 + 1.0) * 0.5).clamp(0.0, 1.0)
    if training and dequant_delta > 0:
        v01 = (v01 + torch.rand_like(v01) * float(dequant_delta)).clamp(0.0, 1.0)
    denom = math.log1p(1.0 / float(scale))
    y01 = torch.log1p(v01 / float(scale)) / float(denom)
    return (y01 * 2.0 - 1.0).clamp(-1.0, 1.0)


def log1p_m11_inverse(y_m11: torch.Tensor, *, scale: float) -> torch.Tensor:
    if scale <= 0:
        raise ValueError("log1p scale must be > 0")
    y01 = ((y_m11 + 1.0) * 0.5).clamp(0.0, 1.0)
    denom = math.log1p(1.0 / float(scale))
    v01 = float(scale) * torch.expm1(y01 * float(denom))
    v01 = v01.clamp(0.0, 1.0)
    return (v01 * 2.0 - 1.0).clamp(-1.0, 1.0)


def to_value_space(
    x_m11: torch.Tensor,
    *,
    value_space: str,
    log1p_scale: float,
    log1p_dequant_delta: float = 0.0,
    training: bool = True,
) -> torch.Tensor:
    if value_space == "raw":
        return x_m11
    if value_space == "log1p":
        return log1p_m11_forward(
            x_m11,
            scale=log1p_scale,
            dequant_delta=log1p_dequant_delta,
            training=training,
        )
    raise ValueError(f"unknown value_space={value_space}")


def from_value_space(
    y_m11: torch.Tensor,
    *,
    value_space: str,
    log1p_scale: float,
) -> torch.Tensor:
    if value_space == "raw":
        return y_m11
    if value_space == "log1p":
        return log1p_m11_inverse(y_m11, scale=log1p_scale)
    raise ValueError(f"unknown value_space={value_space}")
