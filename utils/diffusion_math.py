from __future__ import annotations

import torch


def snr_from_alpha_bar(alpha_bar: torch.Tensor) -> torch.Tensor:
    return alpha_bar / (1.0 - alpha_bar).clamp(min=1e-8)


def p2_weight_from_snr(
    snr: torch.Tensor,
    *,
    k: float = 1.0,
    gamma: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    weight = (snr + float(k)).pow(-float(gamma))
    if normalize:
        weight = weight / weight.mean().detach().clamp(min=1e-12)
    return weight
