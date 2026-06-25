from __future__ import annotations

from typing import Callable, Tuple

import torch

from utils.diffusion_utils import DiffusionConfig, predict_x0_from_eps, q_sample


def _snr_from_alpha_bar(alpha_bar: torch.Tensor) -> torch.Tensor:
    return alpha_bar / (1.0 - alpha_bar).clamp(min=1e-8)


def _p2_weight_from_snr(
    snr: torch.Tensor,
    *,
    k: float = 1.0,
    gamma: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    w = (snr + float(k)).pow(-float(gamma))
    if normalize:
        w = w / (w.mean().detach().clamp(min=1e-12))
    return w


def build_legacy_train_batch(
    *,
    diff: DiffusionConfig,
    device: torch.device,
    x0_raw: torch.Tensor,
    prev_x_m11: torch.Tensor | None,
    to_value_space: Callable[[torch.Tensor], torch.Tensor],
    temporal_enable: bool,
    prev_noised: bool,
    prev_same_t: bool,
    prev_drop_prob: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x0 = to_value_space(x0_raw)
    batch_size = int(x0.shape[0])
    t = torch.randint(0, diff.T, (batch_size,), device=device, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise, diff)

    if not temporal_enable:
        return x0, x_t, noise, t, x_t

    if prev_x_m11 is None:
        raise RuntimeError("[temporal] enabled but feats.prev_x_m11 is None. Check dataset/feature extractor.")

    prev0 = to_value_space(prev_x_m11)
    if prev_noised:
        if prev_same_t:
            prev_t = q_sample(prev0, t, torch.randn_like(prev0), diff)
        else:
            t_prev = torch.randint(0, diff.T, (batch_size,), device=device, dtype=torch.long)
            prev_t = q_sample(prev0, t_prev, torch.randn_like(prev0), diff)
    else:
        prev_t = prev0

    if prev_drop_prob > 0:
        drop_mask = torch.rand((batch_size,), device=device) < float(prev_drop_prob)
        if drop_mask.any():
            prev_t = prev_t.clone()
            prev_t[drop_mask] = 0.0

    return x0, x_t, noise, t, torch.cat([x_t, prev_t], dim=1)


def compute_legacy_timestep_weights(
    *,
    diff: DiffusionConfig,
    t: torch.Tensor,
    p2_k: float,
    p2_gamma: float,
    p2_normalize: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    alpha_bar = diff.abar[t].float()
    snr = _snr_from_alpha_bar(alpha_bar)
    w_p2 = _p2_weight_from_snr(
        snr,
        k=p2_k,
        gamma=p2_gamma,
        normalize=p2_normalize,
    )
    return alpha_bar, snr, w_p2


def predict_x0_raw_legacy(
    *,
    x_t: torch.Tensor,
    t: torch.Tensor,
    pred_main: torch.Tensor,
    diff: DiffusionConfig,
    prediction_type: str,
    from_value_space: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    if prediction_type == "epsilon":
        x0_hat_vs = predict_x0_from_eps(x_t, t, pred_main, diff)
    else:
        x0_hat_vs = pred_main
    return from_value_space(x0_hat_vs)
