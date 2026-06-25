#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


@dataclass
class DiffusionConfig:
    T: int
    betas: torch.Tensor
    alphas: torch.Tensor
    abar: torch.Tensor
    abar_prev: torch.Tensor
    sqrt_abar: torch.Tensor
    sqrt_1m_abar: torch.Tensor
    posterior_var: torch.Tensor

    @staticmethod
    def build(T: int, device: torch.device, beta_start=1e-4, beta_end=2e-2) -> "DiffusionConfig":
        betas = make_beta_schedule(T, beta_start, beta_end).to(device)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        abar_prev = torch.cat([torch.tensor([1.0], device=device), abar[:-1]], dim=0)
        sqrt_abar = torch.sqrt(abar)
        sqrt_1m_abar = torch.sqrt(1.0 - abar)
        posterior_var = betas * (1.0 - abar_prev) / (1.0 - abar + 1e-12)
        posterior_var[0] = betas[0]
        return DiffusionConfig(T, betas, alphas, abar, abar_prev, sqrt_abar, sqrt_1m_abar, posterior_var)

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        ab = self.abar[t]
        return ab / (1.0 - ab + 1e-12)


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, cfg: DiffusionConfig) -> torch.Tensor:
    sa = cfg.sqrt_abar[t].view(-1, 1, 1, 1, 1)
    so = cfg.sqrt_1m_abar[t].view(-1, 1, 1, 1, 1)
    return sa * x0 + so * noise


def predict_x0_from_eps(x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, cfg: DiffusionConfig) -> torch.Tensor:
    sa = cfg.sqrt_abar[t].view(-1, 1, 1, 1, 1)
    so = cfg.sqrt_1m_abar[t].view(-1, 1, 1, 1, 1)
    return (x_t - so * eps) / (sa + 1e-12)


def eps_mse_loss(eps_pred: torch.Tensor, eps_gt: torch.Tensor, t: torch.Tensor, cfg: DiffusionConfig, p2_gamma: float = 0.0) -> torch.Tensor:
    """
    Standard eps MSE, optional p2 weighting:
      w = (1 / (k + snr))^gamma
    """
    if p2_gamma <= 0:
        return F.mse_loss(eps_pred, eps_gt)

    snr = cfg.snr(t).view(-1, 1, 1, 1, 1)
    k = 1.0
    w = (1.0 / (k + snr)).pow(p2_gamma)
    return ((eps_pred - eps_gt) ** 2 * w).mean()

@torch.no_grad()
def p_sample_step(
    model,
    x_t: torch.Tensor,
    t_int: int,
    cfg: DiffusionConfig,
    clamp_x0: bool = True,
    x0_clip=(-1.0, 1.0),
) -> torch.Tensor:
    B = x_t.shape[0]
    t = torch.full((B,), int(t_int), device=x_t.device, dtype=torch.long)

    eps = model(x_t, t)
    # ---- your original DDPM step ----
    x0 = predict_x0_from_eps(x_t, t, eps, cfg)
    if clamp_x0:
        x0 = x0.clamp(float(x0_clip[0]), float(x0_clip[1]))

    if t_int == 0:
        return x0

    abar_t = cfg.abar[t_int]
    abar_prev = cfg.abar_prev[t_int]
    beta_t = cfg.betas[t_int]
    alpha_t = cfg.alphas[t_int]

    coef1 = (torch.sqrt(abar_prev) * beta_t) / (1.0 - abar_t)
    coef2 = (torch.sqrt(alpha_t) * (1.0 - abar_prev)) / (1.0 - abar_t)
    mean = coef1 * x0 + coef2 * x_t

    var = cfg.posterior_var[t_int]
    noise = torch.randn_like(x_t)
    return mean + torch.sqrt(var) * noise

@torch.no_grad()
def p_sample_loop(model, shape, cfg: DiffusionConfig, device: torch.device, clamp_x0: bool = True, x0_clip=(-1.0, 1.0)) -> torch.Tensor:
    x = torch.randn(shape, device=device)
    for t in reversed(range(cfg.T)):
        x = p_sample_step(model, x, t, cfg, clamp_x0=clamp_x0, x0_clip=x0_clip)
    return x
