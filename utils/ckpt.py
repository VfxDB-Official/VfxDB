import os, json, argparse, random, subprocess, sys
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .diffusion_utils import DiffusionConfig


# ============================================================
# EMA
# ============================================================
class EMA:
    def __init__(self, model, decay: float = 0.9999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        sd = model.state_dict()
        for k in self.shadow.keys():
            self.shadow[k].mul_(self.decay).add_(sd[k], alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


# ============================================================
# ckpt + auto eval
# ============================================================
def save_ckpt(path: str, model, opt, ema: EMA, step: int, cfg: DiffusionConfig, extra: dict):
    torch.save({
        "step": int(step),
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "ema": ema.shadow if ema is not None else None,
        "diff": {
            "T": int(cfg.T),
            "beta_start": float(cfg.betas[0].detach().cpu()),
            "beta_end": float(cfg.betas[-1].detach().cpu())
        },
        "extra": extra,
    }, path)


def load_ckpt(path: str, device: torch.device, use_ema: bool) -> Tuple[dict, dict, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    sd = ckpt.get("ema", None) if use_ema else None
    if not isinstance(sd, dict):
        sd = ckpt.get("model", None)
    if not isinstance(sd, dict):
        raise RuntimeError(f"Checkpoint format error: cannot find state dict in {path}")
    extra = ckpt.get("extra", {})
    if not isinstance(extra, dict):
        extra = {}
    return ckpt, sd, extra
