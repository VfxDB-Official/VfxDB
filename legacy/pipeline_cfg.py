from __future__ import annotations

from typing import Optional, Tuple

import torch


@torch.no_grad()
def run_legacy_cfg_step(
    *,
    unet,
    x_t: torch.Tensor,
    prev_t: Optional[torch.Tensor],
    t_batch: torch.Tensor,
    class_labels: torch.Tensor,
    uncond_id: int,
    guidance_scale: float,
    use_occ_head: bool,
    only_cfg_eps: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Legacy CFG path that matches the old sampler's two-pass forward order."""
    x_uncond = x_t
    x_cond = x_t
    if prev_t is not None:
        x_uncond = torch.cat([x_uncond, prev_t], dim=1)
        x_cond = torch.cat([x_cond, prev_t], dim=1)

    batch_size = int(x_t.shape[0])
    out_uncond = unet(
        sample=x_uncond,
        timestep=t_batch,
        class_labels=torch.full((batch_size,), uncond_id, device=x_t.device, dtype=torch.long),
        return_dict=True,
    )
    out_cond = unet(
        sample=x_cond,
        timestep=t_batch,
        class_labels=class_labels,
        return_dict=True,
    )

    main_uncond = out_uncond.sample if hasattr(out_uncond, "sample") else out_uncond["sample"]
    main_cond = out_cond.sample if hasattr(out_cond, "sample") else out_cond["sample"]

    if use_occ_head:
        val_uncond = main_uncond[:, 0:1, ...]
        val_cond = main_cond[:, 0:1, ...]
        occ_uncond = main_uncond[:, 1:2, ...]
        occ_cond = main_cond[:, 1:2, ...]
    else:
        val_uncond = main_uncond
        val_cond = main_cond
        occ_uncond = None
        occ_cond = None

    val_pred = val_uncond + guidance_scale * (val_cond - val_uncond)
    if not use_occ_head:
        return val_pred, None
    if only_cfg_eps:
        return val_pred, occ_uncond
    return val_pred, occ_uncond + guidance_scale * (occ_cond - occ_uncond)
