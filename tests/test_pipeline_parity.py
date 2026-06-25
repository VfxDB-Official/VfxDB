#!/usr/bin/env python
from __future__ import annotations

import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legacy.pipeline_cfg import run_legacy_cfg_step
from models.vfx_model import UNet3DModel
from pipelines.vfxdb_dense_pipeline import VfxDBDensePipeline


@torch.no_grad()
def manual_loop(
    pipe: VfxDBDensePipeline,
    *,
    latents: torch.Tensor,
    class_labels: torch.Tensor,
    prev0: torch.Tensor,
    prev_fixed_noise: torch.Tensor,
    num_inference_steps: int,
    guidance_scale: float,
    only_cfg_eps: bool,
):
    device = latents.device
    model = pipe.unet
    scheduler = pipe.scheduler
    batch_size = int(latents.shape[0])
    use_occ_head = int(model.config.out_channels) >= 2
    uncond_id = int(getattr(model, "uncond_id", int(model.config.num_classes)))

    scheduler.set_timesteps(num_inference_steps, device=device)
    x_t = latents.clone()
    x0_pred = None
    occ_logits_pred = None

    for t in scheduler.timesteps:
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        prev_t = scheduler.add_noise(prev0, prev_fixed_noise, t_batch)

        if guidance_scale > 0 and getattr(pipe, "_scheduler_legacy_align", False):
            val_pred, occ_logits = run_legacy_cfg_step(
                unet=model,
                x_t=x_t,
                prev_t=prev_t,
                t_batch=t_batch,
                class_labels=class_labels,
                uncond_id=uncond_id,
                guidance_scale=guidance_scale,
                use_occ_head=use_occ_head,
                only_cfg_eps=only_cfg_eps,
            )
        else:
            x_in = torch.cat([x_t, x_t], dim=0)
            t_in = torch.cat([t_batch, t_batch], dim=0)
            y_in = torch.cat([
                torch.full((batch_size,), uncond_id, device=device, dtype=torch.long),
                class_labels,
            ], dim=0)
            prev_in = torch.cat([prev_t, prev_t], dim=0)
            x_in = torch.cat([x_in, prev_in], dim=1)

            out = model(sample=x_in, timestep=t_in, class_labels=y_in, return_dict=True)
            main_pred = out["sample"] if isinstance(out, dict) else out.sample
            if use_occ_head:
                val_pred = main_pred[:, 0:1]
                occ_logits = main_pred[:, 1:2]
            else:
                val_pred = main_pred
                occ_logits = None

            val_uncond, val_cond = val_pred.chunk(2)
            val_pred = val_uncond + guidance_scale * (val_cond - val_uncond)
            if use_occ_head:
                occ_uncond, occ_cond = occ_logits.chunk(2)
                occ_logits = occ_uncond if only_cfg_eps else occ_uncond + guidance_scale * (occ_cond - occ_uncond)

        step_res = scheduler.step(val_pred, t, x_t)
        x_t = step_res.prev_sample
        x0_pred = step_res.pred_original_sample
        occ_logits_pred = occ_logits

    return x0_pred, occ_logits_pred


def run_case(*, legacy_align: bool) -> None:
    torch.manual_seed(20260613)
    device = torch.device("cpu")
    model = UNet3DModel(
        in_channels=2,
        out_channels=2,
        base_channels=4,
        channel_mults=(1, 2),
        time_emb_dim=32,
        num_classes=3,
        sample_size=8,
    ).to(device)
    model.eval()

    pipe = VfxDBDensePipeline.from_scheduler_config(
        unet=model,
        scheduler_name="ddpm",
        scheduler_legacy_align=legacy_align,
        num_train_timesteps=8,
        beta_start=1e-4,
        beta_end=2e-2,
        prediction_type="epsilon",
    ).to(device)

    batch_size = 2
    latents = torch.randn((batch_size, 1, 8, 8, 8), device=device)
    prev0 = torch.randn((batch_size, 1, 8, 8, 8), device=device).clamp(-1.0, 1.0)
    prev_noise = torch.randn_like(prev0)
    labels = torch.tensor([0, 2], device=device, dtype=torch.long)

    rng_state = torch.get_rng_state()
    manual_x0, manual_occ = manual_loop(
        pipe,
        latents=latents,
        class_labels=labels,
        prev0=prev0,
        prev_fixed_noise=prev_noise,
        num_inference_steps=4,
        guidance_scale=1.75,
        only_cfg_eps=True,
    )
    torch.set_rng_state(rng_state)
    pipe_out = pipe(
        batch_size=batch_size,
        class_labels=labels,
        num_inference_steps=4,
        guidance_scale=1.75,
        only_cfg_eps=True,
        prev0=prev0,
        prev_noised=True,
        prev_fixed_noise=prev_noise,
        volume_size=8,
        latents=latents.clone(),
    )

    dx = torch.max(torch.abs(manual_x0 - pipe_out.x0_pred)).item()
    do = torch.max(torch.abs(manual_occ - pipe_out.occ_logits)).item()
    print(f"legacy_align={legacy_align} x0_max_diff={dx:.8e} occ_max_diff={do:.8e}")
    assert dx < 1e-6, dx
    assert do < 1e-6, do


if __name__ == "__main__":
    run_case(legacy_align=False)
    run_case(legacy_align=True)
    print("[PASS] pipeline parity")
