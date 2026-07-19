#!/usr/bin/env python
from __future__ import annotations

import datetime
import json
import os
from typing import Optional

import numpy as np
import torch

from models.vfx_model import UNet3DModel
from utils.hf_inference_runtime import load_hf_pipeline_bundle
from utils.ops import log1p_m11_inverse_torch, safe_tag, sigmoid_np
from utils.tools import ensure_dir, parse_args_cfg


def _bool(v, default: bool = False) -> bool:
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)):
        return bool(int(v))
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)


def main() -> None:
    args = parse_args_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = str(getattr(args, "ckpt", "") or "").strip()
    if not ckpt:
        raise RuntimeError(
            "Missing ckpt. Pass a local checkpoint directory or a Hugging Face model repo id."
        )
    ckpt_subfolder = str(getattr(args, "ckpt_subfolder", "") or "").strip() or None
    ckpt_revision = str(getattr(args, "ckpt_revision", "") or "").strip() or None

    out_dir = ensure_dir(str(getattr(args, "out_dir", "results/infer")))
    step_name = str(getattr(args, "step_name", "") or "").strip()
    if not step_name:
        step_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_dir(os.path.join(out_dir, step_name))

    timesteps = getattr(args, "timesteps", None)
    beta_start = getattr(args, "beta_start", None)
    beta_end = getattr(args, "beta_end", None)

    bundle = load_hf_pipeline_bundle(
        model_cls=UNet3DModel,
        ckpt_path=ckpt,
        device=device,
        use_ema=_bool(getattr(args, "use_ema", True), True),
        ckpt_subfolder=ckpt_subfolder,
        ckpt_revision=ckpt_revision,
        scheduler_name=str(getattr(args, "scheduler_name", "ddpm")),
        scheduler_legacy_align=_bool(getattr(args, "scheduler_legacy_align", False), False),
        scheduler_mode=getattr(args, "scheduler_mode", None),
        prediction_type=str(getattr(args, "eval_prediction_type", "auto")),
        train_timesteps_override=None if timesteps is None else int(timesteps),
        beta_start_override=None if beta_start is None else float(beta_start),
        beta_end_override=None if beta_end is None else float(beta_end),
        value_space_fallback=str(getattr(args, "value_space", "raw")),
        log1p_scale_fallback=float(getattr(args, "log1p_scale", 0.02)),
        volume_size_fallback=int(getattr(args, "volume_size", 32)),
    )

    seed = int(getattr(args, "seed", 123))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    num_samples = int(getattr(args, "eval_num_samples", 1))
    batch_size = int(getattr(args, "eval_batch_size", 1))
    seq_len = max(1, int(getattr(args, "eval_seq_len", 1)))
    if not bundle.temporal_model:
        seq_len = 1

    sample_steps = int(getattr(args, "sampling_steps", 0) or 0)
    if sample_steps <= 0:
        sample_steps = int(bundle.train_timesteps)

    cfg_scale = float(getattr(args, "eval_cfg_scale", 0.0) or 0.0)
    cfg_enabled = cfg_scale > 0.0 and bundle.num_classes > 0
    fixed_class_id: Optional[int] = None
    raw_class_id = getattr(args, "eval_cfg_class_id", -1)
    if raw_class_id is not None and int(raw_class_id) >= 0:
        fixed_class_id = int(raw_class_id)

    occ_gating = (not _bool(getattr(args, "eval_disable_occ_gating", False), False)) and (
        _bool(getattr(args, "eval_force_occ_gating", False), False) or bundle.use_occ_head
    )
    occ_thr = float(getattr(args, "eval_occ_gating_thr", 0.5))
    occ_soft_k = float(getattr(args, "eval_occ_soft_k", 12.0))
    occ_mode = str(getattr(args, "eval_occ_gating_mode", "soft")).lower()

    class_names = bundle.inference_meta.get("class_names", None)
    records = []
    seq_base = 0
    while seq_base < num_samples:
        cur = min(batch_size, num_samples - seq_base)
        if cfg_enabled:
            if fixed_class_id is None:
                y = torch.randint(0, int(bundle.num_classes), (cur,), device=device, dtype=torch.long)
            else:
                y = torch.full((cur,), int(fixed_class_id), device=device, dtype=torch.long)
        else:
            y = None

        if bundle.temporal_model:
            prev0 = torch.zeros((cur, 1, bundle.volume_size, bundle.volume_size, bundle.volume_size), device=device)
        else:
            prev0 = None

        for frame in range(seq_len):
            latents = torch.randn((cur, 1, bundle.volume_size, bundle.volume_size, bundle.volume_size), device=device)
            fixed_prev_noise = torch.randn_like(prev0) if (prev0 is not None and _bool(getattr(args, "eval_prev_fixed_eps", True), True)) else None
            out = bundle.pipe(
                batch_size=cur,
                class_labels=y,
                num_inference_steps=sample_steps,
                guidance_scale=cfg_scale if cfg_enabled else 0.0,
                only_cfg_eps=_bool(getattr(args, "only_cfg_eps", False), False),
                prev0=prev0,
                prev_noised=_bool(getattr(args, "eval_prev_noised", True), True),
                prev_fixed_noise=fixed_prev_noise,
                volume_size=bundle.volume_size,
                latents=latents,
            )
            x0 = out.x0_pred
            if bundle.temporal_model:
                prev0 = x0.detach()
            if bundle.value_space == "log1p":
                x0 = log1p_m11_inverse_torch(x0, scale=float(bundle.log1p_scale))
            vol = ((x0.detach().cpu().numpy() + 1.0) * 0.5).clip(0.0, 1.0)
            occ = None
            if out.occ_logits is not None:
                occ = torch.sigmoid(out.occ_logits).detach().cpu().numpy().astype(np.float32)
                if occ_gating:
                    if occ_mode == "hard":
                        gate = (occ > occ_thr).astype(np.float32)
                    else:
                        gate = sigmoid_np((occ - occ_thr) * occ_soft_k)
                    vol = vol * gate

            for i in range(cur):
                seq_id = seq_base + i
                class_id = int(y[i].item()) if y is not None else None
                class_name = None
                if isinstance(class_names, list) and class_id is not None and class_id < len(class_names):
                    class_name = str(class_names[class_id])
                tag = safe_tag(class_name) if class_name else (f"class{class_id}" if class_id is not None else "uncond")
                seq_dir = ensure_dir(os.path.join(out_dir, f"seq_{seq_id:03d}_{tag}"))
                npz_path = os.path.join(seq_dir, f"f{frame:03d}.npz")
                np.savez_compressed(
                    npz_path,
                    volume=vol[i, 0].astype(np.float32),
                    occ_prob=None if occ is None else occ[i, 0].astype(np.float32),
                    class_id=-1 if class_id is None else int(class_id),
                    frame=int(frame),
                    seq_id=int(seq_id),
                )
                records.append({
                    "seq_id": int(seq_id),
                    "frame": int(frame),
                    "class_id": class_id,
                    "class_name": class_name,
                    "path": os.path.relpath(npz_path, out_dir),
                })

        seq_base += cur

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "checkpoint": ckpt,
            "checkpoint_subfolder": ckpt_subfolder,
            "checkpoint_revision": ckpt_revision,
            "seed": seed,
            "num_samples": num_samples,
            "seq_len": seq_len,
            "prediction_type": bundle.prediction_type,
            "scheduler": type(bundle.scheduler).__name__,
            "records": records,
        }, handle, indent=2)
    print(f"[DONE] inference outputs: {out_dir}")


if __name__ == "__main__":
    main()
