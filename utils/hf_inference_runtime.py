from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

import torch

from pipelines.vfxdb_dense_pipeline import VfxDBDensePipeline
from utils.hf_checkpoint import (
    LoadedHFCheckpoint,
    ResolvedHFRuntimeBundle,
    load_hf_checkpoint,
    normalize_hf_prediction_type,
    resolve_hf_runtime_bundle,
)
from utils.hf_scheduler_compat import resolve_scheduler_choice


@dataclass
class LoadedHFPipelineBundle:
    checkpoint: LoadedHFCheckpoint
    runtime: ResolvedHFRuntimeBundle
    pipe: VfxDBDensePipeline
    model: Any
    scheduler: Any
    train_meta: Dict[str, Any]
    inference_meta: Dict[str, Any]
    model_config: Any
    prediction_type: str
    train_timesteps: int
    in_channels: int
    out_channels: int
    volume_size: int
    num_classes: int
    uncond_id: int
    temporal_model: bool
    use_occ_head: bool
    value_space: str
    log1p_scale: float


def load_hf_pipeline_bundle(
    *,
    model_cls: Type,
    ckpt_path: str,
    device: torch.device,
    use_ema: bool,
    ckpt_subfolder: Optional[str] = None,
    ckpt_revision: Optional[str] = None,
    scheduler_name: str = "ddpm",
    scheduler_legacy_align: bool = False,
    scheduler_mode: Optional[str] = None,
    prediction_type: str = "auto",
    train_timesteps_override: Optional[int] = None,
    beta_start_override: Optional[float] = None,
    beta_end_override: Optional[float] = None,
    value_space_fallback: str = "raw",
    log1p_scale_fallback: float = 0.02,
    volume_size_fallback: int = 32,
) -> LoadedHFPipelineBundle:
    scheduler_name, scheduler_legacy_align = resolve_scheduler_choice(
        scheduler_name=str(scheduler_name),
        scheduler_legacy_align=bool(scheduler_legacy_align),
        scheduler_mode=scheduler_mode,
    )
    checkpoint = load_hf_checkpoint(
        model_cls=model_cls,
        ckpt_path=ckpt_path,
        device=device,
        use_ema=use_ema,
        ckpt_subfolder=ckpt_subfolder,
        ckpt_revision=ckpt_revision,
    )
    runtime = resolve_hf_runtime_bundle(
        checkpoint=checkpoint,
        prediction_type=prediction_type,
        train_timesteps_override=train_timesteps_override,
        beta_start_override=beta_start_override,
        beta_end_override=beta_end_override,
    )
    pipe = VfxDBDensePipeline.from_scheduler_config(
        unet=checkpoint.model,
        scheduler_name=scheduler_name,
        scheduler_legacy_align=scheduler_legacy_align,
        scheduler_mode=scheduler_mode,
        num_train_timesteps=runtime.train_timesteps,
        beta_start=runtime.beta_start,
        beta_end=runtime.beta_end,
        prediction_type=runtime.prediction_type,
    ).to(device)

    model = pipe.unet
    model.eval()
    torch.set_grad_enabled(False)

    train_meta = dict(checkpoint.train_meta)
    inference_meta = dict(checkpoint.inference_meta)
    model_config = model.config
    in_channels = int(model_config.in_channels)
    out_channels = int(model_config.out_channels)
    num_classes = int(getattr(model_config, "num_classes", 0))
    uncond_id = int(getattr(model, "uncond_id", num_classes))
    temporal_model = (in_channels == 2)
    use_occ_head = (out_channels >= 2)
    value_space = str(inference_meta.get("value_space", value_space_fallback)).lower()
    if value_space not in ("raw", "log1p"):
        value_space = "raw"
    log1p_scale = float(inference_meta.get("log1p_scale", log1p_scale_fallback))
    volume_size = int(getattr(model_config, "sample_size", volume_size_fallback))

    return LoadedHFPipelineBundle(
        checkpoint=checkpoint,
        runtime=runtime,
        pipe=pipe,
        model=model,
        scheduler=pipe.scheduler,
        train_meta=train_meta,
        inference_meta=inference_meta,
        model_config=model_config,
        prediction_type=normalize_hf_prediction_type(runtime.prediction_type),
        train_timesteps=int(runtime.train_timesteps),
        in_channels=in_channels,
        out_channels=out_channels,
        volume_size=volume_size,
        num_classes=num_classes,
        uncond_id=uncond_id,
        temporal_model=temporal_model,
        use_occ_head=use_occ_head,
        value_space=value_space,
        log1p_scale=log1p_scale,
    )
