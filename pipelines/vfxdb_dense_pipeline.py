import torch
from typing import Any, Dict, Optional, Union, Tuple
from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput

from legacy.pipeline_cfg import run_legacy_cfg_step
from utils.hf_scheduler_compat import build_scheduler_bundle


class VfxDBPipelineOutput(BaseOutput):
    """Minimal tensor output used by the existing eval/bench stack."""

    x0_pred: torch.Tensor
    occ_logits: Optional[torch.Tensor]

class VfxDBDensePipeline(DiffusionPipeline):
    """Diffusers pipeline for dense VDB generation.

    Design note:
    - The pipeline itself should stay scheduler-agnostic.
    - Scheduler selection and legacy/native policy are handled by the scheduler
      factory so the inference loop does not branch on scheduler type.
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self._scheduler_name = type(scheduler).__name__
        self._scheduler_legacy_align = False

    @classmethod
    def from_scheduler_config(
        cls,
        *,
        unet,
        scheduler_name: str = "ddpm",
        scheduler_legacy_align: bool = False,
        scheduler_mode: Optional[str] = None,
        num_train_timesteps: int,
        beta_start: float,
        beta_end: float,
        prediction_type: str,
        **scheduler_kwargs,
    ):
        """Build a pipeline with a scheduler resolved from runtime config."""
        bundle = build_scheduler_bundle(
            scheduler_name=scheduler_name,
            scheduler_legacy_align=scheduler_legacy_align,
            scheduler_mode=scheduler_mode,
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            prediction_type=prediction_type,
            **scheduler_kwargs,
        )
        pipe = cls(unet=unet, scheduler=bundle.scheduler)
        pipe._scheduler_name = bundle.name
        pipe._scheduler_legacy_align = bundle.legacy_align
        return pipe

    def configure_scheduler(
        self,
        *,
        scheduler_name: str = "ddpm",
        scheduler_legacy_align: bool = False,
        scheduler_mode: Optional[str] = None,
        num_train_timesteps: int,
        beta_start: float,
        beta_end: float,
        prediction_type: str,
        **scheduler_kwargs,
    ):
        """Hot-swap scheduler policy on an existing pipeline instance."""
        bundle = build_scheduler_bundle(
            scheduler_name=scheduler_name,
            scheduler_legacy_align=scheduler_legacy_align,
            scheduler_mode=scheduler_mode,
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            prediction_type=prediction_type,
            **scheduler_kwargs,
        )
        self.scheduler = bundle.scheduler
        self._scheduler_name = bundle.name
        self._scheduler_legacy_align = bundle.legacy_align
        return bundle

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        class_labels: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 0.0,
        only_cfg_eps: bool = False,
        prev0: Optional[torch.Tensor] = None,
        prev_noised: bool = True,
        prev_fixed_noise: Optional[torch.Tensor] = None,
        volume_size: int = 32,
        latents: Optional[torch.Tensor] = None,
    ) -> VfxDBPipelineOutput:
        if latents is not None:
            device = latents.device
        else:
            try:
                device = next(self.unet.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        in_ch = self.unet.config.in_channels
        out_ch = self.unet.config.out_channels
        num_classes = getattr(self.unet.config, "num_classes", 0)
        uncond_id = getattr(self.unet, "uncond_id", num_classes)

        temporal_model = int(in_ch) == 2
        use_occ_head = out_ch >= 2
        cfg_enabled = guidance_scale > 0.0 and class_labels is not None

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        if latents is None:
            shape = (batch_size, 1, volume_size, volume_size, volume_size)
            latents = torch.randn(shape, device=device, dtype=self.unet.dtype)

        x_t = latents
        x0_pred = None
        occ_logits_pred = None

        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            prev_t = None
            if temporal_model and prev0 is not None:
                if prev_noised:
                    p_noise = prev_fixed_noise if prev_fixed_noise is not None else torch.randn_like(prev0)
                    prev_t = self.scheduler.add_noise(prev0, p_noise, t_batch)
                else:
                    prev_t = prev0

            if cfg_enabled and self._scheduler_legacy_align:
                val_pred, occ_logits = run_legacy_cfg_step(
                    unet=self.unet,
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
                if cfg_enabled:
                    x_in = torch.cat([x_t, x_t], dim=0)
                    t_in = torch.cat([t_batch, t_batch], dim=0)
                    y_in = torch.cat([
                        torch.full((batch_size,), uncond_id, device=device, dtype=torch.long),
                        class_labels,
                    ], dim=0)
                else:
                    x_in = x_t
                    t_in = t_batch
                    y_in = class_labels

                if prev_t is not None:
                    prev_in = torch.cat([prev_t, prev_t], dim=0) if cfg_enabled else prev_t
                    x_in = torch.cat([x_in, prev_in], dim=1)

                out = self.unet(sample=x_in, timestep=t_in, class_labels=y_in, return_dict=True)
                main_pred = out.sample if hasattr(out, "sample") else out["sample"]

                if use_occ_head:
                    val_pred = main_pred[:, 0:1, ...]
                    occ_logits = main_pred[:, 1:2, ...]
                else:
                    val_pred = main_pred
                    occ_logits = None

                if cfg_enabled:
                    val_uncond, val_cond = val_pred.chunk(2)
                    val_pred = val_uncond + guidance_scale * (val_cond - val_uncond)

                    if use_occ_head:
                        occ_uncond, occ_cond = occ_logits.chunk(2)
                        if only_cfg_eps:
                            occ_logits = occ_uncond
                        else:
                            occ_logits = occ_uncond + guidance_scale * (occ_cond - occ_uncond)

            step_res = self.scheduler.step(val_pred, t, x_t)
            x_t = step_res.prev_sample
            x0_pred = step_res.pred_original_sample
            occ_logits_pred = occ_logits

        return VfxDBPipelineOutput(x0_pred=x0_pred, occ_logits=occ_logits_pred)
