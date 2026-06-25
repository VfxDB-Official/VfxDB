from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from diffusers import DDPMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
from diffusers.utils.torch_utils import randn_tensor


class LegacyAlignedDDPMScheduler(DDPMScheduler):
    """DDPM scheduler that matches the legacy manual sampling loop."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._legacy_dtype = self.betas.dtype

    def _rebuild_legacy_schedule(
        self,
        num_steps: int,
        device: Union[str, torch.device, None] = None,
    ):
        self.betas = torch.linspace(
            float(self.config.beta_start),
            float(self.config.beta_end),
            int(num_steps),
            dtype=self._legacy_dtype,
            device=device,
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0, dtype=self.alphas.dtype, device=device)

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device, None] = None,
        timesteps=None,
    ):
        if timesteps is not None:
            return super().set_timesteps(num_inference_steps=None, device=device, timesteps=timesteps)

        if num_inference_steps is None:
            raise ValueError("num_inference_steps must be provided")

        self.num_inference_steps = int(num_inference_steps)
        self.custom_timesteps = False
        self._rebuild_legacy_schedule(self.num_inference_steps, device=device)
        self.timesteps = torch.arange(
            self.num_inference_steps - 1,
            -1,
            -1,
            dtype=torch.long,
            device=device,
        )

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[DDPMSchedulerOutput, Tuple[torch.Tensor, torch.Tensor]]:
        t = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
        batch_size = int(sample.shape[0])
        t_batch = torch.full((batch_size,), t, device=sample.device, dtype=torch.long)

        alphas_cumprod = self.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        betas = self.betas.to(device=sample.device, dtype=sample.dtype)
        alphas = self.alphas.to(device=sample.device, dtype=sample.dtype)

        alpha_prod_t = alphas_cumprod[t_batch].view(-1, 1, 1, 1, 1)
        alpha_prod_t_prev = (
            alphas_cumprod[(t_batch - 1).clamp(min=0)].view(-1, 1, 1, 1, 1)
            if t > 0
            else torch.ones((batch_size, 1, 1, 1, 1), device=sample.device, dtype=sample.dtype)
        )
        beta_t = betas[t_batch].view(-1, 1, 1, 1, 1)
        alpha_t = alphas[t_batch].view(-1, 1, 1, 1, 1)

        prediction_type = str(self.config.prediction_type)
        if prediction_type == "epsilon":
            pred_original_sample = (sample - (1.0 - alpha_prod_t).sqrt() * model_output) / (alpha_prod_t.sqrt() + 1e-12)
        elif prediction_type == "sample":
            pred_original_sample = model_output
        elif prediction_type == "v_prediction":
            pred_original_sample = alpha_prod_t.sqrt() * sample - (1.0 - alpha_prod_t).sqrt() * model_output
        else:
            raise ValueError(f"Unsupported prediction_type: {prediction_type}")

        pred_original_sample = pred_original_sample.clamp(-1.0, 1.0)

        if t > 0:
            coef1 = alpha_prod_t_prev.sqrt() * beta_t / (1.0 - alpha_prod_t + 1e-12)
            coef2 = alpha_t.sqrt() * (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t + 1e-12)
            pred_prev_sample = coef1 * pred_original_sample + coef2 * sample

            posterior_var = beta_t * (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t + 1e-12)
            noise = randn_tensor(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            pred_prev_sample = pred_prev_sample + (posterior_var + 1e-12).sqrt() * noise
        else:
            pred_prev_sample = pred_original_sample

        if not return_dict:
            return pred_prev_sample, pred_original_sample
        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)
