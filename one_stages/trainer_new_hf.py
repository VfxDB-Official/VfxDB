# one_stages/trainer.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import tqdm

from utils.ops import upsample_nearest
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from utils.diffusion_math import p2_weight_from_snr, snr_from_alpha_bar
from utils.hf_train_runtime import (
    append_eval_process,
    build_project_configuration,
    build_one_stage_inference_meta,
    build_one_stage_train_meta,
    build_one_stage_train_state,
    cleanup_eval_processes,
    close_train_runtime,
    init_train_runtime,
    launch_eval_subprocess,
    log_train_metrics,
    prepare_run_dir,
    resolve_log_backend,
    save_hf_training_checkpoint,
)
from utils.value_space import from_value_space, to_value_space
from legacy.train_align import (
    build_legacy_train_batch,
    compute_legacy_timestep_weights,
    predict_x0_raw_legacy,
)
from tile_dataloader import collate_volume_batch
from one_stages.model_adapter import (
    ModelConditioning,
    ModelPredictions,
    build_default_model_adapter,
)

# -----------------------------------------------------------------------------
# Resume helper: inflate in_channels from 1 -> 2 (temporal warm-start)
# -----------------------------------------------------------------------------
def _inflate_first_conv_in_channels(sd: Dict[str, Any], new_in: int = 2) -> bool:
    """
    If checkpoint was trained with in_channels=1 and current model expects in_channels=2,
    we can warm-start by copying the old weight into channel0 and zeroing channel1.

    Returns True if inflated.
    """
    if not isinstance(sd, dict):
        return False

    # common key candidates
    candidates = [
        "in_conv.weight",
        "input_conv.weight",
        "conv_in.weight",
        "init_conv.weight",
        "stem.weight",
    ]
    for k in candidates:
        if k in sd and torch.is_tensor(sd[k]) and sd[k].ndim == 5 and int(sd[k].shape[1]) == 1:
            w = sd[k]
            out_c, _, kx, ky, kz = w.shape
            w2 = torch.zeros((out_c, new_in, kx, ky, kz), device=w.device, dtype=w.dtype)
            w2[:, 0:1, ...] = w
            sd[k] = w2
            print(f"[temporal][resume] inflated {k}: in_channels 1 -> {new_in} (ch1=0)")
            return True

    # fallback: heuristic scan
    for k, v in list(sd.items()):
        if not (isinstance(k, str) and k.endswith(".weight")):
            continue
        if torch.is_tensor(v) and v.ndim == 5 and int(v.shape[1]) == 1:
            # very likely the first conv (but could be others; still usually ok)
            w = v
            out_c, _, kx, ky, kz = w.shape
            w2 = torch.zeros((out_c, new_in, kx, ky, kz), device=w.device, dtype=w.dtype)
            w2[:, 0:1, ...] = w
            sd[k] = w2
            print(f"[temporal][resume] inflated {k}: in_channels 1 -> {new_in} (heuristic, ch1=0)")
            return True

    return False

# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
@dataclass
class TrainBatchState:
    x0_raw: torch.Tensor
    x0: torch.Tensor
    x_t: torch.Tensor
    noise: torch.Tensor
    t: torch.Tensor
    x_in: torch.Tensor
    occ_gt: Optional[torch.Tensor]
    conditioning: ModelConditioning


@dataclass
class LossBundle:
    total: torch.Tensor
    value: torch.Tensor
    occ: torch.Tensor
    occ_ds: torch.Tensor
    leak: torch.Tensor
    residual: torch.Tensor
    w_p2: torch.Tensor


class Trainer:
    def __init__(
        self,
        cfg,
        device,
        ds,
        splits,
        model,
        opt,
        ema,
        diff,
        feat_extractor,
        use_occupancy: bool,
        out_channels: int,
        class_names,
        model_adapter=None,
    ):
        self.cfg = cfg.raw if hasattr(cfg, "raw") else cfg
        self.run_dir = prepare_run_dir(self.cfg)
        self.accelerator = Accelerator(
            mixed_precision=str(self.cfg.get("mixed_precision", "no")).lower(),
            gradient_accumulation_steps=int(self.cfg.get("gradient_accumulation_steps", 1)),
            log_with=resolve_log_backend(),
            project_config=build_project_configuration(self.run_dir),
        )
        self.device = self.accelerator.device
        self.ds = ds
        self.model = model
        self.opt = opt
        self.ema = ema
        self.diff = diff
        self.noise_scheduler = self._build_noise_scheduler(diff)
        self.feat_extractor = feat_extractor
        self.model_adapter = model_adapter or build_default_model_adapter(use_occupancy=use_occupancy)
        self.use_occupancy = bool(use_occupancy)
        self.out_channels = int(out_channels)
        self.class_names = class_names
        self.train_legacy_align = bool(self.cfg.get("train_legacy_align", False))

        # temporal相关开关
        self.temporal_enable = bool(self.cfg.get("temporal_enable", False))
        self.prev_noised = bool(self.cfg.get("prev_noised", True))               # recommend True
        self.prev_same_t = bool(self.cfg.get("prev_same_t", True))               # recommend True
        self.prev_drop_prob = float(self.cfg.get("prev_drop_prob", 0.0))         # 0~0.1 optional

        # loss配置
        self.loss_type = str(self.cfg.get("loss_function", "l1_l2"))
        self.lambda_l1 = float(self.cfg.get("lambda_l1", 0.2))
        if "occ_deep_supervision" not in self.cfg:
            self.cfg["occ_deep_supervision"] = not bool(self.cfg.get("no_occ_deep_supervision", False))

        # prediction类型 + p2权重
        self.prediction_type = str(self.cfg.get("prediction_type", "epsilon")).lower()
        if self.prediction_type not in ("epsilon", "x0"):
            print(f"[ONE_STAGE] unknown prediction_type={self.prediction_type}, fallback to eps")
            self.prediction_type = "epsilon"

        self.p2_k = float(self.cfg.get("p2_k", 1.0))
        self.p2_gamma = float(self.cfg.get("p2_gamma", 1.0))
        self.p2_normalize = bool(self.cfg.get("p2_normalize", True))
        self.occ_t_weighting = str(self.cfg.get("occ_t_weighting", "alpha_bar")).lower()

        # value-space transform
        self.value_space = str(self.cfg.get("value_space", "raw")).lower()
        self.log1p_scale = float(self.cfg.get("log1p_scale", 0.02))
        self.log1p_dequant_delta = float(self.cfg.get("log1p_dequant_delta", 0.0))

        # residual分支
        self.residual_weight = float(self.cfg.get("residual_weight", 0.0))
        self.residual_leaf_base = int(self.cfg.get("residual_leaf_base", 4))
        self.residual_clip = float(self.cfg.get("residual_clip", 0.0))
        self.residual_mask_by_occ = bool(self.cfg.get("residual_mask_by_occ", False))

        self.loader = self._build_dataloader(ds, device)
        self.runtime = init_train_runtime(self.cfg, self.accelerator, len(ds), run_dir=self.run_dir)

        # resume（支持单帧权重inflate到temporal）
        self.start_step = 0
        self._resume_if_needed(device)

    def _build_noise_scheduler(self, diff):
        if isinstance(diff, DDPMScheduler):
            return diff
        return DDPMScheduler(
            num_train_timesteps=int(self.cfg.get("timesteps", 1000)),
            beta_start=float(self.cfg.get("beta_start", 0.0001)),
            beta_end=float(self.cfg.get("beta_end", 0.02)),
            beta_schedule="linear",
            prediction_type=str(self.cfg.get("prediction_type", "epsilon")).lower(),
            clip_sample=False,
        )

    def _build_dataloader(self, ds, device) -> DataLoader:
        num_workers = int(self.cfg.get("num_workers", 4))
        loader_kwargs = dict(
            dataset=ds,
            batch_size=int(self.cfg.get("batch_size", 8)),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = int(self.cfg.get("prefetch_factor", 4))
            loader_kwargs["persistent_workers"] = True
        if getattr(ds, "return_meta", False):
            loader_kwargs["collate_fn"] = collate_volume_batch
        return DataLoader(**loader_kwargs)

    def _resume_if_needed(self, device) -> None:
        resume = str(self.cfg.get("resume", "")).strip()
        if not resume:
            return

        ckpt = torch.load(resume, map_location=device)
        sd = ckpt.get("model", {})
        if self.temporal_enable:
            _inflate_first_conv_in_channels(sd, new_in=2)

        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[ONE_STAGE] resume strict=False missing={len(missing)} unexpected={len(unexpected)}")
        if "opt" in ckpt:
            self.opt.load_state_dict(ckpt["opt"])
        elif "optimizer" in ckpt:
            self.opt.load_state_dict(ckpt["optimizer"])
        self.start_step = int(ckpt.get("step", 0))
        if "ema" in ckpt and isinstance(ckpt["ema"], dict):
            self.ema.shadow = ckpt["ema"]
        print(f"[ONE_STAGE] resumed from {resume}, step={self.start_step}")

    # -----------------------------
    # value-space transform wrappers
    # -----------------------------
    def _to_value_space(self, x_m11: torch.Tensor) -> torch.Tensor:
        return to_value_space(
            x_m11,
            value_space=self.value_space,
            log1p_scale=self.log1p_scale,
            log1p_dequant_delta=self.log1p_dequant_delta,
            training=self.model.training,
        )

    def _from_value_space(self, y_m11: torch.Tensor) -> torch.Tensor:
        return from_value_space(
            y_m11,
            value_space=self.value_space,
            log1p_scale=self.log1p_scale,
        )

    # -----------------------------
    # weighted voxel loss helpers
    # -----------------------------
    def _voxel_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "l1_l2":
            diff = pred - target
            l2 = diff.pow(2)
            l1 = diff.abs()
            lam = float(self.lambda_l1)
            return (1.0 - lam) * l2 + lam * l1
        else:
            lt = str(self.loss_type).lower()
            diff = pred - target
            if "l1" in lt and "l2" not in lt:
                return diff.abs()
            return diff.pow(2)

    @staticmethod
    def _masked_mean(loss_vox: torch.Tensor, mask_bool: torch.Tensor) -> torch.Tensor:
        if mask_bool is None:
            return loss_vox.mean()
        m = mask_bool.to(loss_vox.dtype)
        denom = m.sum().clamp(min=1.0)
        return (loss_vox * m).sum() / denom

    def _seed_everything(self) -> None:
        seed = int(self.cfg.get("seed", 42))
        set_seed(seed)

    def _setup_training(self):
        if bool(self.cfg.get("tf32", False)) and self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            print("[ONE_STAGE][ACCEL] TF32 enabled")

        scheduler = None
        if bool(self.cfg.get("coslr", False)):
            from torch.optim.lr_scheduler import CosineAnnealingLR

            scheduler = CosineAnnealingLR(
                self.opt,
                T_max=int(self.cfg.get("train_steps", 50000)),
                last_epoch=self.start_step - 1,
            )
            print("[ONE_STAGE] CosineAnnealingLR ON")
        else:
            print("[ONE_STAGE] LR scheduler OFF (constant)")

        torch.backends.cudnn.benchmark = True
        if scheduler is not None:
            self.model, self.opt, self.loader, scheduler = self.accelerator.prepare(
                self.model, self.opt, self.loader, scheduler
            )
        else:
            self.model, self.opt, self.loader = self.accelerator.prepare(
                self.model, self.opt, self.loader
            )
        self.model.train()
        return scheduler

    def _cleanup_eval_processes(self) -> None:
        cleanup_eval_processes(self.runtime)

    def _next_batch(self, data_iter):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(self.loader)
            batch = next(data_iter)
        return batch, data_iter

    def _build_train_batch(self, feats) -> TrainBatchState:
        x0_raw = feats.x_m11
        if self.train_legacy_align:
            x0, x_t, noise, t, x_in = build_legacy_train_batch(
                diff=self.diff,
                device=self.device,
                x0_raw=x0_raw,
                prev_x_m11=feats.prev_x_m11,
                to_value_space=self._to_value_space,
                temporal_enable=self.temporal_enable,
                prev_noised=self.prev_noised,
                prev_same_t=self.prev_same_t,
                prev_drop_prob=self.prev_drop_prob,
            )
            return TrainBatchState(
                x0_raw=x0_raw,
                x0=x0,
                x_t=x_t,
                noise=noise,
                t=t,
                x_in=x_in,
                occ_gt=feats.occ_gt,
                conditioning=self.model_adapter.build_conditioning(feats),
            )

        x0 = self._to_value_space(x0_raw)
        batch_size = int(x0.shape[0])
        t = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )
        noise = torch.randn_like(x0)
        x_t = self.noise_scheduler.add_noise(x0, noise, t)
        x_in = self._build_model_input(x_t, feats, t)
        return TrainBatchState(
            x0_raw=x0_raw,
            x0=x0,
            x_t=x_t,
            noise=noise,
            t=t,
            x_in=x_in,
            occ_gt=feats.occ_gt,
            conditioning=self.model_adapter.build_conditioning(feats),
        )

    def _build_model_input(self, x_t: torch.Tensor, feats, t: torch.Tensor) -> torch.Tensor:
        if not self.temporal_enable:
            return x_t
        if feats.prev_x_m11 is None:
            raise RuntimeError("[temporal] enabled but feats.prev_x_m11 is None. Check dataset/feature extractor.")

        prev0 = self._to_value_space(feats.prev_x_m11)
        if self.prev_noised:
            if self.prev_same_t:
                prev_t = self.noise_scheduler.add_noise(prev0, torch.randn_like(prev0), t)
            else:
                t_prev = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (int(x_t.shape[0]),),
                    device=self.device,
                    dtype=torch.long,
                )
                prev_t = self.noise_scheduler.add_noise(prev0, torch.randn_like(prev0), t_prev)
        else:
            prev_t = prev0

        if self.prev_drop_prob > 0:
            drop_mask = torch.rand((int(x_t.shape[0]),), device=self.device) < float(self.prev_drop_prob)
            if drop_mask.any():
                prev_t = prev_t.clone()
                prev_t[drop_mask] = 0.0

        return torch.cat([x_t, prev_t], dim=1)

    def _forward_model(self, batch_state: TrainBatchState) -> ModelPredictions:
        return self.model_adapter.forward(
            self.model,
            sample=batch_state.x_in,
            timestep=batch_state.t,
            conditioning=batch_state.conditioning,
        )

    def _compute_timestep_weights(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.train_legacy_align:
            return compute_legacy_timestep_weights(
                diff=self.diff,
                t=t,
                p2_k=self.p2_k,
                p2_gamma=self.p2_gamma,
                p2_normalize=self.p2_normalize,
            )
        alpha_bar = self.noise_scheduler.alphas_cumprod.to(self.device)[t].float()
        snr = snr_from_alpha_bar(alpha_bar)
        w_p2 = p2_weight_from_snr(
            snr,
            k=self.p2_k,
            gamma=self.p2_gamma,
            normalize=self.p2_normalize,
        )
        return alpha_bar, snr, w_p2

    def _compute_value_loss(
        self,
        batch_state: TrainBatchState,
        predictions: ModelPredictions,
        w_p2: torch.Tensor,
    ) -> torch.Tensor:
        target = batch_state.noise if self.prediction_type == "epsilon" else batch_state.x0
        loss_vox = self._voxel_loss(predictions.main, target) * w_p2.view(-1, 1, 1, 1, 1)

        if not (self.use_occupancy and batch_state.occ_gt is not None):
            return loss_vox.mean()

        occ_bool = batch_state.occ_gt > 0.5
        emp_bool = ~occ_bool
        loss_occ_val = self._masked_mean(loss_vox, occ_bool)
        loss_emp_val = self._masked_mean(loss_vox, emp_bool)
        return loss_occ_val + float(self.cfg.get("value_empty_weight", 0.05)) * loss_emp_val

    def _compute_occ_losses(
        self,
        batch_state: TrainBatchState,
        predictions: ModelPredictions,
        alpha_bar: torch.Tensor,
        snr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero = torch.zeros([], device=self.device)
        if not (self.use_occupancy and batch_state.occ_gt is not None and predictions.occ_logits is not None):
            return zero, zero

        if self.occ_t_weighting == "snr":
            w_occ = (snr / (snr + 1.0)).view(-1, 1, 1, 1, 1)
        else:
            w_occ = alpha_bar.view(-1, 1, 1, 1, 1)

        dens01_raw = ((batch_state.x0_raw + 1.0) * 0.5).clamp(0.0, 1.0)
        tau = float(self.cfg.get("occ_tau", 10.0))
        occ_soft = 1.0 - torch.exp(-tau * dens01_raw)

        occ_prob = torch.sigmoid(predictions.occ_logits)
        loss_occ = ((occ_prob - occ_soft).pow(2) * w_occ).mean()
        loss_occ_ds = self._compute_occ_deep_supervision_loss(occ_soft, w_occ, predictions.occ_logits)
        return loss_occ, loss_occ_ds

    def _compute_occ_deep_supervision_loss(
        self,
        occ_soft: torch.Tensor,
        w_occ: torch.Tensor,
        occ_logits: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(self.cfg.get("occ_deep_supervision", True)):
            return torch.zeros([], device=self.device)

        aux_list = self.model_adapter.get_occ_aux_logits(self.model)
        main_sp = int(occ_logits.shape[-1])
        aux_list = [
            a for a in aux_list
            if isinstance(a, torch.Tensor) and a.ndim == 5 and int(a.shape[-1]) != main_sp
        ]
        if not aux_list:
            return torch.zeros([], device=self.device)

        ds_losses = []
        for aux_logits in aux_list:
            target_sp = int(aux_logits.shape[-1])
            occ_soft_ds = F.interpolate(
                occ_soft,
                size=(target_sp, target_sp, target_sp),
                mode="trilinear",
                align_corners=False,
            )
            occ_prob_ds = torch.sigmoid(aux_logits)
            ds_losses.append(((occ_prob_ds - occ_soft_ds).pow(2) * w_occ).mean())
        return torch.stack(ds_losses).mean()

    def _predict_x0_raw(
        self,
        batch_state: TrainBatchState,
        predictions: ModelPredictions,
        alpha_bar: torch.Tensor,
    ) -> torch.Tensor:
        if self.train_legacy_align:
            return predict_x0_raw_legacy(
                x_t=batch_state.x_t,
                t=batch_state.t,
                pred_main=predictions.main,
                diff=self.diff,
                prediction_type=self.prediction_type,
                from_value_space=self._from_value_space,
            )
        if self.prediction_type == "epsilon":
            alpha_bar_ = alpha_bar.view(-1, 1, 1, 1, 1)
            sa = alpha_bar_.sqrt()
            so = (1.0 - alpha_bar_).sqrt()
            x0_hat_vs = (batch_state.x_t - so * predictions.main) / (sa + 1e-10)
        else:
            x0_hat_vs = predictions.main
        return self._from_value_space(x0_hat_vs)

    def _compute_empty_leak_loss(
        self,
        occ_gt: Optional[torch.Tensor],
        x0_hat_raw: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if occ_gt is None or x0_hat_raw is None or float(self.cfg.get("lambda_empty_leak", 0.1)) <= 0:
            return torch.zeros([], device=self.device)
        return (torch.abs(x0_hat_raw + 1.0) * (1.0 - occ_gt)).mean()

    def _compute_residual_loss(
        self,
        x0_raw: torch.Tensor,
        occ_gt: Optional[torch.Tensor],
        x0_hat_raw: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.residual_weight <= 0:
            return torch.zeros([], device=self.device)
        if x0_hat_raw is None:
            raise RuntimeError("[ONE_STAGE] residual loss requires x0_hat_raw")
        if self.residual_leaf_base <= 0:
            raise RuntimeError("[ONE_STAGE] residual_leaf_base must be > 0")

        base_m11 = self._build_residual_base(x0_raw)
        res_gt = x0_raw - base_m11
        if self.residual_clip > 0:
            res_gt = res_gt.clamp(-self.residual_clip, self.residual_clip)

        mask_bool = (occ_gt > 0.5) if (self.residual_mask_by_occ and occ_gt is not None) else None
        res_pred = x0_hat_raw - base_m11
        if self.residual_clip > 0:
            res_pred = res_pred.clamp(-self.residual_clip, self.residual_clip)

        loss_res_vox = self._voxel_loss(res_pred, res_gt)
        if mask_bool is not None:
            return self._masked_mean(loss_res_vox, mask_bool)
        return loss_res_vox.mean()

    def _build_residual_base(self, x0_raw: torch.Tensor) -> torch.Tensor:
        resolution = int(x0_raw.shape[-1])
        if resolution % self.residual_leaf_base != 0:
            raise RuntimeError(
                f"[ONE_STAGE] residual_leaf_base={self.residual_leaf_base} must divide volume size R={resolution}"
            )
        vol01 = ((x0_raw + 1.0) * 0.5).clamp(0.0, 1.0)
        mean_leaf01 = F.avg_pool3d(
            vol01,
            kernel_size=self.residual_leaf_base,
            stride=self.residual_leaf_base,
        )
        return upsample_nearest((mean_leaf01 * 2.0 - 1.0).clamp(-1.0, 1.0), target=resolution)

    def _compute_losses(
        self,
        batch_state: TrainBatchState,
        predictions: ModelPredictions,
        alpha_bar: torch.Tensor,
        snr: torch.Tensor,
        w_p2: torch.Tensor,
    ) -> LossBundle:
        loss_value = self._compute_value_loss(batch_state, predictions, w_p2)
        loss_total = loss_value

        loss_occ, loss_occ_ds = self._compute_occ_losses(batch_state, predictions, alpha_bar, snr)
        if self.use_occupancy and batch_state.occ_gt is not None:
            loss_total = loss_total + float(self.cfg.get("lambda_occ", 1.0)) * loss_occ
            loss_total = loss_total + float(self.cfg.get("lambda_occ_ds", 0.5)) * loss_occ_ds

        need_x0_hat = (
            float(self.cfg.get("lambda_empty_leak", 0.1)) > 0 and self.use_occupancy and batch_state.occ_gt is not None
        ) or (self.residual_weight > 0)
        x0_hat_raw = self._predict_x0_raw(batch_state, predictions, alpha_bar) if need_x0_hat else None

        loss_leak = self._compute_empty_leak_loss(batch_state.occ_gt, x0_hat_raw)
        if float(self.cfg.get("lambda_empty_leak", 0.1)) > 0:
            loss_total = loss_total + float(self.cfg.get("lambda_empty_leak", 0.1)) * loss_leak

        loss_res = self._compute_residual_loss(batch_state.x0_raw, batch_state.occ_gt, x0_hat_raw)
        if self.residual_weight > 0:
            loss_total = loss_total + float(self.residual_weight) * loss_res

        return LossBundle(
            total=loss_total,
            value=loss_value,
            occ=loss_occ,
            occ_ds=loss_occ_ds,
            leak=loss_leak,
            residual=loss_res,
            w_p2=w_p2,
        )

    def _optimizer_step(self, loss_total: torch.Tensor, scheduler) -> bool:
        self.accelerator.backward(loss_total)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(),
                max_norm=float(self.cfg.get("grad_clip", 1.0)),
            )
        self.opt.step()
        if (
            scheduler is not None
            and self.accelerator.sync_gradients
            and not bool(getattr(self.accelerator, "optimizer_step_was_skipped", False))
        ):
            scheduler.step()
        self.opt.zero_grad(set_to_none=True)
        if self.accelerator.sync_gradients:
            self.ema.update(self.accelerator.unwrap_model(self.model))
        return bool(self.accelerator.sync_gradients)

    def _log_step(self, global_step: int, losses: LossBundle) -> None:
        log_every = int(self.cfg.get("log_every", 50))
        if global_step % log_every != 0:
            return

        cur_lr = self.opt.param_groups[0]["lr"]
        if self.use_occupancy:
            msg = (
                f"[STEP {global_step}] loss={losses.total.item():.6f} "
                f"value={losses.value.item():.6f} occ={losses.occ.item():.6f} "
                f"occ_ds={losses.occ_ds.item():.6f} leak={losses.leak.item():.6f} "
            )
            if self.residual_weight > 0:
                msg += f"res={losses.residual.item():.6f} "
            if self.temporal_enable:
                msg += f"temporal(prev_noised={self.prev_noised}, drop={self.prev_drop_prob}) "
            msg += f"lr={cur_lr:.6e} p2(mean)={losses.w_p2.mean().item():.3f} "
            print(msg)
        else:
            msg = f"[STEP {global_step}] loss={losses.total.item():.6f} "
            if self.residual_weight > 0:
                msg += f"res={losses.residual.item():.6f} "
            if self.temporal_enable:
                msg += f"temporal(prev_noised={self.prev_noised}, drop={self.prev_drop_prob}) "
            msg += f"lr={cur_lr:.6e} p2(mean)={losses.w_p2.mean().item():.3f}"
            print(msg)

        if self.accelerator.is_main_process:
            log_train_metrics(
                self.accelerator,
                global_step=global_step,
                losses=losses,
                learning_rate=cur_lr,
                use_occupancy=self.use_occupancy,
                residual_weight=self.residual_weight,
            )

    def _build_train_meta(self) -> dict:
        return build_one_stage_train_meta(
            prediction_type=str(self.prediction_type),
            timesteps=int(self.noise_scheduler.config.num_train_timesteps),
            beta_start=float(self.noise_scheduler.config.beta_start),
            beta_end=float(self.noise_scheduler.config.beta_end),
            value_space=str(self.value_space),
            log1p_scale=float(self.log1p_scale),
            log1p_dequant_delta=float(self.log1p_dequant_delta),
            num_classes=int(self.cfg.get("num_classes", 0)),
            class_names=self.class_names,
            volume_size=int(self.cfg.get("volume_size", 32)),
            in_channels=(2 if self.temporal_enable else 1),
            out_channels=int(self.out_channels),
            use_occupancy=bool(self.use_occupancy),
            out_spec=self.cfg.get("out_spec", None),
            train_legacy_align=bool(self.train_legacy_align),
            prev_noised=bool(self.prev_noised),
            prev_same_t=bool(self.prev_same_t),
            prev_drop_prob=float(self.prev_drop_prob),
            only_cfg_eps=bool(self.cfg.get("only_cfg_eps", False)),
        )

    def _build_train_state(self, global_step: int) -> dict:
        return build_one_stage_train_state(
            global_step=global_step,
            opt_state=self.opt.state_dict(),
            ema_shadow=(self.ema.shadow if hasattr(self.ema, "shadow") else None),
            train_meta=self._build_train_meta(),
        )

    def _build_hf_inference_meta(self) -> dict:
        return build_one_stage_inference_meta(
            prediction_type=str(self.prediction_type),
            timesteps=int(self.noise_scheduler.config.num_train_timesteps),
            beta_start=float(self.noise_scheduler.config.beta_start),
            beta_end=float(self.noise_scheduler.config.beta_end),
            value_space=str(self.value_space),
            log1p_scale=float(self.log1p_scale),
            log1p_dequant_delta=float(self.log1p_dequant_delta),
            num_classes=int(self.cfg.get("num_classes", 0)),
            class_names=self.class_names,
            volume_size=int(self.cfg.get("volume_size", 32)),
            in_channels=(2 if self.temporal_enable else 1),
            out_channels=int(self.out_channels),
            use_occupancy=bool(self.use_occupancy),
            out_spec=self.cfg.get("out_spec", None),
            prev_noised=bool(self.prev_noised),
            prev_same_t=bool(self.prev_same_t),
            prev_drop_prob=float(self.prev_drop_prob),
            only_cfg_eps=bool(self.cfg.get("only_cfg_eps", False)),
        )

    def _handle_checkpoint_and_eval(self, global_step: int) -> None:
        ckpt_dir = None
        if global_step % int(self.cfg.get("ckpt_every", 1000)) == 0:
            ckpt_dir = os.path.join(self.runtime.run_dir, "ckpt", f"step_{global_step:06d}")
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                save_hf_training_checkpoint(
                    accelerator=self.accelerator,
                    model=self.model,
                    noise_scheduler=self.noise_scheduler,
                    ckpt_dir=ckpt_dir,
                    inference_meta=self._build_hf_inference_meta(),
                    train_state=self._build_train_state(global_step),
                )
                print(f"[ONE_STAGE] HF Model saved to: {ckpt_dir}")

        eval_every = int(self.cfg.get("eval_every", 0))
        if eval_every <= 0 or (global_step % eval_every != 0):
            return
        if not self.accelerator.is_main_process:
            return
        if not ckpt_dir or not os.path.isdir(ckpt_dir):
            print(f"[ONE_STAGE EVAL] skip step={global_step} (no checkpoint dir for eval)")
            return

        max_conc = int(self.cfg.get("max_concurrent_evals", 0))
        can_launch = (max_conc <= 0) or (len(self.runtime.eval_procs) < max_conc)
        if not can_launch:
            print(f"[ONE_STAGE EVAL] skip (running {len(self.runtime.eval_procs)}/{max_conc})")
            return

        proc = launch_eval_subprocess(ckpt_dir, self.cfg, self.runtime.run_dir, global_step)
        append_eval_process(self.runtime, proc)

    def train(self):
        self._seed_everything()
        scheduler = self._setup_training()

        global_step = int(self.start_step)
        data_iter = iter(self.loader)
        pbar = tqdm.tqdm(
            total=int(self.cfg.get("train_steps", 50000)),
            initial=global_step,
            desc="[ONE_STAGE] train",
            dynamic_ncols=True,
            disable=not self.accelerator.is_local_main_process,
        )

        while global_step < int(self.cfg.get("train_steps", 50000)):
            self._cleanup_eval_processes()
            batch, data_iter = self._next_batch(data_iter)
            with self.accelerator.accumulate(self.model):
                feats = self.feat_extractor(batch, self.device)
                batch_state = self._build_train_batch(feats)
                predictions = self._forward_model(batch_state)
                alpha_bar, snr, w_p2 = self._compute_timestep_weights(batch_state.t)
                losses = self._compute_losses(batch_state, predictions, alpha_bar, snr, w_p2)
                took_step = self._optimizer_step(losses.total, scheduler)

            if not took_step:
                continue

            global_step += 1
            pbar.update(1)
            self._log_step(global_step, losses)
            pbar.set_postfix(loss=f"{losses.total.item():.4f}", lr=f"{self.opt.param_groups[0]['lr']:.2e}")
            self._handle_checkpoint_and_eval(global_step)

        pbar.close()
        close_train_runtime(self.accelerator)
