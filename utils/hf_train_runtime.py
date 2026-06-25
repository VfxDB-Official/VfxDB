from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from utils.hf_checkpoint import attach_hf_inference_meta, build_hf_inference_meta
from utils.run_utils import setup_run_dir


@dataclass
class TrainRuntimeState:
    run_dir: str
    log_with: Optional[str]
    eval_procs: List[subprocess.Popen] = field(default_factory=list)


def prepare_run_dir(cfg: dict) -> str:
    return setup_run_dir(
        cfg["out_dir"],
        cfg.get("exp_name", ""),
        default_prefix="one_stage_%Y%m%d_%H%M%S",
        with_eval=True,
    )


def resolve_log_backend() -> Optional[str]:
    if importlib.util.find_spec("tensorboard") is not None:
        return "tensorboard"
    if importlib.util.find_spec("tensorboardX") is not None:
        return "tensorboard"
    return None


def init_train_runtime(
    cfg: dict,
    accelerator: Accelerator,
    train_len: int,
    *,
    run_dir: Optional[str] = None,
) -> TrainRuntimeState:
    run_dir = str(run_dir or prepare_run_dir(cfg))
    if accelerator.is_main_process:
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as handle:
            json.dump({**cfg, "train_len": train_len}, handle, ensure_ascii=False, indent=2)

        if accelerator.log_with is not None:
            clean_cfg = {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))}
            accelerator.init_trackers(
                project_name=str(cfg.get("exp_name", "one_stage")),
                config=clean_cfg,
            )
    return TrainRuntimeState(run_dir=run_dir, log_with=accelerator.log_with)


def build_project_configuration(run_dir: str) -> ProjectConfiguration:
    return ProjectConfiguration(
        project_dir=run_dir,
        logging_dir=os.path.join(run_dir, "logs"),
        automatic_checkpoint_naming=True,
    )


def cleanup_eval_processes(runtime: TrainRuntimeState) -> None:
    if not runtime.eval_procs:
        return
    alive = []
    for proc in runtime.eval_procs:
        ret = proc.poll()
        if ret is None:
            alive.append(proc)
        elif ret != 0:
            print(f"[ONE_STAGE EVAL WARN] exit code {ret}")
    runtime.eval_procs = alive


def append_eval_process(runtime: TrainRuntimeState, proc: Optional[subprocess.Popen]) -> None:
    if proc is not None:
        runtime.eval_procs.append(proc)


def close_train_runtime(accelerator: Accelerator) -> None:
    accelerator.end_training()


def log_train_metrics(
    accelerator: Accelerator,
    *,
    global_step: int,
    losses,
    learning_rate: float,
    use_occupancy: bool,
    residual_weight: float,
) -> None:
    if not getattr(accelerator, "trackers", None):
        return
    payload = {
        "train/loss_total": float(losses.total.item()),
        "train/loss_value": float(losses.value.item()),
        "train/lr": float(learning_rate),
        "train/p2_w_mean": float(losses.w_p2.mean().item()),
    }
    if use_occupancy:
        payload["train/loss_occ"] = float(losses.occ.item())
        payload["train/loss_occ_ds"] = float(losses.occ_ds.item())
        payload["train/loss_empty_leak"] = float(losses.leak.item())
    if residual_weight > 0:
        payload["train/loss_residual"] = float(losses.residual.item())
    accelerator.log(payload, step=int(global_step))


def launch_eval_subprocess(ckpt_path: str, cfg: dict, run_dir: str, global_step: int):
    eval_root = os.path.join(run_dir, "eval")
    os.makedirs(eval_root, exist_ok=True)
    step_name = f"step_{global_step:06d}"

    eval_config = str(cfg.get("eval_config", "configs/one_stage_multi_ch.yaml"))
    if not os.path.isfile(eval_config):
        alt = os.path.join(os.path.dirname(__file__), "..", eval_config)
        alt = os.path.abspath(alt)
        if os.path.isfile(alt):
            eval_config = alt

    eval_script = str(cfg["eval_script"])
    if os.path.isdir(ckpt_path) and os.path.basename(eval_script) == "eval_mc_one_stage.py":
        eval_script = "eval_one_stage_hf.py"
        print("[ONE_STAGE EVAL] detected HF checkpoint dir, switching eval script to eval_one_stage_hf.py")

    cmd = [
        sys.executable,
        eval_script,
        "--config", str(eval_config),
        "--ckpt", str(ckpt_path),
        "--out-dir", str(eval_root),
        "--step-name", str(step_name),
    ]

    def _add_override(key: str, value):
        if value is None:
            return
        if isinstance(value, bool):
            cmd.extend([f"--{key.replace('_', '-')}", str(value).lower()])
            return
        if isinstance(value, (list, tuple)):
            cmd.extend([f"--{key.replace('_', '-')}", json.dumps(list(value))])
            return
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    _add_override("timesteps", cfg.get("timesteps", 200))
    _add_override("beta_start", cfg.get("beta_start", 1e-4))
    _add_override("beta_end", cfg.get("beta_end", 2e-2))
    _add_override("sampling_steps", cfg.get("sampling_steps", 0))
    _add_override("volume_size", cfg.get("volume_size", 32))
    _add_override("base_channels", cfg.get("base_channels", 48))
    _add_override("time_emb_dim", cfg.get("time_emb_dim", 256))
    _add_override("channel_mults", cfg.get("channel_mults", [1, 2, 4, 4]))
    _add_override("value_space", cfg.get("value_space", "raw"))
    _add_override("log1p_scale", cfg.get("log1p_scale", 0.02))
    _add_override("log1p_dequant_delta", cfg.get("log1p_dequant_delta", 0.0))
    _add_override("prediction_type", cfg.get("prediction_type", "epsilon"))
    _add_override("p2_k", cfg.get("p2_k", 1.0))
    _add_override("p2_gamma", cfg.get("p2_gamma", 1.0))
    _add_override("p2_normalize", cfg.get("p2_normalize", True))
    _add_override("occ_t_weighting", cfg.get("occ_t_weighting", "alpha_bar"))
    _add_override("eval_prediction_type", cfg.get("eval_prediction_type", "auto"))
    _add_override("eval_num_samples", cfg.get("eval_num_samples", 8))
    _add_override("eval_batch_size", cfg.get("eval_batch_size", 4))
    _add_override("eval_vdb_threshold", cfg.get("eval_vdb_threshold", 1e-5))
    _add_override("eval_export_size", cfg.get("eval_export_size", 128))
    _add_override("eval_panel_cols", cfg.get("eval_panel_cols", None))
    _add_override("eval_panel_cell", cfg.get("eval_panel_cell", None))
    _add_override("eval_panel_max_width", cfg.get("eval_panel_max_width", 0))
    _add_override("eval_views_both", cfg.get("eval_views_both", False))
    _add_override("eval_no_3d", cfg.get("eval_no_3d", False))
    _add_override("eval_no_vdb", cfg.get("eval_no_vdb", False))
    _add_override("eval_no_vdb_render", cfg.get("eval_no_vdb_render", False))
    _add_override("eval_vdb_dense_write", cfg.get("eval_vdb_dense_write", False))
    _add_override("eval_mitsuba", cfg.get("eval_mitsuba", True))
    _add_override("eval_mitsuba_spp", cfg.get("eval_mitsuba_spp", 1))
    _add_override("eval_mitsuba_variant", cfg.get("eval_mitsuba_variant", "scalar_rgb"))
    _add_override("eval_mitsuba_scale", cfg.get("eval_mitsuba_scale", 1.0))
    _add_override("eval_force_occ_gating", cfg.get("eval_force_occ_gating", False))
    _add_override("eval_disable_occ_gating", cfg.get("eval_disable_occ_gating", False))
    _add_override("eval_occ_gating_thr", cfg.get("eval_occ_gating_thr", 0.5))
    _add_override("eval_occ_gating_mode", cfg.get("eval_occ_gating_mode", "soft"))
    _add_override("eval_occ_soft_k", cfg.get("eval_occ_soft_k", 12.0))
    _add_override("eval_occ_upsampling", cfg.get("eval_occ_upsampling", "trilinear"))
    _add_override("use_ema", cfg.get("use_ema", True))
    _add_override("eval_cfg_scale", cfg.get("eval_cfg_scale", 0.0))
    _add_override("eval_cfg_class", cfg.get("eval_cfg_class", ""))
    _add_override("eval_cfg_class_id", cfg.get("eval_cfg_class_id", -1))
    _add_override("only_cfg_eps", cfg.get("only_cfg_eps", False))

    print(f"[ONE_STAGE EVAL] launch: {' '.join(cmd)}")
    try:
        return subprocess.Popen(cmd)
    except Exception as exc:
        print("[ONE_STAGE EVAL] failed:", repr(exc))
        return None


def build_one_stage_train_meta(
    *,
    prediction_type: str,
    timesteps: int,
    beta_start: float,
    beta_end: float,
    value_space: str,
    log1p_scale: float,
    log1p_dequant_delta: float,
    num_classes: int,
    class_names: Optional[List[str]],
    volume_size: int,
    in_channels: int,
    out_channels: int,
    use_occupancy: bool,
    out_spec: Any,
    train_legacy_align: bool,
    prev_noised: bool,
    prev_same_t: bool,
    prev_drop_prob: float,
    only_cfg_eps: bool,
) -> Dict[str, Any]:
    return {
        "prediction_type": str(prediction_type),
        "timesteps": int(timesteps),
        "beta_start": float(beta_start),
        "beta_end": float(beta_end),
        "value_space": str(value_space),
        "log1p_scale": float(log1p_scale),
        "log1p_dequant_delta": float(log1p_dequant_delta),
        "num_classes": int(num_classes),
        "class_names": class_names,
        "volume_size": int(volume_size),
        "in_channels": int(in_channels),
        "out_channels": int(out_channels),
        "use_occupancy": bool(use_occupancy),
        "out_spec": out_spec,
        "train_legacy_align": bool(train_legacy_align),
        "prev_noised": bool(prev_noised),
        "prev_same_t": bool(prev_same_t),
        "prev_drop_prob": float(prev_drop_prob),
        "only_cfg_eps": bool(only_cfg_eps),
    }


def build_one_stage_train_state(
    *,
    global_step: int,
    opt_state: Dict[str, Any],
    ema_shadow: Optional[Dict[str, Any]],
    train_meta: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "step": int(global_step),
        "opt": opt_state,
        "ema": ema_shadow,
        "train_meta": dict(train_meta),
    }


def build_one_stage_inference_meta(
    *,
    prediction_type: str,
    timesteps: int,
    beta_start: float,
    beta_end: float,
    value_space: str,
    log1p_scale: float,
    log1p_dequant_delta: float,
    num_classes: int,
    class_names: Optional[List[str]],
    volume_size: int,
    in_channels: int,
    out_channels: int,
    use_occupancy: bool,
    out_spec: Any,
    prev_noised: bool,
    prev_same_t: bool,
    prev_drop_prob: float,
    only_cfg_eps: bool,
) -> Dict[str, Any]:
    return build_hf_inference_meta(
        prediction_type=prediction_type,
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        value_space=value_space,
        log1p_scale=log1p_scale,
        log1p_dequant_delta=log1p_dequant_delta,
        num_classes=num_classes,
        class_names=class_names,
        volume_size=volume_size,
        in_channels=in_channels,
        out_channels=out_channels,
        use_occupancy=use_occupancy,
        out_spec=out_spec,
        prev_noised=prev_noised,
        prev_same_t=prev_same_t,
        prev_drop_prob=prev_drop_prob,
        only_cfg_eps=only_cfg_eps,
    )


def save_hf_training_checkpoint(
    *,
    accelerator: Accelerator,
    model,
    noise_scheduler,
    ckpt_dir: str,
    inference_meta: Dict[str, Any],
    train_state: Dict[str, Any],
) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    attach_hf_inference_meta(unwrapped_model, inference_meta)
    unwrapped_model.save_pretrained(ckpt_dir, safe_serialization=True)
    if hasattr(noise_scheduler, "save_pretrained"):
        scheduler_dir = os.path.join(ckpt_dir, "scheduler")
        noise_scheduler.save_pretrained(scheduler_dir)
    torch.save(train_state, os.path.join(ckpt_dir, "trainer_state.pt"))
