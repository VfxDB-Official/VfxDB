from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Type

import torch


HF_INFERENCE_META_KEY = "vfxdb_inference_meta"
HF_INFERENCE_META_FIELDS = (
    "prediction_type",
    "timesteps",
    "beta_start",
    "beta_end",
    "value_space",
    "log1p_scale",
    "log1p_dequant_delta",
    "num_classes",
    "class_names",
    "volume_size",
    "in_channels",
    "out_channels",
    "use_occupancy",
    "out_spec",
    "prev_noised",
    "prev_same_t",
    "prev_drop_prob",
    "only_cfg_eps",
)


@dataclass
class LoadedHFCheckpoint:
    ckpt_dir: str
    model: Any
    train_state: Optional[Dict[str, Any]]
    train_meta: Dict[str, Any]
    inference_meta: Dict[str, Any]


@dataclass
class ResolvedHFRuntimeBundle:
    checkpoint: LoadedHFCheckpoint
    prediction_type: str
    train_timesteps: int
    beta_start: float
    beta_end: float


def normalize_hf_prediction_type(pred_type: str) -> str:
    pred_type = str(pred_type).strip().lower()
    if pred_type == "eps":
        return "epsilon"
    if pred_type == "x0":
        return "sample"
    if pred_type in ("epsilon", "sample", "v_prediction"):
        return pred_type
    raise RuntimeError(f"[HF] unsupported prediction_type: {pred_type}")


def build_hf_inference_meta(
    *,
    prediction_type: str,
    timesteps: int,
    beta_start: float,
    beta_end: float,
    value_space: str,
    log1p_scale: float,
    log1p_dequant_delta: float,
    num_classes: int,
    class_names,
    volume_size: int,
    in_channels: int,
    out_channels: int,
    use_occupancy: bool,
    out_spec,
    prev_noised: bool,
    prev_same_t: bool,
    prev_drop_prob: float,
    only_cfg_eps: bool,
) -> Dict[str, Any]:
    return {
        "prediction_type": normalize_hf_prediction_type(prediction_type),
        "timesteps": int(timesteps),
        "beta_start": float(beta_start),
        "beta_end": float(beta_end),
        "value_space": str(value_space).lower(),
        "log1p_scale": float(log1p_scale),
        "log1p_dequant_delta": float(log1p_dequant_delta),
        "num_classes": int(num_classes),
        "class_names": class_names,
        "volume_size": int(volume_size),
        "in_channels": int(in_channels),
        "out_channels": int(out_channels),
        "use_occupancy": bool(use_occupancy),
        "out_spec": out_spec,
        "prev_noised": bool(prev_noised),
        "prev_same_t": bool(prev_same_t),
        "prev_drop_prob": float(prev_drop_prob),
        "only_cfg_eps": bool(only_cfg_eps),
    }


def attach_hf_inference_meta(model, inference_meta: Dict[str, Any]) -> Dict[str, Any]:
    meta = dict(inference_meta or {})
    if not hasattr(model, "register_to_config"):
        raise RuntimeError("[HF] model does not support register_to_config")
    model.register_to_config(**{HF_INFERENCE_META_KEY: meta})
    return meta


def read_hf_inference_meta(model_config, train_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    config_meta = getattr(model_config, HF_INFERENCE_META_KEY, None)
    if isinstance(config_meta, Mapping):
        meta.update(dict(config_meta))

    train_meta = train_meta or {}
    for key in HF_INFERENCE_META_FIELDS:
        if key not in meta and key in train_meta:
            meta[key] = train_meta[key]

    if "prediction_type" in meta:
        meta["prediction_type"] = normalize_hf_prediction_type(meta["prediction_type"])
    return meta


def load_hf_checkpoint(
    *,
    model_cls: Type,
    ckpt_path: str,
    device: torch.device,
    use_ema: bool,
) -> LoadedHFCheckpoint:
    ckpt_dir = str(ckpt_path)
    if os.path.isfile(ckpt_dir):
        print(f"[HF WARN] checkpoint path is a file: {ckpt_dir}")
        ckpt_dir = os.path.dirname(ckpt_dir)
        print(f"[HF WARN] using Hugging Face checkpoint directory instead: {ckpt_dir}")

    model = model_cls.from_pretrained(ckpt_dir).to(device)

    state_path = os.path.join(ckpt_dir, "trainer_state.pt")
    train_state = None
    train_meta: Dict[str, Any] = {}
    if os.path.exists(state_path):
        train_state = torch.load(state_path, map_location="cpu")
        if isinstance(train_state, dict) and isinstance(train_state.get("train_meta"), dict):
            train_meta = dict(train_state["train_meta"])

    if use_ema:
        if train_state is None:
            print(f"[HF WARN] use_ema=True but {state_path} not found. Using online weights.")
        else:
            print(f"[HF] Attempting to load EMA weights from {state_path} ...")
            ema_weights = train_state.get("ema", None)
            if ema_weights is None:
                print("[HF WARN] trainer_state.pt found, but no EMA weights inside. Using online weights.")
            else:
                missing, unexpected = model.load_state_dict(ema_weights, strict=False)
                print(f"[HF] EMA load report: missing={len(missing)} unexpected={len(unexpected)}")

    inference_meta = read_hf_inference_meta(model.config, train_meta)
    return LoadedHFCheckpoint(
        ckpt_dir=ckpt_dir,
        model=model,
        train_state=train_state,
        train_meta=train_meta,
        inference_meta=inference_meta,
    )


def resolve_hf_runtime_bundle(
    *,
    checkpoint: LoadedHFCheckpoint,
    prediction_type: str = "auto",
    train_timesteps_override: Optional[int] = None,
    beta_start_override: Optional[float] = None,
    beta_end_override: Optional[float] = None,
) -> ResolvedHFRuntimeBundle:
    inference_meta = checkpoint.inference_meta

    pred_type = prediction_type
    if str(pred_type).lower() == "auto":
        pred_type = inference_meta.get("prediction_type", "epsilon")
    resolved_pred_type = normalize_hf_prediction_type(pred_type)

    train_timesteps = (
        int(train_timesteps_override)
        if train_timesteps_override is not None
        else int(inference_meta.get("timesteps", 1000))
    )
    beta_start = (
        float(beta_start_override)
        if beta_start_override is not None
        else float(inference_meta.get("beta_start", 0.0001))
    )
    beta_end = (
        float(beta_end_override)
        if beta_end_override is not None
        else float(inference_meta.get("beta_end", 0.02))
    )
    return ResolvedHFRuntimeBundle(
        checkpoint=checkpoint,
        prediction_type=resolved_pred_type,
        train_timesteps=train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
    )
