from __future__ import annotations

"""Scheduler factory and compatibility adapters for HF inference.

Long-term API:
- `build_scheduler_bundle(...)`
- `VfxDBDensePipeline.from_scheduler_config(...)`
- `VfxDBDensePipeline.configure_scheduler(...)`

Compatibility / transitional pieces:
- `LegacyAlignedDDPMScheduler`: reproduces the old DDPM sampler semantics.
- `scheduler_mode`: backward-compatible config key kept for old configs.
- `build_diffusers_scheduler(...)`: compatibility alias for older call sites.

Notes for maintainers:
- All schedulers can share one factory / pipeline interface.
- Exact "legacy alignment" is algorithm-specific. Right now it is implemented
  only for DDPM because the historical sampler was DDPM-based.
- To add a new scheduler, prefer registering its native diffusers class first.
  Add a dedicated legacy adapter only if you need to reproduce an older custom
  sampler byte-for-byte.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Type, Union

import torch
from diffusers import DDIMScheduler, DDPMScheduler

from legacy.ddpm_scheduler import LegacyAlignedDDPMScheduler


@dataclass(frozen=True)
class SchedulerRegistryEntry:
    native_cls: Type
    legacy_cls: Optional[Type] = None


@dataclass(frozen=True)
class SchedulerBundle:
    scheduler: Any
    name: str
    legacy_align: bool


# Add new schedulers here. Legacy adapters are optional and should only be
# provided when exact historical reproduction is required.
SCHEDULER_REGISTRY: Dict[str, SchedulerRegistryEntry] = {
    "ddpm": SchedulerRegistryEntry(native_cls=DDPMScheduler, legacy_cls=LegacyAlignedDDPMScheduler),
    "ddim": SchedulerRegistryEntry(native_cls=DDIMScheduler, legacy_cls=None),
}


def resolve_scheduler_choice(
    *,
    scheduler_name: str = "ddpm",
    scheduler_legacy_align: bool = False,
    scheduler_mode: Optional[str] = None,
):
    """Resolve scheduler config while preserving old config compatibility."""
    name = str(scheduler_name).strip().lower()
    legacy_align = bool(scheduler_legacy_align)

    # Backward compatibility for the old config key.
    if scheduler_mode is not None:
        mode = str(scheduler_mode).strip().lower()
        if mode == "legacy_aligned":
            name = "ddpm"
            legacy_align = True
        elif mode == "native":
            name = "ddpm"
            legacy_align = False

    return name, legacy_align


def build_scheduler_bundle(
    *,
    scheduler_name: str = "ddpm",
    scheduler_legacy_align: bool = False,
    scheduler_mode: Optional[str] = None,
    num_train_timesteps: int,
    beta_start: float,
    beta_end: float,
    prediction_type: str,
    **kwargs,
):
    """Create a scheduler and return the resolved runtime metadata.

    This is the preferred factory for new call sites because it returns both the
    scheduler instance and the resolved scheduler identity.
    """
    name, legacy_align = resolve_scheduler_choice(
        scheduler_name=scheduler_name,
        scheduler_legacy_align=scheduler_legacy_align,
        scheduler_mode=scheduler_mode,
    )

    entry = SCHEDULER_REGISTRY.get(name)
    if entry is None:
        raise ValueError(
            f"unsupported scheduler_name: {scheduler_name}. "
            f"available={sorted(SCHEDULER_REGISTRY.keys())}"
        )

    if legacy_align:
        scheduler_cls = entry.legacy_cls
        if scheduler_cls is None:
            raise ValueError(
                f"legacy alignment is not implemented for scheduler_name={name}"
            )
    else:
        scheduler_cls = entry.native_cls

    scheduler = scheduler_cls(
        num_train_timesteps=int(num_train_timesteps),
        beta_start=float(beta_start),
        beta_end=float(beta_end),
        beta_schedule="linear",
        prediction_type=str(prediction_type),
        clip_sample=False,
        **kwargs,
    )
    return SchedulerBundle(scheduler=scheduler, name=name, legacy_align=legacy_align)


def build_diffusers_scheduler(
    *,
    scheduler_name: str = "ddpm",
    scheduler_legacy_align: bool = False,
    scheduler_mode: Optional[str] = None,
    num_train_timesteps: int,
    beta_start: float,
    beta_end: float,
    prediction_type: str,
    **kwargs,
):
    """Compatibility alias kept for older callers.

    Prefer `build_scheduler_bundle(...)` in new code so the caller can also log
    the resolved scheduler identity.
    """
    bundle = build_scheduler_bundle(
        scheduler_name=scheduler_name,
        scheduler_legacy_align=scheduler_legacy_align,
        scheduler_mode=scheduler_mode,
        num_train_timesteps=num_train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        prediction_type=prediction_type,
        **kwargs,
    )
    return bundle.scheduler
