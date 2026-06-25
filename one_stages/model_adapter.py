from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class ModelConditioning:
    class_labels: Optional[torch.Tensor] = None
    encoder_hidden_states: Optional[torch.Tensor] = None
    added_cond_kwargs: Optional[dict[str, Any]] = None
    cross_attention_kwargs: Optional[dict[str, Any]] = None
    attention_mask: Optional[torch.Tensor] = None


@dataclass
class ModelPredictions:
    main: torch.Tensor
    occ_logits: Optional[torch.Tensor]


class DiffusersModelAdapter:
    def __init__(self, *, use_occupancy: bool):
        self.use_occupancy = bool(use_occupancy)

    def build_conditioning(self, feats) -> ModelConditioning:
        return ModelConditioning(
            class_labels=getattr(feats, "y", None),
            encoder_hidden_states=getattr(feats, "encoder_hidden_states", None),
            added_cond_kwargs=getattr(feats, "added_cond_kwargs", None),
            cross_attention_kwargs=getattr(feats, "cross_attention_kwargs", None),
            attention_mask=getattr(feats, "attention_mask", None),
        )

    def forward(
        self,
        model,
        *,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: ModelConditioning,
    ) -> ModelPredictions:
        model_kwargs = {
            "sample": sample,
            "timestep": timestep,
            "return_dict": True,
        }
        if conditioning.class_labels is not None:
            model_kwargs["class_labels"] = conditioning.class_labels
        if conditioning.encoder_hidden_states is not None:
            model_kwargs["encoder_hidden_states"] = conditioning.encoder_hidden_states
        if conditioning.added_cond_kwargs is not None:
            model_kwargs["added_cond_kwargs"] = conditioning.added_cond_kwargs
        if conditioning.cross_attention_kwargs is not None:
            model_kwargs["cross_attention_kwargs"] = conditioning.cross_attention_kwargs
        if conditioning.attention_mask is not None:
            model_kwargs["attention_mask"] = conditioning.attention_mask

        out = model(**model_kwargs)
        sample_out = out["sample"] if isinstance(out, dict) else out.sample
        if self.use_occupancy:
            return ModelPredictions(main=sample_out[:, 0:1, ...], occ_logits=sample_out[:, 1:2, ...])
        return ModelPredictions(main=sample_out, occ_logits=None)

    def get_occ_aux_logits(self, model) -> list[torch.Tensor]:
        aux_list = []
        if hasattr(model, "get_occ_aux_logits"):
            aux_list = model.get_occ_aux_logits() or []
        return list(aux_list)


def build_default_model_adapter(*, use_occupancy: bool) -> DiffusersModelAdapter:
    return DiffusersModelAdapter(use_occupancy=use_occupancy)
