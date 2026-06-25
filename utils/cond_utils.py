from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch


def clean_str_list(xs: Optional[List[str]]) -> Optional[List[str]]:
    if xs is None:
        return None
    out = []
    for s in xs:
        if s is None:
            continue
        ss = str(s).strip()
        if ss == "":
            continue
        out.append(ss)
    return out


def get_by_dotted_key(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and (part in cur):
            cur = cur[part]
        else:
            return None
    return cur


def _infer_names_from_mapping(mapping: Any) -> Optional[List[str]]:
    if not isinstance(mapping, dict) or len(mapping) == 0:
        return None
    ok = True
    max_id = -1
    for k, v in mapping.items():
        if not isinstance(k, str):
            ok = False
            break
        try:
            iv = int(v)
        except Exception:
            ok = False
            break
        max_id = max(max_id, iv)
    if (not ok) or max_id < 0:
        return None
    names = [None] * (max_id + 1)
    for k, v in mapping.items():
        names[int(v)] = str(k)
    if any(n is None for n in names):
        return None
    return names


def infer_class_names(args, ds) -> Optional[List[str]]:
    if getattr(args, "class_names", None):
        return list(args.class_names)
    if getattr(args, "categories", None):
        return list(args.categories)

    for attr in ["class_names", "classes", "categories", "category_names"]:
        if hasattr(ds, attr):
            v = getattr(ds, attr)
            if isinstance(v, (list, tuple)) and len(v) > 0 and all(isinstance(x, str) for x in v):
                return list(v)

    for attr in ["class_to_id", "cat_to_id", "category_to_id", "cat_to_idx"]:
        if hasattr(ds, attr):
            mapping = getattr(ds, attr)
            names = _infer_names_from_mapping(mapping)
            if names is not None:
                return names

    return None


def get_class_id_from_batch(
    batch: Dict[str, Any],
    B: int,
    class_to_id: Optional[Dict[str, int]],
    cond_key: str = "",
) -> torch.Tensor:
    cand = None
    if cond_key.strip():
        cand = get_by_dotted_key(batch, cond_key.strip())

    if cand is None:
        for k in ["class_id", "category_id", "cat_id", "label", "y"]:
            if k in batch:
                cand = batch[k]
                break

    if cand is None:
        for k in ["category", "class_name", "cat"]:
            if k in batch:
                cand = batch[k]
                break

    if cand is None and ("meta" in batch):
        meta = batch["meta"]
        if isinstance(meta, list) and len(meta) > 0 and isinstance(meta[0], dict):
            names = []
            for it in meta:
                v = None
                for kk in ["category", "class_name", "cat", "category_name"]:
                    if kk in it:
                        v = it[kk]
                        break
                names.append(v)
            cand = names
        elif isinstance(meta, dict):
            for kk in ["category", "class_name", "cat", "category_name", "class_id", "category_id"]:
                if kk in meta:
                    cand = meta[kk]
                    break

    if isinstance(cand, torch.Tensor):
        y = cand.view(-1).long()
        if y.numel() == 1 and B > 1:
            y = y.expand(B).contiguous()
        if y.numel() != B:
            raise RuntimeError(f"[cond] tensor label size mismatch: got {tuple(cand.shape)} -> {y.numel()} vs B={B}")
        return y

    if isinstance(cand, (list, tuple)):
        if len(cand) != B:
            raise RuntimeError(f"[cond] list label size mismatch: len={len(cand)} vs B={B}")
        y = []
        for s in cand:
            if s is None:
                raise RuntimeError("[cond] meta/category contains None; fix dataloader meta or set --cond-key to an int id field.")
            if isinstance(s, (int, np.integer)):
                y.append(int(s))
            else:
                if class_to_id is None:
                    raise RuntimeError("[cond] got string labels but class_to_id is None. Please provide --class-names/--categories or dataset mapping.")
                sid = class_to_id.get(str(s), None)
                if sid is None:
                    raise RuntimeError(f"[cond] unknown class name '{s}'. Known: {list(class_to_id.keys())[:50]} ...")
                y.append(int(sid))
        return torch.tensor(y, dtype=torch.long)

    if cand is None:
        raise RuntimeError(
            "[cond] cannot find labels in batch. "
            "Either (1) make dataloader return 'class_id'/'category_id', "
            "or (2) enable return_meta and use --cond-key meta.category, "
            "or (3) pass --cond-key to specify the exact field."
        )

    if isinstance(cand, (int, np.integer)):
        return torch.full((B,), int(cand), dtype=torch.long)
    if isinstance(cand, str):
        if class_to_id is None:
            raise RuntimeError("[cond] got a string label but class_to_id is None. Please provide --class-names/--categories or dataset mapping.")
        sid = class_to_id.get(cand, None)
        if sid is None:
            raise RuntimeError(f"[cond] unknown class name '{cand}'. Known: {list(class_to_id.keys())[:50]} ...")
        return torch.full((B,), int(sid), dtype=torch.long)

    raise RuntimeError(f"[cond] unsupported label type: {type(cand)}")
