from __future__ import annotations

import os
from datetime import datetime
from .tools import ensure_dir


def setup_run_dir(
    out_dir: str,
    exp_name: str,
    default_prefix: str,
    with_logs: bool = True,
    with_eval: bool = False,
    with_ckpt: bool = True,
) -> str:
    ensure_dir(out_dir)
    name = exp_name if exp_name else datetime.now().strftime(default_prefix)
    run = os.path.join(out_dir, name)
    ensure_dir(run)
    if with_ckpt:
        ensure_dir(os.path.join(run, "ckpt"))
    if with_logs:
        ensure_dir(os.path.join(run, "logs"))
    if with_eval:
        ensure_dir(os.path.join(run, "eval"))
    return run
