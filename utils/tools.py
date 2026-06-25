import os
import json
from typing import Optional, List
import yaml
import argparse
from omegaconf import OmegaConf
from argparse import Namespace
from typing import Any, List, Tuple
import sys
from contextlib import contextmanager

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def hard_log(msg, log_name = "temp_log.txt"):
    os.makedirs(".log_cache", exist_ok=True)  # 确保文件夹存在
    if log_name == "temp_log.txt":
        print("[warning] default log_name of \'temp_log.txt\'", flush=True)
    with open(f".log_cache/{log_name}", "a", encoding="utf-8") as f:
        f.write(msg)
        f.flush()             # 极其重要：清空 Python 缓冲区
        os.fsync(f.fileno())  # 极其重要：强制操作系统立刻把数据刻入硬盘

_MISSING = object()

def _flatten_keys(d: Any, prefix: str = "") -> List[str]:
    """把 cfg 的所有可覆盖 key 展平成 dot-path 列表（用于严格校验）。"""
    keys = []
    if isinstance(d, dict):
        for k, v in d.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            keys.append(path)
            keys.extend(_flatten_keys(v, path))
    return keys

def _has_key(flat_keys_set: set, key: str) -> bool:
    # key 可能是 "a.b.c"
    return key in flat_keys_set

def _norm_key(flag: str) -> str:
    """
    把 --data-root => data_root
    把 --eval-vdb-threshold => eval_vdb_threshold
    如果你未来用分组，也支持 --train.steps => train.steps（每段内 -=>_）
    """
    s = flag.lstrip("-")
    parts = s.split(".")
    parts = [p.replace("-", "_") for p in parts]
    return ".".join(parts)

def _parse_value(s: str) -> Any:
    """
    用 yaml.safe_load 做类型推断：
    "true"->True, "1e-4"->float, "[a,b]"->list, "meta.category"->str
    """
    return yaml.safe_load(s)

def _parse_overrides(argv: List[str]) -> List[Tuple[str, Any]]:
    """
    解析形如：
      --k v
      --k            (bool True)
      --no-k         (bool False)
      --k=true       (也支持)
    返回 [(key, value), ...]，key 是 dot-path（已 -=>_）
    """
    out = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            raise ValueError(f"Unexpected token: {tok} (expected --key or --key=value)")

        # --k=v
        if "=" in tok:
            kflag, vstr = tok.split("=", 1)
            neg = kflag.startswith("--no-")
            key = _norm_key(kflag[5:] if neg else kflag[2:])
            val = _parse_value(vstr)
            if neg:
                # --no-k=... 这种一般没意义，直接强制 False
                val = False
            out.append((key, val))
            i += 1
            continue

        # --no-k
        if tok.startswith("--no-"):
            key = _norm_key(tok[5:])
            out.append((key, False))
            i += 1
            continue

        # --k [v] 或 --k (bool True)
        key = _norm_key(tok[2:])
        # 如果下一个也是 -- 开头或者没有下一个 -> 当作 bool True
        if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
            out.append((key, True))
            i += 1
        else:
            vstr = argv[i + 1]
            out.append((key, _parse_value(vstr)))
            i += 2

    return out


@contextmanager
def suppress_stdout_stderr():
    """
    上下文管理器：在操作系统级别劫持并丢弃标准输出和标准错误。
    专门用来对付 Blender 这种 C 底层直接向终端狂刷日志的软件。
    """
    # 刷新当前的 Python 缓冲区
    sys.stdout.flush()
    sys.stderr.flush()

    # 打开黑洞设备 (Linux/Mac下是 /dev/null，Windows下是 NUL)
    devnull = os.open(os.devnull, os.O_RDWR)

    # 备份原始的 文件描述符 (1是stdout, 2是stderr)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)

    try:
        # 将标准输出和错误重定向到黑洞
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        # 退出 with 块时，恢复原始的输出通道
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        # 清理描述符
        os.close(old_stdout)
        os.close(old_stderr)
        os.close(devnull)

def parse_args_cfg() -> Namespace:
    # 只用 argparse 解析 --config（其余全手工解析）
    p = argparse.ArgumentParser("Stage-2 (config-driven)")
    p.add_argument("--config", required=True, type=str, help="full yaml config")
    # 关键：parse_known_args，把剩下的原封不动留给我们
    known, unknown = p.parse_known_args()

    # 1) 读 YAML（唯一真相）
    cfg = OmegaConf.load(known.config)

    # 2) 严格：只允许覆盖 YAML 里存在的 key
    cfg_dict = OmegaConf.to_container(cfg, resolve=False)
    flat_keys = set(_flatten_keys(cfg_dict))

    overrides = _parse_overrides(unknown)
    for k, v in overrides:
        if not _has_key(flat_keys, k):
            raise KeyError(
                f"Unknown override key: '{k}'. Not found in YAML. "
                f"(Did you misspell it? YAML uses keys like: {sorted(list(flat_keys))[:20]} ...)"
            )
        OmegaConf.update(cfg, k, v, merge=False)

    # 3) 强制检查：如果 YAML 有 ???，这里 resolve 会直接报 MissingMandatoryValue
    cfg_final = OmegaConf.to_container(cfg, resolve=True)

    # 4) 仍然给你 args.xxx 的体验
    return Namespace(**cfg_final)
