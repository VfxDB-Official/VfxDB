# one_stages/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from utils.tools import parse_args_cfg


@dataclass
class OneStageCfg:
    # 原始cfg包装，统一提供常用字段入口
    raw: Dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))


def load_cfg() -> OneStageCfg:
    # 从命令行/配置读取并封装
    args = parse_args_cfg()
    return OneStageCfg(raw=vars(args))
