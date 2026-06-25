#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import faulthandler
faulthandler.enable()
from one_stages.config import load_cfg
from one_stages.factory import build_everything
from one_stages.trainer_new_hf import Trainer


def main():
    cfg = load_cfg()
    objs = build_everything(cfg)
    trainer = Trainer(cfg, **objs)
    trainer.train()


if __name__ == "__main__":
    main()
