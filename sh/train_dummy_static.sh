#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python tools/make_dummy_vdbset.py --out dummy_data/vdbset --num-categories 2 --num-sequences 2 --num-frames 4 --size 8
python train_one_stage.py --config configs/train_dummy_static.yaml "$@"
