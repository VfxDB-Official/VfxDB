#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python infer_one_stage_hf.py --config configs/infer_one_stage_hf.yaml "$@"
