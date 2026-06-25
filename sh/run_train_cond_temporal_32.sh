#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -n "${CONDA_PREFIX:-}" && -f "$CONDA_PREFIX/lib/libjemalloc.so.2" ]]; then
  export LD_PRELOAD="$CONDA_PREFIX/lib/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
fi

python train_one_stage.py --config configs/train_temporal_32.yaml "$@"
