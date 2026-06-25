#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

exec ./sh/run_train_cond_static_32.sh "$@"
