#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python tests/test_pipeline_parity.py
