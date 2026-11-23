#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
GPU="${1:-0}"
VARIANT="${2:-production}"
CONFIG="$CONFIG_PRODUCTION"
[[ "$VARIANT" == "smoke" ]] && CONFIG="$CONFIG_SMOKE"
CUDA_VISIBLE_DEVICES="$GPU" "$QG3D_PYTHON" \
  python/scripts/filo_srda_qg/denseobs_conditional_sb_srda_qg/preflight_denseobs_conditional_sb.py \
  --config-path "$CONFIG"
if [[ "$VARIANT" == "production" ]]; then
  touch "$EXP_WORK_ROOT/gates/preflight.passed"
fi
