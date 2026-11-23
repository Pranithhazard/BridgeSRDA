#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
GPU="${1:-0}"
VARIANT="${2:-production}"
CONFIG="$CONFIG_PRODUCTION"
if [[ "$VARIANT" == "smoke" ]]; then
  CONFIG="$CONFIG_SMOKE"
elif [[ "$VARIANT" == "p2_2gpu" ]]; then
  require_production
  CONFIG="$CONFIG_P2_2GPU"
fi
export CUDA_VISIBLE_DEVICES="$GPU"
"$QG3D_PYTHON" python/scripts/filo_srda_qg/sb_prior_retrain_qg/preflight_sb_prior_retrain.py \
  --config-path "$CONFIG"
