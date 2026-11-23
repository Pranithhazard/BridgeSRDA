#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
GPU="${1:-0}"
VARIANT="${2:-smoke}"
CONFIG="$CONFIG_SMOKE"
if [[ "$VARIANT" != "smoke" ]]; then
  require_production
  CONFIG="$CONFIG_PRODUCTION"
fi
export CUDA_VISIBLE_DEVICES="$GPU"
"$QG3D_PYTHON" python/scripts/filo_srda_qg/sb_prior_retrain_qg/overfit_sb_prior_retrain.py \
  --config-path "$CONFIG" \
  --root-dir "$ROOT_DIR"
