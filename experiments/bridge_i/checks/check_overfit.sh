#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
GPU="${1:-0}"
VARIANT="${2:-smoke}"
CONFIG="$CONFIG_SMOKE"
[[ "$VARIANT" == "production" ]] && CONFIG="$CONFIG_PRODUCTION"
CUDA_VISIBLE_DEVICES="$GPU" "$QG3D_PYTHON" python/scripts/filo_srda_qg/conditional_sb_srda_qg/overfit_conditional_sb.py --config-path "$CONFIG" --root-dir "$ROOT_DIR"
