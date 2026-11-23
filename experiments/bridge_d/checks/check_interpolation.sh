#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
VARIANT="${1:-production}"
CONFIG="$CONFIG_PRODUCTION"
[[ "$VARIANT" == "smoke" ]] && CONFIG="$CONFIG_SMOKE"
"$QG3D_PYTHON" python/scripts/filo_srda_qg/denseobs_conditional_sb_srda_qg/audit_observation_interpolation.py \
  --config-path "$CONFIG" --root-dir "$ROOT_DIR" "${@:2}"
if [[ "$VARIANT" == "production" ]]; then
  touch "$EXP_WORK_ROOT/gates/interpolation.passed"
fi
