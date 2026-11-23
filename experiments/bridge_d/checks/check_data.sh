#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
"$QG3D_PYTHON" python/scripts/filo_srda_qg/denseobs_conditional_sb_srda_qg/audit_denseobs_data.py \
  --config-path "$CONFIG_PRODUCTION" --root-dir "$ROOT_DIR"
touch "$EXP_WORK_ROOT/gates/data.passed"
