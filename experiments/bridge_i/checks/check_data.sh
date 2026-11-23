#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
VARIANT="${1:-production}"
CONFIG="$CONFIG_PRODUCTION"
[[ "$VARIANT" == "smoke" ]] && CONFIG="$CONFIG_SMOKE"
"$QG3D_PYTHON" python/scripts/filo_srda_qg/conditional_sb_srda_qg/audit_conditional_data.py --config-path "$CONFIG" --root-dir "$ROOT_DIR"
