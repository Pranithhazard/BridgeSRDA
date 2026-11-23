#!/usr/bin/env bash
set -euo pipefail

EXP_ID="073a_denseobs_conditional_sb_srda_qg"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_WORK_ROOT="${EXP_WORK_ROOT:-$ROOT_DIR/work/experiments/$EXP_ID}"
EXP_LOG_ROOT="$EXP_WORK_ROOT/logs"
CONFIG_PRODUCTION="${CONFIG_PRODUCTION:-python/configs/filo_srda_qg/denseobs_conditional_sb_srda_qg/denseobs_conditional_sb_p2_b32.yml}"
CONFIG_SMOKE="${CONFIG_SMOKE:-python/configs/filo_srda_qg/denseobs_conditional_sb_srda_qg/denseobs_conditional_sb_smoke.yml}"
RUN_NAME="DenseObsConditionalSBUNet3D_Scratch_quad_p2_eps02582_ogx08_ogy08_mixup_b32_ema999"

cd "$ROOT_DIR"
export ROOT_DIR EXP_ID EXP_WORK_ROOT EXP_LOG_ROOT CONFIG_PRODUCTION CONFIG_SMOKE RUN_NAME
export PYTHONPATH="${PYTHONPATH:-$ROOT_DIR/python}"
if [[ -z "${QG3D_PYTHON:-}" ]]; then
  [[ -x "$ROOT_DIR/.venv/bin/python" ]] && QG3D_PYTHON="$ROOT_DIR/.venv/bin/python" || QG3D_PYTHON="python3"
  export QG3D_PYTHON
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p "$EXP_LOG_ROOT"/{preflight,training,evaluation,audits,metrics} \
  "$EXP_WORK_ROOT"/{audits,metrics,samples,checkpoints,evaluation,gates}

require_production() {
  [[ "${QG3D_ALLOW_PRODUCTION:-0}" == "1" ]] || {
    echo "Set QG3D_ALLOW_PRODUCTION=1 for production runs." >&2
    exit 2
  }
}

require_gates() {
  local missing=0
  for gate in interpolation data preflight overfit ddp_smoke regression_tests; do
    if [[ ! -f "$EXP_WORK_ROOT/gates/${gate}.passed" ]]; then
      echo "missing production gate: $EXP_WORK_ROOT/gates/${gate}.passed" >&2
      missing=1
    fi
  done
  [[ "$missing" == "0" ]] || exit 3
}

count_gpus() { awk -F',' '{print NF}' <<<"$1"; }

run_nohup() {
  local name="$1" subdir="$2" command="$3"
  local log="$EXP_LOG_ROOT/$subdir/${name}_$(date +%Y%m%d_%H%M%S).log"
  echo "log_path: $log"
  echo "command: $command"
  [[ "${DRY_RUN:-0}" == "1" ]] && return 0
  nohup bash -lc "$command" >"$log" 2>&1 &
  echo "pid: $!"
}
