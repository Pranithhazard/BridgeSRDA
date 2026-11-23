#!/usr/bin/env bash
set -euo pipefail

EXP_ID="071b_sb_prior_retrain_qg"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_WORK_ROOT="${EXP_WORK_ROOT:-$ROOT_DIR/work/experiments/$EXP_ID}"
EXP_LOG_ROOT="$EXP_WORK_ROOT/logs"
CONFIG_PRODUCTION="${CONFIG_PRODUCTION:-python/configs/filo_srda_qg/sb_prior_retrain_qg/sb_prior_retrain_b32.yml}"
CONFIG_SMOKE="${CONFIG_SMOKE:-python/configs/filo_srda_qg/sb_prior_retrain_qg/sb_prior_retrain_smoke.yml}"
CONFIG_P2="${CONFIG_P2:-python/configs/filo_srda_qg/sb_prior_retrain_qg/sb_prior_retrain_p2_b32.yml}"
CONFIG_P2_2GPU="${CONFIG_P2_2GPU:-python/configs/filo_srda_qg/sb_prior_retrain_qg/sb_prior_retrain_p2_b32_2gpu.yml}"

cd "$ROOT_DIR"
export ROOT_DIR EXP_ID EXP_WORK_ROOT EXP_LOG_ROOT CONFIG_PRODUCTION CONFIG_SMOKE CONFIG_P2 CONFIG_P2_2GPU
export PYTHONPATH="${PYTHONPATH:-$ROOT_DIR/python}"
if [[ -z "${QG3D_PYTHON:-}" ]]; then
  [[ -x "$ROOT_DIR/.venv/bin/python" ]] && QG3D_PYTHON="$ROOT_DIR/.venv/bin/python" || QG3D_PYTHON="python3"
  export QG3D_PYTHON
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p "$EXP_LOG_ROOT"/{preflight,training,evaluation,audits,metrics} "$EXP_WORK_ROOT"/{audits,metrics,samples,checkpoints}

require_production() {
  [[ "${QG3D_ALLOW_PRODUCTION:-0}" == "1" ]] || {
    echo "Set QG3D_ALLOW_PRODUCTION=1 for production runs." >&2
    exit 2
  }
}

count_gpus() {
  local list="$1"
  awk -F',' '{print NF}' <<<"$list"
}

run_nohup() {
  local name="$1"
  local subdir="$2"
  local command="$3"
  local log="$EXP_LOG_ROOT/$subdir/${name}_$(date +%Y%m%d_%H%M%S).log"
  echo "log_path: $log"
  echo "command: $command"
  [[ "${DRY_RUN:-0}" == "1" ]] && return 0
  nohup bash -lc "$command" >"$log" 2>&1 &
  echo "pid: $!"
}
