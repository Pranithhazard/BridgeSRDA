#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
GPUS="${1:-0,1,2,3}"
PHASE="${2:-full}"
WORLD="$(count_gpus "$GPUS")"
[[ "$WORLD" == "2" || "$WORLD" == "4" ]] || {
  echo "smoke requires either two or four GPUs" >&2
  exit 2
}
ACCUMULATION=$((32 / WORLD))
PORT="${MASTER_PORT:-29801}"
[[ "$PHASE" =~ ^(first|resume|full)$ ]] || { echo "phase must be first, resume, or full" >&2; exit 2; }
BASE="'$QG3D_PYTHON' python/scripts/filo_srda_qg/denseobs_conditional_sb_srda_qg/train_denseobs_conditional_sb_ddp.py --config-path '$CONFIG_SMOKE' --world-size '$WORLD' --override-gradient-accumulation-steps '$ACCUMULATION' --root-dir '$ROOT_DIR'"
if [[ "$PHASE" == "first" ]]; then
  TRAIN_COMMAND="$BASE --override-num-epochs 3"
elif [[ "$PHASE" == "resume" ]]; then
  TRAIN_COMMAND="$BASE --override-num-epochs 6 && touch '$EXP_WORK_ROOT/gates/ddp_smoke.passed'"
else
  # The full gate deliberately performs a 3-epoch run followed by a resume to
  # epoch 6, verifying optimizer/EMA/RNG/config restoration rather than merely
  # completing a fresh six-epoch run.
  TRAIN_COMMAND="$BASE --override-num-epochs 3 && $BASE --override-num-epochs 6 && touch '$EXP_WORK_ROOT/gates/ddp_smoke.passed'"
fi
COMMAND="cd '$ROOT_DIR' && export CUDA_VISIBLE_DEVICES='$GPUS' PYTHONPATH='$PYTHONPATH' MASTER_ADDR='${MASTER_ADDR:-127.0.0.1}' MASTER_PORT='$PORT' && $TRAIN_COMMAND"
run_nohup "train_denseobs_conditional_sb_smoke_${PHASE}" training "$COMMAND"
