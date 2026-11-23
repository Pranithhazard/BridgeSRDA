#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
GPUS="${1:-0,1,2,3}"
PHASE="${2:-full}"
WORLD="$(count_gpus "$GPUS")"
[[ "$WORLD" == "4" ]] || { echo "smoke requires four GPUs for global batch 32" >&2; exit 2; }
PORT="${MASTER_PORT:-29791}"
EXTRA=""
if [[ "$PHASE" == "first" ]]; then
  EXTRA="--override-num-epochs 3"
elif [[ "$PHASE" == "resume" || "$PHASE" == "full" ]]; then
  EXTRA="--override-num-epochs 6"
else
  echo "phase must be first, resume, or full" >&2
  exit 2
fi
COMMAND="cd '$ROOT_DIR' && export CUDA_VISIBLE_DEVICES='$GPUS' PYTHONPATH='$PYTHONPATH' MASTER_ADDR='${MASTER_ADDR:-127.0.0.1}' MASTER_PORT='$PORT' && '$QG3D_PYTHON' python/scripts/filo_srda_qg/conditional_sb_srda_qg/train_conditional_sb_ddp.py --config-path '$CONFIG_SMOKE' --world-size 4 --root-dir '$ROOT_DIR' $EXTRA"
run_nohup "train_conditional_sb_smoke_${PHASE}" training "$COMMAND"
