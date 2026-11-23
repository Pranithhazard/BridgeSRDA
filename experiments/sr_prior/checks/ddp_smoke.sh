#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
GPUS="${1:-0,1,2,3}"
WORLD="$(count_gpus "$GPUS")"
PORT="${MASTER_PORT:-29783}"
ADDRESS="${MASTER_ADDR:-127.0.0.1}"
COMMAND="cd '$ROOT_DIR' && export CUDA_VISIBLE_DEVICES='$GPUS' PYTHONPATH='$PYTHONPATH' MASTER_ADDR='$ADDRESS' MASTER_PORT='$PORT' && '$QG3D_PYTHON' python/scripts/filo_srda_qg/sb_prior_retrain_qg/train_sb_prior_retrain_ddp.py --config-path '$CONFIG_SMOKE' --world-size '$WORLD' --root-dir '$ROOT_DIR'"
run_nohup "train_ddp_smoke" training "$COMMAND"
