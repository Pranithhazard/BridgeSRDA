#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/_common.sh"

mkdir -p "$EXP_WORK_ROOT/gates"

"$QG3D_PYTHON" -m pytest -q \
  python/tests/filo_srda_qg/test_denseobs_conditional_sb_srda_qg.py \
  python/tests/filo_srda_qg/test_conditional_sb_srda_qg.py \
  python/tests/filo_srda_qg/test_sb_prior_retrain_qg.py

date -u +"%Y-%m-%dT%H:%M:%SZ" > "$EXP_WORK_ROOT/gates/regression_tests.passed"
echo "Regression tests passed: 073A, 072, and 071B."
