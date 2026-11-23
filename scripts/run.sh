#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
export PYTHONPATH="$root/python"
export MPLBACKEND="${MPLBACKEND:-Agg}"
python="${PYTHON:-python3}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run.sh [--dry-run] COMMAND [arguments]

  smoke
  generate-hr | split-hr | generate-lr
  generate-uhr GROUP
  train-bridge-d | train-bridge-i
  train-prior-p1 | train-prior-p2 | train-4d-srda
  bridge-d SEED CHECKPOINT [NFE] [M] [evaluator options]
  bridge-i SEED CHECKPOINT
  sr-letkf SEED CHECKPOINT
  nowcast SEED CHECKPOINT | sparse-bridge SEED CHECKPOINT
  stride-two SEED CHECKPOINT
  hr-letkf-m30 SEED | hr-letkf-m10 SEED | sparse-hr-letkf SEED
  lr-letkf SEED | lr-free-run | evaluate-4d-srda
  prior-p1-panel RAW_CHECKPOINT | prior-p2-panel RAW_CHECKPOINT
  train-diffusion-unet | train-diffusion-maxvit | diffusion-budget NFE
  benchmark-cycling OUTPUT | benchmark-worker bridge|letkf MEMBERS OUTPUT [worker options]
  score-main [table options]
  summarize-budgets | summarize-nowcasts | summarize-stride-two | summarize-sparse

GROUP is 9990_9992, 9993_9995, 9996_9997 or 9998_9999.
See README.md for inputs and examples.
EOF
}

dry_run=0
if [[ "${1:-}" == --dry-run ]]; then dry_run=1; shift; fi
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
if [[ $# == 0 ]]; then usage >&2; exit 2; fi
action="$1"
shift

fail() { echo "$*" >&2; exit 2; }

run() {
  [[ -f "$1" ]] || fail "Missing entry point: $1"
  if [[ "$dry_run" == 1 ]]; then
    printf '%q ' "$python" "$@"
    printf '\n'
  else
    exec "$python" "$@"
  fi
}

new_output() {
  if [[ "$dry_run" == 0 && ( -e "$1" || -L "$1" ) ]]; then
    fail "Output already exists: $1"
  fi
}

require_gates() {
  local directory="work/experiments/$1/gates"
  shift
  [[ "$dry_run" == 0 ]] || return 0
  for gate in "$@"; do
    [[ -f "$directory/$gate.passed" ]] || fail "Missing production gate: $directory/$gate.passed"
  done
}

seed_group() {
  case "$1" in
    999[0-2]) echo 9990_9992 ;;
    999[3-5]) echo 9993_9995 ;;
    999[6-7]) echo 9996_9997 ;;
    999[8-9]) echo 9998_9999 ;;
    *) fail "Expected a manuscript evaluation seed 9990–9999: $1" ;;
  esac
}

filo=python/scripts/filo_srda_qg
four=python/scripts/four_dim_srda
cfg=python/configs/filo_srda_qg
handover=python/configs/four_dim_srda/handover
cfd="$handover/cfd_simulation/qg_model"
filter="$handover/perform_letkf_hr_using_uhr"
d=denseobs_conditional_sb_srda_qg
i=conditional_sb_srda_qg
p=sb_prior_retrain_qg
diff=diffusion_srda_qg
d_config="$cfg/$d/denseobs_conditional_sb_p2_b32.yml"
i_config="$cfg/$i/conditional_sb_p2_b32.yml"
p1_config="$cfg/$p/sb_prior_retrain_b32.yml"
p2_config="$cfg/$p/sb_prior_retrain_p2_b32_2gpu.yml"
srda_stem=bea2_bed2_dspe360_nsls100_ogx08_ogy08_n3drb3_nmb6_bias0_bs12_lr1e-04
srda_config="$handover/perform_4D_SRDA/UNetMaxVitVer01/$srda_stem.yml"
legacy_filter="$filter/letkf_cfg_ogx08_ogy08_ne100_ch16e-04_cr6e+00_if12e-01_lr57e-01_bs6.yml"
unet_config="$cfg/$diff/diffusion_diffsrda3d_b32_fp32_filledobs.yml"
maxvit_config="$cfg/$diff/diffusion_w144_fp32_reconsel.yml"
comparison=work/experiments/comparative_analysis_sb_srda_qg
presentation="$comparison/metrics/presentation"
followup=work/experiments/073a_denseobs_conditional_sb_srda_qg/evaluation/paper_followup_20261005
data=data/four_dim_srda/handover/cfd_simulation/qg_model

case "$action" in
  smoke) run examples/cpu_smoke.py "$@" ;;
  generate-hr|split-hr|generate-lr)
    case "$action" in
      generate-hr) script=make_hr_training_data; pattern='*_hr_pv_*.npy'; output="$data/hr_pv" ;;
      split-hr) script=split_hr_training_data_in_parallel; pattern='*_hr_pv_*.npy'; output="$data/hr_and_lr_pv" ;;
      generate-lr) script=make_lr_training_data; pattern='*_lr_pv_*.npy'; output="$data/hr_and_lr_pv" ;;
    esac
    if [[ "$dry_run" == 0 && -d "$output" ]]; then
      existing="$(find "$output" -name "$pattern" -print -quit)"
      [[ -z "$existing" ]] || fail "Array already exists: $existing"
    fi
    run "$four/cfd_simulation/qg_model/$script.py" --config_path "$cfd/gpu_make_data_config.yml" ;;
  generate-uhr)
    group="${1:?Provide a seed group}"
    case "$group" in 9990_9992|9993_9995|9996_9997|9998_9999) ;; *) fail "Unknown seed group: $group" ;; esac
    output="$data/uhr_pv_narrow_jet"
    for ((seed=${group%_*}; seed<=${group#*_}; seed++)); do
      new_output "$output/seed0$seed"
    done
    run "$four/cfd_simulation/qg_model/make_uhr_data_using_narrow_jet.py" --config_path "$cfd/gpu_evaluation_uhr_seed$group.yml" ;;
  train-bridge-d)
    require_gates 073a_denseobs_conditional_sb_srda_qg interpolation data preflight overfit ddp_smoke regression_tests
    run "$filo/$d/train_denseobs_conditional_sb_ddp.py" --config-path "$d_config" --world-size 2 --override-gradient-accumulation-steps 16 "$@" ;;
  train-bridge-i)
    run "$filo/$i/train_conditional_sb_ddp.py" --config-path "$i_config" --world-size 4 --override-num-epochs 1200 "$@" ;;
  train-prior-p1|train-prior-p2)
    if [[ "$action" == train-prior-p1 ]]; then config="$p1_config"; ranks=4; else config="$p2_config"; ranks=2; fi
    run "$filo/$p/train_sb_prior_retrain_ddp.py" --config-path "$config" --world-size "$ranks" "$@" ;;
  train-4d-srda)
    run "$four/srda/train_ddp_ml_model.py" --config_path "$srda_config" --world_size 4 --model_name UNetMaxVitVer01 "$@" ;;
  evaluate-4d-srda)
    new_output "python/results/four_dim_srda/handover/srda/UNetMaxVitVer01/use_narrow_jet/$srda_stem"
    run "$four/srda/evaluate_using_fixed_obs_point_and_narrow_jet.py" --config_srda_path "$srda_config" \
      --config_cfd_path "$cfd/gpu_evaluation_config.yml" --model_name UNetMaxVitVer01 "$@" ;;
  bridge-d|bridge-i|sr-letkf|nowcast|sparse-bridge|stride-two)
    seed="${1:?Provide seed}"; checkpoint="${2:?Provide checkpoint}"
    seed_group "$seed" >/dev/null
    shift 2
    ns="$d"; config="$d_config"; epoch=ep2000; mode=sr_only; nfe=20; members=30; extra=()
    case "$action" in
      bridge-d)
        if [[ "${1:-}" =~ ^[0-9]+$ ]]; then nfe="$1"; shift; fi
        if [[ "${1:-}" =~ ^[0-9]+$ ]]; then members="$1"; shift; fi ;;
      bridge-i) ns="$i"; config="$i_config"; epoch=ep1200 ;;
      sr-letkf) ns="$p"; config="$p2_config"; epoch=ep0900; mode=sr_letkf ;;
      nowcast) members=10; extra=(--save-nowcast-metrics) ;;
      sparse-bridge) extra=(--observation-scenario sparse_ogx16_ogy16) ;;
      stride-two) members=10; extra=(--cycling-stride 2 --save-nowcast-metrics) ;;
    esac
    if [[ "$dry_run" == 0 ]]; then [[ -f "$checkpoint" ]] || fail "Missing checkpoint: $checkpoint"; fi
    run "$filo/$ns/evaluate_narrow_jet.py" --config-path "$config" --mode "$mode" --seed "$seed" \
      --weights-kind ema --checkpoint-path "$checkpoint" --checkpoint-label "$epoch" --nfe-budget "$nfe" \
      --ensemble-size "$members" --brownian-base-steps 600 --sampling-seed 7042 ${extra[@]+"${extra[@]}"} "$@" ;;
  hr-letkf-m30|hr-letkf-m10|sparse-hr-letkf)
    seed="${1:?Provide seed}"
    group="$(seed_group "$seed")"
    shift
    members=30; native=030; scenario=canonical_ogx08_ogy08
    output="work/experiments/072_conditional_sb_srda_qg/evaluation/narrow_jet/hr_letkf_m30_native"
    if [[ "$action" == hr-letkf-m10 ]]; then
      members=10; native=010
      case "$seed" in 999[0-4]) group=9990_9994 ;; *) group=9995_9999 ;; esac
      output="${output/m30_native/m10_native}"
    elif [[ "$action" == sparse-hr-letkf ]]; then
      scenario=sparse_ogx16_ogy16
      output="$followup/hr_letkf_m30/$scenario"
    fi
    run "$filo/letkf/evaluate_letkf_uq_narrow_jet.py" --seed "$seed" --matched-ensemble-size "$members" \
      --config-cfd-path "$filter/cfd_cfg_ne${native}_na3e-03_uhr_seed$group.yml" --config-srda-path "$srda_config" \
      --config-letkf-path "$filter/letkf_cfg_ogx08_ogy08_ne${native}_ch16e-04_cr6e+00_if12e-01_lr57e-01_bs6.yml" \
      --observation-scenario "$scenario" --output-root "$output" --skip-canonical-check "$@" ;;
  lr-letkf)
    seed="${1:?Provide seed}"; seed_group "$seed" >/dev/null; shift
    run "$filo/$i/evaluate_lr_letkf_baselines.py" --seed "$seed" --ensemble-size 30 --baseline-kind true_lr_letkf \
      --config-letkf-path "$legacy_filter" --lr-obs-grid-mode physical_matched "$@" ;;
  lr-free-run)
    run "$filo/$d/evaluate_lr_free_run.py" --seed 9991 --output-dir "$followup/lr_free_run/seed09991" "$@" ;;
  prior-p1-panel|prior-p2-panel)
    checkpoint="${1:?Provide raw checkpoint}"; shift
    if [[ "$action" == prior-p1-panel ]]; then config="$p1_config"; else config="$p2_config"; fi
    run "$filo/$p/evaluate_sampler_panel.py" --config-path "$config" --checkpoint-path "$checkpoint" \
      --nfe-budget 20 --ensemble-size 8 --brownian-base-steps 600 --sampling-seed 7042 "$@" ;;
  train-diffusion-unet|train-diffusion-maxvit)
    if [[ "$action" == train-diffusion-unet ]]; then config="$unet_config"; else config="$maxvit_config"; fi
    run "$filo/$diff/train_diffusion_ddp.py" --config-path "$config" --world-size 4 "$@" ;;
  diffusion-budget)
    nfe="${1:?Provide inference NFE}"; shift
    case "$nfe" in 100|200|500) ;; *) fail "Published diffusion budgets are 100, 200 and 500" ;; esac
    run "$filo/$diff/evaluate_diffusion_windows.py" --config-path "$maxvit_config" --protocol matched_clean \
      --weights-kind reconstruction --max-batches 32 --ensemble-size 8 --eta 1 --num-steps "$nfe" "$@" ;;
  benchmark-cycling)
    output="${1:?Provide a new output directory}"; shift
    run "$filo/$d/benchmark_cycling.py" --root "$root" --output "$output" "$@" ;;
  benchmark-worker)
    method="${1:?Provide method}"; members="${2:?Provide members}"; output="${3:?Provide a new output directory}"; shift 3
    run "$filo/$d/benchmark_cycling_worker.py" --root "$root" --output "$output" --method "$method" \
      --ensemble-size "$members" --seed 9991 "$@" ;;
  score-main) run analysis/tables/canonical.py "$@" ;;
  summarize-budgets)
    new_output "$presentation/073a_reduced_cost"
    run "$filo/$d/summarize_reduced_cost_evaluation.py" "$@" ;;
  summarize-nowcasts)
    new_output "$presentation/073a_nowcast_nfe20_m10"
    run "$filo/$d/summarize_nowcast_evaluation.py" "$@" ;;
  summarize-stride-two)
    output="$presentation/073a_stride2_pilot"
    new_output "$output"
    run "$filo/$d/analyze_stride2_cycling.py" --output-dir "$output" "$@" ;;
  summarize-sparse)
    output="$presentation/hr_letkf_sparse_followup"
    new_output "$output"
    run "$filo/$d/summarize_sparse_observations.py" --output-dir "$output" "$@" ;;
  *) usage >&2; fail "Unknown command: $action" ;;
esac
