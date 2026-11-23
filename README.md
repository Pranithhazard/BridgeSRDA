# BridgeSRDA: Efficient Probabilistic Super-Resolution Data Assimilation with Forecast-Informed Stochastic Interpolants

BridgeSRDA combines low-resolution quasi-geostrophic (QG) forecasts with sparse observations to generate ensembles of high-resolution analysis windows. Each member's current analysis starts its next low-resolution QG forecast. BridgeSRDA-D uses interpolated observation fields, while BridgeSRDA-I uses masked observation-minus-forecast differences.

## Install and check

Use Python 3.11 and a PyTorch build appropriate for your platform.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
bash scripts/run.sh smoke
bash scripts/run.sh --help
```

The smoke example runs two small QG steps and checks both samplers on a CPU using generated inputs. The paper experiments use Linux and CUDA, with NCCL for multi-GPU training. The Appendix F benchmark used an NVIDIA RTX A6000 and PyTorch `2.12.0+cu130`.

## Run the paper experiments

Run these commands from the repository root using `bash scripts/run.sh COMMAND`. To see the script and arguments a command will use, put `--dry-run` before the command. Set `CUDA_VISIBLE_DEVICES` for CUDA runs. The solver and networks are in `python/src/`, the scripts that run them are in `python/scripts/`, and the configurations are in `python/configs/`.

| Task | Command |
|---|---|
| Paired training windows | Run `generate-hr`, `split-hr`, then `generate-lr`. |
| Narrow-jet reference | Run `generate-uhr GROUP` for `9990_9992`, `9993_9995`, `9996_9997`, and `9998_9999`. |
| Main training | `train-bridge-d` uses two ranks with gradient accumulation of 16. `train-bridge-i` uses four ranks with accumulation of 8. |
| Main cycling | Run `bridge-d SEED CHECKPOINT`, `bridge-i SEED CHECKPOINT`, and `sr-letkf SEED CHECKPOINT` with the EMA checkpoints from epochs 2000, 1200, and 900, respectively. Each uses NFE20/M30. |
| Baselines | Run `hr-letkf-m30 SEED`, `hr-letkf-m10 SEED`, and `lr-letkf SEED`. Use `train-4d-srda` and `evaluate-4d-srda` for the deterministic baseline and its observation archives. `lr-free-run` produces the forecast without assimilation. |
| Budget, nowcast and sparse results | Use `bridge-d SEED CHECKPOINT NFE M` to select a sampling budget and ensemble size. Use `nowcast SEED CHECKPOINT` for NFE20/M10 with lead times, and `sparse-bridge` and `sparse-hr-letkf` for the spacing-16 comparison. |
| Appendix B | Run `train-prior-p1` and `train-prior-p2`, then use `prior-p1-panel` and `prior-p2-panel` with the raw epoch-525 checkpoints. |
| Appendix C | `stride-two SEED CHECKPOINT` for seeds 9990 and 9991. Compare with the normal `nowcast` runs. |
| Appendix D | `train-diffusion-unet`, `train-diffusion-maxvit`, and `diffusion-budget NFE` for 100, 200 or 500. The epoch-125 comparison uses each model's stored validation panel and its own targets. |
| Appendix F | `benchmark-cycling OUTPUT` runs the four conditions with a two-cycle check before full cycling. It verifies the canonical observation stream and Bridge checkpoint against the hashes in the asset manifest. |

For the required data and weights, see the [data](data/README.md) and [checkpoint](checkpoints/README.md) notes and the [asset manifest](manifests/assets.json). Once those inputs are in place, start with seed 9991. To check two BridgeSRDA cycles, add `--max-analyses 2 --selected-analysis-indices 0 1` to a cycling command. The full main comparison uses seeds 9990–9999 with 199 analyses each.

Before training BridgeSRDA-D, run the relevant data, interpolation, preflight, overfit, DDP smoke and regression checks in `experiments/bridge_d/checks/`. The two-rank DDP check is `ddp_smoke.sh 0,1 full`. BridgeSRDA-I and the SR prior have their own checks in `experiments/bridge_i/` and `experiments/sr_prior/`.

## Scores

```bash
bash scripts/run.sh score-main --root . --device cuda:0 --output runs/canonical
```

This writes scores for each cycle and seed, along with aggregate scores. To run the budget, nowcast, sparse and stride-two summaries, set the canonical output directory to `work/experiments/comparative_analysis_sb_srda_qg/metrics/presentation` in a fresh run workspace. Run the budget and nowcast summaries before the stride-two summary. See the paper for the reported results.

Cycling scores use the full HR grid, including boundary rows. The 4D-SRDA verification retains its historical first-row projection. The sparse comparison uses the same CPU implementation of structural similarity for all methods, so its values can differ from those in the original main table. The LETKF observation-covariance diagonal is 0.022, and generated observations have variance `0.022²`. Nowcast lead four has 198 verified windows per seed. Leads zero through three have 199.

## Citation and release status

See [CITATION.cff](CITATION.cff) for citation details. The QG, 4D-SRDA, LETKF and metric code is based on the Notsu–Yasuda–Onishi implementation. The transport follows the conditional stochastic-interpolant construction of Chen et al. and the LR-to-HR application of Yasuda and Onishi. The diffusion appendix builds on DiffSRDA. The voxel shuffle source credits RLSP and pixelshuffle3d. [Notsu et al.](https://doi.org/10.2151/sola.21B-001) · [Chen et al.](https://proceedings.mlr.press/v235/chen24n.html) · [Yasuda and Onishi](https://doi.org/10.1007/s44393-025-00007-7) · [DiffSRDA](https://github.com/Pranithhazard/DiffSRDA)

