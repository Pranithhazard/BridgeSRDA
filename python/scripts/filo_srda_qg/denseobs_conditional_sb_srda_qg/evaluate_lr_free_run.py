"""Single-seed unassimilated LR control, initialized exactly like 073A.

Run on the server. Never resets to truth after t=0 and never assimilates.
The saved lifted fields are bicubic in x/y, linear in z, for visualization.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.filo_srda_qg.sb_prior_retrain_qg.evaluate_narrow_jet import (
    DEFAULT_CFD_CONFIG, make_filter, make_initial_hr_pv,
)
from src.four_dim_srda.config.experiment_config import CFDConfig
from src.four_dim_srda.utils.random_seed_helper import set_seeds
from src.qg_model.qg_model import QGModel


def lift_bicubic_xy_linear_z(x: torch.Tensor, hr_shape: tuple[int, int, int]) -> torch.Tensor:
    nz, ny, nx = hr_shape
    xy = F.interpolate(x, size=(ny, nx), mode="bicubic", align_corners=True)
    z = F.interpolate(xy[:, None], size=(nz, ny, nx), mode="trilinear", align_corners=True)[:, 0]
    # The native y=0 and y=Ly boundary values are zero; retain them exactly.
    z[:, :, [0, -1], :] = 0
    return z


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=9991)
    parser.add_argument("--config-cfd-path", default=DEFAULT_CFD_CONFIG)
    parser.add_argument("--selected-indices", nargs="+", type=int, default=[160, 500, 792])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Run the physical free-run integration in the server GPU container")
    root = args.root_dir.resolve()
    cfg = CFDConfig.load(root / args.config_cfd_path)
    cfg_base = copy.deepcopy(cfg)
    if cfg_base.hr_base_config.n_batch != 1:
        raise ValueError("Matched 073A initialization requires the one-member CFD base config")
    num_times = int(cfg.time_config.end_time / cfg.time_config.output_hr_dt)
    selected = sorted(set(args.selected_indices))
    if any(i < 0 or i >= num_times for i in selected):
        raise ValueError("Requested field time outside the saved trajectory")
    out = args.output_dir
    if (out / "_COMPLETE").exists():
        print(f"Already complete: {out}")
        return
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite partial output: {out}")
    out.mkdir(parents=True)
    cfg.lr_base_config.n_batch = 1
    cfg.lr_base_config.device = "cuda:0"
    cfg.hr_base_config.device = "cuda:0"
    set_seeds(555 * args.seed, use_deterministic=True)
    initial = make_initial_hr_pv(cfg_base).to("cuda:0").float()
    model = QGModel(cfg.lr_base_config, show_input_cfg_info=False)
    model.initialize_pv(make_filter(cfg).apply(initial))
    dt = float(cfg.time_config.lr_dt)
    spacing = float(cfg.time_config.output_hr_dt)
    steps = round(spacing / dt)
    if not np.isclose(steps * dt, spacing):
        raise ValueError("LR integration dt does not divide saved-frame spacing")
    native, snapshots = [], []
    for i in range(num_times):
        state = model.get_grid_pv()
        if not torch.isfinite(state).all():
            raise FloatingPointError(f"Nonfinite free-run state at index {i}")
        native.append(state[0].float().cpu().numpy())
        if i in selected:
            snapshots.append(lift_bicubic_xy_linear_z(state.float(), (cfg.hr_base_config.nz, cfg.hr_base_config.ny, cfg.hr_base_config.nx))[0].cpu().numpy())
        if i < num_times - 1:
            model.integrate_n_steps(dt_per_step=dt, n_steps=steps)
    np.savez_compressed(out / f"lr_free_run_seed{args.seed:05d}.npz", native_lr=np.stack(native), mean=np.stack(snapshots), cycle_indices=np.array(selected), physical_times=np.array(selected) * spacing)
    metadata = {"seed": args.seed, "initialization": "same analytic HR initial condition and seed multiplier 555 as 073A, then same low-pass LR initialization", "truth_reset_after_initialization": False, "assimilations": 0, "lr_dt": dt, "saved_dt": spacing, "selected_indices": selected, "lifting": "bicubic xy align_corners=True; linear z; exact zero y-boundaries", "interpretation": "single-trajectory phase/structure control; not an ensemble/UQ baseline"}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out / "_COMPLETE").write_text("complete\n")
    print(f"COMPLETE: {out}")


if __name__ == "__main__":
    main()
