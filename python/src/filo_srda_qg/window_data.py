from __future__ import annotations

import dataclasses
import glob
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from src.four_dim_srda.data.dataloader import split_file_paths
from src.four_dim_srda.data.get_path import get_similar_source_lr_path
from torch.utils.data import Dataset


@dataclasses.dataclass
class WindowDatasetConfig:
    data_dir_name: str = "handover/cfd_simulation/qg_model/hr_and_lr_pv"
    split: str = "train"
    train_valid_test_ratios: list[float] = dataclasses.field(
        default_factory=lambda: [0.7, 0.2, 0.1]
    )
    allowed_seed_ids: list[int] | None = None
    allowed_start_indices: list[int] | None = None
    allowed_members: list[int] | None = None
    max_start_time_index: int = 792
    max_files: int | None = None
    samples_per_epoch: int | None = None
    split_seed: int = 42
    beta_dist_alpha: float = 2.0
    beta_dist_beta: float = 2.0
    max_ensemble_number: int = 20
    num_searched_lr_states: int = 100
    nx_hr: int = 128
    ny_hr: int = 65
    nz_hr: int = 16
    obs_grid_interval_x: int = 8
    obs_grid_interval_y: int = 8
    obs_noise_std: float = 0.022
    pv_min: float = -2.88
    pv_max: float = 2.92
    missing_value: float = 0.0
    lr_and_obs_time_interval: int = 4
    is_future_obs_missing: bool = True
    fixed_obs_mask_index: int | None = None
    use_mixup: bool = True
    use_observation: bool = True
    use_ground_truth_clipping: bool | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WindowDatasetConfig":
        return cls(**data)


def split_seed_dirs(
    seed_dirs: list[Path], ratios: list[float]
) -> dict[str, list[Path]]:
    if len(seed_dirs) < 3:
        return {"train": seed_dirs, "valid": seed_dirs, "test": seed_dirs}
    train, valid, test = split_file_paths([str(path) for path in seed_dirs], ratios)
    return {
        "train": [Path(path) for path in train],
        "valid": [Path(path) for path in valid],
        "test": [Path(path) for path in test],
    }


def parse_hr_filename(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    name = path.name
    seed = re.search(r"seed(\d+)", name)
    span = re.search(r"start(\d+)_end(\d+)", name)
    member = re.search(r"hr_pv_(\d+)\.npy$", name)
    return {
        "hr_path": str(path),
        "lr_path": str(path).replace("hr_pv", "lr_pv"),
        "seed": int(seed.group(1)) if seed else None,
        "start": int(span.group(1)) if span else None,
        "end": int(span.group(2)) if span else None,
        "member": int(member.group(1)) if member else None,
    }


def scan_hr_paths(root_dir: str | Path, cfg: WindowDatasetConfig) -> list[Path]:
    data_dir = Path(root_dir) / "data/four_dim_srda" / cfg.data_dir_name
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing Notsu HR/LR data directory: {data_dir}")
    seed_dirs = sorted([Path(p) for p in glob.glob(f"{data_dir}/seed*") if os.path.isdir(p)])
    splits = split_seed_dirs(seed_dirs, cfg.train_valid_test_ratios)
    if cfg.split not in splits:
        raise ValueError(f"Unsupported split: {cfg.split}")
    paths: list[Path] = []
    for seed_dir in splits[cfg.split]:
        paths.extend(sorted(seed_dir.glob("*_hr_pv_*.npy")))
    filtered = []
    for path in paths:
        meta = parse_hr_filename(path)
        if cfg.allowed_seed_ids is not None and meta["seed"] not in set(cfg.allowed_seed_ids):
            continue
        if cfg.allowed_start_indices is not None and meta["start"] not in set(cfg.allowed_start_indices):
            continue
        if cfg.allowed_members is not None and meta["member"] not in set(cfg.allowed_members):
            continue
        if meta["start"] is None or meta["start"] <= cfg.max_start_time_index:
            filtered.append(path)
    if cfg.max_files is not None:
        filtered = filtered[: cfg.max_files]
    if not filtered:
        raise FileNotFoundError(f"No HR windows found under {data_dir} for split={cfg.split}")
    return filtered


class NotsuWindowDataset(Dataset):
    """Notsu-compatible 4D-SRDA windows with an explicit observation mask."""

    def __init__(
        self,
        cfg: WindowDatasetConfig,
        *,
        root_dir: str | Path,
        dtype: torch.dtype = torch.float32,
    ):
        self.cfg = cfg
        self.root_dir = Path(root_dir)
        self.dtype = dtype
        self.all_hr_paths = scan_hr_paths(self.root_dir, cfg)
        self.active_indices = list(range(len(self.all_hr_paths)))
        self.set_epoch(0)

        self.dict_all_lr_data_at_init_time = (
            self._load_all_lr_data_at_init_time() if cfg.use_mixup else {}
        )
        self.is_obses = make_observation_masks(
            nx=cfg.nx_hr,
            ny=cfg.ny_hr,
            nz=cfg.nz_hr,
            obs_grid_interval_x=cfg.obs_grid_interval_x,
            obs_grid_interval_y=cfg.obs_grid_interval_y,
            dtype=self.dtype,
        )

    def set_epoch(self, epoch: int) -> None:
        if self.cfg.samples_per_epoch is None:
            self.active_indices = list(range(len(self.all_hr_paths)))
            return
        rng = np.random.default_rng(self.cfg.split_seed + epoch)
        indices = rng.choice(
            len(self.all_hr_paths),
            size=self.cfg.samples_per_epoch,
            replace=len(self.all_hr_paths) < self.cfg.samples_per_epoch,
        )
        self.active_indices = [int(i) for i in indices]

    def __len__(self) -> int:
        return len(self.active_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.all_hr_paths[self.active_indices[idx]]
        return self.get_by_path(path)

    def get_by_path(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        target_lr, source_lr, hr = self._load_np_data(path)
        obs, mask, obs_mask_index = self._make_observation_and_mask(hr)

        target_lr = self._preprocess(target_lr, use_clipping=True)
        source_lr = self._preprocess(source_lr, use_clipping=True)
        hr = self._preprocess(
            hr,
            use_clipping=bool(self.cfg.use_ground_truth_clipping)
            if self.cfg.use_ground_truth_clipping is not None
            else False,
        )
        obs = self._preprocess(obs, use_clipping=True)
        mask = self._preprocess_mask(mask)

        obs = obs[:: self.cfg.lr_and_obs_time_interval]
        mask = mask[:: self.cfg.lr_and_obs_time_interval]
        if self.cfg.use_observation:
            obs = torch.nan_to_num(obs, nan=self.cfg.missing_value)
        else:
            obs = torch.full_like(obs, fill_value=self.cfg.missing_value)
            mask = torch.zeros_like(mask)

        if self.cfg.is_future_obs_missing:
            center_index = len(obs) // 2
            obs[center_index + 1 :] = self.cfg.missing_value
            mask[center_index + 1 :] = 0.0

        if self.cfg.use_mixup:
            w = np.random.beta(self.cfg.beta_dist_alpha, self.cfg.beta_dist_beta, size=1)[0]
            condition_lr = w * source_lr + (1.0 - w) * target_lr
        else:
            condition_lr = target_lr

        return {
            # ``lr`` remains the backward-compatible name used by the Notsu,
            # latent-SRDA, and Phase-1 code paths.  DirectSI distinguishes the
            # augmented condition from the paired LR used to construct its
            # stochastic source.
            "lr": condition_lr.to(self.dtype),
            "condition_lr": condition_lr.to(self.dtype),
            "source_lr": target_lr.to(self.dtype),
            "obs": obs.to(self.dtype),
            "mask": mask.to(self.dtype),
            "hr": hr.to(self.dtype),
            "metadata": {**parse_hr_filename(path), "obs_mask_index": obs_mask_index},
        }

    def _load_np_data(self, hr_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hr = np.load(hr_path)
        target_lr_path = str(hr_path).replace("hr_pv", "lr_pv")
        target_lr = np.load(target_lr_path)

        if self.cfg.use_mixup:
            key = re.search(r"start\d+_end\d+", Path(target_lr_path).name).group()
            source_lr_path = get_similar_source_lr_path(
                key,
                target_lr[0],
                self.dict_all_lr_data_at_init_time,
                self.cfg.max_ensemble_number,
                self.cfg.num_searched_lr_states,
            )
            source_lr = (
                np.load(source_lr_path) if source_lr_path is not None else target_lr
            )
        else:
            source_lr = target_lr

        return (
            torch.from_numpy(target_lr).to(self.dtype),
            torch.from_numpy(source_lr).to(self.dtype),
            torch.from_numpy(hr).to(self.dtype),
        )

    def _load_all_lr_data_at_init_time(self) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for hr_path in self.all_hr_paths:
            lr_path = str(hr_path).replace("hr_pv", "lr_pv")
            key = re.search(r"start\d+_end\d+", Path(lr_path).name).group()
            out.setdefault(key, [])
            lr = np.load(lr_path, mmap_mode="r")
            out[key].append({"data": np.asarray(lr[0]), "path": lr_path})
        return out

    def _make_observation_and_mask(self, hr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self.cfg.fixed_obs_mask_index is None:
            obs_mask_index = random.randrange(len(self.is_obses))
        else:
            obs_mask_index = int(self.cfg.fixed_obs_mask_index)
            if not 0 <= obs_mask_index < len(self.is_obses):
                raise ValueError(
                    f"fixed_obs_mask_index={obs_mask_index} is outside "
                    f"[0, {len(self.is_obses) - 1}]"
                )
        is_obs = self.is_obses[obs_mask_index]
        # broadcast_to returns an expanded alias.  The future-observation
        # masking below is in-place, so materialize an independent tensor to
        # prevent one sample from corrupting the stored observation layouts.
        is_obs = torch.broadcast_to(is_obs, hr.shape).clone()
        obs = torch.full_like(hr, torch.nan)
        obs = torch.where(is_obs > 0, hr, obs)
        if self.cfg.obs_noise_std > 0:
            noise = torch.normal(mean=0.0, std=self.cfg.obs_noise_std, size=obs.shape, dtype=self.dtype)
            obs = obs + noise
        return obs, is_obs.to(self.dtype), obs_mask_index

    def _preprocess(self, data: torch.Tensor, *, use_clipping: bool) -> torch.Tensor:
        ret = data[:, None, :, :-1, :]
        ret = (ret - self.cfg.pv_min) / (self.cfg.pv_max - self.cfg.pv_min)
        if use_clipping:
            ret = torch.clamp(ret, 0.0, 1.0)
        return ret

    @staticmethod
    def _preprocess_mask(mask: torch.Tensor) -> torch.Tensor:
        return mask[:, None, :, :-1, :]


def make_observation_masks(
    *,
    nx: int,
    ny: int,
    nz: int,
    obs_grid_interval_x: int,
    obs_grid_interval_y: int,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    masks = []
    for init_x in range(obs_grid_interval_x):
        for init_y in range(obs_grid_interval_y):
            mask = torch.zeros(nz, ny, nx, dtype=dtype)
            mask[:, init_y::obs_grid_interval_y, init_x::obs_grid_interval_x] = 1.0
            masks.append(mask)
    return masks
