from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from src.filo_srda_qg.shared_srda.data import make_direct_si_loader
from src.filo_srda_qg.window_data import WindowDatasetConfig, NotsuWindowDataset


def make_sb_dataset(
    *,
    base_config: dict[str, Any],
    root_dir: str | Path,
    split: str,
    use_mixup: bool,
    samples_per_epoch: int | None,
) -> NotsuWindowDataset:
    data = copy.deepcopy(base_config)
    data.update(
        split=split,
        use_mixup=bool(use_mixup),
        samples_per_epoch=samples_per_epoch,
        use_ground_truth_clipping=split in {"train", "valid"},
        use_observation=False,
        obs_noise_std=0.0,
        fixed_obs_mask_index=0,
    )
    return NotsuWindowDataset(
        WindowDatasetConfig.from_dict(data), root_dir=root_dir
    )


make_sb_loader = make_direct_si_loader


def make_sb_observation_dataset(
    *,
    base_config: dict[str, Any],
    root_dir: str | Path,
    split: str,
    fixed_obs_mask_index: int | None,
    obs_noise_std: float,
) -> NotsuWindowDataset:
    data = copy.deepcopy(base_config)
    data.update(
        split=split,
        use_mixup=False,
        samples_per_epoch=None,
        use_ground_truth_clipping=split in {"train", "valid"},
        use_observation=True,
        obs_noise_std=float(obs_noise_std),
        fixed_obs_mask_index=fixed_obs_mask_index,
    )
    return NotsuWindowDataset(
        WindowDatasetConfig.from_dict(data), root_dir=root_dir
    )
