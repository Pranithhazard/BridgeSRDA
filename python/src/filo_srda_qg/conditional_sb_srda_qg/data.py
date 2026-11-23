from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from src.filo_srda_qg.window_data import (
    WindowDatasetConfig,
    NotsuWindowDataset,
)
from src.filo_srda_qg.sr_prior.data import make_sb_loader


def make_conditional_sb_dataset(
    *,
    base_config: dict[str, Any],
    root_dir: str | Path,
    split: str,
    use_mixup: bool,
    samples_per_epoch: int | None,
    deterministic_observations: bool = False,
) -> NotsuWindowDataset:
    data = copy.deepcopy(base_config)
    configured_fixed_index = base_config.get("fixed_obs_mask_index")
    fixed_index = (
        int(configured_fixed_index)
        if configured_fixed_index is not None
        else (
            int(base_config.get("validation_obs_mask_index", 0))
            if deterministic_observations
            else None
        )
    )
    data.update(
        split=split,
        use_mixup=bool(use_mixup),
        samples_per_epoch=samples_per_epoch,
        use_ground_truth_clipping=split in {"train", "valid"},
        use_observation=True,
        obs_noise_std=float(base_config.get("obs_noise_std", 0.022)),
        fixed_obs_mask_index=fixed_index,
    )
    return NotsuWindowDataset(
        WindowDatasetConfig.from_dict(
            {key: value for key, value in data.items() if key != "validation_obs_mask_index"}
        ),
        root_dir=root_dir,
    )


__all__ = ["make_conditional_sb_dataset", "make_sb_loader"]
