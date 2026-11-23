from __future__ import annotations

from pathlib import Path
from typing import Any

import copy
import torch
import torch.nn.functional as F
from src.filo_srda_qg.shared_srda.data import (
    make_direct_si_dataset,
    make_direct_si_loader,
)
from src.filo_srda_qg.window_data import NotsuWindowDataset
from torch.utils.data import DataLoader, Dataset, Sampler


class DiffusionWindowDataset(Dataset):
    """Diffusion-only view over the shared Notsu [0,1] dataset."""

    def __init__(
        self,
        base: NotsuWindowDataset,
        *,
        data_scaling: str,
        diffusion_missing_value: float,
        observation_fill_mode: str = "sparse_sentinel",
    ):
        self.base = base
        self.cfg = base.cfg
        self.data_scaling = data_scaling
        self.diffusion_missing_value = float(diffusion_missing_value)
        self.observation_fill_mode = observation_fill_mode
        if data_scaling not in {"zero_one", "minus_one_one"}:
            raise ValueError(f"Unsupported diffusion data scaling: {data_scaling}")
        if data_scaling == "minus_one_one" and not (
            self.diffusion_missing_value < -1.0
            or self.diffusion_missing_value > 1.0
        ):
            raise ValueError(
                "Diffusion missing value must lie outside [-1,1]"
            )
        if observation_fill_mode not in {
            "sparse_sentinel",
            "bicubic_horizontal",
        }:
            raise ValueError(
                f"Unsupported observation fill mode: {observation_fill_mode}"
            )

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)

    def _transform(self, sample: dict[str, Any]) -> dict[str, Any]:
        output = dict(sample)
        output["sparse_obs_zero_one"] = sample["obs"]
        if self.data_scaling == "minus_one_one":
            for key in ("condition_lr", "source_lr", "hr"):
                output[key] = 2.0 * sample[key] - 1.0
        output["lr"] = output["condition_lr"]
        output["obs"] = prepare_diffusion_observations(
            sample["obs"],
            sample["mask"],
            data_scaling=self.data_scaling,
            diffusion_missing_value=self.diffusion_missing_value,
            observation_fill_mode=self.observation_fill_mode,
        )
        return output

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._transform(self.base[index])

    def get_by_path(self, path: str | Path) -> dict[str, Any]:
        return self._transform(self.base.get_by_path(path))


def diffusion_values_to_zero_one(
    values: torch.Tensor, *, data_scaling: str
) -> torch.Tensor:
    if data_scaling == "zero_one":
        return values
    if data_scaling == "minus_one_one":
        return 0.5 * (values + 1.0)
    raise ValueError(f"Unsupported diffusion data scaling: {data_scaling}")


def zero_one_conditions_to_diffusion(
    condition_lr: torch.Tensor,
    obs: torch.Tensor,
    *,
    data_scaling: str,
    diffusion_missing_value: float,
    observation_mask: torch.Tensor | None = None,
    observation_fill_mode: str = "sparse_sentinel",
) -> tuple[torch.Tensor, torch.Tensor]:
    if data_scaling not in {"zero_one", "minus_one_one"}:
        raise ValueError(f"Unsupported diffusion data scaling: {data_scaling}")
    if observation_mask is None:
        observation_mask = obs != 0.0
    condition = (
        condition_lr
        if data_scaling == "zero_one"
        else 2.0 * condition_lr - 1.0
    )
    return (
        condition,
        prepare_diffusion_observations(
            obs,
            observation_mask,
            data_scaling=data_scaling,
            diffusion_missing_value=diffusion_missing_value,
            observation_fill_mode=observation_fill_mode,
        ),
    )


def _bicubic_fill_unbatched(
    obs: torch.Tensor,
    mask: torch.Tensor,
    *,
    missing_value: float,
) -> torch.Tensor:
    if obs.ndim != 5 or mask.shape != obs.shape:
        raise ValueError(
            "Expected unbatched observations and mask shaped (T,1,Z,Y,X)"
        )
    output = torch.full_like(obs, float(missing_value))
    _, _, _, full_y, full_x = obs.shape
    grid_y = torch.linspace(
        -1.0, 1.0, full_y, device=obs.device, dtype=obs.dtype
    )
    grid_x = torch.linspace(
        -1.0, 1.0, full_x, device=obs.device, dtype=obs.dtype
    )
    mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
    grid = torch.stack([mesh_x, mesh_y], dim=-1)[None]
    for time_index in range(obs.shape[0]):
        spatial_mask = mask[time_index, 0] > 0
        if not torch.any(spatial_mask):
            continue
        reference_mask = spatial_mask[0]
        if not torch.all(spatial_mask == reference_mask[None]):
            raise ValueError("Observation layout must be identical across depth")
        y_indices = torch.where(reference_mask.any(dim=1))[0]
        x_indices = torch.where(reference_mask.any(dim=0))[0]
        cartesian = torch.zeros_like(reference_mask)
        cartesian[y_indices[:, None], x_indices[None, :]] = True
        if not torch.equal(reference_mask, cartesian):
            raise ValueError(
                "bicubic_horizontal requires a regular Cartesian observation grid"
            )
        sparse_values = (
            obs[time_index, 0]
            .index_select(1, y_indices)
            .index_select(2, x_indices)
        )
        filled = F.grid_sample(
            sparse_values[None],
            grid,
            mode="bicubic",
            align_corners=False,
            padding_mode="border",
        )[0]
        output[time_index, 0] = filled
    return output


def prepare_diffusion_observations(
    obs: torch.Tensor,
    mask: torch.Tensor,
    *,
    data_scaling: str,
    diffusion_missing_value: float,
    observation_fill_mode: str,
) -> torch.Tensor:
    if data_scaling not in {"zero_one", "minus_one_one"}:
        raise ValueError(f"Unsupported diffusion data scaling: {data_scaling}")
    if obs.shape != mask.shape:
        raise ValueError("Observation and mask shapes must match")
    has_batch = obs.ndim == 6
    if not has_batch and obs.ndim != 5:
        raise ValueError("Expected observations shaped (T,1,Z,Y,X) or batched")
    batches = zip(obs, mask) if has_batch else [(obs, mask)]
    outputs: list[torch.Tensor] = []
    for batch_obs, batch_mask in batches:
        if observation_fill_mode == "sparse_sentinel":
            observed = batch_mask > 0
            scaled = (
                batch_obs
                if data_scaling == "zero_one"
                else 2.0 * batch_obs - 1.0
            )
            transformed = torch.where(
                observed,
                scaled,
                torch.full_like(batch_obs, float(diffusion_missing_value)),
            )
        elif observation_fill_mode == "bicubic_horizontal":
            filled = _bicubic_fill_unbatched(
                batch_obs,
                batch_mask,
                missing_value=diffusion_missing_value,
            )
            observed_slots = (
                batch_mask.reshape(batch_mask.shape[0], -1).sum(dim=1) > 0
            )
            if data_scaling == "minus_one_one":
                for time_index in torch.where(observed_slots)[0].tolist():
                    filled[time_index] = 2.0 * filled[time_index] - 1.0
            transformed = filled
        else:
            raise ValueError(
                f"Unsupported observation fill mode: {observation_fill_mode}"
            )
        outputs.append(transformed)
    return torch.stack(outputs) if has_batch else outputs[0]


def diffusion_conditions_to_zero_one(
    condition_lr: torch.Tensor,
    obs: torch.Tensor,
    mask: torch.Tensor,
    *,
    data_scaling: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if data_scaling == "zero_one":
        return condition_lr, obs
    if data_scaling != "minus_one_one":
        raise ValueError(f"Unsupported diffusion data scaling: {data_scaling}")
    return (
        0.5 * (condition_lr + 1.0),
        torch.where(mask > 0, 0.5 * (obs + 1.0), torch.zeros_like(obs)),
    )


def make_diffusion_dataset(
    *,
    base_config: dict[str, Any],
    root_dir: str | Path,
    split: str,
    samples_per_epoch: int | None,
) -> DiffusionWindowDataset:
    dataset_config = copy.deepcopy(base_config)
    data_scaling = str(dataset_config.pop("data_scaling", "zero_one"))
    default_missing_value = (
        float(base_config.get("missing_value", 0.0))
        if data_scaling == "zero_one"
        else -1.1
    )
    diffusion_missing_value = float(
        dataset_config.pop(
            "diffusion_missing_value", default_missing_value
        )
    )
    observation_fill_mode = str(
        dataset_config.pop("observation_fill_mode", "sparse_sentinel")
    )
    dataset_config["missing_value"] = 0.0
    base = make_direct_si_dataset(
        base_config=dataset_config,
        root_dir=root_dir,
        split=split,
        use_mixup=split == "train" and bool(base_config.get("use_mixup", True)),
        samples_per_epoch=samples_per_epoch,
    )
    return DiffusionWindowDataset(
        base,
        data_scaling=data_scaling,
        diffusion_missing_value=diffusion_missing_value,
        observation_fill_mode=observation_fill_mode,
    )


def make_diffusion_loader(
    *,
    dataset: DiffusionWindowDataset,
    split: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, Sampler[int] | None]:
    return make_direct_si_loader(
        dataset=dataset,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )
