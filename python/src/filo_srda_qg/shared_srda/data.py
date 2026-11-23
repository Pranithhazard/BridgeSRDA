from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterator

import torch
from src.filo_srda_qg.window_data import (
    WindowDatasetConfig,
    NotsuWindowDataset,
)
from src.four_dim_srda.utils.random_seed_helper import (
    get_torch_generator,
    seed_worker,
)
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler


class DistributedEvalSampler(Sampler[int]):
    """Disjoint deterministic evaluation shards without padding duplicates."""

    def __init__(self, dataset: NotsuWindowDataset, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        return len(range(self.rank, len(self.dataset), self.world_size))


def make_direct_si_dataset(
    *,
    base_config: dict[str, Any],
    root_dir: str | Path,
    split: str,
    use_mixup: bool,
    samples_per_epoch: int | None,
) -> NotsuWindowDataset:
    data = copy.deepcopy(base_config)
    data["split"] = split
    data["use_mixup"] = bool(use_mixup)
    data["samples_per_epoch"] = samples_per_epoch
    data["use_ground_truth_clipping"] = split in {"train", "valid"}
    return NotsuWindowDataset(
        WindowDatasetConfig.from_dict(data),
        root_dir=root_dir,
    )


def make_direct_si_loader(
    *,
    dataset: NotsuWindowDataset,
    split: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, Sampler[int] | None]:
    if split == "train" and world_size > 1:
        sampler: Sampler[int] | None = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
    elif split != "train" and world_size > 1:
        sampler = DistributedEvalSampler(dataset, rank, world_size)
    else:
        sampler = None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train" and sampler is None,
        sampler=sampler,
        drop_last=split == "train",
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=get_torch_generator(seed),
    )
    return loader, sampler

