from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.filo_srda_qg.diffusion_srda_qg.diffusion import (
    DiffusionSampleResult,
    GaussianDiffusion,
)
from src.filo_srda_qg.shared_srda.narrow_jet import (
    normalized_windows_to_physical,
)


def sample_memberwise_in_chunks(
    *,
    diffusion: GaussianDiffusion,
    model: torch.nn.Module,
    condition_lr: torch.Tensor,
    obs: torch.Tensor,
    steps: int,
    eta: float,
    parent_chunk_size: int,
    generator: torch.Generator,
) -> DiffusionSampleResult:
    if parent_chunk_size <= 0:
        raise ValueError("parent_chunk_size must be positive")
    members: list[torch.Tensor] = []
    elapsed = 0.0
    nfe = 0
    for start in range(0, condition_lr.shape[0], parent_chunk_size):
        result = diffusion.sample(
            model,
            condition_lr=condition_lr[start : start + parent_chunk_size],
            obs=obs[start : start + parent_chunk_size],
            ensemble_size=1,
            steps=steps,
            eta=eta,
            chunk_size=1,
            generator=generator,
        )
        members.append(result.samples[:, 0])
        elapsed += result.elapsed_seconds
        nfe = result.nfe
    return DiffusionSampleResult(
        samples=torch.cat(members, dim=0)[:, None],
        nfe=nfe,
        elapsed_seconds=elapsed,
    )


def save_cycle_result(
    *,
    path: str | Path,
    anchor_analysis: torch.Tensor,
    diffusion_mean_analysis: torch.Tensor,
    diffusion_spread_analysis: torch.Tensor,
    selected_members: torch.Tensor | None,
    selected_member_analysis_indices: list[int],
    rank_histogram: torch.Tensor,
    online_metrics: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "anchor_analysis": anchor_analysis.cpu().numpy().astype(np.float32),
        "diffusion_mean_analysis": diffusion_mean_analysis.cpu().numpy().astype(
            np.float32
        ),
        "diffusion_spread_analysis": diffusion_spread_analysis.cpu().numpy().astype(
            np.float32
        ),
        "selected_member_analysis_indices": np.asarray(
            selected_member_analysis_indices, dtype=np.int64
        ),
        "rank_histogram": rank_histogram.cpu().numpy().astype(np.int64),
        "online_metrics": np.array(online_metrics, dtype=object),
        "metadata": np.array(metadata, dtype=object),
    }
    if selected_members is not None:
        arrays["selected_members"] = selected_members.cpu().numpy().astype(np.float32)
    np.savez_compressed(path, **arrays)


__all__ = [
    "normalized_windows_to_physical",
    "sample_memberwise_in_chunks",
    "save_cycle_result",
]
