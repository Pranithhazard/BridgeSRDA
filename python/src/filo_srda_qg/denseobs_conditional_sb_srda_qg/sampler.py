from __future__ import annotations

import time

import torch

from src.filo_srda_qg.sb_prior_retrain_qg.bridge import ContinuousQuadraticBridge
from src.filo_srda_qg.sb_prior_retrain_qg.sampler import (
    BareHeunResult,
    _exact_increment,
    _trajectory_seed,
)

from .conditioning import build_dense_observation_condition


@torch.no_grad()
def sample_denseobs_conditional_bare_heun_exact_noise(
    *,
    model: torch.nn.Module,
    source: torch.Tensor,
    observations: torch.Tensor,
    masks: torch.Tensor,
    bridge: ContinuousQuadraticBridge,
    ensemble_size: int,
    nfe_budget: int,
    member_chunk_size: int,
    sampling_seed: int,
    brownian_base_steps: int = 600,
    interpolation_method: str = "periodic_bilinear",
) -> BareHeunResult:
    if source.ndim != 5 or source.shape[1] != 9:
        raise ValueError(f"Expected source (B,9,Z,Y,X), got {tuple(source.shape)}")
    if observations.shape != masks.shape or observations.shape[:3] != (
        source.shape[0],
        3,
        1,
    ):
        raise ValueError("observations and masks must have shape (B,3,1,Z,Y,X)")
    if ensemble_size <= 0 or member_chunk_size <= 0 or nfe_budget <= 0 or nfe_budget % 2:
        raise ValueError("ensemble/chunk must be positive and Heun NFE must be positive/even")
    intervals = nfe_budget // 2
    if brownian_base_steps % intervals:
        raise ValueError("brownian_base_steps must divide Heun interval count")

    # Critical 073A contract: densify once, not once per neural evaluation.
    dense_observations = build_dense_observation_condition(
        source, observations, masks, method=interpolation_method
    )
    if source.is_cuda:
        torch.cuda.reset_peak_memory_stats(source.device)
    started = time.perf_counter()
    total = source.shape[0] * ensemble_size
    outputs: list[torch.Tensor] = []
    dt = 1.0 / intervals
    for flat_start in range(0, total, member_chunk_size):
        indices = list(range(flat_start, min(flat_start + member_chunk_size, total)))
        parents = [index // ensemble_size for index in indices]
        members = [index % ensemble_size for index in indices]
        selector = torch.tensor(parents, device=source.device, dtype=torch.long)
        state_source = source.index_select(0, selector)
        state_dense = dense_observations.index_select(0, selector)
        state = state_source.clone()
        generators: list[torch.Generator] = []
        for parent, member in zip(parents, members):
            generator = torch.Generator(device=source.device)
            generator.manual_seed(_trajectory_seed(sampling_seed, parent, member))
            generators.append(generator)
        for step in range(intervals):
            t0_value, t1_value = step * dt, (step + 1) * dt
            t0 = torch.full(
                (state.shape[0],), t0_value, device=state.device, dtype=state.dtype
            )
            t1 = torch.full(
                (state.shape[0],), t1_value, device=state.device, dtype=state.dtype
            )
            increment = _exact_increment(
                template=state,
                generators=generators,
                bridge=bridge,
                interval_index=step,
                num_intervals=intervals,
                base_steps=brownian_base_steps,
            )
            drift0 = model(state, state_source, state_dense, t0)
            predictor = state + dt * drift0 + increment
            drift1 = model(predictor, state_source, state_dense, t1)
            state = state + 0.5 * dt * (drift0 + drift1) + increment
        outputs.append(state)
    samples = torch.cat(outputs).reshape(
        source.shape[0], ensemble_size, *source.shape[1:]
    )
    return BareHeunResult(
        samples=samples,
        actual_nfe=nfe_budget,
        num_intervals=intervals,
        elapsed_seconds=time.perf_counter() - started,
        peak_gpu_bytes=(
            int(torch.cuda.max_memory_allocated(source.device)) if source.is_cuda else 0
        ),
    )


__all__ = ["sample_denseobs_conditional_bare_heun_exact_noise"]
