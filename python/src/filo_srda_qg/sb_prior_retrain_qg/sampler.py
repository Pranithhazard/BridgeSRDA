from __future__ import annotations

import dataclasses
import math
import time

import torch

from .bridge import ContinuousQuadraticBridge


@dataclasses.dataclass
class BareHeunResult:
    samples: torch.Tensor
    actual_nfe: int
    num_intervals: int
    elapsed_seconds: float
    peak_gpu_bytes: int


def _trajectory_seed(base_seed: int, parent: int, member: int) -> int:
    return int((base_seed + 1_000_003 * parent + 10_007 * member) % (2**63 - 1))


def _draw_generators(
    template: torch.Tensor, generators: list[torch.Generator]
) -> torch.Tensor:
    return torch.stack(
        [
            torch.randn(
                template.shape[1:],
                device=template.device,
                dtype=template.dtype,
                generator=generator,
            )
            for generator in generators
        ]
    )


def _exact_increment(
    *,
    template: torch.Tensor,
    generators: list[torch.Generator],
    bridge: ContinuousQuadraticBridge,
    interval_index: int,
    num_intervals: int,
    base_steps: int,
) -> torch.Tensor:
    if base_steps % num_intervals:
        raise ValueError("brownian_base_steps must be divisible by interval count")
    refinement = base_steps // num_intervals
    first = interval_index * refinement
    increment = torch.zeros_like(template)
    for local in range(refinement):
        index = first + local
        variance = bridge.interval_variance(index / base_steps, (index + 1) / base_steps)
        increment.add_(_draw_generators(template, generators), alpha=math.sqrt(variance))
    return increment


@torch.no_grad()
def sample_bare_heun_exact_noise(
    *,
    model: torch.nn.Module,
    source: torch.Tensor,
    bridge: ContinuousQuadraticBridge,
    ensemble_size: int,
    nfe_budget: int,
    member_chunk_size: int,
    sampling_seed: int,
    brownian_base_steps: int = 600,
) -> BareHeunResult:
    if source.ndim != 5 or source.shape[1] != 9:
        raise ValueError(f"Expected source (B,9,Z,Y,X), got {tuple(source.shape)}")
    if ensemble_size <= 0 or member_chunk_size <= 0 or nfe_budget <= 0 or nfe_budget % 2:
        raise ValueError("ensemble/chunk must be positive and Heun NFE must be positive/even")
    intervals = nfe_budget // 2
    if brownian_base_steps % intervals:
        raise ValueError("brownian_base_steps must divide Heun interval count")
    if source.is_cuda:
        torch.cuda.reset_peak_memory_stats(source.device)
    started = time.perf_counter()
    total = source.shape[0] * ensemble_size
    outputs: list[torch.Tensor] = []
    dt = 1.0 / intervals
    for flat_start in range(0, total, member_chunk_size):
        flat_indices = list(range(flat_start, min(flat_start + member_chunk_size, total)))
        parents = [index // ensemble_size for index in flat_indices]
        members = [index % ensemble_size for index in flat_indices]
        state_source = source.index_select(
            0, torch.tensor(parents, device=source.device, dtype=torch.long)
        )
        state = state_source.clone()
        generators = []
        for parent, member in zip(parents, members):
            generator = torch.Generator(device=source.device)
            generator.manual_seed(_trajectory_seed(sampling_seed, parent, member))
            generators.append(generator)
        for step in range(intervals):
            t0_value, t1_value = step * dt, (step + 1) * dt
            t0 = torch.full((state.shape[0],), t0_value, device=state.device, dtype=state.dtype)
            t1 = torch.full((state.shape[0],), t1_value, device=state.device, dtype=state.dtype)
            noise = _exact_increment(
                template=state,
                generators=generators,
                bridge=bridge,
                interval_index=step,
                num_intervals=intervals,
                base_steps=brownian_base_steps,
            )
            drift0 = model(state, state_source, t0)
            predictor = state + dt * drift0 + noise
            drift1 = model(predictor, state_source, t1)
            state = state + 0.5 * dt * (drift0 + drift1) + noise
        outputs.append(state)
    samples = torch.cat(outputs).reshape(source.shape[0], ensemble_size, *source.shape[1:])
    return BareHeunResult(
        samples=samples,
        actual_nfe=nfe_budget,
        num_intervals=intervals,
        elapsed_seconds=time.perf_counter() - started,
        peak_gpu_bytes=(
            int(torch.cuda.max_memory_allocated(source.device)) if source.is_cuda else 0
        ),
    )
