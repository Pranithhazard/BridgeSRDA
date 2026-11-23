"""Batched stochastic Heun with the canonical 600-step Brownian refinement."""

from __future__ import annotations

import torch

from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.conditioning import (
    build_dense_observation_condition,
)
from src.filo_srda_qg.sb_prior_retrain_qg.bridge import ContinuousQuadraticBridge
from src.filo_srda_qg.sb_prior_retrain_qg.sampler import (
    _exact_increment,
    _trajectory_seed,
)


@torch.no_grad()
def sample_bridge_heun(
    *,
    model: torch.nn.Module,
    source: torch.Tensor,
    observations: torch.Tensor,
    masks: torch.Tensor,
    bridge: ContinuousQuadraticBridge,
    sampling_seed: int,
    member_chunk_size: int = 1,
    global_member_offset: int = 0,
    nfe_budget: int = 20,
    increments: torch.Tensor | None = None,
    interpolation_method: str = "periodic_bilinear",
) -> tuple[torch.Tensor, int]:
    """Return FP32 ``(samples[M,9,Z,Y,X], actual_nfe_per_member)``.

    The default is 10 Heun intervals / NFE20, with two network evaluations per
    interval (batched calls total NFE * ceil(M / chunk)). A contiguous shard
    starting at global member q MUST pass ``global_member_offset=q``. Member i
    uses the evaluator's seed ``_trajectory_seed(sampling_seed, q+i, 0)``, not
    the distinct parent=0, member=i convention.

    Optional ``increments[K,M,9,Z,Y,X]`` are already-scaled additive interval
    increments, not standard normals. They override RNG; shard
    callers supply the corresponding member slice. Densification runs once per
    call, outside the integration loop. The caller owns timing/synchronization
    and must supply a FP32 eval model with autocast disabled.
    """
    if source.ndim != 5 or source.shape[1] != 9 or min(source.shape) < 1:
        raise ValueError("source must be nonempty (M,9,Z,Y,X)")
    if source.dtype != torch.float32 or observations.dtype != torch.float32:
        raise ValueError("source and observations must be FP32")
    if any(t.device != source.device for t in (observations, masks)):
        raise ValueError("source, observations and masks must share a device")
    if model.training or any(m.training for m in model.modules()):
        raise ValueError("model must be in eval mode")
    if any(t.is_floating_point() and t.dtype != torch.float32
           for t in (*model.parameters(), *model.buffers())):
        raise ValueError("model must be FP32")
    if torch.is_autocast_enabled(source.device.type):
        raise ValueError("autocast must be disabled")
    if type(member_chunk_size) is not int or not 1 <= member_chunk_size <= 30:
        raise ValueError("member_chunk_size must be an integer in 1..30")
    if type(global_member_offset) is not int or global_member_offset < 0:
        raise ValueError("global_member_offset must be a nonnegative integer")
    if type(sampling_seed) is not int:
        raise ValueError("sampling_seed must be an integer")
    if type(nfe_budget) is not int or nfe_budget <= 0 or nfe_budget % 2:
        raise ValueError("Heun nfe_budget must be positive and even")
    intervals = nfe_budget // 2
    if 600 % intervals:
        raise ValueError("base600 requires an interval count dividing 600")
    if increments is not None:
        if tuple(increments.shape) != (intervals, *source.shape):
            raise ValueError("increments must have shape (K,M,9,Z,Y,X)")
        if increments.dtype != source.dtype or increments.device != source.device:
            raise ValueError("increments must share source dtype and device")
        if not torch.isfinite(increments).all():
            raise ValueError("increments must be finite")

    dense = build_dense_observation_condition(
        source, observations, masks, method=interpolation_method
    )
    dt = 1.0 / intervals
    outputs = []
    for start in range(0, source.shape[0], member_chunk_size):
        stop = min(start + member_chunk_size, source.shape[0])
        state_source, state_dense = source[start:stop], dense[start:stop]
        state = state_source.clone()
        generators = []
        if increments is None:
            for index in range(start, stop):
                seed = _trajectory_seed(sampling_seed, global_member_offset + index, 0)
                generators.append(torch.Generator(device=source.device).manual_seed(seed))
        for step in range(intervals):
            t0_value, t1_value = step * dt, (step + 1) * dt
            t0 = torch.full((stop - start,), t0_value, device=state.device, dtype=state.dtype)
            t1 = torch.full((stop - start,), t1_value, device=state.device, dtype=state.dtype)
            if increments is not None:
                increment = increments[step, start:stop]
            else:
                increment = _exact_increment(
                    template=state, generators=generators, bridge=bridge,
                    interval_index=step, num_intervals=intervals, base_steps=600,
                )
            drift0 = model(state, state_source, state_dense, t0)
            predictor = state + dt * drift0 + increment
            drift1 = model(predictor, state_source, state_dense, t1)
            state = state + 0.5 * dt * (drift0 + drift1) + increment
        outputs.append(state)
    samples = torch.cat(outputs)
    if not torch.isfinite(samples).all():
        raise ValueError("nonfinite sampled output")
    return samples, 2 * intervals


__all__ = ["sample_bridge_heun"]
