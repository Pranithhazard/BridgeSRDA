from __future__ import annotations

import dataclasses
import time
from typing import Any

import torch
import torch.nn.functional as F


@dataclasses.dataclass(frozen=True)
class DiffusionConfig:
    num_train_timesteps: int = 1000
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2
    loss_type: str = "l1"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiffusionConfig":
        return cls(**data)


@dataclasses.dataclass(frozen=True)
class DiffusionSampleResult:
    samples: torch.Tensor
    nfe: int
    elapsed_seconds: float


def _wide(values: torch.Tensor, ndim: int) -> torch.Tensor:
    return values.view(values.shape[0], *([1] * (ndim - 1)))


def _repeat_members(values: torch.Tensor, members: int) -> torch.Tensor:
    expanded = values[:, None].expand(values.shape[0], members, *values.shape[1:])
    return expanded.reshape(values.shape[0] * members, *values.shape[1:])


class GaussianDiffusion(torch.nn.Module):
    """DiffSRDA-compatible VP diffusion with continuous noise-level training."""

    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive")
        betas = torch.linspace(
            cfg.beta_start,
            cfg.beta_end,
            cfg.num_train_timesteps,
            dtype=torch.float64,
        ).float()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        gamma = torch.sqrt(alphas_cumprod)
        gamma_boundaries = torch.cat([torch.ones(1), gamma])
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("gamma", gamma)
        self.register_buffer("gamma_boundaries", gamma_boundaries)

    @property
    def num_timesteps(self) -> int:
        return int(self.cfg.num_train_timesteps)

    def sample_noise_level(
        self,
        batch_size: int,
        *,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        # DiffSRDA selects one discrete interval for the minibatch, then
        # samples an independent continuous gamma for each batch element.
        interval = torch.randint(
            1,
            self.num_timesteps + 1,
            (1,),
            device=device,
            generator=generator,
        )
        upper = self.gamma_boundaries[interval - 1]
        lower = self.gamma_boundaries[interval]
        weight = torch.rand(
            batch_size, device=device, generator=generator, dtype=upper.dtype
        )
        return lower + weight * (upper - lower)

    @staticmethod
    def q_sample(
        target: torch.Tensor,
        noise_level: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        if target.shape != noise.shape:
            raise ValueError("target and noise shapes must match")
        gamma = _wide(noise_level.to(target.dtype), target.ndim)
        return gamma * target + torch.sqrt(
            torch.clamp(1.0 - gamma.square(), min=0.0)
        ) * noise

    @staticmethod
    def predict_x0(
        noisy: torch.Tensor,
        noise_level: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        gamma = _wide(noise_level.to(noisy.dtype), noisy.ndim)
        sigma = torch.sqrt(torch.clamp(1.0 - gamma.square(), min=0.0))
        return (noisy - sigma * predicted_noise) / gamma.clamp_min(1.0e-8)

    def training_loss(
        self,
        model: torch.nn.Module,
        *,
        target: torch.Tensor,
        condition_lr: torch.Tensor,
        obs: torch.Tensor,
        generator: torch.Generator | None = None,
        noise_level: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = target.shape[0]
        if noise_level is None:
            noise_level = self.sample_noise_level(
                batch_size, device=target.device, generator=generator
            )
        if noise is None:
            noise = torch.randn(
                target.shape,
                device=target.device,
                dtype=target.dtype,
                generator=generator,
            )
        noisy = self.q_sample(target, noise_level, noise)
        prediction = model(noisy, condition_lr, obs, noise_level)
        if self.cfg.loss_type == "l1":
            return F.l1_loss(prediction, noise)
        if self.cfg.loss_type == "mse":
            return F.mse_loss(prediction, noise)
        raise ValueError(f"Unsupported diffusion loss: {self.cfg.loss_type}")

    def respaced_indices(self, steps: int) -> torch.Tensor:
        if not 1 <= steps <= self.num_timesteps:
            raise ValueError("steps must be in [1, num_train_timesteps]")
        indices = torch.linspace(
            0, self.num_timesteps - 1, steps, dtype=torch.float64
        ).round().long()
        return torch.unique_consecutive(indices)

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        *,
        condition_lr: torch.Tensor,
        obs: torch.Tensor,
        ensemble_size: int,
        steps: int,
        eta: float,
        chunk_size: int,
        generator: torch.Generator | None = None,
    ) -> DiffusionSampleResult:
        if ensemble_size <= 0 or chunk_size <= 0 or not 0.0 <= eta <= 1.0:
            raise ValueError("invalid ensemble_size, chunk_size, or eta")
        device = condition_lr.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        indices = self.respaced_indices(steps).to(device)
        batch_size = condition_lr.shape[0]
        chunks: list[torch.Tensor] = []
        state_shape = (
            batch_size,
            9,
            int(obs.shape[-3]),
            int(obs.shape[-2]),
            int(obs.shape[-1]),
        )
        for member_start in range(0, ensemble_size, chunk_size):
            members = min(chunk_size, ensemble_size - member_start)
            repeated_lr = _repeat_members(condition_lr, members)
            repeated_obs = _repeat_members(obs, members)
            shape = (batch_size * members, *state_shape[1:])
            state = torch.randn(
                shape,
                device=device,
                dtype=condition_lr.dtype,
                generator=generator,
            )
            for position in range(len(indices) - 1, -1, -1):
                index = indices[position]
                alpha = self.alphas_cumprod[index].to(state.dtype)
                gamma = torch.sqrt(alpha).expand(state.shape[0])
                predicted_noise = model(state, repeated_lr, repeated_obs, gamma)
                x0 = self.predict_x0(state, gamma, predicted_noise)
                if position == 0:
                    state = x0
                    continue
                previous_index = indices[position - 1]
                alpha_previous = self.alphas_cumprod[previous_index].to(state.dtype)
                sigma = float(eta) * torch.sqrt(
                    torch.clamp(
                        (1.0 - alpha_previous)
                        / (1.0 - alpha)
                        * (1.0 - alpha / alpha_previous),
                        min=0.0,
                    )
                )
                direction = torch.sqrt(
                    torch.clamp(1.0 - alpha_previous - sigma.square(), min=0.0)
                ) * predicted_noise
                reverse_noise = (
                    torch.randn(
                        state.shape,
                        device=device,
                        dtype=state.dtype,
                        generator=generator,
                    )
                    if eta > 0.0
                    else torch.zeros_like(state)
                )
                state = (
                    torch.sqrt(alpha_previous) * x0
                    + direction
                    + sigma * reverse_noise
                )
            chunks.append(
                state.reshape(batch_size, members, *state.shape[1:]).cpu()
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return DiffusionSampleResult(
            samples=torch.cat(chunks, dim=1),
            nfe=len(indices),
            elapsed_seconds=time.perf_counter() - started,
        )
