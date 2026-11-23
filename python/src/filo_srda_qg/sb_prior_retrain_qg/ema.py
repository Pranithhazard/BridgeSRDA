from __future__ import annotations

import copy

import torch


class EMA:
    """FP32 exponential moving average over floating model state."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = float(decay)
        if not 0.0 < self.decay < 1.0:
            raise ValueError("EMA decay must lie in (0,1)")
        self.shadow = {
            key: value.detach().float().clone()
            for key, value in model.state_dict().items()
            if torch.is_floating_point(value)
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for key, shadow in self.shadow.items():
            shadow.mul_(self.decay).add_(state[key].detach().float(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "shadow": copy.deepcopy(self.shadow)}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.decay = float(state["decay"])
        self.shadow = {
            str(key): value.detach().float().clone()
            for key, value in dict(state["shadow"]).items()
        }

    def copy_to(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for key, shadow in self.shadow.items():
            state[key].copy_(shadow.to(device=state[key].device, dtype=state[key].dtype))
        model.load_state_dict(state)

    def model_copy(self, model: torch.nn.Module) -> torch.nn.Module:
        copied = copy.deepcopy(model)
        self.copy_to(copied)
        return copied
