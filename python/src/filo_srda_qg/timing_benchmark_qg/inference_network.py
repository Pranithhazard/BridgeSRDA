"""Isolated network execution policies; stochastic integration stays FP32."""
from __future__ import annotations

import torch
from torch import nn


class InferenceDrift(nn.Module):
    def __init__(self, model: nn.Module, *, bf16=False, compile_network=False):
        super().__init__()
        if model.training:
            raise ValueError("Inference model must already be in eval mode")
        if any(p.dtype != torch.float32 for p in model.parameters() if p.is_floating_point()):
            raise ValueError("Keep network parameters FP32; use autocast for eligible operations")
        self.bf16 = bf16
        # Appendix F uses a static full graph with CUDA graph replay disabled.
        if compile_network:
            model = torch.compile(model, fullgraph=True, dynamic=False,
                                  options={"triton.cudagraphs": False})
        self.network = model
        self.eval()

    def forward(self, state, source, dense, tau):
        with torch.autocast(device_type=state.device.type, dtype=torch.bfloat16, enabled=self.bf16):
            drift = self.network(state, source, dense, tau)
        # No reduced-precision transport state, Brownian noise or Heun updates.
        return drift.float()
