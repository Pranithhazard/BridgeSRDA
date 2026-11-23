import pytest
import torch

from src.filo_srda_qg.timing_benchmark_qg.inference_network import InferenceDrift
from src.filo_srda_qg.timing_benchmark_qg.bridge_sampler import sample_bridge_heun
from src.filo_srda_qg.sb_prior_retrain_qg.bridge import ContinuousQuadraticBridge, ContinuousQuadraticBridgeConfig


class TinyConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(20, 9, 1)
        self.seen = []

    def forward(self, state, source, dense, tau):
        self.seen.append(state.dtype)
        return self.conv(torch.cat([state, source, dense], 1)) * 0.01


@pytest.mark.parametrize("bf16", [False, True])
def test_autocast_preserves_fp32_integrator_contract(bf16):
    torch.manual_seed(21)
    net = TinyConv().eval()
    wrapped = InferenceDrift(net, bf16=bf16)
    source = torch.randn(2, 9, 2, 3, 4)
    obs = torch.zeros(2, 3, 1, 2, 3, 4)
    masks = torch.zeros_like(obs)
    masks[:, :2, :, :, ::2, ::2] = 1
    bridge = ContinuousQuadraticBridge(ContinuousQuadraticBridgeConfig(epsilon=0.2581988897471611, noise_exponent=2))
    original = source.clone()
    with torch.inference_mode():
        result, nfe = sample_bridge_heun(model=wrapped, source=source, observations=obs, masks=masks,
                                       bridge=bridge, sampling_seed=42, member_chunk_size=2,
                                       nfe_budget=4)
    assert nfe == 4 and result.dtype == torch.float32 and torch.isfinite(result).all()
    assert all(dtype == torch.float32 for dtype in net.seen)
    assert all(p.dtype == torch.float32 for p in wrapped.parameters())
    assert torch.equal(source, original)
    assert not torch.is_autocast_enabled("cpu")


def test_training_network_is_not_silently_accepted():
    with pytest.raises(ValueError, match="eval"):
        InferenceDrift(TinyConv())
