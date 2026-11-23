from pathlib import Path

import numpy as np
import torch

from scripts.filo_srda_qg.denseobs_conditional_sb_srda_qg.summarize_sparse_observations import (
    CASES, case_directory, uniform_ssim_layers,
)
from src.four_dim_srda.utils.ssim import MSSIM


def test_cpu_ssim_uses_official_uniform_metric():
    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    truth = rng.uniform(-1, 1, (6, 12, 13)).astype(np.float32)
    pred = truth + rng.normal(0, .1, truth.shape).astype(np.float32)
    x = torch.from_numpy(np.clip((pred + 2.88) / 5.8, 0, 1))[None, None]
    y = torch.from_numpy(np.clip((truth + 2.88) / 5.8, 0, 1))[None, None]
    metric = MSSIM(window_3d_size=(5, 11, 11), sigma_3d=(.7, 1.5, 1.5),
                   value_magnitude=1., use_gaussian=False)
    with torch.no_grad():
        expected = metric(y, x)[0].numpy()
    np.testing.assert_array_equal(uniform_ssim_layers(pred, truth), expected)
    np.testing.assert_array_equal(uniform_ssim_layers(truth, truth), np.ones(6))


def test_case_roots_distinguish_method_and_layout():
    paths = [case_directory(Path("/workspace"), 9991, case) for case in CASES]
    assert len(set(paths)) == 4
    assert "sparse_ogx16_ogy16/conditional_member_wise/seed09991" in str(paths[3])
    assert paths[0].name == "seed09991"
    assert paths[1].name == "seed09991"
    assert paths[2].name == paths[3].name == "ema_ep2000_nfe20_m30"
