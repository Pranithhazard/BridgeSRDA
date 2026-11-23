from logging import getLogger

import torch

logger = getLogger()


def get_obs_matrix(
    obs: torch.Tensor, nz: int, ny: int, nx: int, device: str
) -> torch.Tensor:
    assert obs.shape == (nz, ny, nx)
    is_obs = torch.where(torch.isnan(obs), torch.zeros_like(obs), torch.ones_like(obs))

    obs_indices = is_obs.reshape(-1)
    obs_indices = torch.where(obs_indices == 1.0)[0]

    num_obs = len(obs_indices)

    obs_matrix = torch.zeros(num_obs, nz * ny * nx, dtype=torch.float64, device=device)

    for i, j in enumerate(obs_indices):
        obs_matrix[i, j] = 1.0

    p = 100 * torch.sum(obs_matrix).item() / (nz * ny * nx)
    logger.debug(f"observatio prob = {p} [%]")

    return obs_matrix
