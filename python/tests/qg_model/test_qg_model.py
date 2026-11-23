import numpy as np
import pytest
import torch
from src.qg_model.qg_model import QGModel
from src.qg_model.utils.config import Config


def test_initialize_without_pv_noise():
    nx, ny, nz = 128, 65, 16
    kx, ky, kz = 2, 3, 0.5
    hz = 2 * torch.pi

    _xs = torch.linspace(0, 2 * torch.pi, nx + 1, dtype=torch.float64)[:-1]
    _ys = torch.linspace(0, torch.pi, ny, dtype=torch.float64)
    _zs = torch.linspace(0, hz, nz, dtype=torch.float64)
    zs, ys, xs = torch.meshgrid(_zs, _ys, _xs, indexing="ij")

    expected_ave_u = torch.ones(size=[nz, 1, 1]) * zs[:, 0:1, 0:1] / hz

    # psi = torch.sin(kx * xs) * torch.sin(ky * ys) * torch.sin(kz * zs)
    expected_total_u = (
        -ky * torch.sin(kx * xs) * torch.cos(ky * ys) * torch.sin(kz * zs)
        + expected_ave_u
    )
    expected_total_v = kx * torch.cos(kx * xs) * torch.sin(ky * ys) * torch.sin(kz * zs)

    config = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "f_0": 1.0,
        "beta": 0.0,
        "Lx": 2 * torch.pi,
        "Ly": torch.pi,
        "Lz": hz,
        "reduced_gravity": float(nz) / hz,
        "diffusion_coefficient": 1e-4,
        "diffusion_exponent": 2,
        "device": "cpu",
        "precision": "double",
    }

    config = Config(**config)
    model = QGModel(config)
    model.initialize(expected_total_u)

    actual_total_u, actual_total_v = model.get_total_grid_velocity()
    actual_ave_u = model.horizontally_averaged_u.detach().cpu()

    assert expected_ave_u.shape == actual_ave_u.shape == (nz, 1, 1)
    assert torch.max(torch.abs(expected_ave_u - actual_ave_u)).item() < 1e-13

    assert expected_total_u.shape == actual_total_u.shape == (nz, ny, nx)
    assert torch.max(torch.abs(expected_total_u - actual_total_u)).item() < 1e-13

    assert expected_total_v.shape == actual_total_v.shape == (nz, ny, nx)
    assert torch.max(torch.abs(actual_total_v - expected_total_v)) < 1e-13


def test_initialize_with_pv_noise():
    nx, ny, nz = 128, 65, 16
    kx, ky, kz = 1, 1, 0.5
    hz = 2 * torch.pi

    _xs = torch.linspace(0, 2 * torch.pi, nx + 1, dtype=torch.float64)[:-1]
    _ys = torch.linspace(0, torch.pi, ny, dtype=torch.float64)
    _zs = torch.linspace(0, hz, nz, dtype=torch.float64)
    zs, ys, xs = torch.meshgrid(_zs, _ys, _xs, indexing="ij")

    # psi = torch.sin(kx * xs) * torch.sin(ky * ys) * torch.sin(kz * zs)
    total_u = -ky * torch.sin(kx * xs) * torch.cos(ky * ys) * torch.sin(kz * zs)

    noise = 1 * torch.sin(5 * kx * xs) * torch.sin(3 * ky * ys)
    expected_total_pv = (
        -(kx**2 + ky**2 + kz**2)
        * torch.sin(kx * xs)
        * torch.sin(ky * ys)
        * torch.sin(kz * zs)
        + noise
    )

    config = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "f_0": 1.0,
        "beta": 0.0,
        "Lx": 2 * torch.pi,
        "Ly": torch.pi,
        "Lz": hz,
        "reduced_gravity": float(nz) / hz,
        "diffusion_coefficient": 1e-4,
        "diffusion_exponent": 2,
        "device": "cpu",
        "precision": "double",
    }

    config = Config(**config)
    model = QGModel(config)
    model.initialize(total_u, grid_pv_noise=noise)

    actual_total_pv = model.get_total_grid_pv()

    assert torch.max(torch.abs(actual_total_pv - expected_total_pv)).item() < 0.25


def test_get_grid_velocity():
    nx, ny, nz = 128, 65, 16
    kx, ky, kz = 4, 3, 0.5
    hz = 2 * torch.pi

    _xs = torch.linspace(0, 2 * torch.pi, nx + 1, dtype=torch.float64)[:-1]
    _ys = torch.linspace(0, torch.pi, ny, dtype=torch.float64)
    _zs = torch.linspace(0, hz, nz, dtype=torch.float64)
    zs, ys, xs = torch.meshgrid(_zs, _ys, _xs, indexing="ij")

    # potential vorticity
    psi = torch.sin(kx * xs) * torch.sin(ky * ys) * torch.sin(kz * zs)
    q = -(kx**2 + ky**2 + kz**2) * psi
    expected_u = -ky * torch.sin(kx * xs) * torch.cos(ky * ys) * torch.sin(kz * zs)
    expected_v = kx * torch.cos(kx * xs) * torch.sin(ky * ys) * torch.sin(kz * zs)

    config = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "f_0": 1.0,
        "beta": 0.0,
        "Lx": 2 * torch.pi,
        "Ly": torch.pi,
        "Lz": hz,
        "reduced_gravity": float(nz) / hz,
        "diffusion_coefficient": 1e-4,
        "diffusion_exponent": 2,
        "device": "cpu",
        "precision": "double",
    }

    model = QGModel(Config(**config))
    model.initialize_pv(q)

    actual_u, actual_v = model.get_grid_velocity()

    # ky and kx are the magnitude of u and v, respectively.
    assert torch.max(torch.abs(actual_u - expected_u)).item() / ky < 1e-2
    assert torch.max(torch.abs(actual_v - expected_v)).item() / kx < 1e-2


def test_integrate_n_steps_single_simulation():
    nx, ny, nz = 64, 33, 16
    hx, hy, hz = 2 * torch.pi, torch.pi, torch.pi
    beta, kx, ky = 1.0, 2.0, 1.0

    config = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "f_0": 1.0 / (hz / nz),
        "beta": beta,
        "Lx": hx,
        "Ly": hy,
        "Lz": hz,
        "reduced_gravity": nz / hz,
        "diffusion_coefficient": 1e-10,
        "diffusion_exponent": 3,
        "device": "cpu",
        "precision": "double",
    }

    model = QGModel(Config(**config))

    _xs = torch.linspace(0, 2 * torch.pi, nx + 1, dtype=torch.float64)[:-1]
    _ys = torch.linspace(0, torch.pi, ny, dtype=torch.float64)
    _zs = torch.linspace(0, 2 * torch.pi, nz, dtype=torch.float64)
    zs, ys, xs = torch.meshgrid(_zs, _ys, _xs, indexing="ij")

    # Rossby wave is an analytic solution for QG model.
    # Numerical solution is compared to it.

    omega = -kx * beta / (kx**2 + ky**2)
    # period (2\pi/omega) is about 15

    def get_qs(t: float):
        # analytic solution.
        q = torch.sin(kx * xs - omega * t) * torch.sin(ky * ys)
        return q.to(torch.float64)

    model.initialize_pv(get_qs(t=0.0))
    dt = 0.05

    for i in range(16):
        actual_qs = model.get_grid_pv()
        expected_qs = get_qs(t=model.t)
        assert torch.max(torch.abs(actual_qs - expected_qs)).item() < 1e-3

        model.integrate_n_steps(dt_per_step=dt, n_steps=20)


def test_integrate_n_steps_batch_simulation():
    nx, ny, nz = 64, 33, 16
    hx, hy, hz = 2 * torch.pi, torch.pi, torch.pi
    beta, kx, ky = 1.0, 2.0, 1.0

    config = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "f_0": 1.0 / (hz / nz),
        "beta": beta,
        "Lx": hx,
        "Ly": hy,
        "Lz": hz,
        "reduced_gravity": nz / hz,
        "diffusion_coefficient": 1e-10,
        "diffusion_exponent": 3,
        "device": "cpu",
        "precision": "double",
    }

    model = QGModel(Config(**config))

    _xs = torch.linspace(0, 2 * torch.pi, nx + 1, dtype=torch.float64)[:-1]
    _ys = torch.linspace(0, torch.pi, ny, dtype=torch.float64)
    _zs = torch.linspace(0, 2 * torch.pi, nz, dtype=torch.float64)
    zs, ys, xs = torch.meshgrid(_zs, _ys, _xs, indexing="ij")

    # Rossby wave is an analytic solution for QG model.
    # Numerical solution is compared to it.

    omega = -kx * beta / (kx**2 + ky**2)
    # period (2\pi/omega) is about 15

    # Here, 3 different simulations are performed.
    phases = torch.Tensor([0, np.sqrt(2), np.sqrt(5)])
    phases = torch.broadcast_to(phases[:, None, None, None], size=(3, nz, ny, nx))

    def get_qs(t: float):
        # analytic solution.
        q = torch.sin(kx * xs - omega * t + phases) * torch.sin(ky * ys)
        return q.to(torch.float64)

    model.initialize_pv(get_qs(t=0.0))
    dt = 0.05

    for i in range(16):
        actual_qs = model.get_grid_pv()
        expected_qs = get_qs(t=model.t)
        assert torch.max(torch.abs(actual_qs - expected_qs)).item() < 1e-3

        model.integrate_n_steps(dt_per_step=dt, n_steps=20)


def test__pv_inversion():
    nx, ny, nz = 128, 65, 16
    kx, ky, kz = 4, 2, 0.5
    hx = 2 * torch.pi
    hy = 1 * torch.pi
    hz = 2 * torch.pi

    _xs = torch.linspace(0, hx, nx + 1, dtype=torch.float64)[:-1]
    _ys = torch.linspace(0, hy, ny, dtype=torch.float64)
    _zs = torch.linspace(0, hz, nz, dtype=torch.float64)
    zs, ys, xs = torch.meshgrid(_zs, _ys, _xs, indexing="ij")

    # potential vorticity
    qs = torch.sin(kx * xs) * torch.sin(ky * ys) * torch.sin(kz * zs)

    # stream function
    expected = -qs / (kx**2 + ky**2 + kz**2)

    config = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "f_0": 1.0,
        "beta": 0.0,
        "Lx": hx,
        "Ly": hy,
        "Lz": hz,
        "reduced_gravity": float(nz) / hz,
        "diffusion_coefficient": 1e-4,
        "diffusion_exponent": 2,
        "device": "cpu",
        "precision": "double",
    }

    model = QGModel(Config(**config))

    spec_qs = model.fft.apply_fft2(qs, is_odd=True)
    actual = model.fft.apply_ifft2(model._pv_inversion(spec_qs))

    assert torch.max(torch.abs(actual - expected)).item() < 1e-3


def test__jacobian():
    nx, ny, nz = 128, 65, 32
    hx = 2 * torch.pi
    hy = 1 * torch.pi
    hz = 2 * torch.pi

    k1x, k1y, k1z = 4, 2, 0.5
    k2x, k2y, k2z = 5, 3, 1.0

    _xs = torch.linspace(0, hx, nx + 1, dtype=torch.float64)[:-1]
    _ys = torch.linspace(0, hy, ny, dtype=torch.float64)
    _zs = torch.linspace(0, hz, nz, dtype=torch.float64)
    zs, ys, xs = torch.meshgrid(_zs, _ys, _xs, indexing="ij")

    f = torch.sin(k1x * xs) * torch.sin(k1y * ys) * torch.sin(k1z * zs)
    g = torch.sin(k2x * xs) * torch.sin(k2y * ys) * torch.sin(k2z * zs)

    f_x = k1x * torch.cos(k1x * xs) * torch.sin(k1y * ys) * torch.sin(k1z * zs)
    f_y = k1y * torch.sin(k1x * xs) * torch.cos(k1y * ys) * torch.sin(k1z * zs)
    g_x = k2x * torch.cos(k2x * xs) * torch.sin(k2y * ys) * torch.sin(k2z * zs)
    g_y = k2y * torch.sin(k2x * xs) * torch.cos(k2y * ys) * torch.sin(k2z * zs)

    # case of beta = 0
    beta = 0.0
    config = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "f_0": 1.0,
        "beta": beta,
        "Lx": hx,
        "Ly": hy,
        "Lz": hz,
        "reduced_gravity": float(nz) / hz,
        "diffusion_coefficient": 1e-4,
        "diffusion_exponent": 2,
        "device": "cpu",
        "precision": "double",
    }

    model = QGModel(Config(**config))

    # set horizontal_averaged_u and background_pv_y manually, because initializer is not called here.
    model.horizontally_averaged_u = torch.full(
        size=[model.nz, 1, 1],
        fill_value=0.0,
        dtype=model.real_dtype,
        device=model.device,
    )
    model.background_pv_y = beta

    spec_f = model.fft.apply_fft2(f, is_odd=True)
    spec_g = model.fft.apply_fft2(g, is_odd=True)

    actual = model._jacobian(spec_f, spec_g)
    expected = f_x * g_y - f_y * g_x

    assert torch.max(torch.abs(expected - actual)).item() < 1e-12

    # case of beta /= 0
    beta = 1.0
    config = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "f_0": 1.0,
        "beta": beta,
        "Lx": hx,
        "Ly": hy,
        "Lz": hz,
        "reduced_gravity": float(nz) / hz,
        "diffusion_coefficient": 1e-4,
        "diffusion_exponent": 2,
        "device": "cpu",
        "precision": "double",
    }

    model = QGModel(Config(**config))

    # set horizontal_averaged_u and background_pv_y manually, because initializer is not called here.
    model.horizontally_averaged_u = torch.full(
        size=[model.nz, 1, 1],
        fill_value=0.0,
        dtype=model.real_dtype,
        device=model.device,
    )
    model.background_pv_y = beta

    # Jacobian is anti-symmetric, so if the same functions are substituted, jacobian gets zero.
    # Then, only the beta effect term remains.
    expected = f_x * beta
    actual = model._jacobian(spec_f, spec_f)

    assert torch.max(torch.abs(expected - actual)).item() < 1e-13


def test__do_diffusion():
    nx, ny, nz = 128, 65, 16

    config = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "f_0": 1.0,
        "beta": 1.0,
        "Lx": 2 * torch.pi,
        "Ly": torch.pi,
        "Lz": float(nz),
        "reduced_gravity": 1.0,
        "diffusion_coefficient": 1.0,
        "diffusion_exponent": 1,
        "device": "cpu",
        "precision": "double",
    }

    model = QGModel(Config(**config))

    x0 = torch.tensor([3.0], dtype=torch.float64)
    friction = torch.tensor([-1.0], dtype=torch.float64)

    model.diffusion_operator = friction
    x = x0
    dt = 0.001
    n = 10000

    for _ in range(n):
        x = model._do_diffusion(dt, x)
    actual = x

    expected = torch.exp(friction * n * dt) * x0

    assert abs(actual.item() - expected.item()) < 1e-16


def test__calc_background_pv_derivative_with_respect_to_y():
    nx, ny, nz = 64, 33, 4
    beta = 2.0

    config = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "f_0": 1.0,
        "beta": beta,
        "Lx": 2 * torch.pi,
        "Ly": torch.pi,
        "Lz": float(nz),
        "reduced_gravity": 1.0,
        "diffusion_coefficient": 1e-10,
        "diffusion_exponent": 3,
        "device": "cpu",
        "precision": "double",
    }

    config = Config(**config)
    model = QGModel(config)

    actual_laplacian_z = model.laplacian_z.numpy().transpose().tolist()

    expected_laplacian_z = [
        [-1.0, 1.0, 0.0, 0.0],
        [1.0, -2.0, 1.0, 0.0],
        [0.0, 1.0, -2.0, 1.0],
        [0.0, 0.0, 1.0, -1.0],
    ]

    assert actual_laplacian_z == expected_laplacian_z

    grids = model.get_grids()
    zs = grids.z
    u = zs**2

    model.initialize(u)

    actual = model.background_pv_y.numpy().tolist()

    # u_zz = [ 2.,  2.,  2., -6.]
    # q_y = beta - u_zz

    expected = (
        (beta - torch.tensor([2.0, 2.0, 2.0, -6.0]))[:, None, None].numpy().tolist()
    )

    assert actual == expected
