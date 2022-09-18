from re import I
import pytest

import numpy as np
from scipy import stats
from emulate import CompoundMesh
from emulate import fourier_transform_partial_wave
from emulate import gaussian_radial_fourier_transform
from emulate.utils import (
    yamaguchi_form_factor_momentum_space,
    yamaguchi_form_factor_position_space,
    yamaguchi_scattering_amplitude,
)
from emulate import NewtonEmulator, BoundaryCondition


def test_fourier_transform():
    mesh = CompoundMesh([0, 0.1, 5, 10], [50, 50, 50, 50])
    r, dr = mesh.x, mesh.w
    a = 0.7
    k_grid = np.linspace(0, 10, 101)
    ell = 0

    f_r = np.exp(-a * r**2)
    f_k = fourier_transform_partial_wave(f=f_r, r=r, dr=dr, k=k_grid, ell=ell)
    f_k_exact = gaussian_radial_fourier_transform(x=k_grid, a=a)

    np.testing.assert_allclose(actual=f_k, desired=f_k_exact, atol=1e-10, rtol=1e-1)


def test_reverse_fourier_transform():
    mesh = CompoundMesh([0, 0.1, 5, 10], [50, 50, 50, 50])
    k, dk = mesh.x, mesh.w
    a = 0.7
    r_grid = np.linspace(0, 10, 101)
    ell = 0

    f_r = np.exp(-a * r_grid**2)
    f_k = gaussian_radial_fourier_transform(x=k, a=a)
    f_r_reverse = fourier_transform_partial_wave(f=f_k, r=k, dr=dk, k=r_grid, ell=ell)

    np.testing.assert_allclose(actual=f_r_reverse, desired=f_r, atol=1e-10, rtol=1e-1)


def test_yamaguchi_position_form_factor():
    mesh = CompoundMesh([0, 0.1, 5, 10], [50, 50, 50, 50])
    k, dk = mesh.x, mesh.w
    r, dr = mesh.x, mesh.w
    beta = 1
    # ell must be >= 2 else the numerical Fourier transform is too noisy.
    # Assume since it works for >=2 then the analytic form works for < 2 as well.
    ell = 2

    # The momentum space is simple. Compute it exactly.
    f_k = yamaguchi_form_factor_momentum_space(k=k, beta=beta, ell=ell)
    # Compute its fourier transform numerically with a trusted function
    f_r_numeric = fourier_transform_partial_wave(f=f_k, r=k, dr=dk, k=r, ell=ell)

    # Ensure our analytic form matches the numeric one
    f_r = yamaguchi_form_factor_position_space(r=r, beta=beta, ell=ell)

    np.testing.assert_allclose(actual=f_r, desired=f_r_numeric, atol=1e-10, rtol=1e3)


def test_yamaguchi_scattering_amplitude():
    mesh = CompoundMesh([0, 0.1, 5, 10], [50, 50, 50, 50])
    k, dk = mesh.x, mesh.w
    beta = 2.0
    q_cm = np.array([0.1, 1, 2])
    strength = np.array([3])

    f_k = yamaguchi_form_factor_momentum_space(k=k, beta=beta, ell=0)
    V1 = (f_k[:, None] * f_k)[..., None]
    # Use emulator class as exact Lippmann Schwinger equation solver
    newton = NewtonEmulator(
        V0=np.zeros_like(V1[..., 0]),
        V1=V1,
        k=k,
        dk=dk,
        q_cm=q_cm,
        boundary_condition=BoundaryCondition.STANDING,
        nugget=1e-10,
    )

    # Compute exact K matrix
    K_ls = newton.predict(strength, full_space=True)
    # Compute K with the analytic function we want to test
    K_analytic = yamaguchi_scattering_amplitude(
        q_cm=q_cm, beta=beta, strength=strength, include_q=True
    )
    np.testing.assert_allclose(actual=K_analytic, desired=K_ls, atol=1e-10, rtol=1e-1)


def test_quadrature_infinite():
    mesh = CompoundMesh([0, 2], [50, 50])
    x, dx = mesh.x, mesh.w

    sigma = 5
    half_gaussian = stats.halfnorm(loc=0, scale=sigma).pdf(x)

    numerical_integral = np.sum(half_gaussian * dx)
    np.testing.assert_allclose(numerical_integral, 1.0)


def test_quadrature_finite():
    mesh = CompoundMesh([-1, 1], [50])
    x, dx = mesh.x, mesh.w

    numerical_integral = np.sum(x**2 * dx)
    np.testing.assert_allclose(numerical_integral, 2 / 3)


def test_quadrature_finite2():
    mesh = CompoundMesh([-1, 0, 1], [50, 50])
    x, dx = mesh.x, mesh.w

    numerical_integral = np.sum(x**2 * dx)
    np.testing.assert_allclose(numerical_integral, 2 / 3)
