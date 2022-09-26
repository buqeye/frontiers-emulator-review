import pytest

import numpy as np
from scipy import stats
from scipy.special import spherical_jn

from emulate import CompoundMesh
from emulate import fourier_transform_partial_wave
from emulate import gaussian_radial_fourier_transform
from emulate.utils import (
    yamaguchi_form_factor_momentum_space,
    yamaguchi_form_factor_position_space,
    yamaguchi_radial_wave_function,
    yamaguchi_scattering_amplitude,
    schrodinger_residual,
)
from emulate import (
    NewtonEmulator,
    BoundaryCondition,
    KohnLippmannSchwingerEmulator,
    SeparableKohnEmulator,
)


def test_lippmann_schwinger_solver():
    pass


def test_separable_potential_solver():
    # Rule: The separable Lippmann-Schwinger solver should match the standard solver

    # Given a quadrature mesh
    n_intervals = 21
    nodes = np.linspace(0, 10, n_intervals)
    n_points = 100 * np.ones(n_intervals - 1, dtype=int)
    mesh = CompoundMesh(nodes, n_points)
    k, dk = mesh.x, mesh.w
    r, dr = mesh.x, mesh.w

    # And other parameters
    beta = 2.0
    ell = 0
    q_cm = np.array([0.1, 1, 2])
    strength = np.array([5])
    hbarsq_over_M = 1

    # And a Yamaguchi potential
    f_k = yamaguchi_form_factor_momentum_space(k=k, beta=beta, ell=ell)
    f_r = yamaguchi_form_factor_position_space(r=r, beta=beta, ell=ell)
    V1_k = (f_k[:, None] * f_k)[..., None]  # Add parameter dimension
    V1_r = (f_r[:, None] * f_r)[..., None]

    # When the Lippmann Schwidnger equation is evaluated to solve for the reactance matrix, K
    newton = NewtonEmulator(
        V0=np.zeros_like(V1_k[..., 0]),
        V1=V1_k,
        k=k,
        dk=dk,
        q_cm=q_cm,
        boundary_condition=BoundaryCondition.STANDING,
        nugget=1e-10,
    )

    separable = SeparableKohnEmulator(
        v_r=[f_r],
        r=r,
        dr=dr,
        v_k=[f_k],
        k=k,
        dk=dk,
        q_cm=q_cm,
        inv_mass=hbarsq_over_M,
        ell=ell,
        nugget=1e-7,
    )

    p = strength * np.array([1])
    K_half_schwinger = newton.reactance(
        p=p, include_q=False, shell="half", return_gradient=False
    )
    K_half_separable = separable.compute_half_on_shell_reactance(p, include_q=False)

    # Then the K matrix from the standard LS equation and the separable version should match
    np.testing.assert_allclose(
        actual=K_half_separable, desired=K_half_schwinger, atol=1e-5, rtol=1e-1
    )


def test_newton_emulator():
    pass


def test_kohn_emulator():
    pass
