import pytest

import numpy as np
from numpy.random import default_rng
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


def create_yamaguchi_kohn_emulator(
    nodes,
    n_points,
    betas,
    ell,
    q_cm,
    hbar2_over_2mu=1,
    nugget=0,
    use_lagrange_multiplier=False,
    is_separable=True,
):
    R"""A helper function to reduce code duplication throughout the tests."""

    # Given a quadrature mesh
    mesh = CompoundMesh(nodes, n_points)
    k, dk = mesh.x, mesh.w
    r, dr = mesh.x, mesh.w

    # And a Yamaguchi potential of rank len(betas)
    f_k = np.array(
        [
            yamaguchi_form_factor_momentum_space(
                k=k, beta=beta, ell=ell, hbar2_over_2mu=hbar2_over_2mu
            )
            for beta in betas
        ]
    )
    f_r = np.array(
        [
            yamaguchi_form_factor_position_space(
                r=r, beta=beta, ell=ell, hbar2_over_2mu=hbar2_over_2mu
            )
            for beta in betas
        ]
    )

    # When a Kohn emulator is created
    if is_separable:
        kohn = SeparableKohnEmulator(
            v_r=f_r,
            r=r,
            dr=dr,
            v_k=f_k,
            k=k,
            dk=dk,
            q_cm=q_cm,
            # inv_mass=hbarsq_over_M,
            ell=ell,
            nugget=nugget,
            use_lagrange_multiplier=use_lagrange_multiplier,
        )
    else:
        V1_k = (f_k[:, None] * f_k)[..., None]  # Add parameter dimension
        V1_r = (f_r[:, None] * f_r)[..., None]
        newton = NewtonEmulator(
            V0=np.zeros_like(V1_k[..., 0]),
            V1=V1_k,
            k=k,
            dk=dk,
            q_cm=q_cm,
            boundary_condition=BoundaryCondition.STANDING,
            nugget=nugget,
        )

        kohn = KohnLippmannSchwingerEmulator(
            V0=np.zeros_like(V1_r[..., 0]),
            V1=V1_r,
            r=r,
            dr=dr,
            NVP=newton,
            # inv_mass=hbarsq_over_M,
            ell=ell,
        )

    # Then something about the emulator will be tested...
    return kohn


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
    hbar2_over_2mu = 1

    # And a Yamaguchi potential
    f_k = yamaguchi_form_factor_momentum_space(
        k=k, beta=beta, ell=ell, hbar2_over_2mu=hbar2_over_2mu
    )
    f_r = yamaguchi_form_factor_position_space(
        r=r, beta=beta, ell=ell, hbar2_over_2mu=hbar2_over_2mu
    )
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
        # inv_mass=hbarsq_over_M,
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


@pytest.mark.parametrize(
    "betas, ell, q_cm, n_train, seed",
    [
        ([2.0, 3.0], 0, [0.1, 1, 2], 4, 1),
        ([2.0, 3.0], 0, [0.1, 1, 2], 3, 2),
        ([2.0, 3.0], 0, [0.1, 1, 2], 2, 3),
        ([2.0, 3.0], 0, [0.1, 1, 2], 1, 4),
    ],
)
def test_kohn_coefficients_sum_to_one(betas, ell, q_cm, n_train, seed):
    # Rule: The coefficients from the Kohn emulator should sum to one

    # Given a quadrature mesh (it can be coarse for this test)
    n_intervals = 5
    nodes = np.linspace(0, 10, n_intervals)
    n_points = 20 * np.ones(n_intervals, dtype=int)

    # And other parameters
    hbar2_over_2mu = 1
    nugget = 0
    q_cm = np.array(q_cm)

    # When a Kohn emulator is created
    kohn = create_yamaguchi_kohn_emulator(
        nodes,
        n_points,
        betas,
        ell,
        q_cm,
        hbar2_over_2mu=hbar2_over_2mu,
        nugget=nugget,
        use_lagrange_multiplier=False,
        is_separable=True,
    )

    # And a random set of training parameters is generated to fit the emulator
    n_parameters = int(len(betas) * (len(betas) + 1) / 2)
    params = default_rng(seed).uniform(-10, 10, size=(n_train, n_parameters))
    kohn.fit(params)

    # And the coefficients are extracted for a set of test parameters
    param_test = np.array([1, 7, -1])
    coeff = kohn.coefficients(param_test)

    # And the coefficients are summed
    coeff_sum = coeff.sum(-1)

    # Then the summed coefficients should be equal to one
    np.testing.assert_allclose(coeff_sum, np.ones_like(coeff_sum))


@pytest.mark.parametrize(
    "betas, ell, q_cm, n_train, seed",
    [
        ([2.0, 3.0], 0, [0.1, 1, 2], 4, 1),
        ([2.0, 3.0], 0, [0.1, 1, 2], 3, 2),
        ([2.0, 3.0], 0, [0.1, 1, 2], 2, 3),
        ([2.0, 3.0], 0, [0.1, 1, 2], 1, 4),
    ],
)
def test_kohn_coefficients_methods_match(betas, ell, q_cm, n_train, seed):
    # Rule: The coefficients found via the explicit Lagrange multiplier method
    # should match the coefficients from the Lagrange-multiplier-free method

    # Given a quadrature mesh (it can be coarse for this test)
    n_intervals = 5
    nodes = np.linspace(0, 10, n_intervals)
    n_points = 20 * np.ones(n_intervals, dtype=int)

    # And other parameters
    hbar2_over_2mu = 1
    nugget = 0
    q_cm = np.array(q_cm)

    # When a Kohn emulator is created
    kohn = create_yamaguchi_kohn_emulator(
        nodes,
        n_points,
        betas,
        ell,
        q_cm,
        hbar2_over_2mu=hbar2_over_2mu,
        nugget=nugget,
        use_lagrange_multiplier=False,
        is_separable=True,
    )

    # And a random set of training parameters is generated to fit the emulator
    n_parameters = int(len(betas) * (len(betas) + 1) / 2)
    params = default_rng(seed).uniform(-10, 10, size=(n_train, n_parameters))
    kohn.fit(params)

    # And the coefficients are extracted for a set of test parameters
    param_test = np.array([1, 7, -1])
    coeff_lagrange = kohn.coefficients_and_multiplier(param_test)[:, :-1]
    coeff_no_lagrange = kohn.coefficients_without_multiplier(param_test)

    # Then the coefficients from each type should match
    np.testing.assert_allclose(coeff_no_lagrange, coeff_lagrange)
