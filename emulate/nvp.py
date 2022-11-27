from __future__ import annotations

import numpy as np
from itertools import product
from typing import Dict

from .constants import pi
from .utils import (
    cubic_spline_matrix,
    BoundaryCondition,
    greens_function_free_space,
    greens_function_free_space_squared,
    t_matrix_incoming_to_standing,
    t_matrix_outgoing_to_standing,
    fix_phases_continuity,
)
from .separable import SeparableYamaguchiMixin


class NewtonEmulator:
    r"""A class that can either simulate or emulate two-body scattering observables via the reactance matrix and the Newton variational principle.

    Depending on the context, the reactance matrix is denoted by K or R.

    Parameters
    ----------
    V0 : ArrayLike, shape = (n_k, n_k)
        The piece of the potential that does not depend of parameters, in units of fermi. This may require
        multiplying the standard momentum space potential (which is in MeV fm^3) by the 2 times the reduced
        mass of the system: 2 * mu / hbar**2.
    V1 : ArrayLike, shape = (n_k, n_k, n_p)
        The piece of the potential that is linear in the parameters p. When multiplied by p, then this
        is expected to be in units of fermi.
    k :
        The momentum mesh, likely created using some quadrature rules, in units of inverse fermi.
    dk :
        The integration measure, likely created using some quadrature rules, in units of inverse fermi.
    t_lab :
        The on-shell energies of interest for computing observables, in units of MeV.
    system :
        The system of particles involved in the collision: 'pp', 'np', 'nn', 'p-alpha',
        or an instance of the Isospin class.
    dwa_wfs :
        The wave functions used for the distorted wave approach.
        Must include 'f', 'g', 'df', 'dg', 'f0', 'g0', 'df0', and 'dg0'. The g's may have an additional negative
        sign compared to some conventions.
    """

    def __init__(
        self,
        V0,
        V1,
        k,
        dk,
        # t_lab,
        q_cm,
        # system: Union[str, ScatteringSystem],
        boundary_condition: BoundaryCondition,
        dwa_wfs: Dict = None,
        is_coupled=False,
        nugget=0,
        is_mesh_semi_infinite=True,
    ):
        # Mass and isospin info
        # system = ScatteringSystem(system)
        # mu = system.reduced_mass
        # inv_mass = hbar_c ** 2 / (2 * mu)

        # Momentum info
        # q_cm = t_lab_to_q_cm(t_lab, isospin)
        # q_cm = system.t_lab_to_q_cm(t_lab=t_lab)
        n_k = len(k)
        n_q = len(q_cm)

        Id = np.identity(n_k, float)

        # In Landau's QM text, it is recommended to create an (n_k+1, n_k+1) matrix where the
        # extra element holds the on-shell part. This works fine, but is annoying because a new
        # matrix must be created for every on-shell piece you want to compute. Instead, we will
        # only create one set of (n_k, n_k) matrices, then upon solving the LS equation for the
        # off-shell reactance matrix, we will interpolate to the on-shell piece via this spline
        # matrix, which is only computed once and stored.
        Sp = cubic_spline_matrix(k, q_cm)

        is_G0_compressed = True
        k_cut = None
        if is_mesh_semi_infinite:
            k_cut = np.inf
        G0 = greens_function_free_space(
            k=k,
            dk=dk,
            q_cm=q_cm,
            spline=Sp,
            boundary_condition=boundary_condition,
            is_compressed=is_G0_compressed,
            k_cut=k_cut,
        )
        G0_sq = greens_function_free_space_squared(
            k=k,
            dk=dk,
            q_cm=q_cm,
            spline=Sp,
            k_cut=k_cut,
        )

        Id_coup = G0_coup = Sp_coup = G0_sq_coup = None
        if is_coupled:
            Id_coup = np.identity(2 * n_k, float)
            G0_coup = np.zeros((n_q, 2 * n_k), float)
            G0_coup[:, :n_k] = G0_coup[:, n_k:] = G0
            Sp_coup = np.zeros((n_q, 2, 2 * n_k))
            Sp_coup[:, 0, :n_k] = Sp_coup[:, 1, n_k:] = Sp

            G0_sq_coup = np.zeros((n_q, 2 * n_k), float)
            G0_sq_coup[:, :n_k] = G0_sq_coup[:, n_k:] = G0_sq

        n_p = V1.shape[-1]
        # Store everything
        self.V0 = V0
        self.V1 = V1
        V0_sub = []
        V1_sub = []
        if is_coupled:
            for i in range(n_q):
                V0_sub.append(Sp_coup[i] @ V0 @ Sp_coup[i].T)
                V1_sub.append(
                    np.stack(
                        [Sp_coup[i] @ V1[..., p] @ Sp_coup[i].T for p in range(n_p)],
                        axis=-1,
                    )
                )
        else:
            for i in range(n_q):
                V0_sub.append(Sp[i] @ V0 @ Sp[i].T)
                V1_sub.append(
                    np.stack(
                        [Sp[i] @ V1[..., p] @ Sp[i].T for p in range(n_p)], axis=-1
                    )
                )
        self.V0_sub = np.stack(V0_sub, axis=0)
        self.V1_sub = np.stack(V1_sub, axis=0)
        # self.inv_mass = inv_mass
        # self.mu = mu
        self.q_cm = q_cm
        self.n_k = n_k
        self.n_q = n_q
        self.n_p = n_p
        self.k = k
        self.dk = dk
        self.Id = Id
        self.G0 = G0
        self.Sp = Sp
        self.Id_coup = Id_coup
        self.G0_coup = G0_coup
        self.Sp_coup = Sp_coup
        self.G0_sq = G0_sq
        self.G0_sq_coup = G0_sq_coup
        self.boundary_condition = boundary_condition
        # self.is_G0_compressed = is_G0_compressed
        self.is_coupled = is_coupled
        self.dwa_wfs = dwa_wfs
        self.nugget = nugget
        self.is_mesh_semi_infinite = is_mesh_semi_infinite

        if boundary_condition is BoundaryCondition.STANDING:
            lippmann_schwinger_dtype = float
        else:
            lippmann_schwinger_dtype = "complex128"
        self.lippmann_schwinger_dtype = lippmann_schwinger_dtype

        # Attributes that will be created during the call to `fit`
        self.p_train = None
        self.K_train = None
        self.K_on_shell_train = None
        self.phase_train = None
        self.m0_vec = None
        self.m1_vec = None
        # self.M_const = None
        self.M0 = None
        self.M1 = None

    def full_potential(self, p):
        r"""Returns the full-space potential in momentum space.

        Parameters
        ----------
        p :
            The parameters upon which the potential depends

        Returns
        -------
        V : shape = (n_k, n_k)
        """
        return self.V0 + self.V1 @ p

    def on_shell_potential(self, p):
        r"""Returns the potential interpolated to the on-shell momenta.

        Parameters
        ----------
        p :
            The parameters upon which the potential depends

        Returns
        -------
        V_sub : shape = (n_t_lab, n_t_lab)
        """
        return self.V0_sub + self.V1_sub @ p

    def m_vec(self, p):
        r"""Returns the on shell part of the m vector.

        It is a vector in the space of training indices, and is defined by

        .. math::
            K_i G_0 V + V G_0 K_i

        where i is the index of the training points.

        Parameters
        ----------
        p :
            The parameters upon which the potential depends

        Returns
        -------
        m_vec : shape = (n_train, n_t_lab)
        """
        return self.m0_vec + self.m1_vec @ p

    def M_mat(self, p):
        r"""Returns the on shell part of the M matrix.

        It is a matrix in the space of training indices, and is defined by

        .. math::
            K_i G_0 K_j + K_j G_0 K_i - K_i G_0 V G_0 K_j - K_j G_0 V G_0 K_i

        where i and j are indices of the training points.

        Parameters
        ----------
        p :
            The parameters upon which the potential depends

        Returns
        -------
        M_mat : shape = (n_t_lab, n_train, n_train)
        """
        M = self.M0 + self.M1 @ p
        if self.nugget != 0:
            M = M + self.nugget * np.eye(M.shape[-1])
        return M

    def coefficients(self, p):
        r"""Returns the coefficients of the reactance matrix expansion.

        The linear combination of these coefficients and the reactance training matrices allows the
        emulation of the reactance at other parameter values.

        Parameters
        ----------
        p :
            The parameters upon which the potential depends

        Returns
        -------
        coefficients
        """
        return np.linalg.solve(self.M_mat(p), self.m_vec(p))

    def predict(
        self,
        p,
        return_phase: bool = False,
        full_space: bool = False,
        return_gradient: bool = False,
    ):
        """Returns the on-shell reactance matrix (or phase shifts) either via emulation or the full-space calculation.

        Parameters
        ----------
        p :
            The parameters upon which the potential depends
        return_phase :
            If True, this will return phase shifts (in degrees) rather than the K matrix. Defaults to False
        full_space :
            If True, this will compute the quantity using the full-space simulator, rather than the emulator.
            Defaults to False.
        return_gradient : bool
            Whether the gradient is to be returned along with the reactance or phase shifts. Defaults to False.

        Returns
        -------
        quantity : shape = (n_t_lab,)
            Either the on-shell reactance matrix or the phase shifts
        """
        if full_space:
            out = self.reactance(p, shell="on", return_gradient=return_gradient)
            if return_gradient:
                K, dK = out
            else:
                K = out
                dK = None
        else:
            V = self.on_shell_potential(p)
            m_vec = self.m_vec(p)
            M = self.M_mat(p)
            Minv_m = np.linalg.solve(M, m_vec)
            K = V + 0.5 * (m_vec * Minv_m).sum(axis=-1)

            dK = None
            if return_gradient:
                dMinv_m = np.sum(self.m1_vec * Minv_m[..., None], axis=1)
                dK = self.V1_sub + dMinv_m
                dK -= 0.5 * np.einsum("qi,qijp,qj->qp", Minv_m, self.M1, Minv_m)

            if self.is_coupled:
                q_cm = self.q_cm[:, None, None]
            else:
                q_cm = self.q_cm
            K *= q_cm * pi / 2
            if return_gradient:
                dK *= q_cm[..., None] * pi / 2

        if self.dwa_wfs is not None:
            from .utils import dwa

            K = dwa(
                K=K,
                f0=self.dwa_wfs["f0"],
                g0=self.dwa_wfs["g0"],
                df0=self.dwa_wfs["df0"],
                dg0=self.dwa_wfs["dg0"],
                f=self.dwa_wfs["f"],
                g=self.dwa_wfs["g"],
                df=self.dwa_wfs["df"],
                dg=self.dwa_wfs["dg"],
                coupled=False,
                dK=dK,
            )

        if return_gradient:
            return K, dK
        if return_phase:
            # TODO: Handle gradients?
            return self.phase_shifts(K)
        return K

    def fit(self, p_train) -> NewtonEmulator:
        """Train the reactance emulator.

        Parameters
        ----------
        p_train : shape = (n_train, n_p)
            The parameters of the potential at which to compute the reactance matrix with full fidelity.
            These reactance matrices will be stored and used to quickly emulate the reactance matrix
            at other parameter values via the `predict` method.

        Returns
        -------
        self
        """
        # Loop over training points and compute the reactance at each point.
        K_train = []
        for p in p_train:
            K_i = self.reactance(p, include_q=False, shell="half")
            K_train.append(K_i)
        K_train = np.stack(K_train, axis=-1)

        # These variables will behave differently depending on is_coupled.
        # Define the up front to make life easier later.
        is_coupled = self.is_coupled
        if is_coupled:
            G0 = self.G0_coup
            Sp = self.Sp_coup
            q_cm = self.q_cm[:, None, None, None]
        else:
            G0 = self.G0
            Sp = self.Sp
            q_cm = self.q_cm[:, None]

        # This is just for convenience for checking the training points.
        # Put q_cm back in so that the phases can be extracted.
        K_on_shell_train = q_cm * np.stack(
            [Sp[i] @ K_train[i] for i in range(self.n_q)], axis=0
        )

        # This matrix product is needed multiple times going forward. Compute it once.
        if is_coupled:
            # G0_K shape = (n_q, 2, 2 * n_k, n_train)
            G0_K = G0[:, None, :, None] * K_train
        else:
            # G0_K shape = (n_q, n_k, n_train)
            G0_K = self.G0[..., None] * K_train

        # =========================
        # The m vector.
        # Calculate the on shell part of the operator: K_i G_0 V + V G_0 K_i
        # where i denotes the training point. This creates a vector indexed by i.
        # =========================
        # We only want the on shell part

        # m0_vec shape = (n_q, n_train) or (n_q, 2, 2, n_train)
        m0_vec = np.stack([Sp[i] @ self.V0 @ G0_K[i] for i in range(self.n_q)])

        # Do the same thing, but loop over the parameter dimension and stack.
        # m1_vec shape = (n_q, n_train, n_params) or (n_q, 2, 2, n_train, n_params)
        m1_vec = np.stack(
            [
                np.stack([Sp[i] @ self.V1[..., p] @ G0_K[i] for i in range(self.n_q)])
                for p in range(self.n_p)
            ],
            axis=-1,
        )
        if is_coupled:
            m0_vec += m0_vec.swapaxes(-2, -3)
            m1_vec += m1_vec.swapaxes(-3, -4)
        else:
            m0_vec *= 2
            m1_vec *= 2

        # =========================
        # The M matrix.
        # Calculate the on shell part of the operator:
        # K_i G_0 K_j + K_j G_0 K_i - K_i G_0 V G_0 K_j - K_j G_0 V G_0 K_i
        # where i, j denote the training points. This creates a matrix indexed by i and j.
        # =========================

        if is_coupled:
            # This is a matrix in the space of training points: (n_t_lab, 2, 2, n_train, n_train)
            M0 = K_train.swapaxes(-1, -2)[:, None, :, :, :] @ G0_K[:, :, None, :, :]
            M0 -= (
                G0_K.swapaxes(-1, -2)[:, :, None, :, :]
                @ self.V0
                @ G0_K[:, None, :, :, :]
            )
        else:
            # This is a matrix in the space of training points: (n_t_lab, n_train, n_train)
            M0 = K_train.swapaxes(-1, -2) @ G0_K
            M0 -= G0_K.swapaxes(-1, -2) @ self.V0 @ G0_K
        M0 += M0.swapaxes(-1, -2)

        if is_coupled:
            M1 = np.stack(
                [
                    G0_K.swapaxes(-1, -2)[:, None, :, :, :]
                    @ self.V1[..., i]
                    @ G0_K[:, :, None, :, :]
                    for i in range(self.n_p)
                ],
                axis=-1,
            )
        else:
            M1 = np.stack(
                [
                    G0_K.swapaxes(-1, -2) @ self.V1[..., i] @ G0_K
                    for i in range(self.n_p)
                ],
                axis=-1,
            )
        M1 += M1.swapaxes(-2, -3)
        M1 *= -1

        # Store the emulator-specific objects
        self.m0_vec = m0_vec
        self.m1_vec = m1_vec
        # self.M_const = M_const
        self.M0 = M0
        self.M1 = M1

        # Store other objects for convenience and debugging
        self.p_train = p_train
        self.K_train = K_train
        self.K_on_shell_train = K_on_shell_train
        self.phase_train = np.stack(
            [
                self.phase_shifts(self.K_on_shell_train[..., i], fix=True)
                for i in range(len(p_train))
            ],
            axis=-1,
        )
        return self

    def setup_uq(self):
        # These variables will behave differently depending on is_coupled.
        # Define the up front to make life easier later.
        is_coupled = self.is_coupled
        if is_coupled:
            G0 = self.G0_coup
            G0_sq = self.G0_sq_coup
            Sp = self.Sp_coup
            q_cm = self.q_cm[:, None, None, None]
        else:
            G0 = self.G0
            G0_sq = self.G0_sq
            Sp = self.Sp
            q_cm = self.q_cm[:, None]

        V0 = self.V0
        V1 = self.V1
        K_train = self.K_train
        # G0_sq *= 4
        # G0_sq *= -1
        # This matrix product is needed multiple times going forward. Compute it once.
        if is_coupled:
            # K_G0 shape = (n_q, 2, 2 * n_k, n_train)
            K_G0 = K_train * G0[:, None, :, None]
        else:
            # K_G0 shape = (n_q, n_k, n_train)
            K_G0 = K_train * self.G0[..., None]

        # shape = (n_q, n_k) or (n_q, 2*n_k)
        # C0 = np.stack([Sp[i] @ self.V0 * G0[i] for i in range(self.n_q)])
        C0 = np.stack([Sp[i] @ V0 for i in range(self.n_q)])
        Cq = [C0]
        for q in range(self.n_p):
            # Ci = np.stack([Sp[i] @ self.V1[..., q] * G0[i] for i in range(self.n_q)])
            Ci = np.stack([Sp[i] @ V1[..., q] for i in range(self.n_q)])
            Cq.append(Ci)
        # Ultimately, Cq shape = (num_params+1, num_t_lab, num_momentum)
        # Cq = 2 * np.stack(Cq)
        Cq = np.stack(Cq)
        Cq = 2 * Cq

        # shape = (num_train, num_params+1, num_t_lab, num_momentum)
        n_train = len(self.p_train)
        Lnq = np.zeros((n_train, self.n_p + 1, *G0.shape))
        for n in range(n_train):
            Kn_G0 = K_G0[..., n]
            Kn = K_train[..., n]
            # Lnq[n, 0] = 2 * (Kn_G0 - np.einsum("ij,jk,ik->ik", Kn_G0, V0, G0))
            # Lnq[n, 0] = 2 * (Kn_G0 - (Kn_G0 @ V0) * G0)
            Lnq[n, 0] = 2 * (Kn - Kn_G0 @ V0)
            for q in range(self.n_p):
                Vq = V1[..., q]
                # Lnq[n, q + 1] = 2 * np.einsum("ij,jk,ik->ik", Kn_G0, Vq, G0)
                # Lnq[n, q + 1] = 2 * (Kn_G0 @ Vq) * G0
                Lnq[n, q + 1] = -2 * Kn_G0 @ Vq

        # # shape = (num_t_lab, num_params+1, num_params+1)
        # CqCq = np.einsum("ijk,ljk->jil", Cq, Cq)
        # # shape = (num_t_lab, num_params+1, num_params+1, num_train)
        # CqLnq = np.einsum("ijk,lmjk->jiml", Cq, Lnq)
        # # shape = (num_t_lab, num_params+1, num_params+1, num_train, num_train)
        # LnqLnq = np.einsum("ijkl,mnkl->kjnim", Lnq, Lnq)

        # shape = (num_t_lab, num_params+1, num_params+1)
        CqCq = np.einsum("ijk,jk,ljk->jil", Cq, G0_sq, Cq)
        # shape = (num_t_lab, num_params+1, num_params+1, num_train)
        CqLnq = np.einsum("ijk,jk,lmjk->jiml", Cq, G0_sq, Lnq)
        # shape = (num_t_lab, num_params+1, num_params+1, num_train, num_train)
        LnqLnq = np.einsum("ijkl,kl,mnkl->kjnim", Lnq, G0_sq, Lnq)
        self.Cq = Cq
        self.Lnq = Lnq
        self.CqCq = CqCq
        self.CqLnq = CqLnq
        self.LnqLnq = LnqLnq

    def predict_uncertainty(self, p):
        p_aug = np.concatenate(([1.0], p), axis=0)
        c = self.coefficients(p)
        unc_sq = (
            self.CqCq @ p_aug @ p_aug
            - 2 * np.einsum("ijkl,il->ijk", self.CqLnq, c) @ p_aug @ p_aug
            + np.einsum("ijklm,il,im->ijk", self.LnqLnq, c, c) @ p_aug @ p_aug
        )
        return unc_sq

    def _compute_reactance_no_q(self, V, G0, Id, Sp, K, shell, dV=None, dK=None):
        return_gradient = dV is not None and dK is not None
        for i in range(self.n_q):
            # This single line solves the LS equation:
            # TODO: Speed this up by pre-computing half on shell potential in init
            M = Id - V * G0[i]
            ket = np.linalg.solve(M, V @ Sp[i].T)
            if shell == "half":
                K[i] = ket.T
            else:
                K[i] = Sp[i] @ ket
                if return_gradient:
                    # TODO: This needs to be adjusted for coupled channels.
                    d_bra = np.linalg.solve(M.T, Sp[i])
                    d_ket = Sp[i] + G0[i] * ket
                    for a in range(self.n_p):
                        dK[i, ..., a] = d_bra @ dV[..., a] @ d_ket
        K *= pi / 2
        if return_gradient:
            # dK is modified in place so does not need to be returned
            dK *= pi / 2
        return K

    def _reactance_coupled(
        self, p, include_q: bool = True, shell="on", return_gradient=False
    ):
        if shell not in ["on", "half"]:
            raise ValueError("shell must be one of 'on' or 'half'.")
        V = self.full_potential(p)
        if shell == "half":
            K = np.zeros((self.n_q, 2, 2 * self.n_k), self.lippmann_schwinger_dtype)
        else:
            K = np.zeros((self.n_q, 2, 2), self.lippmann_schwinger_dtype)

        dK = None
        dV = None
        if return_gradient:
            if shell == "half":
                raise ValueError("If return_gradient is True, then shell must be 'on'.")
            dK = np.zeros((self.n_q, 2, 2, self.n_p), self.lippmann_schwinger_dtype)
            dV = self.V1

        Id = self.Id_coup
        Sp = self.Sp_coup
        G0 = self.G0_coup
        # dK is modified in place, if applicable
        K = self._compute_reactance_no_q(
            V=V, G0=G0, Id=Id, Sp=Sp, K=K, shell=shell, dV=dV, dK=dK
        )
        if include_q:
            q_cm = self.q_cm[:, None, None]
            # For the emulator training matrices, q should not be included
            K *= q_cm
            if return_gradient:
                dK *= q_cm[..., None]
        return K

    def _reactance_uncoupled(
        self, p, include_q: bool = True, shell="on", return_gradient=False
    ):
        if shell not in ["on", "half"]:
            raise ValueError("shell must be one of 'on' or 'half'.")
        V = self.full_potential(p)
        if shell == "half":
            K = np.zeros((self.n_q, self.n_k), self.lippmann_schwinger_dtype)
        else:
            K = np.zeros(self.n_q, self.lippmann_schwinger_dtype)

        dK = None
        dV = None
        if return_gradient:
            if shell == "half":
                raise ValueError("If return_gradient is True, then shell must be 'on'.")
            dK = np.zeros((self.n_q, self.n_p), self.lippmann_schwinger_dtype)
            dV = self.V1

        Id = self.Id
        Sp = self.Sp
        G0 = self.G0
        # dK is modified in place, if applicable
        K = self._compute_reactance_no_q(
            V=V, G0=G0, Id=Id, Sp=Sp, K=K, shell=shell, dV=dV, dK=dK
        )
        if include_q:
            if shell == "half":
                q_cm = self.q_cm[:, None]
            else:
                q_cm = self.q_cm
                if return_gradient:
                    dK *= q_cm[:, None]
            # For the emulator training matrices, q should not be included
            K *= q_cm

        if return_gradient:
            return K, dK
        return K

    def reactance(self, p, include_q: bool = True, shell="on", return_gradient=False):
        """Computes the reactance matrix by solving the Lippmann-Schwinger equation.

        Parameters
        ----------
        p :
            The parameters of the potential at which to compute the reactance matrix with full fidelity.
        include_q :
            Whether the K matrix should be multiplied by the center-of-mass momentum.
            This makes the matrix dimensionless and makes extracting
            phase shifts easier since it just involves an arc-tangent. Defaults to True.
        shell : str
            Whether the reactance matrix should be on-shell or half on-shell. Valid values are ['on', 'half'].

        Returns
        -------
        K :
            The reactance matrix. If shell == 'on', then shape = (n_t_lab,).
            If shell == 'half' then shape = (n_t_lab, n_k).
        """
        if self.is_coupled:
            return self._reactance_coupled(
                p=p, include_q=include_q, shell=shell, return_gradient=return_gradient
            )
        else:
            return self._reactance_uncoupled(
                p=p, include_q=include_q, shell=shell, return_gradient=return_gradient
            )

    def phase_shifts(self, K, fix=True):
        r"""Computes phase shifts in degrees given the solution to the LS equation.

        Parameters
        ----------
        K :
            The on-shell solution to the LS equation. Depending on the choice of boundary condition, this could
            represent the K matrix, or the incoming or outgoing T matrix.
        fix :
            Whether to try to make the phase shifts continuous, as opposed to jump by 180 degrees.
            Defaults to True.

        Returns
        -------
        phase_shifts
        """
        if self.boundary_condition is BoundaryCondition.OUTGOING:
            K = t_matrix_outgoing_to_standing(K)
        if self.boundary_condition is BoundaryCondition.INCOMING:
            K = t_matrix_incoming_to_standing(K)

        if self.is_coupled:
            # Some attempts to make the coupled phases continuous. It doesn't always work.

            def _fix_phases(delta_minus, delta_plus, epsilon, bar=True):
                delta_minus, delta_plus, epsilon = np.atleast_1d(
                    delta_minus, delta_plus, epsilon
                )
                d = delta_minus - delta_plus
                # d must be in range -pi/2 to pi/2 for some reason.
                offset = (d + np.pi / 2) // np.pi
                # Will not affect S since phases are only defined modulo pi
                dm = delta_minus - offset * np.pi
                # epsilon must be in -pi/4 to pi/4
                # e_offset = (epsilon + np.pi / 2) // np.pi
                # e = epsilon - e_offset * np.pi
                # print('e', epsilon - e)
                if bar:
                    # epsilon must be in -pi/4 to pi/4
                    e_offset = (2 * epsilon + np.pi / 2) // np.pi
                    e = epsilon - e_offset * np.pi / 2
                else:
                    e_offset = (epsilon + np.pi / 2) // np.pi
                    e = epsilon - e_offset * np.pi
                # e[offset % 2 == 1] *= -1
                return dm, delta_plus, e

            def transform_phases(delta_minus, delta_plus, epsilon, to_bar=True):
                # delta_minus = delta_minus % np.pi - np.pi/2
                # delta_plus = delta_plus % np.pi - np.pi / 2
                # epsilon = (epsilon % np.pi) - np.pi / 2
                delta_minus, delta_plus, epsilon = np.atleast_1d(
                    delta_minus, delta_plus, epsilon
                )
                # delta_minus, delta_plus, epsilon = _fix_phases(
                #     delta_minus, delta_plus, epsilon, bar=not to_bar)
                d = delta_minus - delta_plus
                s = delta_minus + delta_plus
                # offset = (d + np.pi / 2) // np.pi
                offset = (s + np.pi / 2) // np.pi
                s -= offset * np.pi
                # s = s % np.pi
                if to_bar:
                    # dm, dp, and e are *bar* phase shifts
                    e = 0.5 * np.arcsin(np.sin(2 * epsilon) * np.sin(d))
                    # dm = 0.5 * (s + np.arcsin(np.tan(2 * e) / np.tan(2 * epsilon)))
                    diff = np.arcsin(np.tan(2 * e) / np.tan(2 * epsilon))
                else:
                    # dm, dp, and e are *eigen* phase shifts
                    # e = 0.5 * np.arctan2(np.tan(2 * epsilon), np.sin(d))
                    e = 0.5 * np.arctan(np.tan(2 * epsilon) / np.sin(d))
                    # tan2e = np.tan(2*epsilon) / np.sin(d)
                    # sin2e = tan2e / np.sqrt(1 + tan2e**2)
                    # asinx = 2 * np.arctan2(np.tan(2*epsilon), tan2e + np.sqrt(tan2e**2 - np.tan(2*epsilon)**2))
                    # dm = 0.5 * (s + np.arcsin(np.sin(2 * epsilon) / sin2e))
                    # dm = 0.5 * (s + asinx)
                    diff = np.arcsin(np.sin(2 * epsilon) / np.sin(2 * e))
                # dm -= offset * np.pi
                # dp = s - dm
                dm = 0.5 * (s + diff)
                dp = 0.5 * (s - diff)

                # dm, dp, e = _fix_phases(dm, dp, e, bar=to_bar)

                # dm = dm % np.pi
                # dp = dp % np.pi
                # e = (e % np.pi) - np.pi / 2
                return dm, dp, e

            K00, K01, K11 = K[..., 0, 0], K[..., 0, 1], K[..., 1, 1]
            PT = self.q_cm * 0 + 1
            # Epsilon = arctan2(-2.0 * K01, -(K00 - K11)) / 2.0
            Epsilon = np.arctan(2.0 * K01 / (K00 - K11)) / 2.0
            rEpsilon = (K00 - K11) / np.cos(2.0 * Epsilon)
            Delta_a = -1.0 * np.arctan(PT[:] * (K00 + K11 + rEpsilon) / 2.0)
            Delta_b = -1.0 * np.arctan(PT[:] * (K00 + K11 - rEpsilon) / 2.0)

            Delta_a, Delta_b, Epsilon = transform_phases(
                Delta_a, Delta_b, Epsilon, to_bar=True
            )
            Delta_a, Delta_b, Epsilon = _fix_phases(Delta_a, Delta_b, Epsilon, bar=True)
            ps = np.stack([Delta_a, Delta_b, Epsilon], axis=0) * 180.0 / pi
        else:
            ps = np.arctan(-K) * 180.0 / pi
        if fix:
            ps = fix_phases_continuity(ps, is_radians=False)
        return ps


class SeparableNewtonMixin:
    """Overrides the `fit` method for the NVP to take advantage of the separable structure of the potential.

    Relies on the methods defined in the SeparableMixin,
    which should also be included in any subclasses that use this mixin
    """

    @staticmethod
    def _off_diagonal_multiplier(i, j):
        if i == j:
            return 1
        return 2

    def fit(self, p_train):

        v_q = self.compute_v_on_shell()
        tau = np.array([self.compute_reactance_strength_matrix(p_i) for p_i in p_train])
        vGv = self.compute_vGv_matrix()
        self.n_train = n_train = len(p_train)
        K_train = np.array([self.reactance(p_i, include_q=True) for p_i in p_train])

        n = self.n_form_factors
        m0 = np.zeros((self.n_q, n_train))
        m1 = np.zeros((self.n_q, n_train, self.n_p))
        M0 = np.zeros((self.n_q, n_train, n_train))
        M1 = np.zeros((self.n_q, n_train, n_train, self.n_p))

        upper_triangular_indices = [(xx, yy) for xx in range(n) for yy in range(xx, n)]
        param_idx = dict(
            zip(upper_triangular_indices, np.arange(len(upper_triangular_indices)))
        )

        m1_init = np.einsum("cq,iqcd,qda,bq->qiab", v_q, tau, vGv, v_q) + np.einsum(
            "aq,qbc,iqcd,dq->qiab", v_q, vGv, tau, v_q
        )
        M0 = np.einsum("cq,iqcd,qde,jqef,fq->qij", v_q, tau, vGv, tau, v_q) + np.einsum(
            "cq,jqcd,qde,iqef,fq->qij", v_q, tau, vGv, tau, v_q
        )
        M1_init = -np.einsum(
            "cq,iqcd,qda,qbe,jqef,fq->qijab", v_q, tau, vGv, vGv, tau, v_q
        ) - np.einsum("cq,jqcd,qda,qbe,iqef,fq->qijab", v_q, tau, vGv, vGv, tau, v_q)

        # Turn the matrix quantities into vectors that one can take a dot product with a parameter vector
        for a, b in upper_triangular_indices:
            # The matrices are symmetric, take upper triangular and multiply off-diagonals by 2
            mult_ab = self._off_diagonal_multiplier(a, b)
            pp = param_idx[a, b]
            m1[..., pp] = mult_ab * m1_init[..., a, b]
            M1[..., pp] = mult_ab * M1_init[..., a, b]

        self.p_train = p_train
        self.K_train = K_train
        self.m0_vec = m0
        self.m1_vec = m1
        self.M0 = M0
        self.M1 = M1

    def predict(
        self,
        p,
        return_phase: bool = False,
        full_space: bool = False,
    ):
        """Returns the on-shell reactance matrix (or phase shifts) either via emulation or the full-space calculation.

        Parameters
        ----------
        p :
            The parameters upon which the potential depends
        return_phase :
            If True, this will return phase shifts (in degrees) rather than the K matrix. Defaults to False
        full_space :
            If True, this will compute the quantity using the full-space simulator, rather than the emulator.
            Defaults to False.

        Returns
        -------
        quantity : shape = (n_q_cm,)
            Either the on-shell reactance matrix or the phase shifts
        """
        if full_space:
            # out = self.reactance(p, shell="on", return_gradient=return_gradient)
            K = self.reactance(p, include_q=True)
        else:
            V = self.on_shell_potential(p)
            m_vec = self.m_vec(p)
            M = self.M_mat(p)
            M = M + self.nugget * np.eye(M.shape[-1])  # New, double counts the nugget

            # Old
            # Minv_m = np.linalg.solve(M, m_vec)
            # K = V + 0.5 * (m_vec * Minv_m).sum(axis=-1)

            # New
            c = self.coefficients(p)
            K = V + 0.5 * np.einsum("qi,qij,qj->q", c, M, c)

            q_cm = self.q_cm
            K *= q_cm * pi / 2

        if return_phase:
            # TODO: Handle gradients?
            return self.phase_shifts(K)
        return K


class NewtonYamaguchiEmulator(
    SeparableYamaguchiMixin, SeparableNewtonMixin, NewtonEmulator
):
    def __init__(self, beta, q_cm, nugget=0, hbar2_over_2mu=1):
        self.beta = beta
        self.hbar2_over_2mu = hbar2_over_2mu
        self.q_cm = q_cm
        self.nugget = nugget
        self.n_form_factors = len(beta)
        self.ell = 0
        self.n_q = len(q_cm)
        self.boundary_condition = BoundaryCondition.STANDING
        self.is_coupled = False

        V1_sub = []
        v_q = self.compute_v_on_shell()
        for i in range(self.n_form_factors):
            for j in range(i, self.n_form_factors):
                if i != j:
                    V1_sub.append(v_q[i] * v_q[j] + v_q[j] * v_q[i])
                else:
                    V1_sub.append(v_q[i] * v_q[j])
        V1_sub = np.array(V1_sub).T

        self.n_p = V1_sub.shape[-1]
        self.V0_sub = np.zeros(V1_sub.shape[:-1])
        self.V1_sub = V1_sub
