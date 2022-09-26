import numpy as np
from scipy.special import spherical_jn

from .utils import BoundaryCondition, cubic_spline_matrix, greens_function_free_space


class BaseKohnEmulator:
    r"""A base class that provides methods for emulating scattering systems via the Kohn variational principle.

    The exact method for solving the Schrodinger equation is left for subclasses to implement.

    Parameters
    ----------
    V0 : ArrayLike, shape = (n_k, n_k)
        The piece of the potential that does not depend on parameters, in units of fermi. This may require
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
        r,
        dr,
        q_cm,
        inv_mass,
        nugget=0,
    ):
        self.r = r
        self.dr = dr
        self.V0 = V0
        self.V1 = V1
        self.q_cm = q_cm
        self.inv_mass = inv_mass
        self.nugget = nugget
        self.n_p = V1.shape[-1]
        self.n_q = len(q_cm)

        self.dU0 = None
        self.dU1 = None
        self.p_train = None
        self.psi_train = None
        self.K_train = None

    def fit(self, p_train):
        psi_train = []
        K_train = []
        for p in p_train:
            psi_i, K_i = self.predict_wave_function(p, return_K=True)
            psi_train.append(psi_i)
            K_train.append(K_i)
        psi_train = np.array(psi_train).transpose(2, 0, 1)
        K_train = np.array(K_train)

        n_train = len(p_train)
        n_p = self.n_p
        n_q = self.n_q
        dU0 = np.zeros((n_q, n_train, n_train))
        dU1 = np.zeros((n_q, n_train, n_train, n_p))
        V0 = self.V0
        V1 = self.V1
        dr = self.dr

        V0_sub = np.zeros((n_q, n_train, n_train))
        V1_sub = np.zeros((n_q, n_train, n_train, n_p))
        r = self.r
        for i in range(n_train):
            p_i = p_train[i]
            Vi = V1 @ p_i
            for j in range(i, n_train):
                p_j = p_train[j]
                Vj = V1 @ p_j
                dU0[:, i, j] = dU0[:, j, i] = np.einsum(
                    "ab,bd,ad->a",
                    r**2 * dr * psi_train[:, i],
                    -Vi - Vj,
                    psi_train[:, j],
                )
                dU1[:, i, j] = dU1[:, j, i] = np.einsum(
                    "ab,bdp,ad->ap",
                    r**2 * dr * psi_train[:, i],
                    2 * V1,
                    psi_train[:, j],
                )

                V0_sub[:, i, j] = V0_sub[:, j, i] = np.einsum(
                    "ab,bd,ad->a",
                    r**2 * dr * psi_train[:, i],
                    V0,
                    psi_train[:, j],
                )
                V1_sub[:, i, j] = V1_sub[:, j, i] = np.einsum(
                    "ab,bdp,ad->ap",
                    r**2 * dr * psi_train[:, i],
                    V1,
                    psi_train[:, j],
                )

        self.V0_sub = V0_sub
        self.V1_sub = V1_sub
        inv_mass = self.inv_mass
        dU0 *= 1 / inv_mass
        dU1 *= 1 / inv_mass
        self.dU0 = dU0
        self.dU1 = dU1
        self.p_train = p_train
        self.psi_train = psi_train
        self.K_train = K_train

        return self

    def predict(self, p, full_space=False):
        if full_space:
            return self.predict_wave_function(p)
        else:
            return self.emulate_wave_function(p)

    def predict_wave_function(self, p, return_K=False):
        raise NotImplementedError("Must implement in a subclass")

    def compute_dU(self, p):
        return self.dU0 + self.dU1 @ p

    def coefficients_and_multiplier(self, p):
        dU = self.compute_dU(p)
        n_train = len(self.p_train)
        n_q = self.n_q
        mat = np.block(
            [
                [dU, np.ones((n_q, n_train, 1))],
                [np.ones((n_q, 1, n_train)), np.zeros((n_q, 1, 1))],
            ]
        )
        mat = mat + self.nugget * np.eye(n_train + 1)
        vec = np.block([[-self.K_train], [np.ones(n_q)]])
        c = np.linalg.solve(mat, vec.T)
        return c

    def coefficients_without_multiplier(self, p):
        """Obtain basis coefficients without explicitly solving for the Lagrange multiplier as well.

        An attempt to avoid the need to use a Lagrange multiplier by substituting it back into the variational form.
        """
        # Create Delta U matrix for the choice of parameters
        n_train = len(self.p_train)
        dU = self.compute_dU(p)
        dU_avg = np.sum(dU, axis=-1) / n_train
        tau = -self.K_train.T
        # tau = self.K_train.T
        tau_avg = np.sum(tau, axis=-1, keepdims=True) / n_train  # Could be stored.

        # Create matrix and vector, then solve the system
        mat = dU - dU_avg[..., None] - dU_avg[:, None, :]
        mat = mat + self.nugget * np.eye(n_train)
        vec = tau - dU_avg - tau_avg
        c = np.linalg.solve(mat, vec)
        return c

    def matrix_with_lagrange(self, p):
        dU = self.compute_dU(p)
        n_train = len(self.p_train)
        n_q = self.NVP.n_q
        mat = np.block(
            [
                [dU, np.ones((n_q, n_train, 1))],
                [np.ones((n_q, 1, n_train)), np.zeros((n_q, 1, 1))],
            ]
        )
        mat = mat + self.nugget * np.eye(n_train + 1)
        return mat

    def matrix_without_lagrange(self, p):
        n_train = len(self.p_train)
        dU = self.compute_dU(p)
        dU_avg = np.sum(dU, axis=-1) / n_train
        mat = dU - dU_avg[..., None] - dU_avg[:, None, :]
        mat = mat + self.nugget * np.eye(n_train)
        return mat

    def coefficients(self, p):
        # return self.coefficients_and_multiplier(p)[:, :-1]
        return self.coefficients_without_multiplier(p)

    def emulate_wave_function(self, p, return_K=False):
        c = self.coefficients(p)
        psi = np.einsum("ijk,ij->ki", self.psi_train, c)
        if return_K:
            K = self.emulate_reactance(p, coefficients=c)
            return psi, K
        return psi

    def emulate_reactance(self, p, coefficients=None):
        if coefficients is None:
            coefficients = self.coefficients(p)
        dU = self.compute_dU(p)
        return np.einsum("qn,nq->q", coefficients, self.K_train) - 0.5 * np.einsum(
            "qn,qnm,qm->q", coefficients, dU, coefficients
        )
        # return np.einsum("qn,nq->q", coefficients, -self.K_train) - 0.5 * np.einsum(
        #     "qn,qnm,qm->q", coefficients, dU, coefficients
        # )
        # return np.einsum("qn,nq->q", coefficients, self.K_train)


class KohnLippmannSchwingerEmulator(BaseKohnEmulator):
    def __init__(
        self,
        V0,
        V1,
        r,
        dr,
        NVP,
        inv_mass,
        ell,
    ):
        self.NVP = NVP
        nugget = NVP.nugget
        self.ell = ell
        super().__init__(
            V0=V0, V1=V1, r=r, dr=dr, q_cm=NVP.q_cm, nugget=nugget, inv_mass=inv_mass
        )

    def predict_wave_function(self, p, return_K=False):
        R"""

        References
        ----------
        [1] (1970) Haftel, Michael I. and Tabakin, Frank
            NUCLEAR SATURATION AND THE SMOOTHNESS OF NUCLEON-NUCLEON POTENTIALS
            Nucl. Phys. A
            10.1016/0375-9474(70)90047-3
        """
        K_half = self.NVP.reactance(
            p=p, include_q=False, shell="half", return_gradient=False
        )
        from scipy.special import spherical_jn

        r = self.r
        G0 = self.NVP.G0
        k = self.NVP.k
        q_cm = self.NVP.q_cm
        Sp = self.NVP.Sp

        G0_K = (2 / np.pi) * G0 * K_half
        # G0_K = G0 * K_half
        j_k = spherical_jn(n=self.ell, z=r[:, None] * k)
        j_q = spherical_jn(n=self.ell, z=r[:, None] * q_cm)
        j_G0_K = np.einsum("rk,qk->rq", j_k, G0_K)
        psi = j_q + j_G0_K

        if return_K:
            K_on = np.einsum("ij,ij->i", Sp, K_half)
            return psi, K_on
        return psi


class SeparableKohnEmulator(BaseKohnEmulator):
    def __init__(
        self,
        v_r,
        r,
        dr,
        v_k,
        k,
        dk,
        q_cm,
        inv_mass,
        ell,
        nugget=0,
        is_mesh_semi_infinite=True,
    ):
        n_form_factors = len(v_r)
        V1 = []
        for i in range(n_form_factors):
            for j in range(i, n_form_factors):
                if i != j:
                    V1.append(v_r[i][:, None] * v_r[j] + v_r[j][:, None] * v_r[i])
                else:
                    V1.append(v_r[i][:, None] * v_r[j])
        V1 = np.dstack(V1)
        V0 = np.zeros(V1.shape[:-1])

        self.n_form_factors = n_form_factors
        self.v_r = v_r
        self.ell = ell
        self.v_k = np.array(v_k)
        self.k = k
        self.dk = dk
        self.Sp = cubic_spline_matrix(k, q_cm)

        k_cut = None
        if is_mesh_semi_infinite:
            k_cut = np.inf
        self.G0 = greens_function_free_space(
            k=k,
            dk=dk,
            q_cm=q_cm,
            spline=self.Sp,
            boundary_condition=BoundaryCondition.STANDING,
            k_cut=k_cut,
        )
        super().__init__(
            V0=V0, V1=V1, r=r, dr=dr, q_cm=q_cm, inv_mass=inv_mass, nugget=nugget
        )

        self.vGv = self.compute_vGv_matrix()

    def compute_strength_matrix(self, p):
        n_dim = self.n_form_factors
        mat = np.zeros((n_dim, n_dim))

        idx = 0
        for i in range(n_dim):
            for j in range(i, n_dim):
                mat[i, j] = mat[j, i] = p[idx]
                idx += 1
        return mat

    def compute_vGv_matrix(self):
        n_dim = self.n_form_factors
        G0 = self.G0
        v_k = self.v_k
        vGv = np.zeros((self.n_q, n_dim, n_dim))
        for i in range(n_dim):
            for j in range(i, n_dim):
                vGv[:, i, j] = vGv[:, j, i] = G0 @ (v_k[i] * v_k[j])
        return vGv

    def compute_half_on_shell_reactance(self, p, include_q=True):
        strength = self.compute_strength_matrix(p)
        Id = np.eye(self.n_form_factors)
        tau = np.linalg.solve(Id - strength @ self.vGv, strength[None, ...])
        v_q_cm = np.einsum("nk,qk->nq", self.v_k, self.Sp)
        vTv = np.einsum("nk,qnm,mq->kq", self.v_k, tau, v_q_cm)
        K_half = (np.pi / 2) * vTv
        if include_q:
            K_half *= self.q_cm
        return K_half.T

    def validate_parameters(self, p):
        n_tri = int(self.n_form_factors * (self.n_form_factors + 1) / 2)
        if len(p) != n_tri:
            raise ValueError(
                f"The length of the parameters ({len(p)}) should be {n_tri} to fill the upper triangle of the strength matrix"
            )
        return self

    def predict_wave_function(self, p, return_K=False):
        self.validate_parameters(p)
        K_half = self.compute_half_on_shell_reactance(p, include_q=False)
        K = np.sum(self.Sp * K_half, axis=1)

        r = self.r
        G0_K = (2 / np.pi) * self.G0 * K_half
        # G0_K = self.G0 * K_half
        j_k = spherical_jn(n=self.ell, z=r[:, None] * self.k)
        j_q = spherical_jn(n=self.ell, z=r[:, None] * self.q_cm)
        j_G0_K = np.einsum("rk,qk->rq", j_k, G0_K)
        psi = j_q + j_G0_K

        if return_K:
            return psi, K
        return psi
