import numpy as np
from scipy.special import spherical_jn

from .utils import BoundaryCondition, cubic_spline_matrix, greens_function_free_space
from .separable import (
    SeparableMixin,
    SeparableYamaguchiMixin,
    yamaguchi_radial_wave_function,
)


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
    r :
        The radial coordinate mesh, likely created using some quadrature rules.
    dr :
        The integration measure, likely created using some quadrature rules.
    q_cm :
        The center-of-mass momentum, in units of inverse length (compatible with r).
    """

    def __init__(
        self,
        V0,
        V1,
        r,
        dr,
        q_cm,
        is_local,
        # inv_mass,
        nugget=0,
        use_lagrange_multiplier=False,
        use_momentum_space=False,
    ):
        self.r = r
        self.dr = dr
        self.V0 = V0
        self.V1 = V1
        self.q_cm = q_cm
        # self.inv_mass = inv_mass
        self.nugget = nugget
        self.n_p = V1.shape[-1]
        self.n_q = len(q_cm)
        self.use_lagrange_multiplier = use_lagrange_multiplier
        self.is_local = is_local
        self.use_momentum_space = use_momentum_space

        self.dU0 = None
        self.dU1 = None
        self.p_train = None
        self.psi_train = None
        self.K_train = None

    def fit(self, p_train):
        # psi_train = []
        # K_train = []
        # for p in p_train:
        #     psi_i, K_i = self.predict_wave_function(p, return_K=True)
        #     psi_train.append(psi_i)
        #     K_train.append(K_i)
        # psi_train = np.array(psi_train).transpose(1, 0, 2)
        # K_train = np.array(K_train)

        # n_train = len(p_train)
        # n_p = self.n_p
        # n_q = self.n_q
        # dU0 = np.zeros((n_q, n_train, n_train))
        # dU1 = np.zeros((n_q, n_train, n_train, n_p))
        # V0 = self.V0
        # V1 = self.V1
        # dr = self.dr

        # V0_sub = np.zeros((n_q, n_train, n_train))
        # V1_sub = np.zeros((n_q, n_train, n_train, n_p))
        # r = self.r
        # for i in range(n_train):
        #     p_i = p_train[i]
        #     Vi = V1 @ p_i
        #     for j in range(i, n_train):
        #         p_j = p_train[j]
        #         Vj = V1 @ p_j

        #         psi_left = r * dr * psi_train[:, i]
        #         if self.is_local:
        #             psi_right = r * psi_train[:, j]
        #         else:
        #             psi_right = r * dr * psi_train[:, j]

        #         dU0[:, i, j] = dU0[:, j, i] = np.einsum(
        #             "ab,bd,ad->a",
        #             psi_left,
        #             -Vi - Vj,
        #             psi_right,
        #         )
        #         dU1[:, i, j] = dU1[:, j, i] = np.einsum(
        #             "ab,bdp,ad->ap",
        #             psi_left,
        #             2 * V1,
        #             psi_right,
        #         )

        #         V0_sub[:, i, j] = V0_sub[:, j, i] = np.einsum(
        #             "ab,bd,ad->a",
        #             psi_left,
        #             V0,
        #             psi_right,
        #         )
        #         V1_sub[:, i, j] = V1_sub[:, j, i] = np.einsum(
        #             "ab,bdp,ad->ap",
        #             psi_left,
        #             V1,
        #             psi_right,
        #         )

        # self.V0_sub = V0_sub
        # self.V1_sub = V1_sub
        # self.dU0 = dU0 * self.q_cm[:, None, None]
        # self.dU1 = dU1 * self.q_cm[:, None, None, None]
        # self.p_train = p_train
        # self.psi_train = psi_train
        # self.K_train = K_train

        if self.use_momentum_space:
            self._fit_momentum_space(p_train=p_train)
        else:
            self._fit_position_space(p_train=p_train)

        n_train = self.n_train
        n_q = self.n_q
        # For speed, create the matrices needed for solving for the coefficients once here.
        # We will overwrite the upper left block each time it is needed.
        # The lower right entry does not get a nugget added to it (if it did, the coefficients don't sum to one)
        dU_toy_array = np.empty((n_q, n_train, n_train))
        self._dU_expanded = np.block(
            [
                [dU_toy_array, np.ones((n_q, n_train, 1))],
                [np.ones((n_q, 1, n_train)), np.zeros((n_q, 1, 1))],
            ]
        )
        self._tau_expanded = np.block([[-self.K_train], [np.ones(n_q)]])

        # Determine optimal einsum path once so that it does not unnecessarily compute it every time during emulation
        coeff_toy_array = np.ones((n_q, n_train))
        self._c_dU_c_opt_path = np.einsum_path(
            "qn,qnm,qm->q",
            coeff_toy_array,
            dU_toy_array,
            coeff_toy_array,
            optimize="optimal",
        )[0]

        return self

    def _create_snapshots(self, p_train):
        psi_train = []
        K_train = []
        for p in p_train:
            psi_i, K_i = self.predict_wave_function(p, return_K=True)
            psi_train.append(psi_i)
            K_train.append(K_i)
        psi_train = np.array(psi_train).transpose(1, 0, 2)
        K_train = np.array(K_train)
        return psi_train, K_train

    def _fit_position_space(self, p_train):
        psi_train, K_train = self._create_snapshots(p_train)

        n_train = len(p_train)
        n_p = self.n_p
        n_q = self.n_q
        dU0 = np.zeros((n_q, n_train, n_train))
        dU1 = np.zeros((n_q, n_train, n_train, n_p))
        V0 = self.V0
        V1 = self.V1
        r = self.r
        dr = self.dr

        for i in range(n_train):
            p_i = p_train[i]
            Vi = V1 @ p_i
            for j in range(i, n_train):
                p_j = p_train[j]
                Vj = V1 @ p_j

                psi_left = r * dr * psi_train[:, i]
                if self.is_local:
                    psi_right = r * psi_train[:, j]
                else:
                    psi_right = r * dr * psi_train[:, j]

                dU0[:, i, j] = dU0[:, j, i] = np.einsum(
                    "ab,bd,ad->a",
                    psi_left,
                    -Vi - Vj,
                    psi_right,
                )
                dU1[:, i, j] = dU1[:, j, i] = np.einsum(
                    "ab,bdp,ad->ap",
                    psi_left,
                    2 * V1,
                    psi_right,
                )

        self.dU0 = dU0 * self.q_cm[:, None, None]
        self.dU1 = dU1 * self.q_cm[:, None, None, None]
        self.p_train = p_train
        self.n_train = n_train
        self.psi_train = psi_train
        self.K_train = K_train
        return self

    def _fit_momentum_space(self, p_train):
        psi_train, K_train = self._create_snapshots(p_train)
        n_train = len(p_train)

        n_p = self.n_p
        n_q = self.n_q
        dU0 = np.zeros((n_q, n_train, n_train))
        dU1 = np.zeros((n_q, n_train, n_train, n_p))
        V0 = self.V0
        V1 = self.V1
        k = self.r
        dk = self.dr

        Sp = cubic_spline_matrix(k, self.q_cm)

        for i in range(n_train):
            p_i = p_train[i]
            Vi = V1 @ p_i
            for j in range(i, n_train):
                p_j = p_train[j]
                Vj = V1 @ p_j

                # if self.is_local:
                #     psi_left = k * dk * psi_train[:, i]
                #     psi_right = k * psi_train[:, j]
                # else:
                #     psi_left = k**2 * dk * psi_train[:, i]
                #     psi_right = k**2 * dk * psi_train[:, j]

                psi_left = Sp + (2 / np.pi) * psi_train[:, i]
                psi_right = Sp + (2 / np.pi) * psi_train[:, j]

                # psi_left = psi_left + Sp
                # psi_right = psi_right + Sp

                dU0[:, i, j] = dU0[:, j, i] = np.einsum(
                    "ab,bd,ad->a",
                    psi_left,
                    -Vi - Vj,
                    psi_right,
                )
                dU1[:, i, j] = dU1[:, j, i] = np.einsum(
                    "ab,bdp,ad->ap",
                    psi_left,
                    2 * V1,
                    psi_right,
                )

        self.dU0 = dU0 * self.q_cm[:, None, None] * (np.pi / 2)
        self.dU1 = dU1 * self.q_cm[:, None, None, None] * (np.pi / 2)
        self.p_train = p_train
        self.n_train = n_train
        self.psi_train = psi_train
        self.K_train = K_train

    def predict(self, p, full_space=False):
        if full_space:
            _, K = self.predict_wave_function(p, return_K=True)
            return K
        else:
            return self.emulate_reactance(p)

    def predict_wave_function(self, p, return_K=False):
        if self.use_momentum_space:
            return self._predict_wave_function_momentum_space(p=p, return_K=return_K)
        else:
            return self._predict_wave_function_position_space(p=p, return_K=return_K)

    def _predict_wave_function_position_space(self, p, return_K=False):
        raise NotImplementedError("Must implement in a subclass")

    def _predict_wave_function_momentum_space(self, p, return_K=False):
        raise NotImplementedError("Must implement in a subclass")

    def compute_dU(self, p):
        return self.dU0 + self.dU1 @ p

    def coefficients_and_multiplier(self, p):
        mat = self.matrix_with_lagrange(p)
        vec = self._tau_expanded.T
        c = np.linalg.solve(mat, vec)
        # c = np.zeros_like(vec)
        # for i in range(c.shape[0]):
        #     c[i] = np.linalg.lstsq(mat[i], vec[i], rcond=1e-10)[0]
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

        # c = np.zeros_like(vec)
        # for i in range(c.shape[0]):
        #     c[i] = np.linalg.lstsq(mat[i], vec[i], rcond=1e-10)[0]
        return c

    def matrix_with_lagrange(self, p):
        dU = self.compute_dU(p)
        n_train = len(self.p_train)
        # Overwrite existing matrix so that no new memory needs to be allocated
        # The lower right entry does not get a nugget added to it (if it did, the coefficients don't sum to one)
        self._dU_expanded[:, :n_train, :n_train] = dU + self.nugget * np.eye(n_train)
        return self._dU_expanded

    def matrix_without_lagrange(self, p):
        n_train = len(self.p_train)
        dU = self.compute_dU(p)
        dU_avg = np.sum(dU, axis=-1) / n_train
        mat = dU - dU_avg[..., None] - dU_avg[:, None, :]
        mat = mat + self.nugget * np.eye(n_train)
        return mat

    def coefficients(self, p):
        if self.use_lagrange_multiplier:
            return self.coefficients_and_multiplier(p)[:, :-1]
        return self.coefficients_without_multiplier(p)

    def emulate_wave_function(self, p, return_K=False):
        c = self.coefficients(p)
        psi = np.einsum("kjr,kj->kr", self.psi_train, c)
        if return_K:
            K = self.emulate_reactance(p, coefficients=c)
            return psi, K
        return psi

    def emulate_reactance(self, p, coefficients=None):
        if coefficients is None:
            coefficients = self.coefficients(p)
        dU = self.compute_dU(p)
        # dU = dU + self.nugget * np.eye(self.n_train)
        return np.sum(coefficients * self.K_train.T, axis=-1) + 0.5 * np.einsum(
            "qn,qnm,qm->q",
            coefficients,
            dU,
            coefficients,
            # optimize=self._c_dU_c_opt_path,
        )


class KohnLippmannSchwingerEmulator(BaseKohnEmulator):
    def __init__(
        self,
        V0,
        V1,
        r,
        dr,
        NVP,
        # inv_mass,
        is_local,
        ell,
        use_lagrange_multiplier=False,
        use_momentum_space=False,
    ):
        self.NVP = NVP
        nugget = NVP.nugget
        self.ell = ell
        super().__init__(
            V0=V0,
            V1=V1,
            r=r,
            dr=dr,
            q_cm=NVP.q_cm,
            nugget=nugget,
            # inv_mass=inv_mass,
            is_local=is_local,
            use_lagrange_multiplier=use_lagrange_multiplier,
            use_momentum_space=use_momentum_space,
        )

    def _predict_wave_function_momentum_space(self, p, return_K=False):
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

        G0 = self.NVP.G0
        q_cm = self.NVP.q_cm
        Sp = self.NVP.Sp

        scattered_wave_function = (2 / np.pi) * G0 * K_half

        if return_K:
            K_on = q_cm * np.einsum("ij,ij->i", Sp, K_half)
            return scattered_wave_function, K_on
        return scattered_wave_function

    def _predict_wave_function_position_space(self, p, return_K=False):
        R"""

        References
        ----------
        [1] (1970) Haftel, Michael I. and Tabakin, Frank
            NUCLEAR SATURATION AND THE SMOOTHNESS OF NUCLEON-NUCLEON POTENTIALS
            Nucl. Phys. A
            10.1016/0375-9474(70)90047-3
        """

        out = self._predict_wave_function_momentum_space(p, return_K=return_K)
        if return_K:
            scatt_wf, K_on = out
        else:
            scatt_wf = out
        from scipy.special import spherical_jn

        r = self.r
        k = self.NVP.k
        q_cm = self.NVP.q_cm

        j_k = spherical_jn(n=self.ell, z=r * k[:, None])
        j_q = spherical_jn(n=self.ell, z=r * q_cm[:, None])
        j_G0_K = np.einsum("kr,qk->qr", j_k, scatt_wf)
        psi = j_q + j_G0_K

        if return_K:
            return psi, K_on
        return psi


class SeparableKohnMixin:
    """Overrides the `fit` method for the KVP to take advantage of the separable structure of the potential."""

    def fit(self, p_train):
        v_q = self.compute_v_on_shell()
        tau = np.array([self.compute_reactance_strength_matrix(p_i) for p_i in p_train])
        Lambda = np.array([self.compute_strength_matrix(p_i) for p_i in p_train])
        vGv = self.compute_vGv_matrix()
        self.n_train = n_train = len(p_train)
        K_train = np.array([self.reactance(p_i, include_q=True) for p_i in p_train])

        n = self.n_form_factors
        dU0 = np.zeros((self.n_q, n_train, n_train))
        dU1 = np.zeros((self.n_q, n_train, n_train, self.n_p))

        for i in range(n_train):
            for j in range(i, n_train):
                dU0[:, i, j] = dU0[:, j, i] = -(
                    np.einsum("cq,cd,dq->q", v_q, Lambda[i] + Lambda[j], v_q)
                    + np.einsum("cq,qcd,qde,ef,fq->q", v_q, tau[i], vGv, Lambda[j], v_q)
                    + np.einsum("cq,qcd,qde,ef,fq->q", v_q, tau[j], vGv, Lambda[i], v_q)
                    + np.einsum("cq,cd,qde,qef,fq->q", v_q, Lambda[j], vGv, tau[j], v_q)
                    + np.einsum("cq,cd,qde,qef,fq->q", v_q, Lambda[i], vGv, tau[i], v_q)
                    + np.einsum(
                        "cq,qcd,qde,ef,qfg,qgh,hq->q",
                        v_q,
                        tau[i],
                        vGv,
                        Lambda[j],
                        vGv,
                        tau[j],
                        v_q,
                    )
                    + np.einsum(
                        "cq,qcd,qde,ef,qfg,qgh,hq->q",
                        v_q,
                        tau[j],
                        vGv,
                        Lambda[i],
                        vGv,
                        tau[i],
                        v_q,
                    )
                )

        dU1_init = (
            2 * np.einsum("aq,bq->qab", v_q, v_q)[:, None, None, :, :]
            + np.einsum("cq,iqcd,qda,bq->qiab", v_q, tau, vGv, v_q)[:, :, None, :, :]
            + np.einsum("cq,iqcd,qda,bq->qiab", v_q, tau, vGv, v_q)[:, None, :, :, :]
            + np.einsum("aq,qbe,iqef,fq->qiab", v_q, vGv, tau, v_q)[:, :, None, :, :]
            + np.einsum("aq,qbe,iqef,fq->qiab", v_q, vGv, tau, v_q)[:, None, :, :, :]
            + np.einsum("cq,iqcd,qda,qbg,jqgh,hq->qijab", v_q, tau, vGv, vGv, tau, v_q)
            + np.einsum("cq,jqcd,qda,qbg,iqgh,hq->qijab", v_q, tau, vGv, vGv, tau, v_q)
        )

        upper_triangular_indices = [(xx, yy) for xx in range(n) for yy in range(xx, n)]
        param_idx = dict(
            zip(upper_triangular_indices, np.arange(len(upper_triangular_indices)))
        )

        # Turn the matrix quantities into vectors that one can take a dot product with a parameter vector
        for a, b in upper_triangular_indices:
            # The matrices are symmetric, take upper triangular and multiply off-diagonals by 2
            mult_ab = 1 if a == b else 2
            pp = param_idx[a, b]
            dU1[..., pp] = mult_ab * dU1_init[..., a, b]

        self.p_train = p_train
        self.K_train = K_train
        self.dU0 = dU0 * self.q_cm[:, None, None] * np.pi / 2
        self.dU1 = dU1 * self.q_cm[:, None, None, None] * np.pi / 2

        n_q = self.n_q

        # For speed, create the matrices needed for solving for the coefficients once here.
        # We will overwrite the upper left block each time it is needed.
        # The lower right entry does not get a nugget added to it (if it did, the coefficients don't sum to one)
        dU_toy_array = np.empty((n_q, n_train, n_train))
        self._dU_expanded = np.block(
            [
                [dU_toy_array, np.ones((n_q, n_train, 1))],
                [np.ones((n_q, 1, n_train)), np.zeros((n_q, 1, 1))],
            ]
        )
        self._tau_expanded = np.block([[-self.K_train], [np.ones(n_q)]])

        # Determine optimal einsum path once so that it does not unnecessarily compute it every time during emulation
        coeff_toy_array = np.ones((n_q, n_train))
        self._c_dU_c_opt_path = np.einsum_path(
            "qn,qnm,qm->q",
            coeff_toy_array,
            dU_toy_array,
            coeff_toy_array,
            optimize="optimal",
        )[0]

    def predict(self, p, full_space=False):
        if full_space:
            return self.reactance(p, include_q=True)
        else:
            return self.emulate_reactance(p)

    def predict_wave_function(self, p, r):
        psi = np.array(
            [
                yamaguchi_radial_wave_function(
                    r=r,
                    q_cm=self.q_cm,
                    beta=beta,
                )
                for beta in self.beta
            ]
        )


class SeparableKohnEmulator(SeparableKohnMixin, SeparableMixin, BaseKohnEmulator):
    def __init__(
        self,
        v_r,
        r,
        dr,
        v_k,
        k,
        dk,
        q_cm,
        # inv_mass,
        # is_local,
        ell,
        nugget=0,
        use_lagrange_multiplier=False,
        use_momentum_space=False,
        is_mesh_semi_infinite=True,
    ):
        n_form_factors = len(v_r)
        V1 = []
        if use_momentum_space:
            vv = v_k
        else:
            vv = v_r
        for i in range(n_form_factors):
            for j in range(i, n_form_factors):
                if i != j:
                    V1.append(vv[i][:, None] * vv[j] + vv[j][:, None] * vv[i])
                else:
                    V1.append(vv[i][:, None] * vv[j])
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
            V0=V0,
            V1=V1,
            r=r,
            dr=dr,
            q_cm=q_cm,
            is_local=False,
            # inv_mass=inv_mass,
            nugget=nugget,
            use_lagrange_multiplier=use_lagrange_multiplier,
            use_momentum_space=use_momentum_space,
        )

        self.vGv = self.compute_vGv_matrix()

    # def compute_strength_matrix(self, p):
    #     n_dim = self.n_form_factors
    #     mat = np.zeros((n_dim, n_dim))

    #     idx = 0
    #     for i in range(n_dim):
    #         for j in range(i, n_dim):
    #             mat[i, j] = mat[j, i] = p[idx]
    #             idx += 1
    #     return mat

    # def compute_vGv_matrix(self):
    #     n_dim = self.n_form_factors
    #     G0 = self.G0
    #     v_k = self.v_k
    #     vGv = np.zeros((self.n_q, n_dim, n_dim))
    #     for i in range(n_dim):
    #         for j in range(i, n_dim):
    #             vGv[:, i, j] = vGv[:, j, i] = G0 @ (v_k[i] * v_k[j])
    #     return vGv

    # def compute_half_on_shell_reactance(self, p, include_q=True):
    #     strength = self.compute_strength_matrix(p)
    #     Id = np.eye(self.n_form_factors)
    #     tau = np.linalg.solve(Id - strength @ self.vGv, strength[None, ...])
    #     v_q_cm = np.einsum("nk,qk->nq", self.v_k, self.Sp)
    #     vTv = np.einsum("nk,qnm,mq->kq", self.v_k, tau, v_q_cm)
    #     K_half = (np.pi / 2) * vTv
    #     if include_q:
    #         K_half *= self.q_cm
    #     return K_half.T

    # def validate_parameters(self, p):
    #     n_tri = int(self.n_form_factors * (self.n_form_factors + 1) / 2)
    #     if len(p) != n_tri:
    #         raise ValueError(
    #             f"The length of the parameters ({len(p)}) should be {n_tri} to fill the upper triangle of the strength matrix"
    #         )
    #     return self

    def _predict_wave_function_momentum_space(self, p, return_K=False):
        self.validate_parameters(p)
        K_half = self.compute_half_on_shell_reactance(p, include_q=False)
        K = self.q_cm * np.sum(self.Sp * K_half, axis=1)
        scattered_wf = (2 / np.pi) * self.G0 * K_half
        if return_K:
            return scattered_wf, K
        return scattered_wf

    def _predict_wave_function_position_space(self, p, return_K=False):
        # self.validate_parameters(p)
        # K_half = self.compute_half_on_shell_reactance(p, include_q=False)
        # K = self.q_cm * np.sum(self.Sp * K_half, axis=1)

        # r = self.r
        # G0_K = (2 / np.pi) * self.G0 * K_half

        out = self._predict_wave_function_momentum_space(p, return_K=return_K)
        if return_K:
            scatt_wf, K = out
        else:
            scatt_wf = out

        r = self.r
        j_k = spherical_jn(n=self.ell, z=r * self.k[:, None])
        j_q = spherical_jn(n=self.ell, z=r * self.q_cm[:, None])
        j_G0_K = np.einsum("kr,qk->qr", j_k, scatt_wf)
        psi = j_q + j_G0_K

        if return_K:
            return psi, K
        return psi


class KohnYamaguchiEmulator(
    SeparableYamaguchiMixin, SeparableKohnMixin, BaseKohnEmulator
):
    def __init__(
        self, beta, q_cm, nugget=0, hbar2_over_2mu=1, use_lagrange_multiplier=False
    ):
        self.beta = beta
        self.hbar2_over_2mu = hbar2_over_2mu
        self.q_cm = q_cm
        self.nugget = nugget
        self.n_form_factors = len(beta)
        self.ell = 0
        self.n_q = len(q_cm)
        self.n_p = int(self.n_form_factors * (self.n_form_factors + 1) / 2)
        self.use_lagrange_multiplier = use_lagrange_multiplier


class AlternateKohnEmulator:
    r""" """

    def __init__(
        self,
        V0,
        V1,
        r,
        dr,
        q_cm,
        NVP,
        is_local,
        ell,
        use_lagrange_multiplier=False,
    ):
        self.r = r
        self.dr = dr
        self.V0 = V0
        self.V1 = V1
        self.q_cm = q_cm
        nugget = NVP.nugget
        # self.inv_mass = inv_mass
        self.nugget = nugget
        self.n_p = V1.shape[-1]
        self.n_q = len(q_cm)
        self.use_lagrange_multiplier = use_lagrange_multiplier
        self.is_local = is_local

        self.dU0 = None
        self.dU1 = None
        self.p_train = None
        self.psi_train = None
        self.K_train = None

        self.NVP = NVP

        self.ell = ell

    def fit(self, p_train):
        psi_train = []
        K_train = []
        for p in p_train:
            psi_i, K_i = self.predict_wave_function(p, return_K=True)
            psi_train.append(psi_i)
            K_train.append(K_i)
        psi_train = np.array(psi_train).transpose(1, 0, 2)
        K_train = np.array(K_train)

        n_train = len(p_train)
        n_p = self.n_p
        n_q = self.n_q
        dU0 = np.zeros((n_q, n_train, n_train))
        dU1 = np.zeros((n_q, n_train, n_train, n_p))
        rhs0 = np.zeros((n_q, n_train))
        rhs1 = np.zeros((n_q, n_train, n_p))
        V0 = self.V0
        V1 = self.V1
        r = self.r
        dr = self.dr

        j_q = spherical_jn(n=self.ell, z=r * self.q_cm[:, None])
        if self.is_local:
            free_wf = r * j_q
        else:
            free_wf = r * dr * j_q

        V0_free = np.einsum(
            "ab,bd,ad->a",
            r * dr * j_q,
            V0,
            free_wf,
        )
        V1_free = np.einsum(
            "ab,bdp,ad->ap",
            r * dr * j_q,
            V1,
            free_wf,
        )

        for i in range(n_train):
            p_i = p_train[i]
            Vi = V1 @ p_i

            chi_left = r * dr * (psi_train[:, i] - j_q)

            chi_V_phi_0 = np.einsum(
                "ab,bd,ad->a",
                chi_left,
                V0,
                free_wf,
            )
            chi_V_phi_1 = np.einsum(
                "ab,bdp,ad->ap",
                chi_left,
                V1,
                free_wf,
            )
            rhs0[:, i] = -chi_V_phi_0
            rhs1[:, i] = -chi_V_phi_1
            for j in range(n_train):
                p_j = p_train[j]
                Vj = V1 @ p_j

                if self.is_local:
                    chi_right = r * (psi_train[:, j] - j_q)
                else:
                    chi_right = r * dr * (psi_train[:, j] - j_q)

                chi_V_phi_j = np.einsum(
                    "ab,bd,ad->a",
                    chi_left,
                    Vj,
                    free_wf,
                )

                dU0[:, i, j] = (
                    np.einsum(
                        "ab,bd,ad->a",
                        chi_left,
                        -Vj,
                        chi_right,
                    )
                    - chi_V_phi_0
                    - chi_V_phi_j
                )
                dU1[:, i, j] = np.einsum(
                    "ab,bdp,ad->ap",
                    chi_left,
                    V1,
                    chi_right,
                )

        dU0 = 0.5 * (dU0 + dU0.swapaxes(1, 2))
        dU1 = 0.5 * (dU1 + dU1.swapaxes(1, 2))
        self.dU0 = dU0
        self.dU1 = dU1
        self.rhs0 = rhs0
        self.rhs1 = rhs1
        self.V0_free = V0_free
        self.V1_free = V1_free
        self.p_train = p_train
        self.psi_train = psi_train
        self.chi_train = psi_train - j_q[:, None, :]
        self.j_q = j_q
        self.n_train = n_train
        self.K_train = K_train

        return self

    def compute_dU(self, p):
        return self.dU0 + self.dU1 @ p

    def coefficients(self, p):
        dU = self.compute_dU(p)
        dU = dU + self.nugget * np.eye(dU.shape[-1])
        rhs = self.rhs0 + self.rhs1 @ p

        if self.use_lagrange_multiplier:
            n_q = self.n_q
            n_train = self.n_train
            dU_expanded = np.block(
                [
                    [dU, np.ones((n_q, n_train, 1))],
                    [np.ones((n_q, 1, n_train)), np.zeros((n_q, 1, 1))],
                ]
            )
            rhs_expanded = np.block([[rhs.T], [np.ones((n_q))]]).T
            return np.linalg.solve(dU_expanded, rhs_expanded)[:, :-1]
        return np.linalg.solve(dU, rhs)

    def emulate_wave_function(self, p):
        c = self.coefficients(p)
        j_q = self.j_q
        psi = j_q + np.einsum("kjr,kj->kr", self.chi_train, c)
        return psi

    def emulate_reactance(self, p):
        c = self.coefficients(p)
        j_q = self.j_q
        dU = self.compute_dU(p)
        V_free = self.V0_free + self.V1_free @ p
        rhs = self.rhs0 + self.rhs1 @ p
        K = -V_free + 2 * (rhs * c).sum(axis=-1) - np.einsum("qn,qnm,qm->q", c, dU, c)
        return -self.q_cm * K

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
        j_k = spherical_jn(n=self.ell, z=r * k[:, None])
        j_q = spherical_jn(n=self.ell, z=r * q_cm[:, None])
        j_G0_K = np.einsum("kr,qk->qr", j_k, G0_K)
        psi = j_q + j_G0_K

        if return_K:
            K_on = q_cm * np.einsum("ij,ij->i", Sp, K_half)
            return psi, K_on
        return psi


class AlternateSeparableKohnMixin:
    def fit(self, p_train):
        v_q = self.compute_v_on_shell()
        tau = np.array([self.compute_reactance_strength_matrix(p_i) for p_i in p_train])
        Lambda = np.array([self.compute_strength_matrix(p_i) for p_i in p_train])
        vGv = self.compute_vGv_matrix()
        self.n_train = n_train = len(p_train)
        # K_train = np.array([self.reactance(p_i, include_q=True) for p_i in p_train])

        n = self.n_form_factors
        c0 = np.zeros((self.n_q, n_train))
        c1 = np.zeros((self.n_q, n_train, self.n_p))
        C0 = np.zeros((self.n_q, n_train, n_train))
        C1 = np.zeros((self.n_q, n_train, n_train, self.n_p))

        c1_init = np.einsum("cq,iqcd,qda,bq->qiab", v_q, tau, vGv, v_q)
        C0 = np.einsum(
            "cq,iqcd,qda,jab,bq->qij", v_q, tau, vGv, Lambda, v_q
        ) + np.einsum(
            "cq,iqcd,qda,jab,qbe,jqef,fq->qij", v_q, tau, vGv, Lambda, vGv, tau, v_q
        )
        C1_init = -np.einsum(
            "cq,iqcd,qda,qbe,jqef,fq->qijab", v_q, tau, vGv, vGv, tau, v_q
        )

        upper_triangular_indices = [(xx, yy) for xx in range(n) for yy in range(xx, n)]
        param_idx = dict(
            zip(upper_triangular_indices, np.arange(len(upper_triangular_indices)))
        )

        # Turn the matrix quantities into vectors that one can take a dot product with a parameter vector
        for a, b in upper_triangular_indices:
            # The matrices are symmetric, take upper triangular and multiply off-diagonals by 2
            mult_ab = 1 if a == b else 2
            pp = param_idx[a, b]
            c1[..., pp] = mult_ab * c1_init[..., a, b]
            C1[..., pp] = mult_ab * C1_init[..., a, b]

        self.p_train = p_train
        self.c0 = c0
        self.c1 = c1
        self.C0 = C0
        self.C1 = C1

        V1_on_shell = []
        v_q = self.compute_v_on_shell()
        for i in range(self.n_form_factors):
            for j in range(i, self.n_form_factors):
                if i != j:
                    V1_on_shell.append(v_q[i] * v_q[j] + v_q[j] * v_q[i])
                else:
                    V1_on_shell.append(v_q[i] * v_q[j])
        V1_on_shell = np.array(V1_on_shell).T

        self.V0_on_shell = np.zeros(V1_on_shell.shape[:-1])
        self.V1_on_shell = V1_on_shell


class AlternateKohnYamaguchiEmulator(
    AlternateSeparableKohnMixin, SeparableYamaguchiMixin
):
    def __init__(self, beta, q_cm, nugget=0, hbar2_over_2mu=1):
        self.beta = beta
        self.hbar2_over_2mu = hbar2_over_2mu
        self.q_cm = q_cm
        self.nugget = nugget
        self.n_form_factors = len(beta)
        self.ell = 0
        self.n_q = len(q_cm)
        self.n_p = int(self.n_form_factors * (self.n_form_factors + 1) / 2)

    def coefficients(self, p):
        c = self.c0 + self.c1 @ p
        C = self.C0 + self.C1 @ p
        C = C + self.nugget * np.eye(C.shape[-1])
        return np.linalg.solve(C, c)

    def emulate_reactance(self, p):
        V_on_shell = self.V0_on_shell + self.V1_on_shell @ p
        coeff = self.coefficients(p)
        C = self.C0 + self.C1 @ p
        C = C + 2 * self.nugget * np.eye(C.shape[-1])
        K = V_on_shell + np.einsum("qi,qij,qj->q", coeff, C, coeff)
        K = K * self.q_cm * np.pi / 2
        return K

    def predict(self, p, full_space=False):
        if full_space:
            return self.reactance(p, include_q=True)
        else:
            return self.emulate_reactance(p)
