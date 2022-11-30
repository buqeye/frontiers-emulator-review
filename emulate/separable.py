import numpy as np
from scipy.special import spherical_jn

from .constants import pi


class SeparableMixin:
    """A mixin class for emulators with separable potentials.

    This class does not function by itself, but is instead used for multiple inheritance.
    This provides the separable-potential methods needed by the subclass.

    Attributes
    ----------
    v_k
    n_form_factors
    n_q
    G0
    Sp
    """

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

    def compute_reactance_strength_matrix(self, p):
        strength = self.compute_strength_matrix(p)
        vGv = self.compute_vGv_matrix()
        Id = np.eye(self.n_form_factors)
        return np.linalg.solve(Id - strength @ vGv, strength[None, ...])

    def compute_v_on_shell(self):
        return np.einsum("nk,qk->nq", self.v_k, self.Sp)

    def compute_half_on_shell_reactance(self, p, include_q=True):
        tau = self.compute_reactance_strength_matrix(p)
        v_q_cm = self.compute_v_on_shell()
        vTv = np.einsum("nk,qnm,mq->kq", self.v_k, tau, v_q_cm)
        K_half = (np.pi / 2) * vTv
        if include_q:
            K_half *= self.q_cm
        return K_half.T

    def reactance(self, p, include_q=True):
        tau = self.compute_reactance_strength_matrix(p)
        v_q_cm = self.compute_v_on_shell()
        vTv = np.einsum("nq,qnm,mq->q", v_q_cm, tau, v_q_cm)
        K = (np.pi / 2) * vTv
        if include_q:
            K *= self.q_cm
        return K

    def validate_parameters(self, p):
        n_tri = int(self.n_form_factors * (self.n_form_factors + 1) / 2)
        if len(p) != n_tri:
            raise ValueError(
                f"The length of the parameters ({len(p)}) should be {n_tri} to fill the upper triangle of the strength matrix"
            )
        return self


class SeparableYamaguchiMixin(SeparableMixin):
    """

    Attributes
    ----------
    beta
    """

    def _compute_vGv(self, beta1, beta2):
        return vGv_yamaguchi(
            beta1, beta2, q_cm=self.q_cm, hbar2_over_2mu=self.hbar2_over_2mu
        )
        # q = self.q_cm
        # beta_sq = beta1 * beta2
        # beta = (beta1 + beta2) / 2.0
        # num = -np.pi * (beta_sq - q**2) / self.hbar2_over_2mu
        # den = 4 * beta * (beta1**2 + q**2) * (beta2**2 + q**2)
        # return num / den

    def compute_vGv_matrix(self):
        n_dim = self.n_form_factors
        vGv = np.zeros((self.n_q, n_dim, n_dim))
        for i in range(n_dim):
            for j in range(i, n_dim):
                vGv[:, i, j] = vGv[:, j, i] = self._compute_vGv(
                    self.beta[i], self.beta[j]
                )
        return vGv

    def compute_v(self, k):
        return np.array(
            [
                yamaguchi_form_factor_momentum_space(
                    k, beta, ell=self.ell, hbar2_over_2mu=self.hbar2_over_2mu
                )
                for beta in self.beta
            ]
        )

    def compute_v_on_shell(self):
        return self.compute_v(self.q_cm)

    def compute_half_on_shell_reactance(self, p, include_q=True):
        tau = self.compute_reactance_strength_matrix(p)
        v_q_cm = self.compute_v(self.q_cm)
        v_k = self.compute_v(self.k)
        vTv = np.einsum("nk,qnm,mq->kq", v_k, tau, v_q_cm)
        K_half = (np.pi / 2) * vTv
        if include_q:
            K_half *= self.q_cm
        return K_half.T

    def _predict_wave_function_s_wave(self, p, r):
        tau = self.compute_reactance_strength_matrix(p)
        q = self.q_cm[:, None]
        v_q = self.compute_v_on_shell()
        j0 = spherical_jn(0, r * q)
        chi = np.zeros_like(j0, dtype=np.float64)
        for i in range(self.n_form_factors):
            for j in range(self.n_form_factors):
                K_ij = np.pi / 2 * (v_q[i] * tau[:, i, j] * v_q[j])[:, None]
                chi -= K_ij * (np.cos(q * r) - np.exp(-self.beta[i] * r)) / r
        return j0 + chi


def vGv_yamaguchi(beta1, beta2, q_cm, hbar2_over_2mu):
    q = q_cm
    beta_sq = beta1 * beta2
    beta = (beta1 + beta2) / 2.0
    num = -np.pi * (beta_sq - q**2) / hbar2_over_2mu
    den = 4 * beta * (beta1**2 + q**2) * (beta2**2 + q**2)
    return num / den


def yamaguchi_form_factor_momentum_space(k, beta, ell=0, hbar2_over_2mu=1):
    """

    Comes from Eq (27) in Ref [1].

    References
    ----------
    [1] Momentum-Space Probability Density of 6He in Halo Effective Field Theory
        https://doi.org/10.1007/s00601-019-1528-6
    """
    return hbar2_over_2mu ** (-0.5) * k**ell / (beta**2 + k**2) ** (ell + 1)


def yamaguchi_form_factor_position_space(r, beta, ell=0, hbar2_over_2mu=1):
    """

    The fourier transform of Eq (27) in Ref [1].

    References
    ----------
    [1] Momentum-Space Probability Density of 6He in Halo Effective Field Theory
        https://doi.org/10.1007/s00601-019-1528-6
    """
    from scipy.special import gamma

    return (
        hbar2_over_2mu ** (-0.5)
        * np.sqrt(pi / 2)
        * np.exp(-beta * r)
        * r ** (ell - 1)
        / (gamma(ell + 1) * 2**ell)
    )


def yamaguchi_scattering_amplitude(
    q_cm, beta, strength, include_q=True, hbar2_over_2mu=1
):
    pre = 4 * beta * strength * (pi / 2) / hbar2_over_2mu
    if include_q:
        pre = pre * q_cm
    return pre / (
        4 * beta * (beta**2 + q_cm**2) ** 2
        + pi * strength * (beta**2 - q_cm**2)
    )


def yamaguchi_radial_wave_function(r, q_cm, beta, strength, hbar2_over_2mu=1):
    from scipy.special import spherical_jn

    j_ell = spherical_jn(0, r * q_cm)
    pre = (2 * pi * beta * strength / hbar2_over_2mu) / (
        4 * beta * (beta**2 + q_cm**2) ** 2
        + pi * strength * (beta**2 - q_cm**2)
    )
    psi = j_ell - pre * (np.cos(q_cm * r) - np.exp(-beta * r)) / r
    return psi


# def yamaguchi_radial_wave_function_rank_n(r, q_cm, betas, strength, hbar2_over_2mu=1):
#     from scipy.special import spherical_jn

#     j_ell = spherical_jn(0, r * q_cm)
#     pre = (2 * pi * beta * strength) / (
#         4 * beta * (beta**2 + q_cm**2) ** 2
#         + pi * strength * (beta**2 - q_cm**2)
#     )
#     psi = j_ell - pre * (np.cos(q_cm * r) - np.exp(-beta * r)) / r
#     return psi
