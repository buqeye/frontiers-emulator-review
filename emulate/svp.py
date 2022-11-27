import numpy as np

from .utils import greens_function_free_space
from .utils import cubic_spline_matrix
from .utils import BoundaryCondition
from .separable import SeparableMixin, SeparableYamaguchiMixin


class BaseSchwingerEmulator:
    def __init__(
        self,
        V0,
        V1,
        k,
        dk,
        q_cm,
        nugget=0,
        is_mesh_semi_infinite=True,
    ):
        # self.r = r
        # self.dr = dr
        self.k = k
        self.dk = dk
        self.V0 = V0
        self.V1 = V1
        self.q_cm = q_cm
        self.nugget = nugget
        self.n_p = V1.shape[-1]
        self.n_q = len(q_cm)
        # self.use_lagrange_multiplier = use_lagrange_multiplier
        # self.is_local = is_local

        self.dU0 = None
        self.dU1 = None
        self.p_train = None
        self.psi_train = None

        # In Landau's QM text, it is recommended to create an (n_k+1, n_k+1) matrix where the
        # extra element holds the on-shell part. This works fine, but is annoying because a new
        # matrix must be created for every on-shell piece you want to compute. Instead, we will
        # only create one set of (n_k, n_k) matrices, then upon solving the LS equation for the
        # off-shell reactance matrix, we will interpolate to the on-shell piece via this spline
        # matrix, which is only computed once and stored.
        Sp = cubic_spline_matrix(k, q_cm)

        boundary_condition = BoundaryCondition.STANDING
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
        self.G0 = G0
        self.Sp = Sp

    def predict_wave_function(self, p):
        raise NotImplementedError

    def fit(self, p_train):
        chi_train = []
        for p in p_train:
            chi_i = self.predict_wave_function(p)
            chi_train.append(chi_i)
        chi_train = np.array(chi_train).transpose(1, 0, 2)
        n_train = len(p_train)

        n_q = self.n_q
        n_p = self.n_p
        V0 = self.V0
        V1 = self.V1
        G0 = self.G0
        phi = Sp = self.Sp
        psi_train = phi[:, None, :] + chi_train
        # print("Starting matrix mult")

        self.w0 = np.einsum("qni,ij,qj->qn", psi_train, V0, phi)
        self.w1 = np.einsum("qni,ijp,qj->qnp", psi_train, V1, phi)

        # print("More..")

        G0_V0 = G0[..., None] * V0
        V0_G0_V0 = V0 @ G0_V0
        # V1_G0_V0 = np.einsum("ijp,qjk->qijp", V1, G0_V0)

        V0_psi = np.einsum("ij,qnj->qni", V0, psi_train)
        V1_psi = np.einsum("ijp,qnj->qnip", V1, psi_train)

        # self.W0 = np.einsum("qni,ij,qmj->qnm", psi_train, V0_psi) - np.einsum(
        #     "qni,ij,qjk,qmk->qnm", psi_train, V0, G0_V0, psi_train
        # )
        # self.W1 = (
        #     np.einsum("qni,ijp,qmj->qnmp", psi_train, V1, psi_train)
        #     - np.einsum("qni,ij,qj,jkp,qmk->qnmp", psi_train, V0, G0, V1, psi_train)
        #     - np.einsum("qni,ijp,qjk,qmk->qnmp", psi_train, V1, G0_V0, psi_train)
        # )

        self.W0 = np.einsum("qni,qmi->qnm", psi_train, V0_psi) - np.einsum(
            "qni,qik,qmk->qnm", psi_train, V0_G0_V0, psi_train
        )
        # self.W1 = np.einsum("qni,qmip->qnmp", psi_train, V1_psi) - 2 * np.einsum(
        #     "qnip,qik,qmk->qnmp", V1_psi, G0_V0, psi_train
        # )
        self.W1 = (
            np.einsum("qni,qmip->qnmp", psi_train, V1_psi)
            - np.einsum("qnip,qik,qmk->qnmp", V1_psi, G0_V0, psi_train)
            - np.einsum("qni,qji,qmjp->qnmp", psi_train, G0_V0, V1_psi)
        )
        # print("Final")

        self.W2 = -np.einsum("qnip,qi,qmib->qnmpb", V1_psi, G0, V1_psi)
        # self.W2 = -np.einsum(
        #     "qni,ijp,qj,jkb,qmk->qnmpb", psi_train, V1, G0, V1, psi_train
        # )

        # print("Done")
        self.n_train = n_train
        return self

    def coefficients(self, p):
        w = self.w0 + self.w1 @ p
        W = self.W0 + self.W1 @ p + (self.W2 @ p) @ p
        W += self.nugget * np.eye(W.shape[-1])
        return np.linalg.solve(W, w)

    def emulate_wave_function(self, p):
        pass

    def emulate_reactance(self, p):
        w = self.w0 + self.w1 @ p
        W = self.W0 + self.W1 @ p + (self.W2 @ p) @ p
        W = W + 2 * self.nugget * np.eye(W.shape[-1])
        c = self.coefficients(p)
        # K = 2 * np.sum(c * w, axis=-1) - np.einsum("qi,qij,qj->q", c, W, c)
        # K = np.sum(c * w, axis=-1)
        K = np.einsum("qi,qij,qj->q", c, W, c)
        return self.q_cm * K * np.pi / 2

    def predict(self, p, full_space=False):
        if full_space:
            pass
        else:
            return self.emulate_reactance(p)


class SchwingerLSEmulator(BaseSchwingerEmulator):
    def __init__(self) -> None:
        super().__init__()

    def predict_wave_function(self, p):
        return super().predict_wave_function(p)


class SchwingerSeparableMixin:
    def fit(self, p_train):
        """

        Note that the this was not written or tested for either speed or memory efficiency.
        Cases with high-rank potentials or a large number of training points could be inefficient.
        There are likely better ways to write it in these cases, but it works well enough here.
        """
        v_q = self.compute_v_on_shell()
        tau = np.array([self.compute_reactance_strength_matrix(p_i) for p_i in p_train])
        vGv = self.compute_vGv_matrix()
        n_train = len(p_train)

        n = self.n_form_factors
        w0 = np.zeros((self.n_q, n_train))
        w1 = np.zeros((self.n_q, n_train, self.n_p))
        W0 = np.zeros((self.n_q, n_train, n_train))
        W1 = np.zeros((self.n_q, n_train, n_train, self.n_p))
        W2 = np.zeros((self.n_q, n_train, n_train, self.n_p, self.n_p))

        w1_init = np.einsum("aq,bq->qab", v_q, v_q)[:, None, :, :] + np.einsum(
            "cq,iqcd,qda,bq->qiab", v_q, tau, vGv, v_q
        )
        W1_init = (
            np.einsum("aq,bq->qab", v_q, v_q)[:, None, None, :, :]
            + np.einsum("aq,qbc,jqcd,dq->qjab", v_q, vGv, tau, v_q)[:, None, :, :, :]
            + np.einsum("cq,iqcd,qda,bq->qiab", v_q, tau, vGv, v_q)[:, :, None, :, :]
            + np.einsum("cq,iqcd,qda,qbe,jqef,fq->qijab", v_q, tau, vGv, vGv, tau, v_q)
        )
        W2_init = -(
            np.einsum("aq,qbx,yq->qabxy", v_q, vGv, v_q)[:, None, None, :, :, :, :]
            + np.einsum("aq,qbx,qyc,jqcd,dq->qjabxy", v_q, vGv, vGv, tau, v_q)[
                :, None, :, :, :, :, :
            ]
            + np.einsum("cq,iqcd,qda,qbx,yq->qiabxy", v_q, tau, vGv, vGv, v_q)[
                :, :, None, :, :, :, :
            ]
            + np.einsum(
                "cq,iqcd,qda,qbx,qye,jqef,fq->qijabxy",
                v_q,
                tau,
                vGv,
                vGv,
                vGv,
                tau,
                v_q,
            )
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
            w1[..., pp] = mult_ab * w1_init[..., a, b]
            W1[..., pp] = mult_ab * W1_init[..., a, b]

            for a2, b2 in upper_triangular_indices:
                mult_a2b2 = 1 if a2 == b2 else 2
                pp2 = param_idx[a2, b2]
                W2[..., pp, pp2] = mult_ab * mult_a2b2 * W2_init[..., a, b, a2, b2]

        self.p_train = p_train
        self.n_train = n_train
        # self.K_train = K_train
        self.w0 = w0
        self.w1 = w1
        self.W0 = W0
        self.W1 = W1
        self.W2 = W2


class SchwingerSeparableEmulator(
    SeparableMixin, SchwingerSeparableMixin, BaseSchwingerEmulator
):
    def __init__(
        self,
        v_k,
        k,
        dk,
        q_cm,
        nugget=0,
        is_mesh_semi_infinite=True,
    ) -> None:
        n_form_factors = len(v_k)
        V1 = []
        for i in range(n_form_factors):
            for j in range(i, n_form_factors):
                if i != j:
                    V1.append(v_k[i][:, None] * v_k[j] + v_k[j][:, None] * v_k[i])
                else:
                    V1.append(v_k[i][:, None] * v_k[j])

        V1 = np.dstack(V1)
        V0 = np.zeros(V1.shape[:-1])

        self.v_k = v_k
        self.n_form_factors = n_form_factors
        super().__init__(
            V0,
            V1,
            k,
            dk,
            q_cm,
            nugget=nugget,
            is_mesh_semi_infinite=is_mesh_semi_infinite,
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

    # def predict_wave_function(self, p):
    #     self.validate_parameters(p)
    #     K_half = self.compute_half_on_shell_reactance(p, include_q=False)
    #     scattered_wf = (2 / np.pi) * self.G0 * K_half
    #     return scattered_wf


class SchwingerYamaguchiEmulator(
    SeparableYamaguchiMixin, SchwingerSeparableMixin, BaseSchwingerEmulator
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
        # self.boundary_condition = BoundaryCondition.STANDING
        # self.is_coupled = False
