import numpy as np

from .utils import greens_function_free_space
from .utils import cubic_spline_matrix
from .utils import BoundaryCondition


class BaseSchwingerEmulator:
    def __init__(
        self,
        V0,
        V1,
        # r,
        # dr,
        k,
        dk,
        q_cm,
        is_local,
        nugget=0,
        use_lagrange_multiplier=False,
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
        self.use_lagrange_multiplier = use_lagrange_multiplier
        self.is_local = is_local

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

    def predict_wave_function(self, p):
        raise NotImplementedError

    def fit(self, p_train):
        pass

    def coefficients(self, p):
        pass

    def emulate_wave_function(self, p):
        pass

    def emulate_reactance(self, p):
        pass

    def predict(self, p, full_space=False):
        pass


class SchwingerLSEmulator(BaseSchwingerEmulator):
    def __init__(self) -> None:
        super().__init__()

    def predict_wave_function(self, p):
        return super().predict_wave_function(p)


class SchwingerSeparableEmulator(BaseSchwingerEmulator):
    def __init__(self) -> None:
        super().__init__()

    def predict_wave_function(self, p):
        return super().predict_wave_function(p)
