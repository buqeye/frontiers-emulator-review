from typing import Union, Optional, Dict, List
import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.linalg import eigh, eigvalsh
from scipy.special import eval_genlaguerre, gammaln
from scipy.integrate import quadrature

from .constants import hbar_c, pi
from .types import BoundaryCondition, QuadratureType


def markdown_class_method(cls, method):
    import inspect

    method_lines = inspect.getsource(getattr(cls, method))
    return f"```python\nclass {cls.__name__}:\n...\n{method_lines}...\n```"


def jupyter_show_class_method(cls, method):
    from IPython.display import display, Markdown

    return display(Markdown(markdown_class_method(cls, method)))


def ho_energy(n, ell, omega):
    R"""The energy of the harmonic oscillator

    Note that N = 2 (n - 1) + ell.

    Parameters
    ----------
    n
    ell
    omega

    Returns
    -------

    """
    return omega * (2 * (n - 1) + ell + 3 / 2)


def ho_radial_wf(r, n, ell, b):
    r"""The radial wave function u_{nl} for the 3d isotropic harmonic oscillator.

    These are normalized such that \int |u_nl(r)|**2 dr = 1

    Parameters
    ----------
    r :
        The distance in fm
    n :
        The n quantum number
    ell :
        The angular momentum quantum number
    b :
        The oscillator parameter

    Returns
    -------
    u_nl
    """
    # b = 1 / np.sqrt(mass * omega / hbar_c)
    # N_{nl} = 2 Gamma(n) / [b * Gamma(n + l + 1/2)]
    norm = np.sqrt(2 * np.exp(gammaln(n) - np.log(b) - gammaln(n + ell + 0.5)))
    y = r / b
    y2 = y**2
    laguerre = eval_genlaguerre(n - 1, ell + 0.5, y2)
    return norm * y ** (ell + 1) * np.exp(-y2 / 2) * laguerre


def convert_from_r_to_ho_basis(v, n_max, ell, r, dr, b):
    wfs_ho = np.stack(
        [ho_radial_wf(r, n=i + 1, ell=ell, b=b) for i in range(n_max + 1)]
    )
    out = np.zeros((n_max + 1, n_max + 1))
    for n in range(n_max + 1):
        u_nl = wfs_ho[n]
        for m in range(n, n_max + 1):
            u_ml = wfs_ho[m]
            out[n, m] = out[m, n] = np.sum(v * u_nl * u_ml * dr)
    return out


def fourier_transform_partial_wave(f, r, dr, k, ell):
    R"""Fourier transform the radial component of a partial wave expansion.

    Uses

    .. math::

        \langle r | k\ell m\rangle = i^\ell j_\ell(kr) Y_{\ell m}(\Omega_r)

    With the symmetric convention where each transform has a factor of :math:`(2 \pi)^{3/2}` in the denominator.

    Parameters
    ----------
    f :
        The function to transform
    r :
        The radial variable that f depends on (eg, radial position)
    dr :
        The integration weights, eg, from quadrature
    k :
        The conjugate radial variable that the transformed function depends on (eg, radial momentum)
    ell :
        The partial wave

    Returns
    -------
    The fourier transformed function
    """
    from scipy.special import spherical_jn

    j_l = spherical_jn(ell, np.outer(k, r))
    f_k = j_l @ (r**2 * dr * f)
    f_k *= np.sqrt(2 / pi)  # Comes from: 4 pi / (2 pi)^{3/2}
    return np.squeeze(f_k)


def gaussian_radial_fourier_transform(x, a):
    """The radial fourier transform of a Gaussian exp(-a r^2)"""
    return np.sqrt(2 / pi) * np.exp(-(x**2) / (4.0 * a)) * np.sqrt(pi / a) / (4 * a)


def yamaguchi_form_factor_momentum_space(k, beta, ell=0):
    """

    Comes from Eq (27) in Ref [1].

    References
    ----------
    [1] Momentum-Space Probability Density of 6He in Halo Effective Field Theory
        https://doi.org/10.1007/s00601-019-1528-6
    """
    return k**ell / (beta**2 + k**2) ** (ell + 1)


def yamaguchi_form_factor_position_space(r, beta, ell=0):
    """

    The fourier transform of Eq (27) in Ref [1].

    References
    ----------
    [1] Momentum-Space Probability Density of 6He in Halo Effective Field Theory
        https://doi.org/10.1007/s00601-019-1528-6
    """
    from scipy.special import gamma

    return (
        np.sqrt(pi / 2)
        * np.exp(-beta * r)
        * r ** (ell - 1)
        / (gamma(ell + 1) * 2**ell)
    )


def schrodinger_residual(psi, V, r, dr, q_cm, ell):
    u = r * q_cm * psi
    d_u = np.gradient(u, r, axis=-1)
    d2_u = np.gradient(d_u, r, axis=-1)
    angular = ell * (ell + 1) / r**2
    return -d2_u + angular * u + ((r * dr * u) @ V) - q_cm**2 * u


def yamaguchi_scattering_amplitude(q_cm, beta, strength, include_q=True):
    pre = 4 * beta * strength * (pi / 2)
    if include_q:
        pre = pre * q_cm
    return pre / (
        pi * strength * (beta**2 - q_cm**2)
        + 4 * beta * (beta**2 + q_cm**2) ** 2
    )


def yamaguchi_radial_wave_function(r, q_cm, beta, strength):
    from scipy.special import spherical_jn

    # f = yamaguchi_scattering_amplitude(
    #     q_cm=q_cm, beta=beta, strength=strength, include_q=True
    # )
    # j_ell = spherical_jn(0, r * q_cm)
    # psi = j_ell + f * (j_ell - np.exp(-beta * r)) / r
    # return psi

    # f = yamaguchi_scattering_amplitude(
    #     q_cm=q_cm, beta=beta, strength=strength, include_q=False
    # )
    # print(f)
    # j_ell = spherical_jn(0, r * q_cm)
    # pre = 1
    # # pre = 2 / pi
    # # pre = pi / 2
    # psi = j_ell + pre * f * (j_ell - np.exp(-beta * r)) / r

    j_ell = spherical_jn(0, r * q_cm)
    pre = (2 * pi * beta * strength) / (
        4 * beta * (beta**2 + q_cm**2) ** 2
        + pi * strength * (beta**2 - q_cm**2)
    )
    # pre *= pi / 2
    psi = j_ell - pre * (np.cos(q_cm * r) - np.exp(-beta * r)) / r
    return psi


def greens_function_free_space(
    k,
    dk,
    q_cm,
    spline,
    boundary_condition: BoundaryCondition,
    k_cut: Optional[float] = None,
    is_compressed: bool = True,
):
    r"""Computes the partial-wave Green's function for free-space scattering.

    This is not really G_0, because it contains the integration measure dk and k^2.
    It also includes the subtraction of the 0 integral for numerical stability of the principle value.
    The factor of 2*mu is not included either, it is instead expected to be a part of the potential and K.
    This is still convenient to treat as G_0, because this is how G_0 acts between two partial wave matrices.

    Parameters
    ----------
    k : shape = (n_k,)
        The momentum grid in inverse fermi
    dk : shape = (n_k,)
        The integration measure
    q_cm : shape = (n_q_cm,)
        The on-shell center-of-mass momenta
    spline : shape = (n_q_cm, n_k)
        The interpolation matrix that maps from k -> q_cm
    k_cut :
        The cutoff of the momentum grid. It will be chosen automatically if omitted.
    boundary_condition :
        Whether the Green's function represents incoming, outgoing, or standing boundary conditions.
    is_compressed :
        Whether the shape of the output should be compressed to shape (n_q_cm, n_k)
        or a matrix with shape (n_q_cm, n_k, n_k), with zeros everywhere except the diagonal.
        Defaults to True, which compresses the output.

    Returns
    -------
    G0 : shape = (n_q_cm, n_k) or (n_q_cm, n_k, n_k)
        The free-space Green's function.
    """
    n_k = len(k)
    n_q = len(q_cm)
    if k_cut is None:
        # this is usually good for Gauss-Legendre quadrature rules
        k_cut = k[-1] + 0.25 * (k[-1] - k[-2])

    if boundary_condition is BoundaryCondition.STANDING:
        bc_term = 0.0
        dtype = float
    else:
        if boundary_condition is BoundaryCondition.OUTGOING:
            sgn = +1
        elif boundary_condition is BoundaryCondition.INCOMING:
            sgn = -1
        else:
            raise ValueError(
                "Boundary condition must be standing, incoming, or outgoing"
            )
        bc_term = sgn * 1j * pi / 2.0
        dtype = "complex128"

    G0 = np.zeros((n_q, n_k), dtype)
    for i, p in enumerate(q_cm):
        if p > 0:
            G0[i] = k**2 * dk / (p**2 - k**2)
            G0[i] -= (
                (p**2 * np.sum(dk / (p**2 - k**2))) + (p * bc_term)
            ) * spline[i]
            if k_cut > 0 and np.isfinite(k_cut):
                G0[i] -= 0.5 * p * np.log(np.abs(k_cut - p) / (k_cut + p)) * spline[i]
        elif p == 0:
            G0[i] = -dk
        else:
            G0[i] = -(k**2) * dk / (p**2 + k**2)

    if is_compressed:
        return G0
    else:
        return np.stack([np.diag(G0_i) for G0_i in G0], axis=0)


def greens_function_free_space_squared(
    k,
    dk,
    q_cm,
    spline,
    k_cut: Optional[float] = None,
):
    r"""Computes the squared partial-wave Green's function for free-space scattering.

    This is not really G_0, because it contains the integration measure dk.
    It also includes the subtraction of the 0 integral for numerical stability of the principle value.
    The factor of 2*mu is not included either, it is instead expected to be a part of the potential and K.
    This is still convenient to treat as G_0, because this is how G_0 acts between two partial wave matrices.

    Parameters
    ----------
    k : shape = (n_k,)
        The momentum grid in inverse fermi
    dk : shape = (n_k,)
        The integration measure
    q_cm : shape = (n_q_cm,)
        The on-shell center-of-mass momenta
    spline : shape = (n_q_cm, n_k)
        The interpolation matrix that maps from k -> q_cm
    k_cut :
        The cutoff of the momentum grid. It will be chosen automatically if omitted.

    Returns
    -------
    G0 : shape = (n_q_cm, n_k) or (n_q_cm, n_k, n_k)
        The free-space Green's function.
    """
    n_k = len(k)
    n_q = len(q_cm)
    if k_cut is None:
        # this is usually good for Gauss-Legendre quadrature rules
        k_cut = k[-1] + 0.25 * (k[-1] - k[-2])

    G0_sq = np.zeros((n_q, n_k), float)
    for i, p in enumerate(q_cm):
        if p > 0:
            G0_sq[i] = -2 * dk / (p**2 - k**2)
            G0_sq[i] += 2 * np.sum(dk / (p**2 - k**2)) * spline[i]

            # no singularity for this term
            G0_sq[i] += dk / (p + k) ** 2

            # This term has a singularity. Add and subtract the same quantity at the singular point
            G0_sq[i] += dk / (p - k) ** 2
            # Subtract discrete form
            G0_sq[i] -= np.sum(dk / (p - k) ** 2) * spline[i]
            # Add exact form
            # The integral of 1 / (p - k + ie)^2 from 0 to inf equals -1/p
            G0_sq[i] += -1 / p * spline[i]
            if k_cut > 0 and np.isfinite(k_cut):
                G0_sq[i] += -1 / p * np.log(np.abs(k_cut - p) / (k_cut + p)) * spline[i]
        elif p == 0:
            G0_sq[i] = -dk
        else:
            # TODO: Verify this.
            G0_sq[i] = 2 * dk / (p**2 + k**2)

    G0_sq = G0_sq / 4.0
    return G0_sq


def cubic_spline_matrix(old_mesh, new_mesh):
    r"""Computes a cubic spline matrix that only references the input and output locations, not the y values.

    This is useful because it can be computed once up front and stored, so long as the meshes remain constant.
    This code was originally written by Kyle Wendt.

    Parameters
    ----------
    old_mesh :
        The points where the function is already computed
    new_mesh :
        The points to interpolate towards

    Returns
    -------
    S : shape = (n_new, n_old)
        An interpolation matrix that will compute `f_new = S @ f_old`

    Notes
    -----
    This uses a technique called quasi-interpolation. See Ref [1]_.

    References
    ----------
    .. [1] Glöckle, W., Hasberg, G. & Neghabian, A.R.
       Numerical treatment of few body equations in momentum space by the Spline method.
       Z Physik A 305, 217–221 (1982). https://doi.org/10.1007/BF01417437
    """
    from numpy import zeros

    n = len(old_mesh)

    # All notation follows from the reference in the docstring.
    S = zeros((len(new_mesh), len(old_mesh)), float)

    B = zeros((n, n), float)
    A = zeros((n, n), float)
    C = zeros((n, n), float)
    h = zeros(n + 1, float)
    p = zeros(n, float)
    q = zeros(n, float)
    lam = zeros(n, float)
    mu = zeros(n, float)

    for i in range(1, n):
        h[i] = old_mesh[i] - old_mesh[i - 1]

    for i in range(1, n - 1):
        B[i, i] = -6.0 / (h[i] * h[i + 1])
    for i in range(1, n):
        B[i - 1, i] = 6.0 / ((h[i - 1] + h[i]) * h[i])
        B[i, i - 1] = 6.0 / ((h[i + 1] + h[i]) * h[i])

    for j in range(1, n):
        lam[j] = h[j + 1] / (h[j] + h[j + 1])
        mu[j] = 1.0 - lam[j]
        p[j] = mu[j] * q[j - 1] + 2.0
        q[j] = -lam[j] / p[j]
        A[j, :] = (B[j, :] - mu[j] * A[j - 1, :]) / p[j]

    for i in range(n - 2, -1, -1):
        C[i, :] = q[i] * C[i + 1, :] + A[i, :]

    imin = old_mesh.argmin()
    imax = old_mesh.argmax()
    xmin = old_mesh[imin]
    xmax = old_mesh[imax]
    for yi, y in enumerate(new_mesh):
        if y <= xmin:
            S[yi, :] = 0
            S[yi, imin] = 1.0
        elif y >= xmax:
            S[yi, :] = 0
            S[yi, imax] = 1.0
        else:
            j = 0
            while old_mesh[j + 1] < y:
                j += 1
            dx = y - old_mesh[j]
            S[yi, :] += dx * (
                -(h[j + 1] / 6.0) * (2.0 * C[j, :] + C[j + 1, :])
                + dx
                * (
                    0.5 * C[j, :]
                    + dx * (1.0 / (6.0 * h[j + 1])) * (C[j + 1, :] - C[j, :])
                )
            )
            S[yi, j] += 1.0 - dx / h[j + 1]
            S[yi, j + 1] += dx / h[j + 1]
    return S


def t_matrix_outgoing_to_standing(T):
    r"""Converts the outgoing on-shell T matrix to its standing wave (principal value) form, aka the K or R matrix.

    Parameters
    ----------
    T :
        The outgoing (+ie) on-shell T matrix

    Returns
    -------
    reactance :
        The K (or R) matrix.
    """
    return np.real(T / (1 - 1j * T))


def t_matrix_incoming_to_standing(T):
    r"""Converts the incoming on-shell T matrix to its standing wave (principal value) form, aka the K or R matrix.

    Parameters
    ----------
    T :
        The incoming (-ie) on-shell T matrix

    Returns
    -------
    reactance :
        The K (or R) matrix.
    """
    return np.real(T / (1 + 1j * T))


def fix_phases_continuity(phases, n0=None, is_radians=True):
    """Returns smooth phase shifts by removing jumps by multiples of pi.

    Parameters
    ----------
    phases : array, shape = (..., N)
        Phase shifts that vary as a function in their right-most length-N axis. arctan2 may
        have caused jumps by multiples of pi in this axis.
    n0 : int, optional
        If given, shifts the initial value of the smooth phases (phases[..., 0]) to be in
        the range (n0-1/2, n0+1/2) * pi. Else, the smooth phase is defined
        to leave phases[..., -1] fixed.
    is_radians : bool
        Expects phases to be in radians if True, otherwise degrees.

    Returns
    -------
    smooth_phases : array, shape = (..., N)
        Phase shifts with jumps of pi smoothed in the right-most axis.
    """
    from numpy import pi, round, zeros_like

    if is_radians:
        factor = pi
    else:
        factor = 180.0
    n = zeros_like(phases)
    # Find all jumps by multiples of pi.
    # Store cumulative number of jumps from beginning to end of phase array
    n[..., 1:] = (
        round((phases[..., 1:] - phases[..., :-1]) / factor).cumsum(-1) * factor
    )
    # Make the jumps be relative to the final value of the phase shift
    # i.e., don't adjust phases[..., -1]
    n -= n[..., [-1]]
    # Subtract away the jumps
    smooth_phases = phases.copy()
    smooth_phases[...] -= n
    if (
        n0 is not None
    ):  # If the initial (rather than final) value of phases is constrained
        # Now move the entire phase shift at once so it starts in the range (n0-1/2, n0+1/2) * pi.
        smooth_phases[...] -= (round(smooth_phases[..., 0] / factor) - n0) * factor
    return smooth_phases


def leggauss_shifted(deg, a=-1, b=1):
    """Obtain the Gaussian quadrature points and weights when the limits of integration are [a, b]

    Parameters
    ----------
    deg : int
        The degree of the quadrature
    a : float
        The lower limit of integration. Defaults to -1, the standard value.
    b : float
        The upper limit of integration. Defaults to +1, the standard value.

    Returns
    -------
    x : The integration locations
    w : The weights
    """
    x, w = leggauss(deg)
    w *= (b - a) / 2.0
    x = ((b - a) * x + (b + a)) / 2.0
    return x, w


class CompoundMesh:
    """A 1D quadrature mesh that concatenate various types of meshes.

    Each interval in this compound mesh could use a different number of points, and could even use
    different quadrature rules. Currently supported quadrature types are Gauss-Legendre (finite and semi-infinite)
    and Gauss-Lobatto. The QuadratureType enum is used to describe these types.

    Heavily inspired by Kyle Wendt's quadrature code.

    Parameters
    ----------
    nodes :
        The endpoints of the intervals where independent quadrature rules are applied.
    n_points :
        The number of points to be used in each interval. If two adjacent intervals both use Gauss-Lobatto quadrature
        then the meeting point of the two intervals will contain the sum of the weights from each interval.
    kinds :
        The kind of quadrature rule to apply in each interval. If this is omitted, then Gauss-Legendre rules
        are used in each interval. If len(nodes) == len(n_points), it is assumed that the final interval
        is semi-infinite.

    Attributes
    ----------
    x : array
        The computed quadrature locations.
    w : array
        The computed quadrature weights.
    is_semi_infinite : bool
        Whether the compound mesh is semi-infinite in extent.
    kinds :
        The (possibly inferred) quadrature rules in each interval.
    nodes :
        The nodes fed to the initializer.
    n_points :
        The n_points fed to the initializer.
    n_intervals :
        The number of intervals within which to compute quadrature points and weights.
    total_points : The total number of x locations and weights to be returned. If there are adjacent
        Gauss-Lobatto rules, then this will be less than the sum of n_points.
    """

    def __init__(
        self,
        nodes: List[int],
        n_points: List[int],
        kinds: Optional[List[QuadratureType]] = None,
    ):
        self.nodes = nodes
        self.n_points = n_points

        if kinds is None:
            # Default to Gauss-Legendre in each interval
            kinds = [QuadratureType.GaussLegendre for _ in self.n_points]
            # But if the final interval has no end point, assume it is semi-infinite
            if len(self.nodes) == len(self.n_points):
                kinds[-1] = QuadratureType.SemiInfinite

        for k in kinds[:-1]:
            if k == QuadratureType.SemiInfinite:
                raise ValueError("Only the final entry in kinds can be semi-infinite")

        if len(n_points) != len(kinds):
            raise ValueError("len(n_points) must be equal to len(kinds)")

        total_points = n_points[0]
        for i in range(1, len(n_points)):
            n_i = n_points[i]
            total_points += n_i
            if kinds[i] == kinds[i - 1] == QuadratureType.GaussLobatto:
                total_points -= 1

        self.kinds = kinds
        self.n_intervals = len(kinds)
        self.total_points = total_points
        self.is_semi_infinite = kinds[-1] == QuadratureType.SemiInfinite

        x, w = self.compute_mesh()
        self.x = x
        self.w = w

    def gauss_legendre_mesh(self, n, a, b):
        """The Gauss-Legendre points and weights for an integrand in [a, b].

        Parameters
        ----------
        n : int
            The number of points
        a : float
            The lower limit of integration
        b : float
            The upper limit of integration. Can be infinite.


        Returns
        -------
        x : The points
        w : The weights
        """
        from numpy.polynomial.legendre import leggauss

        x_orig, w_orig = leggauss(n)
        if np.isinf(b):
            x, w = self.semi_infinite_transform(x=x_orig, w=w_orig, a=a)
        else:
            x, w = self.linear_transform(x=x_orig, w=w_orig, a=a, b=b)
        return x, w

    def gauss_lobatto_mesh(self, n, a, b):
        """The Gauss-Lobatto points and weights for an integrand in [a, b].

        Parameters
        ----------
        n : int
            The number of points
        a : float
            The lower limit of integration
        b : float
            The upper limit of integration


        Returns
        -------
        x : The points
        w : The weights
        """
        x_orig, w_orig = self.gauss_lobatto_mesh_default(n)
        if np.isinf(b):
            raise ValueError("Semi-infinite mesh is not supported for Gauss-Lobatto")
        else:
            x, w = self.linear_transform(x=x_orig, w=w_orig, a=a, b=b)
        return x, w

    @staticmethod
    def gauss_lobatto_mesh_default(n):
        """The standard Gauss Lobatto mesh. From Kyle Wendt."""
        from numpy import arange, array, diag, ones_like, sqrt, stack, zeros

        if n < 2:
            raise ValueError("n must be > 1")
        if n == 2:
            xi = array((-1.0, +1.0))
            wi = array((+1.0, +1.0))
            return xi, wi

        xi = zeros(n)
        wi = zeros(n)
        # Pn = zeros(n)
        i = arange(1, n - 2)
        # coefficient for Jacobi Poly with a=b=1
        b = sqrt((i * (2.0 + i)) / (3.0 + 4.0 * i * (2.0 + i)))

        M = diag(b, -1) + diag(b, 1)
        xi[1 : n - 1] = eigvalsh(M)
        xi[0] = -1.0
        xi[-1] = 1.0

        Pim2 = ones_like(xi)  # P_{i-2}
        Pim1 = xi  # P_{i-1}
        for j in range(2, n):  # want P_{n-1}
            wi = (1.0 / j) * ((2 * j - 1) * xi * Pim1 - (j - 1) * Pim2)
            Pim2 = Pim1
            Pim1 = wi
        wi = 2.0 / (n * (n - 1) * wi**2)
        wi[0] = wi[-1] = 2.0 / (n * (n - 1))
        return xi, wi

    @staticmethod
    def linear_transform(x, w, a, b):
        """Convert from the standard [-1, 1] interval to [a, b]"""
        x_new = x * (b - a) / 2.0 + (a + b) / 2
        w_new = w * (b - a) / 2.0
        return x_new, w_new

    @staticmethod
    def semi_infinite_transform(x, w, a):
        """Convert from the standard [-1, 1] interval to [a, inf].

        Created with the u-substitution of u = a * ( 1 + (1+x) / (1-x) ).
        """
        x_new = a * (1.0 + (1.0 + x) / (1.0 - x))
        w_new = 2.0 * a * w / (1 - x) ** 2
        return x_new, w_new

    def compute_mesh(self):
        """Creates a compound mesh by appending the x and w from each interval."""
        x = np.zeros(self.total_points)
        w = np.zeros(self.total_points)
        kind_prev = None
        idx = 0

        for i in range(self.n_intervals):
            kind = self.kinds[i]
            n_pts = self.n_points[i]
            a = self.nodes[i]
            if kind == QuadratureType.SemiInfinite:
                b = np.inf
            else:
                b = self.nodes[i + 1]

            if kind in [QuadratureType.GaussLegendre, QuadratureType.SemiInfinite]:
                x_i, w_i = self.gauss_legendre_mesh(n=n_pts, a=a, b=b)
            elif kind == QuadratureType.GaussLobatto:
                x_i, w_i = self.gauss_lobatto_mesh(n=n_pts, a=a, b=b)
            else:
                raise ValueError(
                    "Only Gauss-Lagendre, semi-infinite Gauss-Legendre, and Gauss-Lobatto are supported."
                )

            if i != 0 and kind_prev == kind == QuadratureType.GaussLobatto:
                # Adjacent Gauss-Lobatto meshes share a common point.
                # Add to the existing weight index rather than duplicating the same x location at a new index.
                # This is not relevant for the first interval.
                n_pts = n_pts - 1
                w[idx - 1] += w_i[0]
                x_i = x_i[1:]
                w_i = w_i[1:]

            # Include this interval's points and weights in total array. Move on to next interval.
            idx_end = idx + n_pts
            x[idx:idx_end] = x_i
            w[idx:idx_end] = w_i
            idx = idx_end
            kind_prev = kind
        return x, w
