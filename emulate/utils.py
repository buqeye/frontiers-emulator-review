import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_genlaguerre, gammaln
from scipy.integrate import quadrature

from .constants import hbar_c, pi

def markdown_class_method(cls, method):
    import inspect

    method_lines = inspect.getsource(getattr(cls, method))
    return (
        "```python\n"
        f"class {cls.__name__}:\n"
        "...\n"
        f"{method_lines}...\n"
        "```"
    )


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
    y2 = y ** 2
    laguerre = eval_genlaguerre(n - 1, ell + 0.5, y2)
    return norm * y ** (ell + 1) * np.exp(-y2 / 2) * laguerre


def convert_from_r_to_ho_basis(v, n_max, ell, r, dr, b):
    wfs_ho = np.stack([ho_radial_wf(r, n=i+1, ell=ell, b=b) for i in range(n_max+1)])
    out = np.zeros((n_max + 1, n_max + 1))
    for n in range(n_max + 1):
        u_nl = wfs_ho[n]
        for m in range(n, n_max + 1):
            u_ml = wfs_ho[m]
            out[n, m] = out[m, n] = np.sum(v * u_nl * u_ml * dr)
    return out


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
