import numpy as np
from scipy.integrate import quad, dblquad
import scipy.special
from numpy.linalg import cholesky
from .misc import norm_pdf
import logging
logger = logging.getLogger(__name__)


def is_pos_def(x):
    return np.all(np.linalg.eigvalsh(x) > 0)

def vonmises_measure(k, f):
    """Computes one-dimensional integral against the Von Mises distribution.

    Parameters
    ----------
    - k : complex number representing the natural parameter of the Von Mises
    - f : function (R -> R) to integrate

    Returns
    -------
    - integral of VM(x | k) f(x)
    """
    disp = np.abs(k)
    theta = np.abs(np.log(k / np.abs(k)))
    Z = 2 * np.pi * scipy.special.i0(disp)
    pdf = lambda x: np.exp(disp * np.cos(x - theta)) / Z

    def integrand(x):
        return f(np.exp(1j * x)) * pdf(x)

    return quad(integrand, -np.pi, np.pi)[0]

def gaussian_measure(m, s, f):
    """Computes one-dimensional gaussian integral.

    Parameters
    ----------
    - m, s : mean and std of gaussian measure
    - f : function (R -> R) to integrate

    Returns
    -------
    - integral of N(x | m, s) f(x)
    """
    def integrand(x):
        return norm_pdf(x) * f(m + s * x)
    integral = quad(integrand, -10, 10)[0]
    return integral


def gaussian_measure_2d(m1, s1, m2, s2, f):
    """Computes two-dimensional gaussian integral.

    Parameters
    ----------
    - m1, s1 : mean and std of gaussian measure 1st dimension
    - m2, s2 : mean and std of gaussian measure 2nd dimension
    - f : function (R^2 -> R) to integrate

    Returns
    -------
    - integral of N(x1 | m1, s1) N(x2 | m2, s2) f(x1, x2)
    """
    def integrand(x2, x1):
        return norm_pdf(x1) * norm_pdf(x2) * f(m1 + s1 * x1, m2 + s2 * x2)
    integral = dblquad(integrand, -10, 10, -10, 10)[0]
    return integral

def cpx_gaussian_measure_2d(m1, s1, m2, s2, f):
    """Computes two-dimensional gaussian integral of complex valued function.

    Parameters
    ----------
    - m1, s1 : real mean and std of gaussian measure 1st dimension
    - m2, s2 : real mean and std of gaussian measure 2nd dimension
    - f : function (R^2 -> C) to integrate

    Returns
    -------
    - integral of N(x1 | m1, s1) N(x2 | m2, s2) f(x1, x2) in rectangular (ie (real, imag) format)
    """
    f_re = lambda *x: np.real(f(*x))
    f_im = lambda *x: np.imag(f(*x))

    return gaussian_measure_2d(m1, s1, m2, s2, f_re) + 1j * gaussian_measure_2d(m1, s1, m2, s2, f_im)

def gaussian_measure_2d_full(cov, mean, f):
    """Computes 2-dimensional gaussian integral (full covariance).

    Parameters
    ----------
    - mean : float or array of size 2
        mean vector
    - cov : array of shape (2, 2)
        full covariance matrix
    - f : function (R^2 -> R) to integrate

    Returns
    -------
    - integral of N(x1, x2 | m, cov) f(x1, x2)
    """
    if not is_pos_def(cov):
        logger.warn(f"cov={cov} not positive definite")
    L = cholesky(cov)

    def integrand(x2, x1):
        y1, y2 = L @ [x1, x2] + mean
        return norm_pdf(x1) * norm_pdf(x2) * f(y1, y2)
    integral = dblquad(integrand, -10, 10, -10, 10)[0]
    return integral


def exponential_measure(m, f):
    """Computes one-dimensional exponential integral.

    Parameters
    ----------
    - m : mean of exponential measure
    - f : function (R -> R) to integrate

    Returns
    -------
    - integral of lambd*exp(-lambd*x) f(x)
    """
    def integrand(x):
        return 1/m * exp(-x/m) * f(x)
    integral = quad(integrand, 0, 10)[0]
    return integral
