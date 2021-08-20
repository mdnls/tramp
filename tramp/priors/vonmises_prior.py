import numpy as np
from .base_prior import Prior
from ..utils.truncated_normal import log_Phi
from ..utils.misc import phi_1


class VonMisesPrior(Prior):
    def __init__(self, size, k):
        '''
        Isotropic Von Mises prior with natural parameter k.

        Parameters
        ----------
        size: dimension of the distribution as a tuple, where each coordinate is a complex number
        k: natural parameter for a single coordinate represented as a complex number
        '''
        self.size = size
        self.k = k
        self.repr_init()

        self.angle = np.abs(np.log(k))
        self.disperson = np.abs(k)

    def sample(self):
        return np.exp(1j * np.random.vonmises(mu=self.angle, kappa=self.disperson, size=self.size))

    def math(self):
        return r"\text{VM}"

    def second_moment(self):
        return 1

    def measure(self, f):
        return exponential_measure(m=self.mean)

    def compute_log_partition(self, ax, bx):
        a = ax
        b = bx + self.b
        A = b**2/(2*a) + 1/2 * np.log(2*pi/a)
        z_pos = b / np.sqrt(a)
        A_pos = A + log_Phi(z_pos)
        A_z = -np.log(-self.b)
        logZ_i = A_pos - A_z
        logZ = np.sum(logZ_i)
        return logZ

    def compute_forward_posterior(self, ax, bx):
        a = ax
        b = bx + self.b
        z_pos = b / np.sqrt(a)
        # Use phi_1(x) = x + N(x) / Phi(x)
        pdf_cdf = phi_1(z_pos)-z_pos
        rx = 1/np.sqrt(a) * (z_pos + pdf_cdf)
        v = 1/a * (
            1 - z_pos * pdf_cdf - pdf_cdf**2)
        vx = np.mean(v)
        return rx, vx

    def beliefs_measure(self, ax, f):
        raise NotImplementedError
