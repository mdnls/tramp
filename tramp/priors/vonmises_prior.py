import numpy as np
from .base_prior import Prior
from ..utils.misc import vonmises_v_to_b, rect_to_cpx, cpx_to_rect
from ..utils.integration import vonmises_measure
import scipy.special as sp


class VonMisesPrior(Prior):
    def __init__(self, size, b):
        '''
        Isotropic Von Mises prior with natural parameter b.

        Note: in general, complex natural parameters are represented as [2, ...] dimensional objects
        with two coordinates for real and imaginary parts.

        Parameters
        ----------
        size: dimension of the distribution as a tuple, where each coordinate is a complex number
        b: natural parameter for a single coordinate represented as a complex number
        '''
        self.size = size
        self.b = b
        self.repr_init()

        self.angle = np.abs(np.log(b))
        self.disperson = np.abs(b)

    def sample(self):
        return cpx_to_rect(np.exp(1j * np.random.vonmises(mu=self.angle, kappa=self.disperson, size=self.size)))

    def math(self):
        return r"\text{VM}"

    def second_moment(self):
        return 1

    def measure(self, f):
        return vonmises_measure(k=self.k, f=f)

    def compute_log_partition(self, ax, bx):
        def vonmises_partition(k):
            return np.sum(2 * np.pi * sp.i0(np.abs(k)))
        a = ax
        b = rect_to_cpx(bx) + self.b
        A_post = vonmises_partition(b)
        A_pri = vonmises_partition(self.b * np.ones_like(bx))
        A_gauss = 0.5 * a
        return A_post - A_pri - A_gauss

    def compute_forward_posterior(self, ax, bx):
        b = rect_to_cpx(bx) + self.b
        k = np.abs(b)

        rx_norm = sp.i1(k)/sp.i0(k)
        rx = cpx_to_rect(rx_norm * (b / np.abs(b)))
        vx = np.mean(0.5 * (1 - rx_norm**2))
        return rx, vx


    def compute_forward_message(self, ax, bx):
        k = np.abs(self.b)
        vx_new = np.mean(0.5 * (1 - (sp.i1(k) / sp.i0(k))**2 ))
        ax_new = 1 / vx_new
        bx_new = self.b * np.ones_like(rect_to_cpx(bx))
        return ax_new, cpx_to_rect(bx_new)

    def compute_forward_state_evolution(self, ax):
        k = np.abs(self.b)
        vx_new = np.mean(0.5 * (1 - (sp.i1(k) / sp.i0(k))**2))
        ax_new = 1 / vx_new
        return ax_new

    def beliefs_measure(self, ax, f):
        ''' From gauss bernoulli. Also see the replica example from Antoine's notes.
        mu_0 = gaussian_measure(0, np.sqrt(ax), f)
        mu_1 = gaussian_measure(
            ax * self.mean, np.sqrt(ax + (ax**2) * self.var), f
        )
        mu = (1 - self.rho) * mu_0 + self.rho * mu_1
        '''
        raise NotImplementedError


    def compute_forward_error(self, ax):
        raise NotImplementedError

    def compute_mutual_information(self, ax):
        raise NotImplementedError

    def compute_free_energy(self, ax):
        raise NotImplementedError
