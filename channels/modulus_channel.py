import numpy as np
from ..base import Channel
from ..utils.misc import complex2array, array2complex
import logging


class ModulusChannel(Channel):
    """Modulus channel x = |z|.

    Notes
    -----
    For message passing it is more convenient to represent a complex array x
    as a real array X where X[0] = x.real and X[1] = x.imag

    In particular:
    - input of sample(): Z array of shape (2, z.shape)
    - message bz, posterior rz: real arrays of shape (2, z.shape)
    """

    def __init__(self):
        self.repr_init()

    def sample(self, Z):
        "We assume Z[0] = Z.real and Z[1] = Z.imag"
        Z = array2complex(Z)
        X = np.absolute(Z)
        return X

    def math(self):
        return r"$|\cdot|$"

    def second_moment(self, tau):
        return 2 * tau

    def compute_forward_posterior(self, az, bz, ax, bx):
        bz = array2complex(bz)
        raise NotImplementedError

    def compute_backward_posterior(self, az, bz, ax, bx):
        bz = array2complex(bz)
        raise NotImplementedError

    def beliefs_measure(self, az, ax, tau, f):
        raise NotImplementedError

    def measure(self, f, zmin, zmax):
        raise NotImplementedError
