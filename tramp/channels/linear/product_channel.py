import numpy as np
from ..base_channel import SOFactor
from ...utils.integration import gaussian_measure_2d
from ...utils.misc import rect_to_cpx, cpx_to_rect
import logging

logger = logging.getLogger(__name__)

class ProductChannel(SOFactor):
    n_prev = 2

    def __init__(self, shape, layer_idx=None):
        '''
        This channel represents the operation x = z*s where x, z, s are complex vectors
            and the complex multiplication is performed elementwise.

        Parameters
        ----------
        shape: the shape of x, z, s
        layer_idx: the index of this layer, used for its math representation
        '''
        self.shape = shape
        self.layer_idx = layer_idx
        self.repr_init()

    def sample(self, s, z):
        if s.shape != self.shape:
            raise ValueError("Bad shape for x")
        if z.shape != self.shape:
            raise ValueError("Bad shape for z")
        X = z * s
        return X

    def math(self):
        if(self.layer_idx is not None):
            return f"$D_{self.layer_idx}$"
        else:
            return "D"

    def compute_forward_posterior(self, az, bz, ax, bx):
        aS, aZ = az
        bS, bZ = bz

        rZ = bZ/aZ
        vZ = 1/aZ

        def rX_statistic(z):
            z_ = rect_to_cpx(z)
            return z_ * (bS + z_ * bx)/(aS + (np.conj(z_) * z_) * ax)

        def vX_statistic(z):
            z_ = rect_to_cpx(z)
            return (np.conj(z_) * z_)/(aS + (np.conj(z_) * z_) * ax)

        rX = gaussian_measure_2d(rZ[0, ...], vZ, rZ[1, ...], vZ, rX_statistic)
        vX = gaussian_measure_2d(rZ[0, ...], vZ, rZ[1, ...], vZ, vX_statistic)

        return rX, vX

    def compute_backward_posterior(self, az, bz, ax, bx):
        aS, aZ = az
        bS, bZ = bz

        rZ = bZ / aZ
        vZ = 1 / aZ

        rS = bS / aS
        vS = 1 / aS

        # Plug in either a_ = aS, b_ = bS and integrate over z, or vice versa
        def r_statistic(z, a_, b_):
            z_ = rect_to_cpx(z)
            return z_ * (b_ + z_ * bx) / (a_ + (np.conj(z_) * z_) * ax)

        def v_statistic(z, a_):
            z_ = rect_to_cpx(z)
            return (np.conj(z_) * z_) / (a_ + (np.conj(z_) * z_) * ax)

        rZ_statistic = lambda s: r_statistic(s, aZ, bZ)
        vZ_statistic = lambda s: v_statistic(s, aZ, bZ)
        rS_statistic = lambda z: r_statistic(z, aS, bS)
        vS_statistic = lambda z: v_statistic(z, aS, bS)

        rZ_new = gaussian_measure_2d(rS[0, ...], vS, rS[1, ...], vS, rZ_statistic)
        vZ_new = gaussian_measure_2d(rS[0, ...], vS, rS[1, ...], vS, vZ_statistic)
        rS_new = gaussian_measure_2d(rZ[0, ...], vZ, rZ[1, ...], vZ, rS_statistic)
        vS_new = gaussian_measure_2d(rZ[0, ...], vZ, rZ[1, ...], vZ, vS_statistic)

        return (rS_new, rZ_new), (vS_new, vZ_new)

    def compute_forward_error(self, az, ax, tau_z):
        raise NotImplementedError

    def compute_backward_error(self, az, ax, tau_z):
        raise NotImplementedError

    def second_moment(self, tau_z):
        raise NotImplementedError

