import numpy as np
from ..base_channel import SOFactor
from ...utils.integration import cpx_gaussian_measure_2d, gaussian_measure_2d
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
        self.vec_forward_mean = self._vectorize(self.coordinate_forward_mean)
        self.vec_forward_var = self._vectorize(self.coordinate_forward_var)
        self.vec_backward_mean = self._vectorize(self.coordinate_backward_mean_unary)
        self.vec_backward_var = self._vectorize(self.coordinate_backward_var_unary)

    def _vectorize(self, F):
        '''
        Assume F takes as input a sequence of k vectors of shape either [d1, d2, ..., dj] or scalar [,]
        Where F : C^k -> C is a complex valued function

        This method first broadcasts all scalars to have shape [d1, ..., dj] then evaluates F on every coordinate (ie. assignment of (d1, d2, ..., dj)).
        '''
        def is_scalar(x):
            return isinstance(x, int) or isinstance(x, float) or isinstance(x, complex) or (isinstance(x, np.ndarray) and x.flatten().shape == (1,))

        def _apply(*args):
            non_scalar_shapes = [s.shape for s in args if not is_scalar(s)]
            assert all([non_scalar_shapes[i] == non_scalar_shapes[0] for i in range(len(non_scalar_shapes))]), "Shape error in vectorization"
            if (len(non_scalar_shapes) >= 1):
                shape = non_scalar_shapes[0]
            else:
                shape = (1,)
            _args = np.stack([(a * np.ones(shape)).flatten() for a in args], axis=0) # having shape (d1d2...dj, k)
            return np.apply_along_axis(func1d=lambda A: F(*A), axis=0, arr=_args).reshape(shape)
        return _apply

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

    def coordinate_forward_var(self, aZ, bZ_cpx, aS, bS_cpx, ax, bX_cpx):
        # aZ, aS, ax are real scalar precisions
        # bZ_cpx, bS_cpx, bX_cpx are complex valued scalar means
        rZ = bZ_cpx/aZ
        vZ = 1/aZ

        def vX_statistic(z_real, z_imag):
            z_ = rect_to_cpx(np.stack((z_real, z_imag), axis=0))
            return (np.conj(z_) * z_)/(aS + (np.conj(z_) * z_) * ax)

        vX = np.real_if_close(gaussian_measure_2d(np.real(rZ), vZ, np.imag(rZ), vZ, vX_statistic))
        return vX

    def coordinate_forward_mean(self, aZ, bZ_cpx, aS, bS_cpx, ax, bX_cpx):
        # aZ, aS, ax are real scalar precisions
        # bZ_cpx, bS_cpx, bX_cpx are complex valued scalar means
        rZ = bZ_cpx/aZ
        vZ = 1/aZ

        def rX_statistic(z_real, z_imag):
            z_ = rect_to_cpx(np.stack((z_real, z_imag), axis=0))
            return z_ * (bS_cpx + np.conj(z_) * bX_cpx)/(aS + (np.conj(z_) * z_) * ax)

        rX = cpx_gaussian_measure_2d(np.real(rZ), vZ, np.imag(rZ), vZ, rX_statistic)
        return rX

    def coordinate_backward_var_unary(self, aZ, bZ_cpx, aS, bS_cpx, ax, bX_cpx, reverse=True):
        # aZ, aS, ax are real scalar precisions
        # bZ_cpx, bS_cpx, bX_cpx are complex valued scalar means
        rZ = bZ_cpx / aZ
        vZ = 1 / aZ

        rS = bS_cpx / aS
        vS = 1 / aS

        # Plug in either a_ = aS, b_ = bS and integrate over z, or vice versa
        def v_statistic(z_real, z_imag, a_):
            z_ = rect_to_cpx(np.stack((z_real, z_imag), axis=0))
            return 1 / (a_ + (np.conj(z_) * z_) * ax)

        vZ_statistic = lambda s1, s2: v_statistic(s1, s2, aZ)
        vS_statistic = lambda z1, z2: v_statistic(z1, z2, aS)

        if(reverse):
            return np.real_if_close(cpx_gaussian_measure_2d(np.real(rS), vS, np.imag(rS), vS, vZ_statistic))
        else:
            return np.real_if_close(cpx_gaussian_measure_2d(np.real(rZ), vZ, np.imag(rZ), vZ, vS_statistic))

    def coordinate_backward_mean_unary(self, aZ, bZ_cpx, aS, bS_cpx, ax, bX_cpx, reverse=True):
        '''
        Computes the backward mean for one of two channel inputs as a complex number.
        That is, for x=zs, compute the posterior mean of z. If reverse then compute mean for s instead.
        '''
        rZ = bZ_cpx/aZ
        vZ = 1/aZ

        rS = bS_cpx / aS
        vS = 1 / aS

        # Plug in either a_ = aS, b_ = bS and integrate over z, or vice versa
        def r_statistic(z_real, z_imag, a_, b_):
            z_ = rect_to_cpx(np.stack((z_real, z_imag), axis=0))
            return (b_ + np.conj(z_) * bX_cpx) / (a_ + (np.conj(z_) * z_) * ax)

        rZ_statistic = lambda s1, s2: r_statistic(s1, s2, aZ, bZ_cpx)
        rS_statistic = lambda z1, z2: r_statistic(z1, z2, aS, bS_cpx)

        if(reverse):
            return cpx_gaussian_measure_2d(np.real(rZ), vZ, np.imag(rZ), vZ, rS_statistic)
        else:
            return cpx_gaussian_measure_2d(np.real(rS), vS, np.imag(rS), vS, rZ_statistic)

    def compute_forward_posterior(self, az, bz, ax, bx):
        aZ, aS = az
        bZ, bS = bz

        bZ_cpx = rect_to_cpx(bZ)
        bS_cpx = rect_to_cpx(bZ)
        bX_cpx = rect_to_cpx(bx)

        rX = cpx_to_rect(self.vec_forward_mean(aZ, bZ_cpx, aS, bS_cpx, ax, bX_cpx))
        vX = self.vec_forward_var(aZ, bZ_cpx, aS, bS_cpx, ax, bX_cpx).mean()
        return rX, vX

    def compute_backward_posterior(self, az, bz, ax, bx):
        aZ, aS = az
        bZ, bS = bz

        bZ_cpx = rect_to_cpx(bZ)
        bS_cpx = rect_to_cpx(bS)
        bX_cpx = rect_to_cpx(bx)

        aZ = np.ones_like(bZ_cpx) * aZ
        aS = np.ones_like(bS_cpx) * aS

        both_inp_b = np.stack([bZ_cpx, bS_cpx], axis=0)
        both_inp_b_swap = np.stack([bS_cpx, bX_cpx], axis=0)
        both_inp_a = np.stack([aZ, aS], axis=0)
        both_inp_a_swap = np.stack([aS, aZ], axis=0)
        duplicate_bx = np.stack([bX_cpx, bX_cpx], axis=0)

        '''
        Something potentially confusing about this implementation: it provides two ways to compute the posterior 
        means, and the one used here is more obscure. 
            1. Used here: _vectorize evaluates the correct gaussian integral coordinatewise, so here
                we stack [rZ, rS] parameters and compute all at once coordinatewise. 
            2. Not used, but possible: coordinate_backward_mean_unary has the reverse option which would allow one
                to compute [rZ] with only Z parameters, then do it again with rS parameters using reverse=True.
        '''

        # vec_backward_mean returns [rZ, rS] as complex numbers. We convert to rectangular form.
        rz_cpx = self.vec_backward_mean(both_inp_a, both_inp_b, both_inp_a_swap, both_inp_b_swap, ax, duplicate_bx)
        rz = np.stack([cpx_to_rect(rz_cpx[0]), cpx_to_rect(rz_cpx[1])], axis=0)

        # vz is returned as a real number
        vz = self.vec_backward_var(both_inp_a, both_inp_b, both_inp_a_swap, both_inp_b_swap, ax, duplicate_bx).reshape((2, -1)).mean(axis=1)
        return rz, vz

    def compute_forward_error(self, az, ax, tau_z):
        raise NotImplementedError

    def compute_backward_error(self, az, ax, tau_z):
        raise NotImplementedError

    def second_moment(self, tau_z):
        raise NotImplementedError

