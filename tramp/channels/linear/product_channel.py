import numpy as np
from ..base_channel import SOFactor
from ...utils.integration import cpx_gaussian_measure_2d, gaussian_measure_2d
from ...utils.misc import complex2array, array2complex
import logging


def dot(x, y):
    return np.real(x) * np.real(y) + np.imag(x) * np.imag(y)

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
        self.vec_forward_mean = self._vectorize(self._coordinate_forward_mean)
        self.vec_forward_var = self._vectorize(self._coordinate_forward_var)
        self.vec_backward_mean_S = self._vectorize(self._coordinate_backward_mean_S)
        self.vec_backward_var_S = self._vectorize(self._coordinate_backward_var_S)
        self.vec_backward_mean_Z = self._vectorize(self._coordinate_backward_mean_Z)
        self.vec_backward_var_Z = self._vectorize(self._coordinate_backward_var_Z)
        self.vec_partition = self._vectorize(self._partition)
    def _vectorize(self, F):
        '''
        Assume F takes as input a sequence of k vectors of shape either [d1, d2, ..., dj] or scalar [,]
        Where F : C^k -> C is a complex valued function

        This method first broadcasts all scalars to have shape [d1, ..., dj] then evaluates F on every coordinate (ie. choice of indices (d1, d2, ..., dj)).
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
        if tuple(s.shape) != tuple(self.shape):
            raise ValueError("Bad shape for s")
        if tuple(z.shape) != tuple(self.shape):
            raise ValueError("Bad shape for z")
        X = z * s
        return X

    def math(self):
        if(self.layer_idx is not None):
            return f"$D_{self.layer_idx}$"
        else:
            return "D"

    def _partition(self, aZ, bZ, aS, bS, aX, bX):
        return self._integrate(None, aZ, bZ, aS, bS, aX, bX, real_only=True)

    def _integrate(self, F, aZ, bZ, aS, bS, aX, bX, real_only=False):
        ''' integrate F(s, b_Zhat, a_Zhat) with respect to the unnormalized posterior measure of this factor.
        this integration is factorized as ∫ds ∫dz (...) where the inner integral is gaussian with parameter dependence on s. '''

        def integrand(u_Re, u_Im, _F=None):
            u = u_Re + 1j * u_Im
            s = bS/aS + u / np.sqrt(aS)
            b_Zhat = (bZ + np.conj(s) * bX)
            a_Zhat = (aZ + np.conj(s) * s * aX)
            x = _F(s, b_Zhat, a_Zhat) if (_F is not None) else 1
            return (1/np.sqrt(aS)) * np.exp(b_Zhat * np.conj(b_Zhat) / (2 * a_Zhat)) * (2 * np.pi / a_Zhat) * x

        return cpx_gaussian_measure_2d(0, 1, 0, 1, lambda u1, u2: integrand(u1, u2, F), real_only=real_only)

    '''WARNING: the _coordinate methods compute quantities that are not normalized by the partition function'''
    def _coordinate_forward_var(self, aZ, bZ_cpx, aS, bS_cpx, aX, bX_cpx):
        F = lambda s, b_Zhat, a_Zhat: np.real(s * np.conj(s) * 1/a_Zhat)
        return self._integrate(F, aZ, bZ_cpx, aS, bS_cpx, aX, bX_cpx, real_only=True)

    def _coordinate_forward_mean(self, aZ, bZ_cpx, aS, bS_cpx, aX, bX_cpx):
        F = lambda s, b_Zhat, a_Zhat: s * b_Zhat/a_Zhat
        return self._integrate(F, aZ, bZ_cpx, aS, bS_cpx, aX, bX_cpx)

    def _coordinate_backward_var_Z(self, aZ, bZ_cpx, aS, bS_cpx, aX, bX_cpx):
        F = lambda s, b_Zhat, a_Zhat: np.real(1/a_Zhat)
        return self._integrate(F, aZ, bZ_cpx, aS, bS_cpx, aX, bX_cpx, real_only=True)

    def _coordinate_backward_mean_Z(self, aZ, bZ_cpx, aS, bS_cpx, aX, bX_cpx):
        F = lambda s, b_Zhat, a_Zhat: b_Zhat/a_Zhat
        return self._integrate(F, aZ, bZ_cpx, aS, bS_cpx, aX, bX_cpx)

    def _coordinate_backward_var_S(self, aZ, bZ_cpx, aS, bS_cpx, aX, bX_cpx):
        F = lambda s, b_Zhat, a_Zhat: np.real(s * np.conj(s))
        return self._integrate(F, aZ, bZ_cpx, aS, bS_cpx, aX, bX_cpx, real_only=True)

    def _coordinate_backward_mean_S(self, aZ, bZ_cpx, aS, bS_cpx, aX, bX_cpx):
        F = lambda s, b_Zhat, a_Zhat: s
        return self._integrate(F, aZ, bZ_cpx, aS, bS_cpx, aX, bX_cpx)

    def compute_forward_posterior(self, az, bz, ax, bx):
        aZ, aS = az
        bZ, bS = bz

        bZ_cpx = array2complex(bZ)
        bS_cpx = array2complex(bS)
        bX_cpx = array2complex(bx)

        Z = self.vec_partition(aZ, bZ_cpx, aS, bS_cpx, ax, bX_cpx)
        rX = complex2array(self.vec_forward_mean(aZ, bZ_cpx, aS, bS_cpx, ax, bX_cpx)) / Z
        vX = (self.vec_forward_var(aZ, bZ_cpx, aS, bS_cpx, ax, bX_cpx).mean() / Z).mean()
        return rX, vX

    def compute_backward_posterior(self, az, bz, ax, bx):
        aZ, aS = az
        bZ, bS = bz

        bZ_cpx = array2complex(bZ)
        bS_cpx = array2complex(bS)
        bX_cpx = array2complex(bx)

        Z = self.vec_partition(aZ, bZ_cpx, aS, bS_cpx, ax, bX_cpx)
        rZ = complex2array(self.vec_backward_mean_Z(aZ, bZ_cpx, aS, bS_cpx, ax, bX_cpx) / Z)
        vZ = (self.vec_backward_var_Z(aZ, bZ_cpx, aS, bS_cpx, ax, bX_cpx) / Z).mean()
        rS = complex2array(self.vec_backward_mean_S(aZ, bZ_cpx, aS, bS_cpx, ax, bX_cpx) / Z)
        vS = (self.vec_backward_var_S(aZ, bZ_cpx, aS, bS_cpx, ax, bX_cpx) / Z).mean()

        rz = np.stack([rZ, rS], axis=0)
        vz = np.stack([vZ, vS], axis=0)
        return rz, vz

    def compute_forward_error(self, az, ax, tau_z):
        raise NotImplementedError

    def compute_backward_error(self, az, ax, tau_z):
        raise NotImplementedError

    def second_moment(self, tau_z):
        raise NotImplementedError

