import numpy as np
import scipy.linalg
from ..base_channel import Channel
import logging
from .linear_channel import LinearChannel
logger = logging.getLogger(__name__)



def svd(X):
    "Compute SVD of X = U S V.T"
    U, s, VT = np.linalg.svd(X, full_matrices=True)
    V = VT.T
    S = np.zeros((U.shape[0], V.shape[0]))
    S[:len(s), :len(s)] = np.diag(s)
    svd_X = (U, S, V)
    return svd_X

def as_full_cov(ax, bx):
    dim = len(bx.flatten())
    if(np.isscalar(ax)):
        return (np.eye(dim) * ax, bx)
    else:
        return (ax, bx)

class AnisoLinearChannel(LinearChannel):
    """Linear channel x = W z.

    Parameters
    ----------
    - W: array of shape (Nx, Nz)
    - precompute_svd: bool
        if True precompute SVD of W = U S V.T
    - name: str
        name of weight matrix W for display
    """

    def compute_n_eff(self, az, ax):
        raise NotImplementedError("No compute n eff")

    def compute_backward_variance(self, az, ax):
        if(np.isscalar(az)):
            az = np.eye(self.Nz) * az
        elif(np.isscalar(ax)):
            ax = np.eye(self.Nx) * ax

        a = az + self.W.T @ ax @ self.W
        return np.linalg.pinv(a)

    def compute_forward_variance(self, az, ax):
        a_bwd = self.compute_backward_variance(az, ax)
        return self.W @ a_bwd @ self.W.T

    def compute_mutual_information(self, az, ax, tau_z):
        raise NotImplementedError("No mutual information")

    def compute_free_energy(self, az, ax, tau_z):
        raise NotImplementedError("No free energy")

    def compute_backward_mean(self, az, bz, ax, bx):
        ax, bx = as_full_cov(ax, bx)
        az, bz = as_full_cov(az, bz)

        a = az + self.W.T @ ax @ self.W
        b = (bz + self.W.T @ bx)
        rz = np.linalg.solve(a, b)
        return rz

    def compute_log_partition(self, az, bz, ax, bx):
        ax, bx = as_full_cov(ax, bx)
        az, bz = as_full_cov(az, bz)
        b = bz + self.W.T @ bx
        a = az + self.W.T @ ax @ self.W
        dim = len(b)
        logZ = 0.5 * b.reshape((1, -1)) @ a @ b.reshape((-1, 1)) + 0.5 * dim * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(a))
        return logZ

    def compute_ab_new(self, r, v, a, b):
        "Compute a_new and b_new ensuring that a_new is between AMIN and AMAX"
        # Occasionally, this method may receive a, v as scalars if the corresponding
        #    AnisoGaussianVariable was initialized at a constant scalar.
        if(np.isscalar(a) and np.isscalar(v)):
            a_new = np.clip(1/v - a, self.AMIN, self.AMAX)
            v_inv = a + a_new
            a_new = np.eye(self.Nz) * a_new
            b_new = (v_inv * r) - b
        else:
            if(np.isscalar(v)):
                v = np.eye(len(a)) * v
            if(np.isscalar(a)):
                a = np.eye(len(v)) * a
            a_new = np.clip(np.linalg.inv(v) - a, self.AMIN, self.AMAX)
            v_inv = (a + a_new)
            b_new = (v_inv @ r) - b

        return a_new, b_new
