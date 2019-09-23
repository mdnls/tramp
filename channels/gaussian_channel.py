import numpy as np
from ..base import Channel


class GaussianChannel(Channel):
    def __init__(self, var=1):
        self.var = var
        self.repr_init()
        self.sigma = np.sqrt(var)
        self.a = 1 / var

    def sample(self, Z):
        noise = self.sigma * np.random.standard_normal(Z.shape)
        X = Z + noise
        return X

    def math(self):
        return r"$\mathcal{N}$"

    def second_moment(self, tau):
        return tau + self.var

    def compute_forward_message(self, az, bz, ax, bx):
        kz = self.a / (self.a + az)
        ax_new = kz * az
        bx_new = kz * bz
        return ax_new, bx_new

    def compute_backward_message(self, az, bz, ax, bx):
        kx = self.a / (self.a + ax)
        az_new = kx * ax
        bz_new = kx * bx
        return az_new, bz_new

    def compute_forward_state_evolution(self, az, ax, tau):
        kz = self.a / (self.a + az)
        ax_new = kz * az
        return ax_new

    def compute_backward_state_evolution(self, az, ax, tau):
        kx = self.a / (self.a + ax)
        az_new = kx * ax
        return az_new

    def log_partition(self, az, bz, ax, bx):
        az_new, bz_new = self.compute_backward_message(az, bz, ax, bx)
        rz = (bz_new + bz) / (az_new + az)
        ax_new, bx_new = self.compute_forward_message(az, bz, ax, bx)
        rx = (bx_new + bx) / (ax_new + ax)
        d = ax + az + ax*az*self.var
        logZ = 0.5 * np.sum(rz*bz + rx*bx + np.log(2 * np.pi / d))
        return logZ

    def free_energy(self, az, ax, tau):
        tau_x = self.second_moment(tau)
        K = np.log(2*np.pi / (ax + az + ax*az/self.a))
        A = 0.5*(az*tau + ax*tau_x - 1 + K.mean())
        return A
