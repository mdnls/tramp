import numpy as np
from .linear_channel import LinearChannel




class AnisoToIso(LinearChannel):
    def __init__(self, dim, name="A->I"):
        '''
        Given input z, output x, represent the relationship x = z.
        It is assumed that z has anisotropic beliefs and x has isotropic beliefs.
        '''
        super().__init__(W = np.eye(dim), name=name, precompute_svd=False)

    def compute_backward_variance(self, az, ax):
        ax = np.eye(self.Nx) * ax
        if(np.isscalar(az)):
            az = np.eye(self.Nz) * az
        a = az + ax
        return np.linalg.pinv(a)

    def compute_forward_variance(self, az, ax):
        a_bwd = self.compute_backward_variance(az, ax)
        return np.mean(np.diag(a_bwd)) # tr(A) / d

    def compute_mutual_information(self, az, ax, tau_z):
        raise NotImplementedError("No mutual information")

    def compute_free_energy(self, az, ax, tau_z):
        raise NotImplementedError("No free energy")

    def compute_backward_mean(self, az, bz, ax, bx):
        ax = ax * np.eye(self.Nx)
        if(np.isscalar(az)):
            az = np.eye(self.Nz) * az

        a = az + ax
        b = bz + bx
        rz = np.linalg.solve(a, b)
        return rz

    def compute_forward_mean(self, az, bz, ax, bx):
        return self.compute_backward_mean(az, bz, ax, bx)

    def compute_backward_posterior(self, az, bz, ax, bx):
        rz = self.compute_backward_mean(az, bz, ax, bx)
        vz = self.compute_backward_variance(az, ax)
        return rz, vz

    def compute_forward_posterior(self, az, bz, ax, bx):
        rx = self.compute_forward_mean(az, bz, ax, bx)
        vx = self.compute_forward_variance(az, ax)
        return rx, vx

    def compute_log_partition(self, az, bz, ax, bx):
        ax = np.eye(self.Nx) * ax
        if(np.isscalar(az)):
            az = np.eye(self.Nz) * az
        b = bz + bx
        a = az + ax
        logZ = 0.5 * b.reshape((1, -1)) @ a @ b.reshape((-1, 1)) + 0.5 * self.Nx * np.log(2 * np.pi) - 0.5 * np.log(
            np.linalg.det(a))
        return logZ

    def compute_n_eff(self, az, ax):
        raise NotImplementedError("No compute n eff")

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
                v = np.eye(self.Nz) * v
            if(np.isscalar(a)):
                a = np.eye(self.Nz) * a
            a_new = np.clip(np.linalg.inv(v) - a, self.AMIN, self.AMAX)
            v_inv = (a + a_new)
            b_new = (v_inv @ r) - b

        return a_new, b_new

    def compute_forward_message(self, az, bz, ax, bx):
        rx, vx = self.compute_forward_posterior(az, bz, ax, bx)
        ax_new, bx_new = self.compute_ab_new(rx, vx, ax, bx)
        return np.mean(np.diag(ax_new)), bx_new

    def compute_backward_message(self, az, bz, ax, bx):
        rz, vz = self.compute_backward_posterior(az, bz, ax, bx)
        az_new, bz_new = self.compute_ab_new(rz, vz, az, bz)
        return az_new, bz_new

class IsoToAniso(AnisoToIso):
    def __init__(self, dim, name="I->A"):
        '''
        Given input z, output x, represent the relationship x = z.
        It is assumed that z has anisotropic beliefs and x has isotropic beliefs.
        '''
        super().__init__(name=name, dim=dim)

    def compute_backward_variance(self, az, ax):
        return np.mean(np.diag(super().compute_backward_variance(az, ax)))

    def compute_forward_variance(self, az, ax):
        return super().compute_backward_variance(az, ax)

    def compute_backward_mean(self, az, bz, ax, bx):
        # note that AnisoToIso uses the same method for backward and forward mean
        # calling compute_backward_mean in the following line is necessary to avoid recursion error
        return super().compute_backward_mean(az, bz, ax, bx)

    def compute_forward_mean(self, az, bz, ax, bx):
        return super().compute_backward_mean(az, bz, ax, bx)

    def compute_forward_message(self, az, bz, ax, bx):
        return super().compute_backward_message(az, bz, ax, bx)

    def compute_backward_message(self, az, bz, ax, bx):
        return super().compute_forward_message(az, bz, ax, bx)
