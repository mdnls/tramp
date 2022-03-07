from ..base import Variable
import numpy as np

class AnisoGaussianVariable(Variable):

    def compute_mutual_information(self, ax, tau_x):
        raise NotImplementedError("No mutual information")

    def compute_free_energy(self, ax, tau_x):
        raise NotImplementedError("No free energy")

    def compute_dual_mutual_information(self, vx, tau_x):
        raise NotImplementedError("No dual mutual information")

    def compute_dual_free_energy(self, mx, tau_x):
        raise NotImplementedError("No dual free energy")

    def compute_log_partition(self, ax, bx):
        if ax<=0:
            return np.inf
        inv_ax = np.linalg.pinv(ax)
        d = len(bx.flatten())
        logZ = 0.5*(np.sum(inv_ax * (bx.reshape(-1, 1) @ bx.reshape((1, -1)))) + d * np.log(2*np.pi) - np.log(np.linalg.det(ax)))
        return logZ

    def posterior_rv(self, message):
        a_hat, b_hat = self.posterior_ab(message)
        r_hat = (a_hat @ b_hat.reshape((-1, 1))).flatten()
        v_hat = np.linalg.pinv(a_hat)
        return r_hat, v_hat

    def posterior_a(self, message):
        a_hat = sum(data["a"] for source, target, data in message)
        return a_hat

    def posterior_v(self, message):
        a_hat = self.posterior_a(message)
        v_hat = np.linalg.pinv(a_hat)
        return v_hat

    def log_partition(self, message):
        ax, bx = self.posterior_ab(message)
        logZ = self.compute_log_partition(ax, bx)
        return logZ

    def free_energy(self, message):
        ax = self.posterior_a(message)
        tau_x = self._parse_tau(message)
        A = self.compute_free_energy(ax, tau_x)
        return A

