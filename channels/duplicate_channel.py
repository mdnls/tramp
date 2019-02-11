from ..base import Factor


class DuplicateChannel(Factor):
    n_prev = 1

    def __init__(self, n_next):
        self.n_next = n_next
        self.repr_init()

    def sample(self, Z):
        return (Z,) * self.n_next

    def math(self):
        return r"$\delta$"

    def second_moment(self, tau):
        return (tau,) * self.n_next

    def compute_forward_posterior(self, az, bz, ax, bx):
        "estimate x = {xk} from (xk = z for all k)"
        rz, vz = self.compute_backward_posterior(az, bz, ax, bx)
        rx = [rz] * self.n_next
        vx = [vz] * self.n_next
        return rx, vx

    def compute_backward_posterior(self, az, bz, ax, bx):
        "estimate z from (xk = z for all k)"
        a = ax + sum(az)
        b = bx + sum(bz)
        rz = b / a
        vz = 1. / a
        return rz, vz

    def compute_forward_error(self, az, ax, tau):
        vz = self.compute_backward_error(az, ax, tau)
        vx = [vz] * self.n_next
        return vx

    def compute_backward_error(self, az, ax, tau):
        a = ax + sum(az)
        vz = 1. / a
        return vz
