from ..base import Factor, filter_message
import numpy as np

class ConcatChannel(Factor):
    n_next = 1

    def __init__(self, Ns, axis=0):
        self.Ns = Ns
        self.axis = axis
        self.repr_init()
        self.n_prev = len(Ns)
        self.N = sum(Ns)

    def sample(self, *Zs):
        if len(Zs) != self.n_prev:
            raise ValueError(f"expect {self.n_prev} arrays")
        for k, Z in enumerate(Zs):
            if (Z.shape[self.axis] != self.Ns[k]):
                raise ValueError(
                    f"expect Z k={k} array of dimension {self.Ns[k]} "
                    f"along axis {self.axis} "
                    f"but got array of dimension {Z.shape[self.axis]}"
                )
        X = np.concatenate(Zs, axis=self.axis)
        assert X.shape[self.axis] == self.N
        return X

    def math(self):
        return r"$\oplus$"

    def second_moment(self, *taus):
        if len(taus) != self.n_prev:
            raise ValueError(f"expect {self.n_prev} taus")
        tau_x = sum(N * tau for N, tau in zip(self.Ns, taus)) / self.N
        return tau_x

    def compute_forward_posterior(self, az, bz, ax, bx):
        "estimate x = [zk] from z={zk}"
        rz, vz = self.compute_backward_posterior(az, bz, ax, bx)
        rx = np.concatenate(rz, axis=self.axis)
        vx = sum(N * v for N, v in zip(self.Ns, vz)) / self.N
        return rx, vx

    def compute_backward_posterior(self, az, bz, ax, bx):
        "estimate z={zk} from x = [zk]"
        for N, Z in zip(self.Ns, bz):
            assert bz.shape[self.axis]==N
        assert bx.shape[self.axis]==self.N
        idx = [0]+list(np.cumsum(self.Ns))
        bx_subs = [
            np.take(bx, range(idx_min, idx_max), axis=self.axis)
            for idx_min, idx_max in zip(idx[:-1], idx[1:])
        ]
        ak = [a + ax for a in az]
        bk = [b + bx_sub for b, bx_sub in zip(bz, bx_subs)]
        vz = [1 / a for a in ak]
        rz = [b / a for a, b in zip(ak, bk)]
        return rz, vz

    def compute_forward_error(self, az, ax, tau):
        vz = self.compute_backward_error(az, ax, tau)
        vx = sum(N * v for N, v in zip(self.Ns, vz)) / self.N
        return vx

    def compute_backward_error(self, az, ax, tau):
        ak = [a + ax for a in az]
        vz = [1 / ak for a in ak]
        return vz
