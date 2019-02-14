import numpy as np
from ..base import Channel
from .complex_linear_channel import complex2array, array2complex


def check_unitary(U):
    if (U.shape[0] != U.shape[1]):
        raise ValueError(f"U.shape = {U.shape}")
    N = U.shape[0]
    if not np.allclose(U @ U.H, np.identity(N)):
        raise ValueError("U not unitary")


class UnitaryChannel(Channel):
    """Unitary channel x = U z.

    Parameters
    ----------
    - U: unitary matrix
    - U_name: str
        name of unitary matrix U for display

    Notes
    -----
    For message passing it is more convenient to represent a complex array x
    as a real array X where X[0] = x.real and X[1] = x.imag

    In particular:
    - input  of sample(): Z array of shape (2, z.shape)
    - output of sample(): X array of shape (2, x.shape)
    - message bz, posterior rz: real arrays of shape (2, z.shape)
    - message bx, posterior rx: real arrays of shape (2, x.shape)
    """

    def __init__(self, U, U_name="U"):
        U = np.matrix(U)
        check_unitary(U)
        self.U_name = U_name
        self.N = U.shape[0]
        self.repr_init()
        self.U = U

    def sample(self, Z):
        "We assume Z[0] = Z.real and Z[1] = Z.imag"
        Z = array2complex(Z)
        X = self.U @ Z
        X = complex2array(X)
        assert X.shape == (2, self.N)
        return X

    def math(self):
        return r"$"+self.U_name+"$"

    def second_moment(self, tau):
        return tau

    def compute_forward_message(self, az, bz, ax, bx):
        # x = U z
        ax_new = az
        bz = array2complex(bz)
        bx_new = self.U @ bz
        bx_new = complex2array(bx_new)
        return ax_new, bx_new

    def compute_backward_message(self, az, bz, ax, bx):
        # z = U.H x
        az_new = ax
        bx = array2complex(bx)
        bz_new = self.U.H @ bx
        bz_new = complex2array(bz_new)
        return az_new, bz_new

    def compute_forward_state_evolution(self, az, ax, tau):
        ax_new = az
        return ax_new

    def compute_backward_state_evolution(self, az, ax, tau):
        az_new = ax
        return az_new
