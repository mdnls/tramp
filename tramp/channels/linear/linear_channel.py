import numpy as np
from ..base_channel import Channel
import logging
logger = logging.getLogger(__name__)



def svd(X):
    "Compute SVD of X = U S V.T"
    U, s, VT = np.linalg.svd(X, full_matrices=True)
    V = VT.T
    S = np.zeros((U.shape[0], V.shape[0]))
    S[:len(s), :len(s)] = np.diag(s)
    svd_X = (U, S, V)
    return svd_X


class LinearChannel(Channel):
    """Linear channel x = W z.

    Parameters
    ----------
    - W: array of shape (Nx, Nz)
    - precompute_svd: bool
        if True precompute SVD of W = U S V.T
    - name: str
        name of weight matrix W for display
    """

    def __init__(self, W, precompute_svd=True, name="W"):
        self.name = name
        self.Nx = W.shape[0]
        self.Nz = W.shape[1]
        self.precompute_svd = precompute_svd
        self.repr_init()
        self.W = W
        self.rank = np.linalg.matrix_rank(W)
        self.alpha = self.Nx / self.Nz
        if precompute_svd:
            self.U, self.S, self.V = svd(W)
            self.spectrum = np.diag(self.S.T @ self.S)
        else:
            self.C = W.T @ W
            self.spectrum = np.linalg.eigvalsh(self.C)
        assert self.spectrum.shape == (self.Nz,)
        self.singular = self.spectrum[:self.rank]

    def sample(self, Z):
        X = self.W @ Z
        return X

    def math(self):
        return r"$" + self.name + "$"

    def second_moment(self, tau_z):
        return tau_z * self.spectrum.sum() / self.Nx

    def compute_n_eff(self, az, ax):
        "Effective number of parameters = overlap in z"
        if ax == 0:
            logger.info(f"ax=0 in {self} compute_n_eff")
            return 0.
        if az / ax == 0:
            logger.info(f"az/ax=0 in {self} compute_n_eff")
            return self.rank / self.Nz
        n_eff_trace = np.sum(self.singular / (az / ax + self.singular))
        return n_eff_trace / self.Nz

    def compute_backward_mean(self, az, bz, ax, bx):
        # estimate z from x = Wz
        if self.precompute_svd:
            bx_svd = self.U.T @ bx
            bz_svd = self.V.T @ bz
            resolvent = 1 / (az + ax * self.spectrum)
            if bz.ndim > 1:
                resolvent = resolvent[:, np.newaxis]
            rz_svd = resolvent * (bz_svd + self.S.T @ bx_svd)
            rz = self.V @ rz_svd
        else:
            a = az * np.identity(self.Nz) + ax * self.C
            b = (bz + self.W.T @ bx)
            rz = np.linalg.solve(a, b)
        return rz

    def compute_forward_mean(self, az, bz, ax, bx):
        # estimate x from x = Wz we have rx = W rz
        rz = self.compute_backward_mean(az, bz, ax, bx)
        rx = self.W @ rz
        return rx

    def compute_backward_variance(self, az, ax):
        if az==0:
            logger.info(f"az=0 in {self} compute_backward_variance, clipping to 1e-11")
        az = np.maximum(1e-11, az)
        n_eff = self.compute_n_eff(az, ax)
        vz = (1 - n_eff) / az
        return vz

    def compute_forward_variance(self, az, ax):
        if ax == 0:
            s_mean = np.mean(self.singular)
            return s_mean * self.rank / (self.Nx * az)
        n_eff = self.compute_n_eff(az, ax)
        vx = n_eff / (self.alpha * ax)
        return vx

    def compute_backward_posterior(self, az, bz, ax, bx):
        # estimate z from x = Wz
        rz = self.compute_backward_mean(az, bz, ax, bx)
        vz = self.compute_backward_variance(az, ax)
        return rz, vz

    def compute_forward_posterior(self, az, bz, ax, bx):
        # estimate x from x = Wz
        rx = self.compute_forward_mean(az, bz, ax, bx)
        vx = self.compute_forward_variance(az, ax)
        return rx, vx

    def compute_backward_error(self, az, ax, tau_z):
        vz = self.compute_backward_variance(az, ax)
        return vz

    def compute_forward_error(self, az, ax, tau_z):
        vx = self.compute_forward_variance(az, ax)
        return vx

    def compute_log_partition(self, az, bz, ax, bx):
        rz = self.compute_backward_mean(az, bz, ax, bx)
        b = bz + self.W.T @ bx
        a = az + ax * self.spectrum
        logZ = 0.5 * np.sum(b * rz) + 0.5 * np.sum(np.log(2 * np.pi / a))
        return logZ

    def compute_mutual_information(self, az, ax, tau_z):
        I = 0.5*np.log((az + ax * self.spectrum)*tau_z)
        I = I.mean()
        return I

    def compute_free_energy(self, az, ax, tau_z):
        tau_x = self.second_moment(tau_z)
        I = self.compute_mutual_information(az, ax, tau_z)
        A = 0.5*(az*tau_z + self.alpha*ax*tau_x) - I + 0.5*np.log(2*np.pi*tau_z/np.e)
        return A

class DiagonalChannel(LinearChannel):
    """
    Linear channel x = diag(S)z where S is a tensor of shape matching z.

    Parameters
    ----------
    - S: tensor whose shape matches shape of inputs z.
    - name: str
        name of weight matrix W for display

    Notes
    -----
    This class behaves as LinearChannel but it does not compute diag(S) as a dense matrix.
    """
    def __init__(self, S, name="W"):
        # This class uses LinearChannel method implementations but will completely overwrite its fields.
        super().__init__(W = np.eye(2), precompute_svd=False, name=name)
        del self.name, self.Nx, self.Nz, self.precompute_svd, self.W, \
                self.rank, self.alpha, self.C, self.spectrum, self.singular

        self.name = name
        self.Nx = np.prod(S.shape)
        self.Nz = self.Nx
        self.precompute_svd = True
        self.repr_init()
        self.W = _VirtualDiagMatrix(S)
        self.rank = np.sum(S != 0)
        self.alpha = 1
        self.U = _VirtualDiagMatrix(np.ones_like(S))
        self.V = _VirtualDiagMatrix(np.ones_like(S))
        self.S = _VirtualDiagMatrix(S)
        self.singular = S**2
        self.spectrum = S**2


    def compute_backward_mean(self, az, bz, ax, bx):
        bx_svd = self.U.T @ bx
        bz_svd = self.V.T @ bz
        resolvent = 1 / (az + ax * self.spectrum)
        rz_svd = resolvent * (bz_svd + self.S.T @ bx_svd)
        rz = self.V @ rz_svd
        return rz

class ColorwiseLinearChannel(LinearChannel):
    def __init__(self, input_shape, output_shape, W, name="W"):
        '''
        Sparsely apply the given linear transformation to each color of a multi-color signal.

        Given an N by M matrix W, this channel takes as input a tensor of dimensions (c, d1, d2, ..., dk),
        where it is assumed M = d1 d2 ... dk. This channel will reshape its input to size (c, d1 ... dk) and apply
        W to the second dimension.

        Parameters
        ----------
        W: colorwise linear transormation.
        input_shape: the shape (c, d1, ..., dk) of input data.
        output_shape: the shape (c, f1, ..., fk) of output data.
        name: name of this operator.
        '''
        super().__init__(W = np.eye(2), precompute_svd=False, name=name)
        del self.name, self.Nx, self.Nz, self.precompute_svd, self.W, \
            self.rank, self.alpha, self.C, self.spectrum, self.singular

        n_colors = input_shape[0]
        data_shape = input_shape[1:]
        N, M = W.shape
        assert M == np.prod(data_shape), "W must have input dimension matching the dimension of the data."
        assert N == np.prod(output_shape[1:]), "W must have output dimension matching the dimension of the data."
        self.name = name
        self.Nz = n_colors * M
        self.Nx = n_colors * N
        self.repr_init()

        self.precompute_svd = True
        W_U, W_S, W_Vt = np.linalg.svd(W)
        W_V = np.conj(W_Vt).T

        self.W = _VirtualBlockDiagMatrix(W, input_shape=input_shape, output_shape=output_shape)
        self.rank = n_colors * np.sum(W_S != 0)
        self.alpha = 1

        self.U = _VirtualBlockDiagMatrix(W_U, input_shape=output_shape, output_shape=output_shape)
        self.V = _VirtualBlockDiagMatrix(W_V, input_shape=input_shape, output_shape=input_shape)

        W_S_dense = np.zeros_like(W)
        W_S_dense[0:len(W_S), 0:len(W_S)] = W_S
        self.S = _VirtualBlockDiagMatrix(W_S_dense, input_shape=input_shape, output_shape=output_shape)

        self.singular = np.tile(W_S[np.newaxis, :], (n_colors, 1))
        self.spectrum = np.tile(W_S[np.newaxis, :], (n_colors, 1)).reshape(input_shape)

    def compute_backward_mean(self, az, bz, ax, bx):
        bx_svd = self.U.T @ bx
        bz_svd = self.V.T @ bz
        resolvent = 1 / (az + ax * self.spectrum)
        rz_svd = resolvent * (bz_svd + self.S.T @ bx_svd)
        rz = self.V @ rz_svd
        return rz

class _VirtualDiagMatrix():
    def __init__(self, S, input_shape=None, output_shape=None):
        '''
        Virtual diagonal matrix. Scale coordintes of inputs by the elements of S. If inp_shape and outp_shape are
        provided, this class implements a rectangular matrix by chopping or adding dimensions where necessary.

        Args:
            S: diagonal coefficients of the matrix.
            inp_shape: shape of inputs.
            outp_shape: shape of outputs
        '''
        assert (input_shape is None and output_shape is None) or ( (not input_shape is None ) and (not output_shape is None) ),\
            "You must path either both or none of the arguments [inp_shape, outp_shape]."
        self.S = S
        self.inp_shape = input_shape
        self.outp_shape = output_shape

    def __matmul__(self, z):
        if(self.inp_shape is not None):
            m, n = np.prod(self.outp_shape), np.prod(n)
            z = z.flatten()[:r].reshape(self.S.shape)
            scaled = self.S * z
            if(m < r):
                reshaped = np.pad(scaled.flatten(), [(0, m-n)])
            elif(r < m):
                reshaped = scaled.flatten()[:m]
            else:
                reshaped = scaled
            return reshaped.reshape(self.outp_shape)
        else:
            return self.S * z

    @property
    def T(self):
        return _VirtualDiagMatrix(np.conj(self.S), input_shape=self.outp_shape, output_shape=self.inp_shape)

class _VirtualBlockDiagMatrix():
    def __init__(self, M, input_shape, output_shape):
        self.M = M
        self.input_shape = input_shape
        self.outp_shape = output_shape

    def __matmul__(self, z):
        c = self.input_shape[0]
        return (self.M @ z.reshape((c, -1)).T).T.reshape(self.outp_shape)

    @property
    def T(self):
        return _VirtualBlockDiagMatrix(np.conj(self.M.T), input_shape = self.outp_shape, output_shape=self.input_shape)
