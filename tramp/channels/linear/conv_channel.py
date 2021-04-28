import numpy as np
from numpy.fft import fftn, ifftn
from ..base_channel import Channel
from tramp.utils.conv_filters import (
    gaussian_filter, differential_filter, laplacian_filter
)
from tramp.channels import LinearChannel
from tramp.utils.misc import complex2array, array2complex
import logging
logger = logging.getLogger(__name__)





class ConvChannel(Channel):
    """Conv (complex or real) channel x = w * z.

    Parameters
    ----------
    - filter: real or complex array
        Filter weights. The conv weights w are given by w[u] = f*[-u].
        The conv and filter weights ffts are conjugate.
    - real: bool
        if True assume x, w, z real
        if False assume x, w, z complex

    Notes
    -----
    For message passing it is more convenient to represent a complex array x
    as a real array X where X[0] = x.real and X[1] = x.imag

    In particular when real=False (x, w, z complex):
    - input  of sample(): Z real array of shape (2, z.shape)
    - output of sample(): X real array of shape (2, x.shape)
    - message bz, posterior rz: real arrays of shape (2, z.shape)
    - message bx, posterior rx: real arrays of shape (2, x.shape)
    """

    def __init__(self, filter, real=True):
        self.shape = filter.shape
        self.real = real
        self.repr_init()
        self.filter = filter
        # conv weights and filter ffts are conjugate
        self.w_fft_bar = fftn(filter)
        self.w_fft = np.conjugate(self.w_fft_bar)
        self.spectrum = np.absolute(self.w_fft)**2

    def convolve(self, z):
        "We assume x,z,w complex for complex fft or x,w,z real for real fft"
        z_fft = fftn(z)
        x_fft = self.w_fft * z_fft
        x = ifftn(x_fft)
        if self.real:
            x = np.real(x)
        return x

    def sample(self, Z):
        "When real=False we assume Z[0] = Z.real and Z[1] = Z.imag"
        if not self.real:
            Z = array2complex(Z)
        X = self.convolve(Z)
        if not self.real:
            X = complex2array(X)
        return X

    def math(self):
        return r"$\ast$"

    def second_moment(self, tau_z):
        return tau_z * self.spectrum.mean()

    def compute_n_eff(self, az, ax):
        "Effective number of parameters = overlap in z"
        if ax == 0:
            logger.info(f"ax=0 in {self} compute_n_eff")
            return 0.
        if az / ax == 0:
            logger.info(f"az/ax=0 in {self} compute_n_eff")
            return 1.
        n_eff = np.mean(self.spectrum / (az / ax + self.spectrum))
        return n_eff

    def compute_backward_mean(self, az, bz, ax, bx, return_fft=False):
        # estimate z from x = Wz
        if not self.real:
            bz = array2complex(bz)
            bx = array2complex(bx)
        bx_fft = fftn(bx)
        bz_fft = fftn(bz)
        resolvent = 1 / (az + ax * self.spectrum)
        rz_fft = resolvent * (bz_fft + self.w_fft_bar * bx_fft)
        if return_fft:
            return rz_fft
        rz = ifftn(rz_fft)
        if self.real:
            rz = np.real(rz)
        else:
            rz = complex2array(rz)
        return rz

    def compute_forward_mean(self, az, bz, ax, bx):
        # estimate x from x = Wz we have rx = W rz
        rz_fft = self.compute_backward_mean(az, bz, ax, bx, return_fft=True)
        rx_fft = self.w_fft * rz_fft
        rx = ifftn(rx_fft)
        if self.real:
            rx = np.real(rx)
        else:
            rx = complex2array(rx)
        return rx

    def compute_backward_variance(self, az, ax):
        assert az > 0
        n_eff = self.compute_n_eff(az, ax)
        vz = (1 - n_eff) / az
        return vz

    def compute_forward_variance(self, az, ax):
        if ax == 0:
            s_mean = np.mean(self.spectrum)
            return s_mean / az
        n_eff = self.compute_n_eff(az, ax)
        vx = n_eff / ax
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
        rx = self.compute_forward_mean(az, bz, ax, bx)
        a = az + ax * self.spectrum
        coef = 0.5 if self.real else 1
        logZ = (
            0.5 * np.sum(bz * rz) + 0.5 * np.sum(bx*rx) +
            coef * np.sum(np.log(2 * np.pi / a))
        )
        return logZ

    def compute_mutual_information(self, az, ax, tau_z):
        I = 0.5*np.log((az + ax * self.spectrum)*tau_z)
        I = I.mean()
        return I

    def compute_free_energy(self, az, ax, tau_z):
        tau_x = self.second_moment(tau_z)
        I = self.compute_mutual_information(az, ax, tau_z)
        A = 0.5*(az*tau_z + ax*tau_x) - I + 0.5*np.log(2*np.pi*tau_z/np.e)
        return A


class DifferentialChannel(ConvChannel):
    def __init__(self, D1, D2, shape, real=True):
        self.D1 = D1
        self.D2 = D2
        self.repr_init()
        f = differential_filter(shape=shape, D1=D1, D2=D2)
        super().__init__(filter=f, real=real)

    def math(self):
        return r"$\partial$"


class LaplacianChannel(ConvChannel):
    def __init__(self, shape, real=True):
        self.repr_init()
        f = laplacian_filter(shape)
        super().__init__(filter=f, real=real)

    def math(self):
        return r"$\Delta$"


class Blur1DChannel(ConvChannel):
    def __init__(self, sigma, N, real=True):
        self.sigma = sigma
        self.repr_init()
        f = gaussian_filter(sigma=sigma, N=N)
        super().__init__(filter=f, real=real)


class Blur2DChannel(ConvChannel):
    def __init__(self, sigma, shape, real=True):
        if len(sigma) != 2:
            raise ValueError("sigma must be a length 2 array")
        if len(shape) != 2:
            raise ValueError("shape must be a length 2 tuple")
        self.sigma = sigma
        self.repr_init()
        f0 = gaussian_filter(sigma=sigma[0], N=shape[0])
        f1 = gaussian_filter(sigma=sigma[1], N=shape[1])
        f = np.outer(f0, f1)
        super().__init__(filter=f, real=real)


class MultiConvChannel(LinearChannel):
    """
    Convolution channel for (real valued) multi-color data.

    Inputs and outputs x = [x1 ... xn] and z = [z1 ... zm] have block structure with n, m blocks
        respectively. If we define the product of two blocks as their convolution xi * zj, then
        this layer implements a dense matrix product of blocks:

    [x1 x2 ... xm] = [ w11 * z1 + w12 * z2 + ... + w1n * zn,
                       w21 * z1 + ...            + w2n * zn,
                       ...
                       wm1 * z1 + ...            + wmn * zn]

    Each block xi or zj is called a 'color' in reference to color channels of RGB images. In
        other sources, 'colors' may be called 'channels' instead.

    Parameters
    ---------
    - filters: real or complex array
        Filter weights of dimensions (m, n, r1, ..., rk).
    - block_shape:  Dimensions of each block. All blocks are assumed to be (d1, ..., dk) dimensional
        tensors.
    Notes
    -----
    This layer implements the common operation of neural networks called 'convolution'
        even though, technically, it is a cross correlation.
    """
    def __init__(self, filter, block_shape, name="M"):
        # This class uses LinearChannel method implementations but will completely overwrite its fields.
        super().__init__(W = np.eye(2), precompute_svd=False, name=name)
        del self.name, self.Nx, self.Nz, self.precompute_svd, self.W, \
                self.rank, self.alpha, self.C, self.spectrum, self.singular

        self.name = name
        m, n = filter.shape[:2]
        filter_shape = filter.shape[2:]

        self.macro_shape = (m, n)
        self.block_shape = block_shape
        self.block_order = len(block_shape)
        self.filter_shape = filter_shape

        self.Nz = n * np.prod(block_shape)
        self.Nx = m * np.prod(block_shape)
        self.repr_init()

        if(self.block_order > 24):
            raise ValueError(f"Input data blocks have tensor order {self.block_order} > 24 which will \
            break einstein sums used in this implementation.")

        self.filter = filter

        U, S, V = self._svd(filter)
        self.U = U
        self.S = S
        self.V = V

        self.singular = np.zeros(block_shape + (min(m, n), ))
        self.spectrum = np.zeros(block_shape + (max(m, n), ))
        self.singular[tuple(slice(0, k) for k in self.S.shape)] = S**2
        self.spectrum[tuple(slice(0, k) for k in self.S.shape)] = S**2

        self.alpha = self.Nx / self.Nz
        self.rank = np.sum(self.S != 0)


    def sample(self, Z):
        X = self @ Z
        return X

    def second_moment(self, tau_z):
        return tau_z * self.S.sum() / self.Nx

    def compute_backward_mean(self, az, bz, ax, bx):
        bx_svd = self.U.T(bx)
        bz_svd = self.V.T(bz)
        resolvent = 1/(az + ax * self.spectrum)
        rz_svd = resolvent * (bz_svd + self._scale(bx_svd, transpose=True))
        rz = self.V(rz_svd)
        return rz

    def compute_forward_mean(self, az, bz, ax, bx):
        rz = self.compute_backward_mean(az, bz, ax, bx)
        rx = self @ rz
        return rx

    def compute_forward_variance(self, az, ax):
        if ax == 0:
            s_mean = np.mean(self.singular[self.singular > 0])
            return s_mean * self.rank / (self.Nx * az)
        n_eff = self.compute_n_eff(az, ax)
        vx = n_eff / (self.alpha * ax)
        return vx

    def compute_log_partition(self, az, bz, ax, bx):
        rz = self.compute_backward_mean(az, bz, ax, bx)
        b = bz + self.T(bx)
        a = az + ax * self.spectrum
        logZ = 0.5 * np.sum(b * rz) + 0.5 * np.sum(np.log(2 * np.pi / a))
        return logZ

    def compute_free_energy(self, az, ax, tau_z):
        tau_x = self.second_moment(tau_z)
        I = self.compute_mutual_information(az, ax, tau_z)
        A = 0.5*(az*tau_z + self.alpha*ax*tau_x) - I + 0.5*np.log(2*np.pi*tau_z/np.e)
        return A

    def __matmul__(self, z):
        return self.U(self._scale(self.V.T(z)))

    def __tmatmul__(self, z):
        return self.V(self._scale(self.U.T(z), transpose=True))

    def __call__(self, z):
        return self @ z

    def T(self, z):
        return self.__tmatmul__(z)

    def __str__(self):
        return self.name

    def _scale(self, z, transpose=False):
        '''
        Apply diagonal singular values to input z. (ie. compute S z).

        Note: _scale is not using a VirtualMCCMatrix because that would require densifying S
        '''
        idxs = "abcdefghijklmnopqrstuvwx"[:self.block_order]

        m, n = self.macro_shape
        if(transpose):
            m, n = n, m

        if(m < n): # input dim < output dim, ignore some dimensions
            z = z[ (slice(None),)*self.block_order + (slice(0,m),) ]

        if(transpose):
            z = np.conj(self.S) * z
        else:
            z = self.S * z

        if(m > n): # output dim > input dim, add some zero padding
            z = np.pad(z, [(0, 0)]*self.block_order + [(0, m-n)])

        return z

    def _svd(self, conv_filter):
        '''
        Compute quantities for a sparse svd of the MCC matrix.
        '''
        macro_indices = [0, 1]
        block_indices = [2+r for r in range(self.block_order)]
        m, n = self.macro_shape

        conv_fft = np.flip(conv_filter, axis=block_indices)

        zeropad = [(0, 0), (0, 0)] + [ (0, self.block_shape[i] - self.filter_shape[i]) for i in range(self.block_order)]
        roll = [-(self.filter_shape[i] - 1)//2 for i in range(self.block_order)]

        conv_fft = np.pad(conv_fft, zeropad)
        conv_fft = np.roll(conv_fft, roll, axis=block_indices)

        # Note: the SVD of a convolution matrix is F @ diag(F'^T c) @ F^T
        # where F^T is the unitary DFT matrix, and F'^T is the non-unitary DFT matrix
        conv_fft = np.fft.fftn(conv_fft, axes=block_indices).transpose(block_indices + macro_indices)

        U, S, Vt = np.linalg.svd(conv_fft)
        V = np.conj(np.swapaxes(Vt, -2, -1))
        return _VirtualMCCFactor(U), S, _VirtualMCCFactor(V)

    def densify(self):
        U = self.U.densify()
        Vt = np.conj(self.V.densify()).T

        m, n = self.macro_shape

        if (m < n):  # input dim < output dim, ignore some dimensions
            Vt = Vt.reshape((np.prod(self.block_shape), n, n * np.prod(self.block_shape)))
            Vt = Vt[:, 0:m, :].reshape(np.prod(self.block_shape) * m, np.prod(self.block_shape) * n)

        SVt = self.S.reshape((-1, 1)) * Vt
        return np.real(U @ SVt[0:U.shape[1], :]) # account for m > n case.

class _VirtualMCCFactor:
    def __init__(self, sparse_mat, with_fft=True):
        '''
        This nested class behaves like a numpy matrix and wraps sparse MCC matrices resulting from SVD.

        Parameters
        ----------
        sparse_mat: sparse MCC factor tensor of the form (d1, ..., dk, m, n)
        with_fft: if True, __matmul__ applies a block inverse fourier transform after multiplication by sparse_mat
            and __tmatmul__ applies the corresponding block fourier transform before multiplication by sparse_mat^T.
        '''
        self.with_fft = with_fft
        self.sparse_mat = sparse_mat
        self.block_shape = sparse_mat.shape[:-2]
        self.block_order = len(self.block_shape)
        self.macro_shape = sparse_mat.shape[-2:]

    def __tmatmul__(self, z):
        # transpose matrix multiplication
        if(self.with_fft):
            block_indices = [1 + r for r in range(self.block_order)]
            z = np.transpose(np.fft.fftn(z, axes=block_indices, norm="ortho"), block_indices + [0])

        idxs = "abcdefghijklmnopqrstuvw"[:self.block_order]
        return np.einsum(f'{idxs}yz,{idxs}y->{idxs}z', np.conj(self.sparse_mat), z) # implicit transpose

    def __matmul__(self, z):
        idxs = "abcdefghijklmnopqrstuvw"[:self.block_order]
        z = np.einsum(f'{idxs}yz,{idxs}z->{idxs}y', self.sparse_mat, z)

        if(self.with_fft):
            block_indices = list(range(self.block_order))
            macro_idx = 1 + block_indices[-1]
            z = np.real(np.fft.ifftn(np.transpose(z, [macro_idx] + block_indices), axes=[1+r for r in block_indices], norm="ortho"))
        return z

    def __call__(self, z):
        return self @ z

    def T(self, z):
        return self.__tmatmul__(z)

    def densify(self):
        # Technically, this method densifies the transpose and returns a double transpose
        dim = np.prod(self.block_shape) * self.macro_shape[-1]
        natural_basis = np.eye(dim).reshape( (self.macro_shape[-1],) + self.block_shape + (-1,) )

        if(self.with_fft):
            block_indices = [1 + r for r in range(self.block_order)]
            natural_basis = np.transpose(np.fft.fftn(natural_basis, axes=block_indices, norm="ortho"), block_indices + [0, -1])

        idxs = "abcdefghijklmnopqrstuvw"[:self.block_order]
        mat = np.einsum(f'{idxs}yz,{idxs}yx->{idxs}zx', np.conj(self.sparse_mat), natural_basis)
        return np.conj(mat.reshape((-1, dim))).T

