import numpy as np
import scipy.linalg
from .base_ensemble import Ensemble

class ConvEnsemble(Ensemble):
    def __init__(self, N, k):
        '''
        Random convolutional matrix ensemble. Samples W are scaled so E[ |Wx|^2 ] = |x|^2

        Parameters
        ----------
        N - Signal dimensionality
        k - Size of random convolutional filters
        '''
        self.N = N
        self.k = k
        self.repr_init()

    def generate(self):
        sigma_x = 1 / np.sqrt(self.k)
        filter = np.random.normal(size=(self.k,), scale=sigma_x)
        padded_filter = np.zeros(shape=(self.N,))
        padded_filter[0:self.k] = filter
        X = scipy.linalg.circulant(np.roll(padded_filter, shift=-(self.k-1)//2)).T
        return X

class Multi2dConvEnsemble(Ensemble):
    def __init__(self, width, height, in_channels, out_channels, k):
        '''
        Ensemble of random multichannel convolution matrix for 2d images.

        Parameters
        ----------
        width: input image width
        height: input image height
        in_colors: number of input image color channels
        out_colors: number of output image color channels.
        k: side length of convolutional filters. Must be an odd integer.
        '''
        assert isinstance(k, int) and k % 2 == 1, "k must be an odd integer."
        self.W = width
        self.H = height
        self.C_in = in_channels
        self.C_out = out_channels
        self.k = k

    def generate(self, with_filter=False):
        '''
        Generate a random multichannel convolution matrix.

        Returns
        -------
        Sample of the ensemble.
        '''
        sigma_x = 1/np.sqrt(self.k**2 * self.C_in)
        filter = sigma_x * np.random.normal(size=(self.C_out, self.C_in, self.k, self.k))
        return filter
