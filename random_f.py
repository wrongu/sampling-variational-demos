from typing import Union
import numpy as np


def power_law(freqs: np.ndarray, alpha: float, normalize: Union[None, str] = 'max') -> np.ndarray:
    if normalize is None or normalize.lower() == 'none':
        return (freqs ** alpha)
    elif normalize.lower() == 'max':
        return (freqs ** alpha) / (freqs[0] ** alpha)
    elif normalize.lower() == 'mean':
        return (freqs ** alpha) / np.sum(freqs[0] ** alpha)


class RandomMixtureOfSinusoids(object):
    def __init__(self, dim, freqs, alpha, norm='max'):
        self._dim = dim
        self._omega = np.array(freqs)
        self._norm = norm
        self.set_alpha(alpha)
        self.randomize()
    
    def set_alpha(self, a):
        self._alpha = a
        self._weights = power_law(self._omega, a, self._norm)
    
    def randomize(self):
        self._phases = np.random.rand(len(self._omega)) * 2 * np.pi
        self._t = np.random.randn(len(self._omega), self._dim)
        self._t = self._t / np.linalg.norm(self._t, axis=1, keepdims=True)
    
    def __call__(self, x, total=True):
        dot_tx = self._t @ x
        if total:
            return np.sum(self._weights[:,None] * np.sin(dot_tx*self._omega[:,None] + self._phases[:,None]), axis=0)
        else:
            return self._weights[:,None], np.sin(dot_tx*self._omega[:,None] + self._phases[:,None])

    def gauss_expectation(self, mean, cov, total=True):
        dot_tmu = self._t @ mean
        tCt = np.sum(self._t * (self._t @ cov), axis=1)
        sin_part = np.sin(dot_tmu*self._omega + self._phases)
        exp_part = np.exp(-0.5 * tCt * self._omega**2)
        if total:
            return np.sum(self._weights * sin_part * exp_part)
        else:
            return self._weights, sin_part * exp_part


class LazyMixtureOfSinusoids(object):
    def __init__(self, dim, freqs, x=None, mus=None, covs=None, t=None):
        self._dim = dim
        self._omega = np.array(freqs)
    
        if t is not None:
            self.set_t(t)
        else:
            self.randomize_t()
        
        if x is not None:
            self.update_table_x(x)
        elif mus is not None:
            self.update_table_gauss(mus, covs)
    
    def randomize_t(self):
        # Choose random unit directions
        self._t = np.random.randn(len(self._omega), self._dim)
        self._t = self._t / np.linalg.norm(self._t, axis=1, keepdims=True)
        # Invalidate precomputed table
        self._sin_table = None
        self._cos_table = None
        return self
    
    def set_t(self, t):
        self._t = np.copy(t)
        # Invalidate precomputed table
        self._sin_table = None
        self._cos_table = None
        return self
    
    def update_table_x(self, x):
        dot_tx = self._t @ x
        self._sin_table = np.sin(dot_tx*self._omega[:,None])
        self._cos_table = np.cos(dot_tx*self._omega[:,None])
        return self
    
    def update_table_gauss(self, mus, covs):
        tmu = np.einsum('fa,ax->fx', self._t, mus)
        tCt = np.einsum('fa,fb,abx->fx', self._t, self._t, covs)
        self._sin_table = np.sin(tmu*self._omega[:,None]) * np.exp(-0.5 * tCt * self._omega[:,None]**2)
        self._cos_table = np.cos(tmu*self._omega[:,None]) * np.exp(-0.5 * tCt * self._omega[:,None]**2)
        return self
    
    def apply(self, weights, phases):
        """Use precomputed sin and cos tables to get sum of sin(omega*x+phase) using the identity
        sin(ax+b) = cos(ax)sin(b) + sin(ax)cos(b)
        """
        if self._sin_table is None:
            raise RuntimeError("Must call an update_table method first!")
        return weights @ (self._cos_table * np.sin(phases[:, None]) + self._sin_table * np.cos(phases[:, None]))
