import numpy as np


class Adaptive:


    def _vector_space(self, dimension, vectors, function):
        n, result = vectors[0], np.full(dimension.shape[0], np.nan)
        for vector in range(n, dimension.shape[0]):
            epoch = max(n, vector - vectors[vector] + 1)
            subspace = dimension[epoch:vector + 1]
            result[vector] = function(subspace)
        return result


    def _bivariate_vector_space(self, dx, dy, vectors, fx, fy=None):
        n    = vectors[0]
        x, y = np.full(dx.shape[0], np.nan), np.full(dy.shape[0], np.nan)
        if fy is None: fy = fx
        for vector in range(n, dx.shape[0]):
            epoch = max(n, vector - vectors[vector] + 1)
            subspaceX = dx[epoch:vector + 1]
            subspaceY = dy[epoch:vector + 1]
            x[vector] = fx(subspaceX)
            y[vector] = fy(subspaceY)
        return x, y

        
    def _ema_vector_space(self, dimension, n, vectors, ema):
        for vector in range(n, dimension.shape[0]):
            alpha = 2 / (vectors[vector] + 1)
            ema[vector] = alpha * dimension[vector] + (1-alpha) * ema[vector-1]
        return ema


    def _vectors(self, matrix, n, minlen=5, maxlen=390):
        vectors = np.full(matrix.shape[0], n, dtype=int)
        atr = self._atr(matrix, n)
        atr = self._tools.rescale(atr)
        idx = np.argmax(~np.isnan(atr))
        vectors[idx:] = (atr[idx:] * (maxlen - minlen) + minlen).astype(int)
        vectors[0] = n
        return vectors


    def _std_weighted(self, timeseries, feature, n, omega):
        factor = np.sqrt(self._variance(timeseries, n))
        return self._weighted(feature, factor, omega)


    def _weighted(self, dx, dy, omega):
        weighted = (dx * (1 - omega)) + (dy * omega)
        return weighted
