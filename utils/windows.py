from numpy.lib.stride_tricks import sliding_window_view
import numpy as np


class Windows:


    def rolling(self, dimension, n):
        container = np.full(dimension.shape[0], np.nan)
        shape     = dimension.shape[0] - n + 1, n
        strides   = np.full(2, dimension.strides[0])
        params    = {"shape": shape, "strides": strides}
        try: rolling = np.lib.stride_tricks.as_strided(dimension, **params)
        except ValueError: rolling = container
        return rolling, container


    def rolling_function(self, dimension, n, function):
        windows  = sliding_window_view(dimension, window_shape=(n,), axis=0)
        computed = np.apply_along_axis(function, axis=1, arr=windows)
        result   = np.full_like(dimension, fill_value=np.nan)
        result[n-1:] = computed
        return result


    def expanding_function(self, dimension, n, function):
        result = [funcion(dimension[:i]) for i in range(1, dimension.shape[0])]
        return np.hstack(((np.nan,), np.array(result)))
