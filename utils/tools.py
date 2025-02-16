from sklearn.preprocessing import MinMaxScaler
import numpy as np
import warnings


class Tools:


    def __init__(self):
        self.error = "`n` must be greater than 3."


    def fibonacci(self, n, size):
        if size == n: return None
        elif n < 3: raise ValueError(self.error)
        numbers = np.full(n, 1)
        for i in range(2, numbers.shape[0]):
            numbers[i] = numbers[i - 2] + numbers[i - 1]
        return numbers[2:]


    def rescale(self, timeseries, minima=None, maxima=None):
        imputed = np.where(np.isfinite(timeseries), timeseries, np.nan)
        if np.isnan(imputed).all(): return np.full_like(timeseries, np.nan)
        minima  = minima if minima is not None else np.nanmin(imputed)
        maxima  = maxima if maxima is not None else np.nanmax(imputed)
        return self.divide(timeseries - minima, maxima - minima)


    def sklearn_minmaxscaler(self, timeseries, minima=None, maxima=None):
        imputed = np.where(np.isfinite(timeseries), timeseries, np.nan)
        if np.isnan(imputed).all(): return np.full_like(timeseries, np.nan)
        minima  = minima if minima is not None else np.nanmin(imputed)
        maxima  = maxima if maxima is not None else np.nanmax(imputed)
        scaler  = MinMaxScaler(feature_range=(minima, maxima))
        scaler.fit(timeseries.reshape(-1, 1))
        return scaler
        

    def nanmax_ignore_nan(self, array, axis=0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            minima  = np.nanmin(array)
            imputed = np.where(np.isnan(array), minima, array)
            maxima  = np.nanmax(imputed, axis=axis)
        return maxima


    def nanmin_ignore_nan(self, array, axis=0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            maxima  = np.nanmax(array)
            imputed = np.where(np.isnan(array), maxima, array)
            minima  = np.nanmin(imputed, axis=axis)
        return minima


    def resize(self, dimension, indicator):
        difference = dimension.shape[0] - indicator.shape[0]
        if 0 <= difference: 
            pad = np.full(difference, np.nan)
            indicator = np.hstack((pad, indicator))
        else: 
            indicator = indicator[abs(difference):]
        return indicator


    def divide(self, dividend, divisor):
        dimension = np.full_like(dividend, np.nan)
        theta = (divisor != 0) & (~np.isnan(divisor)) & (~np.isnan(dividend))
        return np.divide(dividend, divisor, where=theta, out=dimension)
