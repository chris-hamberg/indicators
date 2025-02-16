from indicators.base.core import CoreIndicators
from scipy.signal import argrelextrema
from scipy.stats import linregress
import numpy as np


class Denoiser(CoreIndicators):
    """
    Denoise a financial time series.

    This denoising method uses linear regression to fit lines to local minima and maxima
    in the time series, creating a smoothed version of the data.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): The input financial time series.
    - n (int): The order parameter for finding local extrema.
    - dimension (str): The column name representing the dimension in the timeseries.

    Returns:
    np.array: A denoised version of the input time series.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n, dimension):
        regression = []
        dimension  = self._extractor.extract_dimension(timeseries, dimension)
        minima     = argrelextrema(dimension, np.less,    order=n)[0]
        maxima     = argrelextrema(dimension, np.greater, order=n)[0]
        pts = [0] + sorted(list(set(minima) | set(maxima))) + [dimension.shape[0]]
        for i in range(len(pts)-1):
            start, end = pts[i], pts[i+1]
            X = np.arange(start, end)
            if X.shape[0] < 2: continue
            y = dimension[start:end]
            slope, intercept, _, _, _ = linregress(X, y)
            fit = slope * X + intercept
            regression.extend(fit)
        regression = np.array(regression)
        return regression
