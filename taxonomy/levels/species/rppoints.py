from indicators.base.core import CoreIndicators
import numpy as np


class RollingPivotPoints(CoreIndicators):
    """
    Rolling Pivot Points indicator.

    Pivot Points are used in technical analysis to determine potential support 
    and resistance levels. They are calculated based on the previous day's high, 
    low, and close prices. Pivot Points can help traders identify key price levels 
    for making trading decisions.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): historical price data.
    - scale (int): Scale factor for the calculation.
    - days (int): Number of days to consider for the calculation.

    Returns:
    np.ndarray: Array containing the calculated pivot point levels.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, scale, days, pad):
        matrix = self._extractor.extract_matrix(timeseries)
        n = min(int(days * 390) // scale, matrix.shape[0])
        indices = self.indices(matrix, n, pad)
        subsets = self.subsets(matrix, indices, n)
        high, low, close = self.ohlc(subsets)
        levels = self.math(high, low, close)
        return np.array(levels).T


    def indices(self, matrix, n, pad):
        indices = np.arange(matrix.shape[0] + pad)
        indices = indices[:,None,None] - np.arange(2*n, n, -1)[None,:,None]
        indices = np.clip(indices, 0, matrix.shape[0] - 1)
        return indices


    def subsets(self, matrix, indices, n):
        subsets = matrix[indices]
        subsets[:2*n] = np.nan
        return subsets


    def ohlc(self, subsets):
        high  = self._tools.nanmax_ignore_nan(subsets[:,:,0,2], axis=1)
        low   = self._tools.nanmin_ignore_nan(subsets[:,:,0,3], axis=1)
        close = subsets[:,-1,0,4]
        return high, low, close


    def math(self, high, low, close):
        pp = (high + low + close) / 3
        r1 = (2 * pp) - low
        r2 = pp + (high - low)
        s1 = (2 * pp) - high 
        s2 = pp - (high - low)
        return s1, s2, pp, r1, r2
