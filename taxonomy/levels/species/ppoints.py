from indicators.base.core import CoreIndicators
from indicators.stock import Stock
import numpy as np


class PivotPoints(CoreIndicators):
    """
    Pivot Points indicator.

    Pivot Points are used in technical analysis to determine potential support 
    and resistance levels. They are calculated based on the previous day's high, 
    low, and close prices. Pivot Points can help traders identify key price levels 
    for making trading decisions.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): historical price data.

    Returns:
    np.ndarray: Array containing the calculated pivot point levels.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n=390):
        if not isinstance(timeseries, Stock):
            timeseries = Stock("", dataframe=timeseries)
        scaled = timeseries.scale(n)
        matrix = self._extractor.extract_matrix(scaled)
        levels = self.math(matrix[:,2], matrix[:,3], matrix[:,4])
        levels = self.expand(levels, n)
        levels = self.shift(levels, n)
        return levels


    def math(self, high, low, close):
        pp = (high + low + close) / 3
        r1 = (2 * pp) - low
        r2 = pp + (high - low)
        s1 = (2 * pp) - high 
        s2 = pp - (high - low)
        return s1, s2, pp, r1, r2


    def expand(self, levels, n):
        expanded = np.full((levels[0].shape[0] * n, 5), np.nan)
        for e, level in enumerate(levels):
            expanded[:,e] = np.repeat(level, n)
        return expanded


    def shift(self, levels, n):
        pad = np.full((n, levels.shape[1]), np.nan)
        shifted = np.vstack((pad, levels))
        return shifted
