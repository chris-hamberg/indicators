from indicators.base.core import CoreIndicators
import numpy as np


class PercentTrueRange(CoreIndicators):
    """
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n):

        matrix = self._extractor.extract_matrix(timeseries)
        try: high, low = matrix[:,2], matrix[:,3]
        except KeyError: return timeseries

        high = self._maxima(high, n)
        low  = self._minima(low,  n)

        return np.divide(high - low, low + 1e-20) * 100
