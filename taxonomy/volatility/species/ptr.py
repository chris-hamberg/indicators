from indicators.base.core import CoreIndicators
import numpy as np


class PercentTrueRange(CoreIndicators):
    """
    Percent True Range (PTR)

    Computes the Percent True Range (PTR) indicator by measuring the percentage 
    difference between the highest high and the lowest low over a specified 
    number of periods. This indicator is useful for evaluating the relative 
    volatility of a financial instrument.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input financial time series data.
    - n (int): The periods over which to calculate the high and low extremes.

    Returns:
    numpy.ndarray: An array containing the computed Percent True Range values.
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
