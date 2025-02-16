from indicators.base.core import CoreIndicators
import numpy as np


class VWMA(CoreIndicators):
    """
    Volume Weighted Moving Average (VWMA) indicator.

    VWMA calculates the average price of a financial instrument based on both 
    volume and price. It gives more weight to periods with higher volume.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input financial time series, which 
    should include columns for 'open', 'high', 'low', 'close', and 'volume'.
    - n (int): The number of periods to consider in the calculation.

    Returns:
    numpy.ndarray: An array containing the computed VWMA values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n):
        matrix = self._extractor.extract_matrix(timeseries)
        price, volume = matrix[:,4], matrix[:,5]
        X      = self._summation(price * volume, n)
        V      = self._summation(volume, n)
        vwma   = self._tools.divide(X, V)
        vwma   = self._tools.resize(matrix, vwma)
        return vwma
