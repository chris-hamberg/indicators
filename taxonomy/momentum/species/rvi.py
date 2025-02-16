from indicators.base.core import CoreIndicators
import numpy as np


class RelativeVigorIndex(CoreIndicators):
    """
    Relative Vigor Index (RVI) indicator.

    The Relative Vigor Index (RVI) is a technical indicator that measures the 
    strength of a trend by comparing the closing price to the trading range of a 
    security. It is used to identify overbought and oversold conditions and 
    potential trend reversals.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): Historical price data.
    - n (int): Number of periods to consider for the RVI calculation (default is 14).
    - divisor (int): Divisor used in the RVI calculation (default is 6).

    Returns:
    (np.ndarray, np.ndarray): Tuple containing the RVI values and the signal values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n=14, divisor=6):

        matrix = self._extractor.extract_matrix(timeseries)
        open, high, low, close = matrix[:,1],matrix[:,2],matrix[:,3],matrix[:,4]

        a = self.zeta(close - open, divisor)
        b = self.zeta(high  - low,  divisor)

        a = self._sma(a, n)
        b = self._sma(b, n)

        rvi = self._tools.divide(a, b)

        signal = self.zeta(rvi, divisor)

        return rvi, signal


    def zeta(self, dimension, divisor):
        inner_term = 2 * (np.roll(dimension, 1) + np.roll(dimension, 2))
        y = dimension + inner_term + np.roll(dimension, 3)
        y /= divisor
        return y
