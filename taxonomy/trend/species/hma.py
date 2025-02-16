from indicators.base.core import CoreIndicators
import numpy as np


class HMA(CoreIndicators):
    """
    Hull Moving Average (HMA) Indicator.

    Hull Moving Average (HMA) is a type of moving average that reduces lag and improves 
    smoothing. It is calculated using weighted moving averages to provide a more 
    responsive trend-following indicator.

    Parameters:
    timeseries (pd.DataFrame, or Stock): The input financial time series.
    n (int): The period of the HMA.
    dimension (str): The column name for calculation.

    Returns:
    np.ndarray: The Hull Moving Average values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n, dimension):
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        omega   = np.arange(1, n // 2 + 1)
        alpha   = np.convolve(dimension, omega[::-1], mode="valid") / omega.sum()
        alpha   = self._tools.resize(dimension, alpha)
        omega   = np.arange(1, n + 1)
        beta    = np.convolve(dimension, omega[::-1], mode="valid") / omega.sum()
        beta    = self._tools.resize(dimension, beta)
        omega   = np.arange(1, int(np.sqrt(n)) + 1)
        delta   = np.abs(2 * alpha - beta)
        epsilon = np.convolve(delta, omega[::-1], mode="valid") / omega.sum()
        wma     = self._tools.resize(dimension, epsilon)
        return wma
