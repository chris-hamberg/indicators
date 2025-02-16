from indicators.base.core import CoreIndicators
import numpy as np


class WMA(CoreIndicators):
    """
    Weighted Moving Average (WMA) Indicator.


    Weighted Moving Average (WMA) is a type of moving average that assigns more weight
    to recent data points, making it more responsive to price changes.

    Parameters:
    timeseries (pd.DataFrame, or Stock): The input financial time series.
    n (int): The period of the WMA, indicating the number of data points to include in the calculation.
    dimension (str): The target column in the timeseries to use for calculation.

    Returns:
    np.ndarray: The Weighted Moving Average values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n, dimension):
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        omega = np.arange(1, n + 1)
        wma   = np.convolve(dimension, omega[::-1], mode="valid") / omega.sum()
        wma   = self._tools.resize(dimension, wma)
        return wma
