from indicators.base.core import CoreIndicators
import numpy as np


class Slope(CoreIndicators):
    """
    Rolling Slope indicator for trend detection.

    This indicator calculates the rolling slope of a financial time series, which 
    is useful for detecting trends. The slope is calculated as the ratio of the 
    covariance of the series with a sequence of integers to the variance of the 
    sequence of integers.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input financial time series.
    - n (int): The number of periods for the rolling slope calculation.
    - dimension (str): The column name representing the dimension of interest in 
    the timeseries.

    Returns:
    numpy.ndarray: An array containing the computed slopes, indicating the trend 
    direction.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n, dimension):
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        dx, dy, divisor = self.deviations(dimension, n)
        slope = np.where(divisor != 0, np.sum(dx*dy, axis=1)/divisor, np.nan)
        slope = self._tools.resize(dimension, slope)
        return slope


    def deviations(self, dimension, n): 
        (*x,), (y, _) = np.arange(n), self._windows.rolling(dimension, n)
        x = np.array(x)
        dx, dy  = x - x.mean(), y - y.mean(axis=1)[:, None]
        divisor = np.sum(dx**2)
        return dx, dy, divisor
