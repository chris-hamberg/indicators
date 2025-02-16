from indicators.base.core import CoreIndicators
import numpy as np


class PhiWave(CoreIndicators):
    """
    PhiWave Indicator.

    PhiWave is a moving average indicator that looks back over Fibonacci intervals 
    instead of fixed periods. It calculates the average of the data points over 
    Fibonacci intervals to provide a unique perspective on trend analysis, and 
    smooth out Fibonacci fractals.

    Parameters:
    timeseries (pd.DataFrame, or Stock): The input financial time series.
    n (int): The number of Fibonacci intervals to consider.
    dimension (str): The target column in the timeseries to use for calculation.

    Returns:
    np.ndarray: The PhiWave values, representing the moving average over Fibonacci intervals.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n, dimension, _cls):
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        phiWave   = np.full(dimension.shape[0], np.nan)
        if len(_cls._fibonacci) != n: 
            _cls._update_fibonacci(self, n)
        for i in range(_cls._fibonacci[-1], dimension.shape[0]):
            phiWave[i] = np.mean(dimension[:i][-1 * _cls._fibonacci])
        return phiWave
