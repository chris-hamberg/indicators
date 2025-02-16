from indicators.base.core import CoreIndicators
import numpy as np


class PhiBands(CoreIndicators):
    """
    PhiBands indicator.

    PhiBands (Fibonacci Bands) is a technical indicator that uses Fibonacci retracement levels 
    to create bands around a simple moving average (SMA). These bands can help traders identify 
    potential support and resistance levels.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input financial time series, which should include 
    columns for 'open', 'high', 'low', 'close', and 'volume'.
    - n (int): The number of periods to consider for the moving average.
    - dimension (str): The column name representing the dimension in the timeseries. Typically, 
    'close' is used for this parameter.
    - factor (float): The multiplier factor to apply to the standard deviation to calculate the 
    bands.

    Returns:
    numpy.ndarray: An array containing the calculated PhiBands values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n, dimension, factor):
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        bands     = self.container(dimension)
        sma, std  = self.components(n, dimension, factor)
        base      = 2 * CoreIndicators._retracements.shape[0]
        bands     = self.make(bands, sma, std, base) 
        return bands.T


    def container(self, dimension):
        shape = dimension.shape[0], 2 * CoreIndicators._retracements.shape[0]
        bands = np.full(shape, np.nan)
        return bands


    def components(self, n, dimension, factor):
        sma = self._sma(dimension, n), 
        var = self._variance(dimension, n)
        std = np.sqrt(var) * factor
        return sma, std


    def make(self, bands, sma, std, base):
        for e, k in enumerate(CoreIndicators._retracements):
            d = base - (e + 1)
            bands[:, d] = sma + k * std
            bands[:, e] = sma - k * std
        return bands
