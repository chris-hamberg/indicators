from indicators.base.core import CoreIndicators
import pandas as pd
import numpy as np


class ResonantTrendVelocity(CoreIndicators):
    """
    Resonant Trend Velocity.

    This indicator calculates the rolling Pearson Correlation Coefficient between 
    price values and timestamps mapped to integers. If both increase at the same 
    time, it suggests an uptrend in the market.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input financial time series.
    - n (int): The number of periods for the rolling correlation calculation.
    - dimension (str): The column name representing the target dimension in the 
    timeseries.
    - y (pd.DataFrame or Stock): The optional input for the second variable in 
    the correlation calculation. If None, it defaults to an increasing series 
    of integers.
    - y_dimension (str): The column name representing the second variable in the 
    correlation calculation, if `y` is provided.
    - smoothing (int): The number of periods for smoothing the result using either 
    SMA or EMA.
    - mode (str): The smoothing mode, either "sma" for Simple Moving Average or 
      "ema" for Exponential Moving Average.

    Returns:
    numpy.ndarray: An array containing the Resonant Trend Velocity values, which 
    indicate the correlation between price values and timestamps.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n, dimension, y, y_dimension, smoothing, mode):
        if not mode in CoreIndicators._modes: raise ValueError(self._error)
        x    = self._extractor.extract_dimension(timeseries, dimension)
        x, y = pd.Series(x), self.y(x, y, y_dimension, dimension)
        resonantTrendVelocity = x.rolling(n).corr(y).values
        if smoothing: 
            function = getattr(self, f"_{mode}")
            resonantTrendVelocity = function(resonantTrendVelocity, smoothing)
        return resonantTrendVelocity


    def y(self, x, y, y_dimension, x_dimension):
        if y is not None:
            if y_dimension is None: y_dimension = x_dimension
            y = self._extractor.extract_dimension(y, y_dimension)
        else: 
            y = np.arange(x.shape[0])
        return pd.Series(y)
