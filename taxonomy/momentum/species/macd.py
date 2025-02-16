from indicators.base.core import CoreIndicators
import numpy as np


class MACD(CoreIndicators):
    """
    Moving Average Convergence Divergence (MACD) Indicator.

    Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator
    that shows the relationship between two moving averages of a security’s price.
    It consists of the MACD line, Signal line, and Histogram.

    Parameters:
    timeseries (pd.DataFrame, or Stock): The input financial time series.
    short (int): The short-term Exponential Moving Average (EMA) period.
    long (int): The long-term EMA period.
    signal (int): The signal EMA period.
    dimension (str): The target column in the timeseries to use for calculation.

    Returns:
    np.ndarray: Containing the MACD line, Signal line, and Histogram values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, short, long, signal, dimension):
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        short = self._ema(dimension, short)
        long  = self._ema(dimension, long)
        macd  = short - long
        sig   = self._ema(macd, signal)
        hist  = macd - sig
        return np.array((macd, sig, hist)).T
