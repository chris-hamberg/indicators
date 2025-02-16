from indicators.base.core import CoreIndicators
import numpy as np


class AbsolutePriceOscillator(CoreIndicators):
    """
    Absolute Price Oscillator (APO) indicator.

    APO is a technical analysis indicator that measures the absolute difference between two
    moving averages of an asset's price, typically SMA. It is used to identify changes in
    the strength and direction of a trend.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input financial time series.
    - m (int): The number of periods for the fast SMA. Default is 13.
    - n (int): The number of periods for the slow SMA. Default is 55.
    - dimension (str): The target column in the timeseries for the price data.
    - mode (str): The smoothing mode, either "sma", "ema", or "tema". Default is "sma".

    Returns:
    numpy.ndarray: An array containing the Absolute Price Oscillator values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, m, n, dimension, mode):
        if mode not in CoreIndicators._modes: raise ValueError(self._error)
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        fast = getattr(self, f"_{mode}")(dimension, n=m)
        slow = getattr(self, f"_{mode}")(dimension, n=n)
        apo  = np.abs(fast - slow)
        return apo
