from indicators.base.core import CoreIndicators
import numpy as np


class PercentagePriceOscillator(CoreIndicators):
    """
    Percentage Price Oscillator (PPO) Indicator.

    Calculates the Percentage Price Oscillator of a given timeseries. The PPO is a
    momentum oscillator that measures the difference between two moving averages
    as a percentage of the slower moving average. It is used to identify trend
    strength, direction, and potential reversal points.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input financial time series.
    - fast (int): The number of periods for the fast EMA calculation.
    - slow (int): The number of periods for the slow EMA calculation.
    - signal (int): The number of periods for the signal EMA calculation.
    - dimension (str): The column name representing the dimension in the timeseries.
                       Typically, 'Close' is used for this parameter.
    - mode (str): The smoothing mode to use for the EMA calculations. Supported
                  modes are "sma" (Simple Moving Average), "ema" (Exponential
                  Moving Average), etc.

    Returns:
    numpy.ndarray: An array containing the Percentage Price Oscillator values
                   and the signal line values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, fast, slow, signal, dimension, mode):
        if mode not in CoreIndicators._modes: raise ValueError(self._error)
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        fast   = getattr(self, f"_{mode}")(dimension, fast)
        slow   = getattr(self, f"_{mode}")(dimension, slow)
        ppo    = (fast - slow) / slow * 100
        signal = getattr(self, f"_{mode}")(ppo, signal)
        return np.array((ppo, signal)).T
