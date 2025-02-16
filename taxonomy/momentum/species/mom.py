from indicators.base.core import CoreIndicators
import numpy as np


class MOM(CoreIndicators):
    """
    Momentum (MOM) Indicator.

    Calculates the Momentum of a given timeseries. Momentum is a leading indicator
    that measures the rate of change of a security's price. It is used to detect
    trend strength and potential reversal points.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input financial time series.
    - n (int): The number of periods to consider for the Momentum calculation.
    - dimension (str): The column name representing the dimension in the timeseries.
                       Typically, 'Close' is used for this parameter.
    - smoothing (int): The number of periods to consider for smoothing the MOM
                       values. Default is 0 (no smoothing).
    - mode (str): The smoothing mode to use. Supported modes are "sma" (Simple
                  Moving Average), "ema" (Exponential Moving Average), etc.

    Returns:
    numpy.ndarray: An array containing the Momentum values.
    """ 
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n, dimension, smoothing, mode):
        if mode not in CoreIndicators._modes: raise ValueError(self._error)
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        mom = np.full(dimension.shape[0], np.nan)
        mom[n:] = (dimension - np.roll(dimension, n))[n:]
        scaler = self._tools.sklearn_minmaxscaler(mom, -1, 1)
        if smoothing: mom = getattr(self, f"_{mode}")(mom, smoothing)
        mom = scaler.transform(mom.reshape(-1, 1)).flatten()
        return mom
