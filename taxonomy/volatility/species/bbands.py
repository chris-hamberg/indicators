from indicators.base.core import CoreIndicators
import numpy as np


class BBands(CoreIndicators):
    """
    Bollinger Bands (BBands) indicator.

    Bollinger Bands are volatility bands placed above and below a moving average. 
    They are used to measure volatility and identify overbought or oversold 
    conditions.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input financial time series, which 
    should include columns for 'open', 'high', 'low', 'close', and 'volume'.
    - n (int): The number of periods to consider for the moving average.
    - k (float): The number of standard deviations to add or subtract from the 
    moving average to calculate the upper and lower bands.
    - dimension (str): The column name representing the dimension in the 
    timeseries. Typically, 'close' is used for this parameter.

    Returns:
    numpy.ndarray: An array containing the lower band, the moving average (mid), 
    and the upper band.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n, k, dimension):
        mid = self._sma(timeseries, n, dimension)
        var = self._variance(mid, n, dimension)
        std = np.sqrt(var)
        try: 
            upper,  lower = mid + (k * std), mid - (k * std)
        except ValueError:
            upper = lower = np.full(dimension.shape[0], np.nan)
        return np.array((lower, mid, upper))
