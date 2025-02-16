from indicators.base.core import CoreIndicators
from pandas.core.frame import DataFrame
from indicators.stock import Stock
import numpy as np


class Density(CoreIndicators):
    """
    Trading Density Indicator.

    Density measures the trading inactivity by calculating the density of non-zero 
    values in a sliding window of volume data. It provides insights into periods of 
    low trading activity.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): The input financial time series. For a 
    DataFrame, the data should be structured with columns for 'open', 'high', 'low', 
    'close', and 'volume'. For a Stock object, the data should be in OHLCV format.
    - n (int): The number of periods to consider in the calculation. This determines 
    the window size for the sliding window.

    Returns:
    - numpy.ndarray: An array containing the calculated density values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n):
        if isinstance(timeseries, DataFrame):
            timeseries = Stock(dataframe=timeseries)
        dimension = self._extractor.extract_dimension(timeseries, "Volume")
        density   = np.full(dimension.shape[0], np.nan)
        for i in range(n, dimension.shape[0]):
            subspace   = dimension[i-n+1:i+1]
            activity   = np.where(subspace != 0, 1, 0).sum()
            density[i] = activity / n * 100
        return density
