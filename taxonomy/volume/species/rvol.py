from indicators.base.core import CoreIndicators
import numpy as np


class RVOL(CoreIndicators):
    """
    RVOL (Relative Volume) indicator.

    RVOL is a technical indicator that measures the current 390-min cumulative
    volume over the 10-day average volume.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input financial time series, which 
    should include columns for 'open', 'high', 'low', 'close', and 'volume'.
    - n (int): The average volume lookback (default is 10 (days)).

    Returns:
    numpy.ndarray: rvol, rolling daily, n-day mean (rolling)
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n):
        volume = self._extractor.extract_dimension(timeseries, 
                dimension="Volume")
        daily  = self._summation(volume, 390)
        mean   = self._sma(daily, 390 * n)
        rvol   = daily / mean
        return np.column_stack((rvol, daily, mean))
