from indicators.base.core import CoreIndicators
import numpy as np


class VWAP(CoreIndicators):
    """
    VWAP (Volume-Weighted Average Price) indicator.

    VWAP is a technical indicator that calculates the average price a security has traded at 
    throughout the day, based on both volume and price. It is often used as a trading benchmark.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input financial time series, which should include 
    columns for 'open', 'high', 'low', 'close', and 'volume'.
    - n (int): The number of periods to consider for the calculation.
    - mode (str): The mode for calculating VWAP. Must be one of 'rolling' or 'flat'.

    Returns:
    numpy.ndarray: An array containing the calculated VWAP values.
    """
    modes = ["rolling", "flat"]


    def __init__(self):
        super().__init__()
        self.error = f"Mode must be one of: {VWAP.modes}"


    def __call__(self, timeseries, n, mode):
        if mode not in VWAP.modes: raise ValueError(self.error)
        matrix = self._extractor.extract_matrix(timeseries)
        if mode == "flat": n = matrix.shape[0] - 1
        typical_price   = (matrix[:,2] + matrix[:,3] + matrix[:,4]) / 3
        volume_weighted = typical_price * matrix[:,5]
        cumvolume       = np.cumsum(matrix[:,5])
        cumweighted     = np.cumsum(volume_weighted)
        volume_diff     = self.difference(cumvolume, n)
        weighted_diff   = self.difference(cumweighted, n)
        vwap            = self._tools.divide(weighted_diff, volume_diff)
        distance        = timeseries.shape[0] - vwap.shape[0]
        vwap            = np.hstack(((np.nan,) * distance, vwap))
        return vwap


    def difference(self, series, n):
        return np.insert(series[n:], 0, 0) - np.insert(series[:-n], 0, 0)
