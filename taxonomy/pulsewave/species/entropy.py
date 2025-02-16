from indicators.base.core import CoreIndicators
from scipy.stats import entropy
import numpy as np


class Entropy(CoreIndicators):
    """
    Entropy calculation for financial time series.

    The Entropy class calculates the entropy of a financial time series, which
    represents the degree of randomness or unpredictability in the series. It
    provides methods to calculate entropy in different modes: "flat", "rolling",
    and "expanding".

    Parameters:
    - timeseries (pd.DataFrame, or Stock): Historical price data.
    - n (int): Number of periods to consider for entropy calculation.
    - dimension (str): The data dimension to use for entropy calculation.
    - mode (str): Mode of entropy calculation, one of: "flat", "rolling", "expanding".

    Returns:
    np.ndarray: Array of entropy values calculated for each period in the time series.
    """
    modes = ["flat", "rolling", "expanding"]


    def __init__(self):
        super().__init__()
        self.error = f"Mode must be one of: {Entropy.modes}"


    def __call__(self, timeseries, n, dimension, mode):
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        if mode not in Entropy.modes: raise ValueError(self.error)
        elif mode == "flat":    return self.flat(dimension, n)
        n       = 34 if n < 34 else n
        space   = range(n, dimension.shape[0])
        if mode == "rolling": entropy = self.rolling(  dimension, n, space)
        else:                 entropy = self.expanding(dimension, n, space)
        return np.hstack((np.full(n, np.nan), np.array(entropy)))


    def flat(self, dimension, n):
        minima  = self._tools.nanmin_ignore_nan(dimension)
        maxima  = self._tools.nanmax_ignore_nan(dimension)
        bins    = np.linspace(minima, maxima, n)
        hist, _ = np.histogram(dimension, bins=bins, density=True)
        return entropy(hist, base=2)


    def rolling(self, dimension, n, space):
        return [self.flat(dimension[i-n+1:i+1], n//6) for i in space]


    def expanding(self, dimension, n, space):
        return [self.flat(dimension[:i], n//6) for i in space]
