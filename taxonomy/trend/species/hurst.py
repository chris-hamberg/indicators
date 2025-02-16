from indicators.base.core import CoreIndicators
from scipy.stats import linregress
import numpy as np


class Hurst(CoreIndicators):
    """
    Calculate the Hurst exponent for a financial time series.

    The Hurst exponent is a measure of long-term memory in time series data.
    It quantifies the relative tendency of a time series to either cluster in
    a direction (trend) or to revert to a mean.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): The input financial time series.
    - n (int): The number of periods over which to calculate the Hurst exponent.
    - dimension (str): The dimension of interest in the time series, e.g., 'close' for 
      closing prices.
    - mode (str): The computation mode, which can be 'flat' (the entire dataset) or 
      'rolling' (over a rolling window).

    Returns:
    - float or numpy.ndarray: The computed Hurst exponent. If mode is 'rolling', an 
      array of Hurst exponent values is returned.
    """
    modes = ["flat", "rolling"]


    def __init__(self): 
        super().__init__()
        self.error = f"Mode must be one of: {Hurst.modes}"


    def __call__(self, timeseries, n, dimension, mode):

        if not mode in self.modes: raise ValueError(self.error)
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        dimension = np.where(np.isnan(dimension), 0, dimension)

        lags = np.arange(1, n + 1)
        n *= 2

        if mode == "flat":
            dimension = dimension[-n:]
            n = dimension.shape[0]

        taus  = self.taus(dimension, lags, n)

        hurst = self.hurst(dimension, lags, taus, n)

        return hurst if mode == "rolling" else hurst[-1]
    

    def taus(self, dimension, lags, n):
        shape   = dimension.shape[0] - n + 1, n
        strides = np.full(2, dimension.strides[0])
        params  = {"shape": shape, "strides": strides}
        windows = np.lib.stride_tricks.as_strided(dimension, **params)
        taus    = np.full((windows.shape[0], lags.shape[0]), np.nan)
        # Vectorize this loop
        for lag in lags:
            taus[:,lag - 1] = np.nanstd(windows[:,lag:] - windows[:,:-lag], axis=1)
        return taus


    def hurst(self, dimension, lags, taus, n):
        lags, tau = np.log(lags), np.log(taus)
        hurst = np.full_like(dimension, np.nan)
        for i in range(taus.shape[0]):
            hurst[i + n - 1] = linregress(lags, tau[i,:]).slope
        return hurst


    def __taus_optimization(self, dimension, lags, n):
        shape   = dimension.shape[0] - n + 1, n
        strides = np.full(2, dimension.strides[0])
        params  = {"shape": shape, "strides": strides}
        windows = np.lib.stride_tricks.as_strided(dimension, **params)

        differences = windows[:, lags[:,None]] - windows[:, lags[:,None] - lags]
        
        import pdb; pdb.set_trace()

        taus = np.nanstd(taus, axis=2)

        return taus
