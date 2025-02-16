from indicators.base.core import CoreIndicators
import numpy as np


class Coeffvar(CoreIndicators):
    """
    Coefficient of Variation (Coeffvar) indicator.

    The Coeffvar indicator calculates the coefficient of variation, which is a 
    normalized measure of dispersion of a probability distribution. It is useful 
    for comparing the volatility or risk between different instruments or markets.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input financial time series, which 
      should include columns for 'open', 'high', 'low', 'close', and 'volume'.
    - n (int): The number of periods to consider for the moving average and variance.
    - dimension (str): The column name representing the dimension in the 
      timeseries. This could be 'close' for price data or 'volume' for volume data.
    - mode (str): The method for computing cv ("rolling", "expanding", or "flat".)

    Returns:
    numpy.ndarray: An array containing the calculated Coeffvar values as a percentage.
    """

    modes = ["rolling", "expanding", "flat"]

    def __init__(self):
        super().__init__()
        self.error = f"`mode` must be one of: {Coeffvar.modes}"


    def __call__(self, timeseries, n, dimension, mode="rolling"):
        if mode not in Coeffvar.modes: raise ValueError(self.error)
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        if   mode == "rolling":   mean, var = self.rolling(dimension, n)
        elif mode == "expanding": mean, var = self.expanding(dimension)
        elif mode == "flat":      
            mean, var = np.nanmean(dimension), np.nanvar(dimension)
        std       = np.sqrt(var)
        coeffvar  = self._tools.divide(std, mean)
        coeffvar  = np.where(np.isnan(coeffvar), 0, coeffvar)
        return np.round(coeffvar * 100, 2)


    def rolling(self, dimension, n):
        sma      = self._sma(dimension, n)
        var      = self._variance(dimension, n)
        return sma, var


    def expanding(self, dimension):
        mean = np.full(dimension.shape[0], np.nan)
        var  = np.full(dimension.shape[0], np.nan)
        for i in range(1, dimension.shape[0]):
            mean[i] = np.nanmean(dimension[:i + 1])
            var[i]  = np.nanvar(dimension[:i + 1])
        return mean, var
