from indicators.base.core import CoreIndicators
import numpy as np


class CommodityChannelIndex(CoreIndicators):
    """
    Commodity Channel Index (CCI) indicator with optional smoothing.

    The Commodity Channel Index (CCI) is a versatile indicator that can be used 
    to identify overbought and oversold levels, as well as trend strength. It 
    calculates the difference between the typical price and its simple moving 
    average, normalized by the mean absolute deviation to account for volatility.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): Historical price data.
    - n (int): Number of periods to consider for the calculation.
    - smoothing (int): Optional smoothing parameter.
    - mode (str): Smoothing mode, one of: "sma", "ema", or "tema".

    Returns:
    np.ndarray: Array containing the CCI values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n, smoothing, mode):
        if mode not in CoreIndicators._modes: raise ValueError(self._error)
        matrix    = self._extractor.extract_matrix(timeseries)
        high, low, close = matrix[:,2], matrix[:,3], matrix[:,4]
        theta     = (high + low + close) / 3
        mu        = self._sma(theta, n)
        deviation = self._sma(np.abs(theta - mu), n)
        cci       = self._tools.divide(theta - mu, 0.015 * deviation)
        if smoothing: 
            scaler = self._tools.sklearn_minmaxscaler(cci)
            cci    = getattr(self, f"_{mode}")(cci, smoothing)
            cci    = scaler.transform(cci.reshape(-1, 1)).flatten()
        return cci
