from indicators.base.core import CoreIndicators
import numpy as np


class IchimokuCloud(CoreIndicators):
    """
    Ichimoku Cloud Indicator.

    Ichimoku Cloud is a charting system that provides valuable support and resistance levels.
    It calculates several lines:
    - Tenkan Sen (Conversion Line): Short-term equilibrium level, similar to a fast-moving average.
    - Kijun Sen (Base Line): Medium-term equilibrium level, similar to a slow-moving average.
    - Senkou Span A: Average of Tenkan Sen and Kijun Sen, projected forward to indicate future potential support or resistance.
    - Senkou Span B: Average of the highest high and lowest low over a longer period, also projected forward.
    - Chikou Span: Closing price plotted in the past, used to identify recent historical support or resistance.

    Traders typically look for the following signals:
    - Price above the cloud (Span A and Span B) indicates a bullish trend, with the cloud acting as support.
    - Price below the cloud indicates a bearish trend, with the cloud acting as resistance.
    - The Tenkan Sen and Kijun Sen lines can act as immediate support or resistance levels.
    - Crossovers of the Tenkan Sen and Kijun Sen lines can signal potential trend changes.

    Parameters:
    timeseries (pd.DataFrame, or Stock): The input financial time series.
    tenkan (int): The Tenkan Sen period, typically set to 9.
    kijun (int): The Kijun Sen period, typically set to 26.
    
    Returns:
    np.ndarray: A 2D array containing the Tenkan Sen, Kijun Sen, Senkou Span A, Senkou Span B, and Chikou Span values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, tenkan, kijun):
        dimension = self._extractor.extract_dimension(timeseries)
        matrix = self._extractor.extract_matrix(timeseries)
        
        high, low, close = matrix[:,2], matrix[:,3], matrix[:,4]
        
        tenkan_high   = self._maxima(high, tenkan)
        tenkan_low    = self._minima(low,  tenkan)
        tenkan_series = (tenkan_high + tenkan_low) / 2

        kijun_high    = self._maxima(high, kijun)
        kijun_low     = self._minima(low,  kijun)
        kijun_series  = (kijun_high + kijun_low) / 2

        span_a = (tenkan_series + kijun_series) / 2
        span_a = np.roll(span_a, kijun)[kijun:]
        span_a = self._tools.resize(dimension, span_a)

        high_b = self._maxima(high, kijun * 2)
        low_b  = self._minima(low,  kijun * 2)
        span_b = (high_b + low_b) / 2
        span_b = np.roll(span_b, kijun)[kijun:]
        span_b = self._tools.resize(dimension, span_b)
        
        pad = np.full(tenkan, np.nan)
        chikou = np.hstack((pad, close))
        
        tenkan_series = np.hstack((tenkan_series, pad))
        kijun_series  = np.hstack((kijun_series, pad))
        span_a        = np.hstack((span_a, pad))
        span_b        = np.hstack((span_b, pad))
    
        return np.column_stack((tenkan_series, kijun_series, span_a, span_b, chikou))
