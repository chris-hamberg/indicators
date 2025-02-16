from indicators.base.core import CoreIndicators
import numpy as np


class IsothermalSlopes(CoreIndicators):
    """
    Isothermal Slopes Indicator.

    Isothermal slopes are slopes that fall within a certain range of the mean slope
    to filter out extreme values. This indicator helps to identify stable trends
    in a time series.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): Historical market data.
    - n (int): The number of periods over which to calculate the mean and standard deviation.
    - dimension (str): The column name of interest in the time series.
    - clip (float): The number of standard deviations to clip the slopes.
    - smoothing (int): The number of periods over which to smooth the isothermal slopes.
    - mode (str): The mode of smoothing, which can be 'sma' (simple moving average) or 'ema' 
      (exponential moving average).

    Returns:
    - numpy.ndarray: The filtered isothermal slopes.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, slopes, clip, smoothing, mode):
        if mode not in CoreIndicators._modes: raise ValueError(self._error)
        slopes = np.where(np.isnan(slopes), 0, slopes)
        mu, sigma = slopes.mean(), slopes.std()
        isoslopes = np.clip(slopes, mu - clip * sigma, mu + clip * sigma)
        if smoothing: 
            isoslopes = getattr(self, f"_{mode}")(isoslopes, smoothing)
        return isoslopes
