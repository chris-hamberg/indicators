from indicators.base.core import CoreIndicators
import numpy as np


class CleanPivotPoints(CoreIndicators):
    """
    Clean Pivot Points indicator.

    Clean Pivot Points apply smoothing to Pivot Points using a Hamming window. 
    This smoothing can help reduce noise and make the Pivot Points more suitable 
    for trend analysis and decision-making in trading.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): historical price data.
    - scale (int): Scale factor for the calculation.
    - days (int): Number of days to consider for the calculation.
    - pad (int): Number of elements to pad the Hamming window.

    Returns:
    np.ndarray: Array containing the smoothed Pivot Points.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, pps, pad=42):
        pps    = pps.T
        kernel = np.hamming(pad)
        kws    = {"axis":1, "arr":pps, "kernel": kernel}
        dividends = np.apply_along_axis(self.dividends, **kws)
        divisors  = np.apply_along_axis(self.divisors,  **kws)
        pps = self._tools.divide(dividends, divisors)
        levels = pps[:,:-pad]
        return levels.T


    def dividends(self, pps, kernel):
        return np.convolve(pps, kernel, mode="same")


    def divisors(self, pps, kernel):
        return np.convolve(np.ones_like(pps), kernel, mode="same")
