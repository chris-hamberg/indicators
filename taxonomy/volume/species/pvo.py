from indicators.base.core import CoreIndicators
import numpy as np


class PVO(CoreIndicators):
    """
    PVO (Price-Volume Oscillator) indicator.

    PVO is a technical indicator that attempts to forecast rapid and large price 
    moves.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input financial time series, which 
    should include columns for 'open', 'high', 'low', 'close', and 'volume'.
    - k (int): The threshold (default is 60).

    Returns:
    numpy.ndarray: pvo, signal
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, k):
        matrix = self._extractor.extract_matrix(timeseries)
        fast = self._sma(matrix[:,5],  780)
        slow = self._sma(matrix[:,5], 1950)
        pvo  = (fast / slow - 1) * 100
        return pvo, np.where(pvo >= k, 1, 0)
