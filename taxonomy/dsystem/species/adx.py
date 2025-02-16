from indicators.base.core import CoreIndicators
import numpy as np


class ADX(CoreIndicators):
    """
    Average Directional Index (ADX) indicator.

    The Average Directional Index (ADX) is a technical indicator used to measure 
    the strength of a trend. It is part of the Directional Movement System and 
    measures the strength of the trend regardless of its direction. 

    Parameters:
    - timeseries (pd.DataFrame, or Stock): Array of Directional Movement Index (DX) values.
    - n (int): Number of periods to consider for the ADX calculation.

    Returns:
    np.ndarray: Array containing the ADX values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, dx, n):
        adx     = np.full(dx.shape[0], 14, dtype=float)
        adx[n:] = self._wilders(dx[n:])
        return adx
