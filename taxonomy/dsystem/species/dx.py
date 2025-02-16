from indicators.base.core import CoreIndicators
import numpy as np


class DX(CoreIndicators):
    """
    Directional Movement Index (DX) indicator.

    The Directional Movement Index (DX) is used to quantify the strength of a 
    price trend. It is part of the Average Directional Index (ADX) system and is 
    calculated using the Positive Directional Index (+DI) and Negative Directional 
    Index (-DI).

    Parameters:
    - timeseries (pd.DataFrame, or Stock): historical price data.
    - n (int): Number of periods to consider for the calculation.

    Returns:
    np.ndarray: Array containing the DX values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, pdi, ndi):
        pdx = np.abs(pdi - ndi)
        ndx = np.abs(pdi + ndi)
        dx  = self._tools.divide(pdx, ndx) * 100
        return dx
