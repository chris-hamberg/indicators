from indicators.base.core import CoreIndicators
import numpy as np


class DM(CoreIndicators):
    """
    Directional Movement (DM) indicator.

    The Directional Movement (DM) indicator is used to identify the strength of 
    price movement in a particular direction. It consists of two components: 
    Positive Directional Movement (+DM) and Negative Directional Movement (-DM). 
    These components are used in conjunction with the True Range (TR) to calculate 
    the Average True Range (ATR) and the Directional Movement Index (DX).

    Parameters:
    - timeseries (pd.DataFrame, or Stock): historical price data.
    - n (int): Number of periods to consider for the calculation.

    Returns:
    (np.ndarray, np.ndarray): Tuple containing the +DM and -DM values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n):
        matrix = self._extractor.extract_matrix(timeseries)
        high, low, close = matrix[:,2], matrix[:,3], matrix[:,4]
        highs, lows      = self.rules(high, low)
        pdm, ndm         = self.xdms(highs, lows, n)
        return pdm, ndm

    
    def rules(self, high, low):
        #highs = high - np.roll(high, 1)
        highs = high[1:] - high[:-1]
        highs = np.hstack(((np.nan,), highs))
        #lows  = np.roll(low, 1) - low
        lows  = low[1:] - low[:-1]
        lows  = np.hstack(((np.nan,), lows))
        return highs, lows


    def xdms(self, highs, lows, n):
        pdm = np.where(highs > lows, highs, 0)
        ndm = np.where(lows > highs, lows, 0)
        pdm = self._wilders(pdm, n)
        ndm = self._wilders(ndm, n)
        return pdm, ndm
