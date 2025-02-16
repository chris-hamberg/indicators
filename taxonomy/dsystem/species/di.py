from indicators.base.core import CoreIndicators


class DI(CoreIndicators):
    """
    Directional Movement Index (DI).

    The Directional Movement Index (DI) is used to determine the direction of the 
    trend. It consists of two lines: the Positive Directional Index (+DI) and the 
    Negative Directional Index (-DI). These lines are used to determine the 
    strength of the trend based on the price movements.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): Array of historical price data.
    - n (int): Number of periods to consider for the ATR calculation.

    Returns:
    (np.ndarray, np.ndarray): Tuple containing the +DI and -DI values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, pdm, ndm, n):
        atr = self._atr(timeseries, n)
        pdi = self._tools.divide(pdm, atr) * 100
        ndi = self._tools.divide(ndm, atr) * 100
        return pdi, ndi
