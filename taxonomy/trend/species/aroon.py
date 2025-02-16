from indicators.base.core import CoreIndicators
import numpy as np


class AROON(CoreIndicators):
    """
    AROON (Aroon Indicator) is used to identify trend changes in the price of an asset 
    and to gauge the strength of the trend. Consisting of two lines: Aroon Up and Aroon Down, 
    Aroon Up measures the number of periods since the highest high within the given period, 
    while Aroon Down measures the number of periods since the lowest low within the same period.

    When Aroon Up crosses above Aroon Down, it is considered a bullish signal, indicating a 
    potential uptrend. Conversely, when Aroon Down crosses above Aroon Up, it is considered a 
    bearish signal, indicating a potential downtrend.

    Aroon values range from 0 to 100. A reading of 100 indicates that a new high (Aroon Up) or 
    low (Aroon Down) has occurred within the specified number of periods, while a reading of 0 
    indicates that no new high or low has occurred.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input financial time series.
    - n (int): The number of periods to consider for calculating the Aroon indicator.

    Returns:
    - numpy.ndarray: A 2D NumPy array containing the Aroon Up and Aroon Down values for each 
    period in the input time series. Each row of the array corresponds to a period, and the 
    columns represent Aroon Up and Aroon Down values respectively.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n):
        matrix = self._extractor.extract_matrix(timeseries)
        high, low = matrix[:,2], matrix[:,3]
        rolling_high, u = self._windows.rolling(high, n + 1)        
        rolling_low,  d = self._windows.rolling(low,  n + 1)
        u[n:] = np.argmax(rolling_high[:,::-1], axis=1)
        d[n:] = np.argmin(rolling_low[:,::-1],  axis=1)
        u = (n - u) / n * 100
        d = (n - d) / n * 100
        return np.column_stack((u, d))
