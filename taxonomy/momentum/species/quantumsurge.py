from indicators.taxonomy.momentum.core.qstrend import SurgeTrend
from indicators.taxonomy.momentum.core.obv import CoreOBV
from indicators.base.core import CoreIndicators
import numpy as np


class QuantumSurge(CoreIndicators):
    """Quantum Surge indicator, a rolling On-Balance Volume (OBV) with trend integration.

    Quantum Surge is a variation of the On-Balance Volume (OBV) indicator that 
    includes a trend factor in its calculation. It calculates OBV values over a 
    rolling window and integrates trend information to provide insights into 
    buying and selling pressures.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): Historical price data.
    - n (int): Number of periods to consider for the rolling calculation.
    - trend_factor (int): Non-negative integer representing the trend factor.
    - weight (float): Weighting factor for volume, ranging from 0.0 to 1.0.
    - normalize (bool): Flag indicating whether to normalize the OBV values.

    Returns:
    np.ndarray: Array containing the Quantum Surge values.
    """
    def __init__(self):
        super().__init__()
        self.trend = SurgeTrend()
        self.core  = CoreOBV()


    def __call__(self, timeseries, n, trend_factor, weight, normalize):
        self.trend.factor = trend_factor
        matrix = self._extractor.extract_matrix(timeseries)
        obv    = self.rolling(matrix, n, weight)
        return self._tools.rescale(obv) if normalize else obv


    def rolling(self, matrix, n, weight):
        obv = np.full(matrix.shape[0], np.nan)
        for i in range(n, matrix.shape[0]):
            subset = matrix[i-n+1:i+1]
            obv[i], theta = self.core.compute(subset, weight, mode="rolling")
            self.trend_factor(obv, theta, i, n)
        return obv


    def trend_factor(self, obv, theta, i, n):
        uptrend, downtrend = self.trend.direction(theta)
        strength = self.trend.strength(obv, i, n)
        self.trend.update(uptrend, downtrend, strength, obv, i)
