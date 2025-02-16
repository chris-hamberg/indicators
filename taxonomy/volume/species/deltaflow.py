from indicators.base.core import CoreIndicators
import numpy as np


class DeltaFlow(CoreIndicators):
    """
    Delta Flow indicator.

    Delta Flow is an innovative indicator that measures the difference between buy and sell 
    pressures in the market, providing insight into market sentiment and potential price 
    movements. It is a powerful tool for traders looking to gauge market momentum.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input financial time series, which should include 
    columns for 'open', 'high', 'low', 'close', and 'volume'.
    - smoothing (int): The number of periods to use for smoothing the indicator.
    - mode (str): The mode for smoothing the indicator. Must be one of 'sma', 'ema', or 'tema'.
    - normalization (bool): Applies z-score normalization, if true.

    Returns:
    numpy.ndarray: An array containing the calculated DeltaFlow values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, smoothing, mode):
        if mode not in CoreIndicators._modes: raise ValueError(self._error)
        matrix = self._extractor.extract_matrix(timeseries)
        delta  = matrix[:,4] - matrix[:,1]
        buy_pressure  = np.where(0 < delta, matrix[:,5], 0)
        sell_pressure = np.where(delta < 0, matrix[:,5], 0)
        if smoothing:
            buy_pressure = getattr(self, f"_{mode}")(buy_pressure, smoothing)
            sell_pressure = getattr(self, f"_{mode}")(sell_pressure, smoothing)
        imbalance = np.column_stack((buy_pressure, sell_pressure))
        return imbalance
