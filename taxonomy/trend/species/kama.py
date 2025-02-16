from indicators.base.core import CoreIndicators
import numpy as np


class KAMA(CoreIndicators):
    """
    Kaufman's Adaptive Moving Average (KAMA) Indicator.

    KAMA is a moving average designed to account for market noise or volatility.
    It closely follows prices when the price swings are relatively small and the noise is low,
    and adjusts when the price swings widen, following prices from a greater distance.
    This trend-following indicator can be used to identify the overall trend, time turning points,
    and filter price movements.

    Args:
        timeseries (pd.DataFrame or Stock): The input financial time series.
        n (int): The number of periods to consider for the KAMA calculation.
        dimension (str): The column name representing the dimension in the timeseries.
                     Typically, 'Close' is used for this parameter.
        fast (int): The fast MA period.
        slow (int): The slow MA period.

    Returns:
        numpy.ndarray: An array containing the Kaufman Adaptive Moving Average values.

    Example:
        kama = ix.trend.kama(stock)
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n, dimension, fast, slow):
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        kama = np.full(dimension.shape[0], np.nan)
        kama[:n] = dimension[:n]
        eratio   = self._er(dimension, n)
        fsmooth_const = 2/(fast + 1)
        ssmooth_const = 2/(slow + 1)
        sc = (eratio * (fsmooth_const - ssmooth_const) + ssmooth_const) ** 2
        for i in range(n, dimension.shape[0]):
            kama[i] = kama[i-1] + sc[i] * (dimension[i] - kama[i - 1])
        return kama
