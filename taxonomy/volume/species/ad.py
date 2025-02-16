from indicators.base.core import CoreIndicators
import numpy as np


class ChaikinAD(CoreIndicators):
    """
    Chaikin Accumulation/Distribution (AD) Line indicator.

    The Chaikin AD Line is a volume-based indicator that measures the cumulative flow
    of money into and out of a security based on the volume and the close price changes.
    It is used to confirm price trends, as divergences between the AD Line and price
    movements can signal potential trend reversals.

    A rising AD Line indicates buying pressure, suggesting bullish strength, while a
    falling AD Line indicates selling pressure, suggesting bearish sentiment. Traders
    often look for divergences between the AD Line and price movements to anticipate
    changes in trend direction.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input financial time series.

    Returns:
    numpy.ndarray: An array containing the Chaikin AD Line values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, _mfv):
        matrix = self._extractor.extract_matrix(timeseries)
        ad = np.full(matrix.shape[0], np.nan)
        ad[0] = _mfv(matrix[:1], smoothing=0, mode="sma")[0]
        for i in range(1, matrix.shape[0]):
            ad[i] = _mfv(matrix[:i+1], smoothing=0, mode="sma")[i] + ad[i-1]
        return ad
