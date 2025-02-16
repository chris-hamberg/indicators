from indicators.base.core import CoreIndicators


class Chaikin(CoreIndicators):
    """
    Chaikin Oscillator indicator.

    The Chaikin Oscillator is a momentum oscillator that measures the 
    accumulation/distribution of a security by comparing the closing price to the 
    trading range over a specific period of time. It is used to confirm trends and 
    potential reversals.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input financial time series, which 
      should include columns for 'open', 'high', 'low', 'close', and 'volume'.
    - n (int): The number of periods to consider for the EMA calculation.

    Returns:
    numpy.ndarray: An array containing the calculated Chaikin Oscillator values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n):
        matrix = self._extractor.extract_matrix(timeseries)
        high, low = matrix[:,2], matrix[:,3]
        iota      = self._ema(high - low, n)
        chaikin   = self._roc(iota, n)
        return chaikin
