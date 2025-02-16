from indicators.base.core import CoreIndicators


class MFV(CoreIndicators):
    """
    Money Flow Volume (MFV) indicator.

    MFV is a technical analysis indicator that combines volume and the Money
    Flow Multiplier (MFM) to measure the strength of money flowing in and out
    of a financial asset. It is calculated as the product of the volume and
    the Money Flow Multiplier (MFM).

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input financial time series.
    - smoothing (int): The number of periods to consider for smoothing the MFV
                       values. Default is 0 (no smoothing).
    - mode (str): The smoothing mode to use. Supported modes are "sma" (Simple
                  Moving Average), "ema" (Exponential Moving Average), etc.

    Returns:
    numpy.ndarray: An array containing the Money Flow Volume (MFV) values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, smoothing, mode):
        if mode not in self._modes: raise ValueError(self._error)
        matrix = self._extractor.extract_matrix(timeseries)
        volume = matrix[:,5]
        mfm    = self._mfm(timeseries, smoothing=0)
        mfv    = volume * mfm
        if smoothing: mfv = getattr(self, f"_{mode}")(mfv, smoothing)
        return mfv
