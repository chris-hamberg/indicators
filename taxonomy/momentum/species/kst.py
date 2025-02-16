from indicators.base.core import CoreIndicators


class KnowSureThing(CoreIndicators):
    """
    Know Sure Thing (KST) indicator.

    The Know Sure Thing (KST) indicator is a momentum oscillator that combines 
    four different smoothed rate of change (ROC) indicators with different periods 
    into a single indicator. It is used to identify potential trend reversals and 
    to confirm the strength of a current trend.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): Historical price data.
    - dimension (str): Dimension of the timeseries to consider, e.g., "close" or "high".
    - roc (tuple): Tuple of four integers representing the periods for the ROC calculations.
    - sma (tuple): Tuple of four integers representing the periods for the SMA calculations.
    - signal (int): Period for the signal line calculation (default is 9).

    Returns:
    (np.ndarray, np.ndarray): Tuple containing the KST values and the signal line values.
    """
    def __init__(self):
        super().__init__()
        self.defaults = {
                "roc": (10, 15, 20, 30),
                "sma": (10, 10, 10, 15)}


    def __call__(self, timeseries, dimension, **kwargs):
        dimension = self._extractor.extract_dimension(timeseries, dimension)
        roc = kwargs.get("roc", self.defaults.get("roc"))
        sma = kwargs.get("sma", self.defaults.get("sma"))
        sig = kwargs.get("signal", 9)
        f1  = self._sma(self._roc(dimension, roc[0]), sma[0])
        f2  = self._sma(self._roc(dimension, roc[1]), sma[1]) * 2
        f3  = self._sma(self._roc(dimension, roc[2]), sma[2]) * 3
        f4  = self._sma(self._roc(dimension, roc[3]), sma[3]) * 4
        kst = f1 + f2 + f3 + f4
        sig = self._sma(kst, sig)
        return kst, sig
