from indicators.base.core import CoreIndicators


class ADOscillator(CoreIndicators):
    """
    Accumulation/Distribution Oscillator (AD Oscillator).

    The AD Oscillator is a volume-based indicator that measures the difference between
    a fast and a slow moving average of the Accumulation/Distribution (AD) Line. It helps
    traders identify potential trends and reversals based on changes in buying and selling
    pressure indicated by volume.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input financial time series.
    - m (int): The number of periods for the fast moving average.
    - n (int): The number of periods for the slow moving average.
    - mode (str): The smoothing mode for the moving averages ('sma', 'ema', 'tema').

    Returns:
    numpy.ndarray: An array containing the values of the AD Oscillator.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, m, n, mode, _ad):
        if mode not in CoreIndicators._modes: raise ValueError(self._error)
        matrix = self._extractor.extract_matrix(timeseries)
        ad     = _ad(timeseries)
        fast   = getattr(self, f"_{mode}")(ad, m)
        slow   = getattr(self, f"_{mode}")(ad, n)
        return fast - slow
