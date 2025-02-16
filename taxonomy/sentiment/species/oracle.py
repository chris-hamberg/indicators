from indicators.taxonomy.crayon.interface import Crayon
from indicators.base.core import CoreIndicators
import numpy as np


class Oracle(CoreIndicators):
    """
    The Oracle Indicator 

    The Oracle Indicator predicts market sentiment based on the balance between green and red 
    candle bodies.

    This leading indicator uses an algorithm to calculate the ratio between green and red candle 
    bodies, normalizing it to a value between -1 and 1. A negative value indicates a predominance 
    of red candles, while a positive value indicates a predominance of green candles. A value of 
    0 suggests an equal balance.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input time series data.
    - n (int): The window size for the calculation.
    - scale (int): The timescale of the data (default is 1 (1-min)).
    - smoothing (int): The amount of smoothing to apply (0 for none).
    - mode (str): The smoothing mode ('sma', 'ema', 'tema', or 'wilders').

    Returns:
    - numpy.ndarray: An array of normalized ratios representing market sentiment.

    Example:
    ```python
    sentiment = ix.sentiment.oracle(timeseries, n=78, smoothing=78, mode='ema')
    ```
    """
    def __init__(self):
        super().__init__()
        self.crayon = Crayon()


    def __call__(self, timeseries, n, scale, smoothing, mode):
        if mode not in CoreIndicators._modes: raise ValueError(self._error)
        pattern = self.pattern_detection(timeseries, n, scale)
        CR      = self.process(pattern, n)
        scaler  = self._tools.sklearn_minmaxscaler(CR)
        if smoothing: CR = getattr(self, f"_{mode}")(CR, smoothing)
        CR      = scaler.transform(CR.reshape(-1, 1)).flatten()
        return CR


    def pattern_detection(self, timeseries, n, scale):
        bullish_setting = self.crayon.configuration["bullish"]
        bearish_setting = self.crayon.configuration["bearish"]
        bullish = dict.fromkeys(self.crayon.configuration["bullish"], False)
        bearish = dict.fromkeys(self.crayon.configuration["bearish"], False)
        self.crayon.configuration["bullish"] = bullish
        self.crayon.configuration["bearish"] = bearish
        self.crayon.configuration["bullish"]["WhiteBody"] = True
        self.crayon.configuration["bearish"]["BlackBody"] = True
        pattern = self.crayon.detection(timeseries, n, scale)
        self.crayon.configuration["bullish"] = bullish_setting
        self.crayon.configuration["bearish"] = bearish_setting
        return pattern


    def process(self, pattern, n):
        imputed = np.where(np.isnan(pattern), 0, pattern)
        white   = self._summation(imputed[:,0], n)
        black   = self._summation(imputed[:,1], n)
        WBR     = self._tools.divide(white, black)
        BWR     = self._tools.divide(black, white)
        CR      = self._tools.divide((WBR - BWR), (WBR + BWR))
        return CR
