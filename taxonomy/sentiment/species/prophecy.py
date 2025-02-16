from indicators.taxonomy.crayon.interface import Crayon
from indicators.base.core import CoreIndicators
import numpy as np


class Prophecy(CoreIndicators):
    """
    Prophecy Indicator 

    A predictive sentiment indicator that combines bullish and bearish candlestick 
    patterns.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input timeseries data.
    - n (int): The window size for computing the indicator.
    - scale (int): The timescale, in minutes, of the data (default is 1 (for 1-min)).
    - smoothing (int): 0 for no smoothing, for smoothing the indicator output.
    - mode (str): The smoothing mode ('sma', 'ema', 'tema', or 'wilders').

    Returns:
    - np.array: the normalized predictive sentiment indicator values.
    """
    def __init__(self):
        super().__init__()
        self.crayon = Crayon()

        # Pattern imbalance normalizers
        self.bullish = 10
        self.bearish = 15


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
        bullish = dict.fromkeys(self.crayon.configuration["bullish"], True)
        bearish = dict.fromkeys(self.crayon.configuration["bearish"], True)
        self.crayon.configuration["bullish"] = bullish
        self.crayon.configuration["bearish"] = bearish
        self.crayon.configuration["bullish"]["WhiteBody"] = False
        self.crayon.configuration["bearish"]["BlackBody"] = False
        pattern = self.crayon.detection(timeseries, n, scale)
        self.crayon.configuration["bullish"] = bullish_setting
        self.crayon.configuration["bearish"] = bearish_setting
        return pattern


    def process(self, pattern, n):
        imputed = np.where(np.isnan(pattern), 0, pattern)
        white   = self._summation(imputed[:,0], n)
        black   = self._summation(imputed[:,1], n)
        white  /= self.bullish
        black  /= self.bearish
        black  *= -1
        CR      = white + black
        return CR
