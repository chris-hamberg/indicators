from indicators.taxonomy.levels.interface import Levels
from indicators.base.core import CoreIndicators
import pandas as pd
import numpy as np


class MetaCrayon(CoreIndicators):

    
    def __init__(self):
        super().__init__()
        self.levels = Levels()


    def __call__(self, timeseries, n, scale, days):
        matrix     = self._extractor.extract_matrix(timeseries)
        timespace  = pd.Series(np.arange(timeseries.shape[0]))
        self.trend = pd.Series(matrix[:,4]).rolling(n).corr(timespace).values
        self.sr    = self.levels.rppoints(timeseries, scale, days)
        self.atr   = self._atr(timeseries, n)

        self.open, self.high = matrix[:,1], matrix[:,2]
        self.low, self.close = matrix[:,3], matrix[:,4]

        # Compute meta
        self.body_size         = np.abs(self.open - self.close)
        self.range_size        = np.abs(self.high - self.low)
        self.average_body_size = self._sma(self.body_size, n)
        self.std_body_size     = self._std(self.body_size, n)
        self.market_top        = self._maxima(self.close)
        self.market_bottom     = self._minima(self.close)


        A = np.where(self.open > self.close, self.open,  0)
        B = np.where(self.open < self.close, self.close, 0)
        subtrahend = A + B
        self.upper_shadow = self.high - subtrahend
        

        A = np.where(self.open < self.close, self.open,  0)
        B = np.where(self.open > self.close, self.close, 0)
        minuend = A + B
        self.lower_shadow = minuend - self.low
