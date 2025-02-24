from indicators.base.core import CoreIndicators
import numpy as np


class InteractiveCore(CoreIndicators):


    def __init__(self):
        super().__init__()

   
    def __repr__(self):
        return (f"\n  Class: {self.__class__.__name__}\n\n"
            f"  Methods: {self.cluster}")


    def sum(self, timeseries, n=14, dimension="Close"):
        return self._summation(timeseries, n, dimension)


    def min(self, timeseries, n=14, dimension="Close"):
        return self._minima(timeseries, n, dimension)


    def max(self, timeseries, n=14, dimension="Close"):
        return self._maxima(timeseries, n, dimension)


    def variance(self, timeseries, n=14, dimension="Close"):
        return self._variance(timeseries, n, dimension)


    def std(self, timeseries, n=14, dimension="Close"):
        variance = self._variance(timeseries, n, dimension)
        return np.sqrt(variance)


    def roc(self, timeseries, n=10, dimension="Close"):
        return self._roc(timeseries, n, dimension)


    def LogR(self, timeseries, dimension="Close"):
        return self._LogR(timeseries, dimension)


    def inverseLogR(self, LogR, initial_price):
        return self._inverseLogR(LogR, initial_price)


    def sma(self, timeseries, n=14, dimension="Close", adaptive=False):
        return self._sma(timeseries, n, dimension, adaptive)


    def smm(self, timeseries, n=14, dimension="Close"):
        return self._smm(timeseries, n, dimension)


    def ema(self, timeseries, n=14, dimension="Close", adaptive=False):
        return self._ema(timeseries, n, dimension, adaptive)


    def tema(self, timeseries, n=14, dimension="Close", adaptive=False):
        return self._tema(timeseries, n, dimension, adaptive)


    def wilders(self, timeseries, n=14, dimension="Close"):
        return self._wilders(timeseries, n, dimension)


    def tr(self, timeseries):
        return self._tr(timeseries)


    def atr(self, timeseries, n=14):
        return self._atr(timeseries, n)


    def mfm(self, timeseries, smoothing=0, mode="sma"):
        return self._mfm(timeseries, smoothing, mode)


    def er(self, timeseries, n=10, dimension="Close", smoothing=0, mode="ema"):
        return self._er(timeseries, n, dimension, smoothing, mode)
