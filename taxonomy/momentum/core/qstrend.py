import numpy as np


class SurgeTrend:
    

    @property
    def factor(self):
        return self._factor

    @factor.setter
    def factor(self, v):
        self._factor = v


    def __init__(self, factor=0):
        self._factor = factor


    def direction(self, direction):
        uptrend   = self.factor and 0 < np.nansum(direction)
        downtrend = self.factor and np.nansum(direction) < 0
        return uptrend, downtrend


    def strength(self, obv, i, n):
        subset = np.where(obv[i-n:i] != 0, obv[i-n:i], np.nan)
        if np.nansum(np.abs(subset)) != 0:
            strength = np.abs(obv[i]) / np.nansum(np.abs(subset))
        else:
            strength = np.nan
        return strength


    def update(self, uptrend, downtrend, strength, obv, i):
        if uptrend:     obv[i] *= self.factor + strength
        elif downtrend: obv[i] *= self.factor - strength
