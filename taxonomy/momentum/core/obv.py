from indicators.base.core import CoreIndicators
import numpy as np


class CoreOBV(CoreIndicators):


    def compute(self, matrix, weight, mode):
        positive = np.where(np.diff(matrix[:,4]) > 0,  1, 0)
        negative = np.where(np.diff(matrix[:,4]) < 0, -1, 0)
        trend    = (positive + negative)
        obv      = matrix[1:,5] * trend
        if mode == "flat": obv = np.cumsum(obv)
        obv = self._weighted(obv, matrix[1:,5], weight)
        if mode == "rolling": obv = np.nansum(obv)
        return obv, trend
