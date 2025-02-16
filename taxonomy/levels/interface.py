from indicators.taxonomy.levels.species.rppoints import RollingPivotPoints
from indicators.taxonomy.levels.species.cppoints import CleanPivotPoints
from indicators.taxonomy.levels.species.ppoints import PivotPoints
from indicators.taxonomy.levels.species.ichimoku import IchimokuCloud
from indicators.taxonomy.levels.species.phibands import PhiBands
from indicators.utils.description import Indicator
from indicators.base.core import CoreIndicators


class Levels(CoreIndicators):


    def __init__(self):
        super().__init__()
        self._phiBands = PhiBands()
        self._ppoints  = PivotPoints()
        self._ichimoku = IchimokuCloud()
        self._ppoints  = PivotPoints()
        self._cppoints = CleanPivotPoints()
        self._rppoints = RollingPivotPoints()


    @Indicator.description(PhiBands)
    def phiBands(self, timeseries, n=200, dimension="Close", factor=3.0):
        return self._phiBands(timeseries, n, dimension, factor).T


    @Indicator.description(PivotPoints)
    def ppoints(self, timeseries):
        return self._ppoints(timeseries)


    @Indicator.description(RollingPivotPoints)
    def rppoints(self, timeseries, scale=5, days=1):
        return self._rppoints(timeseries, scale, days, pad=0)


    @Indicator.description(CleanPivotPoints)
    def cppoints(self, timeseries, scale=5, days=1, pad=42):
        pps = self._rppoints(timeseries, scale, days, pad)
        return self._cppoints(pps, pad)


    @Indicator.description(IchimokuCloud)
    def ichimoku(self, timeseries, tenkan=9, kijun=26):
        return self._ichimoku(timeseries, tenkan, kijun)
