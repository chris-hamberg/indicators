from indicators.taxonomy.dsystem.species.abands import ATRBands
from indicators.taxonomy.dsystem.species.adx import ADX
from indicators.taxonomy.dsystem.species.dm import DM
from indicators.taxonomy.dsystem.species.di import DI
from indicators.taxonomy.dsystem.species.dx import DX
from indicators.utils.description import Indicator
from indicators.base.core import CoreIndicators


class Dsystem(CoreIndicators):


    def __init__(self):
        super().__init__()
        # tr, and atr are provided by CoreIndicators
        self._atrBands = ATRBands()
        self._adx      = ADX()
        self._dm       = DM()
        self._di       = DI()
        self._dx       = DX()


    @Indicator.description(CoreIndicators._tr)
    def tr(self, timeseries): 
        return self._tr(timeseries)


    @Indicator.description(CoreIndicators._atr)
    def atr(self, timeseries, n=20): 
        return self._atr(timeseries, n)


    @Indicator.description(DM)
    def dm(self, timeseries, n=14): 
        return self._dm(timeseries, n)


    @Indicator.description(DI)
    def di(self, timeseries, n=14):  
        pdm, ndm = self.dm(timeseries, n)
        return self._di(timeseries, pdm, ndm, n)


    @Indicator.description(DX)
    def dx(self, timeseries, n=14):
        pdi, ndi = self.di(timeseries, n)
        return self._dx(timeseries, pdi, ndi)


    @Indicator.description(ADX)
    def adx(self, timeseries, n=14):
        dx = self.dx(timeseries, n)
        return self._adx(dx, n)


    @Indicator.description(ATRBands)
    def atrBands(self, timeseries, n=14, k=3):
        atr = self.atr(timeseries, n)
        return self._atrBands(timeseries, atr, n, k)
