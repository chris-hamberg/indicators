from indicators.taxonomy.trend.species.resonantvelocity import ResonantTrendVelocity
from indicators.taxonomy.trend.species.quantumvortex import QuantumVortex
from indicators.taxonomy.trend.species.isothermal import IsothermalSlopes
from indicators.taxonomy.trend.species.gradientflux import GradientFlux
from indicators.taxonomy.trend.species.aroonosc import AROONOscillator
from indicators.taxonomy.trend.species.phitensor import PhiTensor
from indicators.taxonomy.trend.species.phiwave import PhiWave
from indicators.taxonomy.trend.species.slope import Slope
from indicators.taxonomy.trend.species.aroon import AROON
from indicators.taxonomy.trend.species.hurst import Hurst
from indicators.taxonomy.trend.species.vwma import VWMA
from indicators.taxonomy.trend.species.kama import KAMA
from indicators.taxonomy.trend.species.mode import Mode
from indicators.taxonomy.trend.species.hma import HMA
from indicators.taxonomy.trend.species.wma import WMA
from indicators.utils.description import Indicator
from indicators.base.core import CoreIndicators


class Trend(CoreIndicators):


    def __init__(self):
        super().__init__()
        # sma, ema, tema, and Wilder's are provided by CoreIndicators
        self._wma                   = WMA()
        self._hma                   = HMA()
        self._vwma                  = VWMA()
        self._kama                  = KAMA()
        self._mode                  = Mode()
        self._hurst                 = Hurst()
        self._aroon                 = AROON()
        self._aroonosc              = AROONOscillator()
        self._slope                 = Slope()
        self._phiWave               = PhiWave()
        self._phiTensor             = PhiTensor()
        self._gradientFlux          = GradientFlux()
        self._quantumVortex         = QuantumVortex()
        self._isothermalSlopes      = IsothermalSlopes()
        self._resonantTrendVelocity = ResonantTrendVelocity()


    @Indicator.description(CoreIndicators._sma)
    def sma(self, timeseries, n=14, dimension="Close", adaptive=False):        
        return self._sma(timeseries, n, dimension, adaptive)       


    @Indicators.description(CoreIndicators._smm)
    def smm(self, timeseries, n=14, dimension="Close"):
        return self._smm(timeseries, n, dimension, adaptive)


    @Indicator.description(Mode)
    def mode(self, timeseries, n=14, dimension="Close"):
        return self._mode(timeseries, n, dimension)             


    @Indicator.description(CoreIndicators._ema)
    def ema(self, timeseries, n=14, dimension="Close", adaptive=False):       
        return self._ema(timeseries, n, dimension, adaptive, None)   
 
 
    @Indicator.description(CoreIndicators._tema)
    def tema(self, timeseries, n=14, dimension="Close", adaptive=False):
        return self._tema(timeseries, n, dimension, adaptive)


    @Indicator.description(CoreIndicators._wilders)
    def wilders(self, timeseries, n=14, dimension="Close"):
        return self._wilders(timeseries, n, dimension)


    @Indicator.description(WMA)
    def wma(self, timeseries, n=14, dimension="Close"):
        return self._wma(timeseries, n, dimension)


    @Indicator.description(HMA)
    def hma(self, timeseries, n=14, dimension="Close"):
        return self._hma(timeseries, n, dimension)


    @Indicator.description(VWMA)
    def vwma(self, timeseries, n=3): return self._vwma(timeseries, n)


    @Indicator.description(KAMA)
    def kama(self, timeseries, n=10, dimension="Close", fast=2, slow=30):
        return self._kama(timeseries, n, dimension, fast, slow)


    @Indicator.description(Slope)
    def slope(self, timeseries, n=5, dimension="Close"): 
        return self._slope(timeseries, n, dimension)


    @Indicator.description(PhiWave)
    def phiWave(self, timeseries, n=13, dimension="Close"):
        return self._phiWave(timeseries, n, dimension, self)


    @Indicator.description(PhiTensor)
    def phiTensor(self, timeseries, n=8, dimension="Close"):
        return self._phiTensor(timeseries, n, dimension, self)


    @Indicator.description(GradientFlux)
    def gradientFlux(self, timeseries, n=78, dimension="Close", mode="linear"):
        return self._gradientFlux(timeseries, n, dimension, mode)


    @Indicator.description(IsothermalSlopes)
    def isothermalSlopes(self, timeseries, n=5, dimension="Close", clip=2, 
            smoothing=252, mode="sma"):
        slope = self.slope(timeseries, n, dimension)
        return self._isothermalSlopes(slope, clip, smoothing, mode)


    @Indicator.description(QuantumVortex)
    def quantumVortex(self, timeseries, n=None, dimension="Close"):
        return self._quantumVortex(timeseries, n, dimension, self)
    
   
    @Indicator.description(ResonantTrendVelocity)
    def resonantTrendVelocity(self, timeseries, n=377, dimension="Close", 
            y=None, y_dimension=None, smoothing=377, mode="sma"):
        return self._resonantTrendVelocity(timeseries, n, dimension, y, 
                y_dimension, smoothing, mode)


    @Indicator.description(AROON)
    def aroon(self, timeseries, n=21):
        return self._aroon(timeseries, n)


    @Indicator.description(AROONOscillator)
    def aroonosc(self, timeseries, n=21):
        aroon = self._aroon(timeseries, n).T
        return self._aroonosc(aroon[0], aroon[1])


    @Indicator.description(Hurst)
    def hex(self, timeseries, n=377, dimension="Close", mode="rolling"):
        return self._hurst(timeseries, n, dimension, mode)
