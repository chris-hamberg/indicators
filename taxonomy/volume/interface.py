from indicators.taxonomy.volume.species.deltaflow import DeltaFlow
from indicators.taxonomy.volume.species.adosc import ADOscillator
from indicators.taxonomy.volume.species.ad import ChaikinAD
from indicators.taxonomy.volume.species.vwap import VWAP
from indicators.taxonomy.volume.species.rvol import RVOL
from indicators.taxonomy.volume.species.mfv import MFV
from indicators.taxonomy.volume.species.pvo import PVO
from indicators.utils.description import Indicator
from indicators.base.core import CoreIndicators


class Volume(CoreIndicators):


    def __init__(self):
        super().__init__()
        self._mfv       = MFV()
        self._pvo       = PVO()
        self._rvol      = RVOL()
        self._vwap      = VWAP()
        self._ad        = ChaikinAD()
        self._deltaflow = DeltaFlow()
        self._adosc     = ADOscillator()


    @Indicator.description(MFV)
    def mfv(self, timeseries, smoothing=21, mode="sma"):
        return self._mfv(timeseries, smoothing, mode)


    @Indicator.description(PVO)
    def pvo(self, timeseries, k=60):
        return self._pvo(timeseries, k)


    @Indicator.description(VWAP)
    def vwap(self, timeseries, n=20, mode="rolling"):
        return self._vwap(timeseries, n, mode)


    @Indicator.description(RVOL)
    def rvol(self, timeseries, n=10):
        return self._rvol(timeseries, n)


    @Indicator.description(ChaikinAD)
    def ad(self, timeseries):
        return self._ad(timeseries, self._mfv)


    @Indicator.description(ADOscillator)
    def adosc(self, timeseries, m=5, n=21, mode="sma"):
        return self._adosc(timeseries, m, n, mode, self.ad)


    @Indicator.description(DeltaFlow)
    def deltaFlow(self, timeseries, smoothing=390, mode="sma"):
        return self._deltaflow(timeseries, smoothing, mode)
