from indicators.taxonomy.signal.species.seismology import Seismology
from indicators.taxonomy.signal.species.specter import WaveSpecter
from indicators.taxonomy.signal.species.denoiser import Denoiser
from indicators.taxonomy.signal.species.wavelet import Wavelet
from indicators.taxonomy.signal.species.txdecay import txDecay
from indicators.taxonomy.signal.species.dxdecay import dxDecay
from indicators.taxonomy.signal.species.xdecay import xDecay
from indicators.utils.description import Indicator
from indicators.base.core import CoreIndicators


class Signal(CoreIndicators):


    def __init__(self):
        super().__init__()
        # LogR and inverseLogR are provided by CoreIndicators
        self._xdecay     = xDecay()
        self._dxdecay    = dxDecay()
        self._txdecay    = txDecay()
        self._wavelet    = Wavelet()
        self._denoiser   = Denoiser()
        self._specter    = WaveSpecter()
        self._seismology = Seismology()


    @Indicator.description(CoreIndicators._LogR)
    def LogR(self, timeseries, dimension="Close"):
        return self._LogR(timeseries, dimension)


    @Indicator.description(CoreIndicators._inverseLogR)
    def inverseLogR(self, LogR, initial_price):
        return self._inverseLogR(LogR, initial_price)


    @Indicator.description(Denoiser)
    def denoiser(self, timeseries, n=10, dimension="Close"):
        return self._denoiser(timeseries, n, dimension)


    @Indicator.description(Seismology)
    def seismology(self, timeseries, n=60, dimension="Volume"):
        return self._seismology(timeseries, n, dimension)


    @Indicator.description(WaveSpecter)
    def waveSpecter(self, timeseries, dimension="Volume", wavelet="cmor1.5-1.0", 
            cmap="magma", extent=None):
        return self._specter(timeseries, dimension, wavelet, cmap, extent)


    @Indicator.description(Wavelet)
    def wavelet(self, timeseries, dimension="Close", wavelet="db6", level=5):
        return self._wavelet(timeseries, dimension, wavelet, level)


    @Indicator.description(xDecay)
    def xDecay(self, timeseries, dimension="Close", alpha=0.2):
        return self._xdecay(timeseries, dimension, alpha)


    @Indicator.description(dxDecay)
    def dxDecay(self, timeseries, dimension="Close", alpha=0.005, beta=0.005):
        return self._dxdecay(timeseries, dimension, alpha, beta)


    @Indicator.description(txDecay)
    def txDecay(self, timeseries, dimension="Close", alpha=0.15, beta=0.2,
            gamma=0.2, seasonality=13):
        return self._txdecay(timeseries, dimension, alpha, beta, gamma,
                seasonality)
