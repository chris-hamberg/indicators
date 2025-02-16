from indicators.taxonomy.volatility.species.quarkspectrum import QuarkSpectrum
from indicators.taxonomy.volatility.species.dimensions import Dimensions
from indicators.taxonomy.volatility.species.coeffvar import Coeffvar
from indicators.taxonomy.volatility.species.clusters import Clusters
from indicators.taxonomy.volatility.species.chaikin import Chaikin
from indicators.taxonomy.volatility.species.bbands import BBands
from indicators.taxonomy.volatility.species.bbandw import BBandW
from indicators.taxonomy.volatility.species.garch import GARCH
from indicators.taxonomy.volatility.species.beta import Beta
from indicators.utils.description import Indicator
from indicators.base.core import CoreIndicators
import numpy as np


class Volatility(CoreIndicators):


    def __init__(self):
        super().__init__()
        # variance, and std are provided by CoreIndicators
        self._beta          = Beta()
        self._garch         = GARCH()
        self._bbands        = BBands()
        self._bbandw        = BBandW()
        self._chaikin       = Chaikin()
        self._clusters      = Clusters()
        self._coeffvar      = Coeffvar()
        self._dimensions    = Dimensions()
        self._quarkSpectrum = QuarkSpectrum()


    @Indicator.description(CoreIndicators._variance)
    def variance(self, timeseries, n=14, dimension="Close"):
        return self._variance(timeseries, n, dimension)


    @Indicator.description(CoreIndicators._std)
    def std(self, timeseries, n=14, dimension="Close"):
        return self._std(timeseries, n, dimension)


    @Indicator.description(Beta)
    def beta(self, timeseriesA, timeseriesB, n=78, dimensionA="Close", 
            dimensionB=None, LogR=True):
        return self._beta(timeseriesA, timeseriesB, n, dimensionA, dimensionB,
                LogR)


    @Indicator.description(BBands)
    def bbands(self, timeseries, n=20, k=2, dimension="Close"):
        return self._bbands(timeseries, n, k, dimension).T


    @Indicator.description(BBandW)
    def bbandw(self, timeseries, n=20, k=2, dimension="Close"):
        bands = self._bbands(timeseries, n, k, dimension)
        return self._bbandw(bands)


    @Indicator.description(Chaikin)
    def chaikin(self, timeseries, n=10):
        return self._chaikin(timeseries, n)


    @Indicator.description(Coeffvar)
    def cv(self, timeseries, n=34, dimension="Close", mode="rolling"):
        return self._coeffvar(timeseries, n, dimension, mode)


    @Indicator.description(QuarkSpectrum)
    def quarkSpectrum(self, timeseries, n=13, dimension="Close"):
        return self._quarkSpectrum(timeseries, n, dimension)


    @Indicator.description(GARCH)
    def garch(self, timeseries, n=20, dimension="Close", sims=1000, LogR=True):
        return self._garch(timeseries, n, dimension, sims, LogR)


    @Indicator.description(Dimensions)
    def dimensions(self, timeseries, n=377, dimension="Volume", mode="flat"):
        return self._dimensions(timeseries, n, dimension, mode)


    @Indicator.description(Clusters)
    def clusters(self, timeseries, k=2, dimension="Volume"):
        return self._clusters(timeseries, k, dimension)
