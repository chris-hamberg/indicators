from indicators.taxonomy.pulsewave.species.heteroscedasticity import Heteroscedasticity
from indicators.taxonomy.pulsewave.species.stationarity import Stationarity
from indicators.taxonomy.pulsewave.species.kurtosis import Kurtosis
from indicators.taxonomy.pulsewave.species.entropy import Entropy
from indicators.taxonomy.pulsewave.species.skew import Skew
from indicators.utils.description import Indicator
from indicators.base.core import CoreIndicators


class Pulsewave(CoreIndicators):


    def __init__(self):
        super().__init__()
        self._skew               = Skew()
        self._entropy            = Entropy()
        self._kurtosis           = Kurtosis()
        self._stationarity       = Stationarity()
        self._heteroscedasticity = Heteroscedasticity()


    @Indicator.description(Entropy)
    def entropy(self, timeseries, n=34, dimension="Close", mode="rolling"):
        return self._entropy(timeseries, n, dimension, mode)


    @Indicator.description(Kurtosis)
    def kurtosis(self, timeseries, n=34, dimension="Volume", mode="rolling"):
        return self._kurtosis(timeseries, n, dimension, mode)


    @Indicator.description(Skew)
    def skew(self, timeseries, n=34, dimension="Volume", mode="rolling"):
        return self._skew(timeseries, n, dimension, mode)


    @Indicator.description(Stationarity)
    def stationarity(self, timeseries, dimension="Close"):
        return self._stationarity(timeseries, dimension)


    @Indicator.description(Heteroscedasticity)
    def heteroscedasticity(self, model):
        return self._heteroscedasticity(model)
