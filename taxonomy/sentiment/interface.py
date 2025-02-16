from indicators.taxonomy.sentiment.species.pwv import PriceWeightedVolume
from indicators.taxonomy.sentiment.species.prophecy import Prophecy
from indicators.taxonomy.sentiment.species.oracle import Oracle
from indicators.taxonomy.sentiment.species.tarot import Tarot
from indicators.utils.description import Indicator
from indicators.base.core import CoreIndicators


class Sentiment(CoreIndicators):


    def __init__(self):
        super().__init__()
        self._tarot    = Tarot()
        self._oracle   = Oracle()
        self._prophecy = Prophecy()
        self._pwv      = PriceWeightedVolume()


    @Indicator.description(Tarot)
    def tarot(self, timeseries, n=21, scale=1, smoothing=0, mode="sma"):
        return self._tarot(timeseries, n, scale, smoothing, mode)


    @Indicator.description(Oracle)
    def oracle(self, timeseries, n=21, scale=1, smoothing=0, mode="sma"):
        return self._oracle(timeseries, n, scale, smoothing, mode)


    @Indicator.description(Prophecy)
    def prophecy(self, timeseries, n=21, scale=1, smoothing=0, mode="sma"):
        return self._prophecy(timeseries, n, scale, smoothing, mode)
    

    @Indicator.description(PriceWeightedVolume)
    def pwv(self, timeseries, n=10): return self._pwv(timeseries, n)
