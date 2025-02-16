from indicators.taxonomy.temporal.species.sinusoidal import Sinusoidal
from indicators.taxonomy.temporal.species.density import Density
from indicators.utils.description import Indicator
from indicators.base.core import CoreIndicators


class Temporal(CoreIndicators):


    def __init__(self):
        super().__init__()
        self._density    = Density()
        self._sinusoidal = Sinusoidal()


    @Indicator.description(Density)
    def density(self, timeseries, n=60):
        return self._density(timeseries, n)


    @Indicator.description(Sinusoidal)
    def sinusoidal(self, timeseries, cycle=6.5, granularity=60, amplitude=1.0, 
            offset=0.0):
        return self._sinusoidal(timeseries, cycle, granularity, amplitude, 
                offset)
