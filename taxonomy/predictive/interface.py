from indicators.taxonomy.predictive.species.recombinant import RecombinantR
from indicators.taxonomy.predictive.species.mcforecast import MCForecast
from indicators.utils.description import Indicator
from indicators.base.core import CoreIndicators


class Predictive(CoreIndicators):


    def __init__(self):
        super().__init__()
        self._mcforecast = MCForecast()
        self._recombinantR = RecombinantR()


    @Indicator.description(MCForecast)
    def MCForecast(self, timeseries, horizon=78, lookback=78, 
            dimension="Close", sims=50000, method="median", test=False): 
        return self._mcforecast(timeseries, horizon, lookback, dimension, sims, 
                method, test)


    @Indicator.description(RecombinantR)
    def recombinantR(self, timeseries, n=252, dimension="Close", degree=3, 
            s_weight=0.6, r_weight=0.3, d_weight=0.1):
        return self._recombinantR(timeseries, n, dimension, degree, s_weight, 
                r_weight, d_weight)
