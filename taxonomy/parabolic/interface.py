from indicators.taxonomy.parabolic.species.sar import ParabolicSAR
from indicators.utils.description import Indicator
from indicators.base.core import CoreIndicators


class Parabolic(CoreIndicators):


    def __init__(self):
        super().__init__()
        self._parabolicSAR = ParabolicSAR()


    @Indicator.description(ParabolicSAR)
    def sar(self, timeseries, AF=0.02, increment=0.02, limit=0.02):
        return self._parabolicSAR(timeseries, AF, increment, limit)
