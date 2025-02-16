from indicators.taxonomy.momentum.species.ppo import PercentagePriceOscillator
from indicators.taxonomy.momentum.species.apo import AbsolutePriceOscillator
from indicators.taxonomy.momentum.species.quantumsurge import QuantumSurge
from indicators.taxonomy.momentum.species.cci import CommodityChannelIndex
from indicators.taxonomy.momentum.species.rsi import RelativeStrengthIndex
from indicators.taxonomy.momentum.species.rvi import RelativeVigorIndex
from indicators.taxonomy.momentum.species.stochastic import Stochastic
from indicators.taxonomy.momentum.species.chande import ChandeMomentum
from indicators.taxonomy.momentum.species.obv import OnBalancedVolume
from indicators.taxonomy.momentum.species.bop import BalanceOfPower
from indicators.taxonomy.momentum.species.kst import KnowSureThing
from indicators.taxonomy.momentum.species.macd import MACD
from indicators.taxonomy.momentum.species.mom import MOM
from indicators.utils.description import Indicator
from indicators.base.core import CoreIndicators
import numpy as np


class Momentum(CoreIndicators):


    def __init__(self):
        super().__init__()
        # ROC, MFM, and ER are provided by CoreIndicators
        self._mom  = MOM()
        self._macd = MACD()
        self._kst  = KnowSureThing()
        self._bop  = BalanceOfPower()
        self._obv  = OnBalancedVolume()
        self._stochastic = Stochastic()
        self._chande = ChandeMomentum()
        self._rvi = RelativeVigorIndex()
        self._rsi = RelativeStrengthIndex()
        self._cci = CommodityChannelIndex()
        self._quantumSurge = QuantumSurge()
        self._apo = AbsolutePriceOscillator()
        self._ppo = PercentagePriceOscillator()


    @Indicator.description(CoreIndicators._er)
    def er(self, timeseries, n=10, dimension="Close", smoothing=0, mode="ema"):
        return self._er(timeseries, n, dimension, smoothing, mode)


    @Indicator.description(OnBalancedVolume)
    def obv(self, timeseries, weight=0.0, normalize=True):
        return self._obv(timeseries, weight, normalize)


    @Indicator.description(RelativeStrengthIndex)
    def rsi(self, timeseries, n=14, dimension="Close", adaptive=False, 
            weight=0, smoothing=None):
        return self._rsi(timeseries, n, dimension, adaptive, weight, smoothing)


    @Indicator.description(MOM)
    def mom(self, timeseries, n=60, dimension="Close", smoothing=0, mode="ema"):
        return self._mom(timeseries, n, dimension, smoothing, mode)


    @Indicator.description(MACD)
    def macd(self, timeseries, short=12, long=26, signal=9, dimension="Close"):
        return self._macd(timeseries, short, long, signal, dimension)


    @Indicator.description(PercentagePriceOscillator)
    def ppo(self, timeseries, short=12, long=26, signal=9, dimension="Close", 
            mode="ema"):
        return self._ppo(timeseries, short, long, signal, dimension, mode)


    @Indicator.description(Stochastic)
    def stochastic(self, timeseries, n=14, adaptive=False, weight=0, 
            dsmoothing=0, ksmoothing=14, kmode="tema", dmode="sma"):
        return self._stochastic(timeseries, n, adaptive, weight, dsmoothing, 
                ksmoothing, kmode, dmode)


    @Indicator.description(CommodityChannelIndex)
    def cci(self, timeseries, n=14, smoothing=0, mode="ema"):
        return self._cci(timeseries, n, smoothing, mode)


    @Indicator.description(CoreIndicators._roc)
    def roc(self, timeseries, n=10, dimension="Close"):
        return self._roc(timeseries, n, dimension)


    @Indicator.description(CoreIndicators._mfm)
    def mfm(self, timeseries, smoothing=0, mode="sma"):
        return self._mfm(timeseries, smoothing, mode)


    @Indicator.description(RelativeVigorIndex)
    def rvi(self, timeseries, n=14, divisor=6):
        return self._rvi(timeseries, n, divisor)


    @Indicator.description(ChandeMomentum)
    def cmo(self, timeseries, n=14, dimension="Close"):
        return self._chande(timeseries, n, dimension)


    @Indicator.description(KnowSureThing)
    def kst(self, timeseries, dimension="Close", **kwargs):
        return self._kst(timeseries, dimension, **kwargs)


    @Indicator.description(AbsolutePriceOscillator)
    def apo(self, timeseries, m=13, n=55, dimension="Close", mode="sma"):
        return self._apo(timeseries, m, n, dimension, mode)


    @Indicator.description(QuantumSurge)
    def quantumSurge(self, timeseries, n=34, trend_factor=1, weight=0.5, 
            normalize=True):
        return self._quantumSurge(timeseries, n, trend_factor, weight, 
                normalize)


    @Indicator.description(BalanceOfPower)
    def bop(self, timeseries, smoothing=0, mode="sma"):
        return self._bop(timeseries, smoothing, mode)
