from indicators.taxonomy.crayon.genus.bullish.threewhitesoldiers import ThreeWhiteSoldiers
from indicators.taxonomy.crayon.genus.bullish.engulfingbullish import EngulfingBullish
from indicators.taxonomy.crayon.genus.bullish.longlowershadow import LongLowerShadow
from indicators.taxonomy.crayon.genus.bullish.biggreencrayon import BigGreenCrayon
from indicators.taxonomy.crayon.genus.bullish.bullishharami import BullishHarami
from indicators.taxonomy.crayon.genus.bullish.piercingline import PiercingLine
from indicators.taxonomy.crayon.genus.bullish.risingwindow import RisingWindow
from indicators.taxonomy.crayon.genus.bullish.dragonfly import DragonflyDoji
from indicators.taxonomy.crayon.genus.bullish.whitebody import WhiteBody
from indicators.taxonomy.crayon.genus.bullish.hammer import Hammer

from indicators.taxonomy.crayon.genus.bearish.threeblackcrows import ThreeBlackCrows
from indicators.taxonomy.crayon.genus.bearish.engulfingbearish import EngulfingBearish
from indicators.taxonomy.crayon.genus.bearish.darkcloudcover import DarkCloudCover
from indicators.taxonomy.crayon.genus.bearish.longuppershadow import LongUpperShadow
from indicators.taxonomy.crayon.genus.bearish.invertedhammer import InvertedHammer
from indicators.taxonomy.crayon.genus.bearish.fallingwindow import FallingWindow
from indicators.taxonomy.crayon.genus.bearish.bearishharami import BearishHarami
from indicators.taxonomy.crayon.genus.bearish.bigredcrayon import BigRedCrayon
from indicators.taxonomy.crayon.genus.bearish.shootingstar import ShootingStar
from indicators.taxonomy.crayon.genus.bearish.gravestone import GravestoneDoji
from indicators.taxonomy.crayon.genus.bearish.judascandle import JudasCandle
from indicators.taxonomy.crayon.genus.bearish.onneckline import OnNeckline
from indicators.taxonomy.crayon.genus.bearish.eveningstar import EveningStar
from indicators.taxonomy.crayon.genus.bearish.hangingman import HangingMan
from indicators.taxonomy.crayon.genus.bearish.blackbody import BlackBody

from indicators.taxonomy.crayon.core.metacrayon import MetaCrayon
from indicators.taxonomy.crayon.genus.doji import Doji
from indicators.taxonomy.levels.interface import Levels
from indicators.base.core import CoreIndicators
import pandas as pd
import numpy as np


class Crayon(CoreIndicators):


    configuration = {


            "bullish": {

                "Doji"              : True,

                "Dragonfly"         : True,

                "Hammer"            : True,

                "RisingWindow"      : True,

                "WhiteBody"         : False,

                "PiercingLine"      : True,

                "BigGreenCrayon"    : True,

                "BullishHarami"     : True,

                "LongLowerShadow"   : True,
                
                "EngulfingBullish"  : True,

                "ThreeWhiteSoldiers": True,

                },
            

            "bearish": {

                "Doji"            : True,

                "Gravestone"      : True,

                "HangingMan"      : True,

                "ShootingStar"    : True,

                "InvertedHammer"  : True,

                "FallingWindow"   : True,
                    
                "BlackBody"       : False,

                "BigRedCrayon"    : True,

                "BearishHarami"   : True,

                "EveningStar"     : True,

                "OnNeckline"      : True,

                "JudasCandle"     : True,

                "DarkCloudCover"  : True,

                "ThreeBlackCrows" : True,

                "LongUpperShadow" : True,

                "EngulfingBearish": True,

                }
                
            }


    def __init__(self):
        super().__init__()
        self._metacrayon         = MetaCrayon()
        self._levels             = Levels()

        # Patterns
        self._doji               = Doji()
        self._hammer             = Hammer()
        self._blackbody          = BlackBody()
        self._whitebody          = WhiteBody()
        self._eveningstar        = EveningStar()
        self._dragonfly          = DragonflyDoji()
        self._gravestone         = GravestoneDoji()
        self._hangingman         = HangingMan()
        self._onneckline         = OnNeckline()
        self._piercingline       = PiercingLine()
        self._bigredcrayon       = BigRedCrayon()
        self._biggreencrayon     = BigGreenCrayon()
        self._shootingstar       = ShootingStar()
        self._judascandle        = JudasCandle()
        self._risingwindow       = RisingWindow()
        self._fallingwindow      = FallingWindow()
        self._invertedhammer     = InvertedHammer()
        self._bearishharami      = BearishHarami()
        self._bullishharami      = BullishHarami()
        self._darkcloudcover     = DarkCloudCover()
        self._threeblackcrows    = ThreeBlackCrows()
        self._longuppershadow    = LongUpperShadow()
        self._longlowershadow    = LongLowerShadow()
        self._engulfingbearish   = EngulfingBearish()
        self._engulfingbullish   = EngulfingBullish()
        self._threewhitesoldiers = ThreeWhiteSoldiers()


    def detection(self, timeseries, n=55, scale=5, days=1, mode="binary"):
        bullish   = np.zeros(timeseries.shape[0])
        bearish   = np.zeros(timeseries.shape[0])
        bullkey = [k for k, v in Crayon.configuration["bullish"].items() if v]
        bearkey = [k for k, v in Crayon.configuration["bearish"].items() if v]
        self._metacrayon(timeseries, n, scale, days)
        
        for key in bearkey: 
            if key == "Doji":
                bearish += getattr(self, f"_{key}".lower())(self._metacrayon, "bearish")
            else:
                bearish += getattr(self, f"_{key}".lower())(self._metacrayon)

        for key in bullkey:
            if key == "Doji":
                bullish += getattr(self, f"_{key}".lower())(self._metacrayon, "bullish")
            else:
                bullish += getattr(self, f"_{key}".lower())(self._metacrayon)

        if mode == "binary":
            bullish = np.where(bullish != 0, 1, 0)
            bearish = np.where(bearish != 0, 1, 0)

        return np.array((bullish, bearish)).T
