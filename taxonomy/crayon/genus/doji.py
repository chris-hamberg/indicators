import numpy as np


class Doji:


    def test(self, matrix, n, i, trend, sr, atr):
        open, high, low, close = matrix[i,1], matrix[i,2], matrix[i,3], matrix[i,4]
        trend = trend[i]

        # Rule out Dragonfly Doji
        assert open  != high
        assert close != high

        # Rule out Gravestone Doji
        assert open  != low
        assert close != low

        body  = np.abs(open - close)
        range = np.abs(high - low)
        threshold = range * 0.1

        assert body <= threshold

        if 0.5 < trend:    return "bearish"
        elif trend < -0.5: return "bullish"
        else:              assert False


    def __call__(self, metacrayon, sentiment):

        # Rule out Dragonfly
        dragonfly = metacrayon.open != metacrayon.high
        dragonfly = (dragonfly) & (metacrayon.close != metacrayon.high)
        dragonfly = ~dragonfly

        # Rule out Gravestone
        gravestone = metacrayon.open != metacrayon.low
        gravestone = (gravestone) & (metacrayon.close != metacrayon.low)
        gravestone = ~gravestone

        # Small body
        body_size = metacrayon.body_size <= metacrayon.range_size * 0.1

        if   sentiment == "bullish": 
            trend = metacrayon.trend < -0.3
        elif sentiment == "bearish":
            trend = 0.3 < metacrayon.trend
        else:
            trend = np.full(metacrayon.trend.shape[0], False)

        return np.where(~(dragonfly) & ~(gravestone) & (trend) & (body_size), 
                1, 0)
