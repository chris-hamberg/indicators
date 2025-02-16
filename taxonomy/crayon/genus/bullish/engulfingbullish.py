import numpy as np


class EngulfingBullish:


    def __call__(self, metacrayon):
        prev_open      = np.hstack((metacrayon.open[1:],      (np.nan,)))
        prev_close     = np.hstack((metacrayon.close[1:],     (np.nan,)))
        prev_body_size = np.hstack((metacrayon.body_size[1:], (np.nan,)))

        # Downtrend?
        trend = metacrayon.trend < -0.2

        # First color is red?
        fcolor = prev_close < prev_open

        # Second color is green?
        scolor = metacrayon.open < metacrayon.close

        # Second is large?
        ssize = metacrayon.average_body_size < metacrayon.body_size

        # First is small?
        fsize = prev_body_size < metacrayon.average_body_size

        # Is engulfing?
        e1 = prev_open < metacrayon.close
        e2 = metacrayon.open < prev_close
        engulfing = (e1) & (e2)

        return np.where(
                (trend) & (fcolor) & (scolor) & (ssize) & (fsize) & (engulfing),
                1, 0)
