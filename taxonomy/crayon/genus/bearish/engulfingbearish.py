import numpy as np


class EngulfingBearish:


    def __call__(self, metacrayon):
        
        prev_open      = np.hstack((metacrayon.open[1:],      (np.nan,)))
        prev_close     = np.hstack((metacrayon.close[1:],     (np.nan,)))
        prev_body_size = np.hstack((metacrayon.body_size[1:], (np.nan,)))

        trend = 0.2 < metacrayon.trend
        
        # First is green?
        fcolor = prev_open < prev_close
        
        # Second is red?
        scolor = metacrayon.close < metacrayon.open

        # Large second candle
        ssize = metacrayon.average_body_size < metacrayon.body_size

        # Small first candle
        fsize = prev_body_size < metacrayon.average_body_size

        # Engulfing?
        e1 = prev_close < metacrayon.open
        e2 = metacrayon.close < prev_open
        engulfing = (e1) & (e2)
        
        return np.where(
                (trend) & (fcolor) & (scolor) & (ssize) & (fsize) & (engulfing),
                1, 0)
