import numpy as np


class BullishHarami:


    def __call__(self, metacrayon):

        # First candle properties.
        prev_body_size = np.hstack((metacrayon.body_size[1:], (np.nan,)))
        prev_open      = np.hstack((metacrayon.open[1:],      (np.nan,)))
        prev_close     = np.hstack((metacrayon.close[1:],     (np.nan,)))

        # Check for downtrend.
        trend = metacrayon.trend < -0.2

        # First is unusally large?
        size = 2 * metacrayon.average_body_size <= prev_body_size

        # First is red?
        fcolor = prev_open > prev_close

        # Second is green?
        scolor = metacrayon.close > metacrayon.open

        # Large followed by small?
        relation = metacrayon.body_size < prev_body_size

        # Small contained by large?
        harami = (metacrayon.open < prev_close) & (prev_open < metacrayon.close)

        return np.where(
            (trend) & (size) & (fcolor) & (scolor) & (relation) & (harami),
            1, 0)
