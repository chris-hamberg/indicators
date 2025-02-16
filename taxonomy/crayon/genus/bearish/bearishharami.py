import numpy as np


class BearishHarami:


    def __call__(self, metacrayon):

        # First candle properties
        prev_body_size = np.hstack((metacrayon.body_size[1:], (np.nan,)))
        prev_open      = np.hstack((metacrayon.open[1:], (np.nan,)))
        prev_close     = np.hstack((metacrayon.close[1:], (np.nan,)))

        # Check for uptrend
        trend = 0.2 < metacrayon.trend

        # First is unusally large?
        size = 2 * metacrayon.average_body_size <= prev_body_size

        # First is green
        fcolor = prev_open < prev_close

        # Second is red
        scolor = metacrayon.close < metacrayon.open

        # Large followed by small
        relation = metacrayon.body_size < prev_body_size

        # Small contained by large
        harami = np.where((
            metacrayon.open < prev_close) & (prev_open < metacrayon.close),
            True, False)

        return np.where((trend) & (size) & (fcolor) & (scolor) & (relation) & 
                (harami), 1, 0)
