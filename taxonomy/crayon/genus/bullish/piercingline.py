import numpy as np


class PiercingLine:


    def __call__(self, metacrayon):

        # Pattern metaproperties
        open_shift_1 = np.hstack((metacrayon.open[1:],      (np.nan,)))
        low_shift_1  = np.hstack((metacrayon.low[1:],       (np.nan,)))
        close_shift_1 = np.hstack((metacrayon.close[1:],    (np.nan,)))
        size_shift_1 = np.hstack((metacrayon.body_size[1:], (np.nan,)))

        # Market bottom
        bottom = metacrayon.close == metacrayon.market_bottom

        # First is red
        first_color = close_shift_1 < open_shift_1

        # Second is green
        second_color = metacrayon.open < metacrayon.close

        # Second open lower than first low
        second_low = metacrayon.open < low_shift_1

        # Second closes more than halfway into the body of the first
        second_close = (close_shift_1 + open_shift_1) / 2 < metacrayon.close

        return np.where((bottom) & (first_color) & (second_color) & (
            second_low) & (second_close), 1, 0)
