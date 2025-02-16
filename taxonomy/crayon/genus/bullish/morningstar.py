import numpy as np


class MorningStar:


    def __call__(self, metacrayon):

        # Pattern metaproperties
        open_shift_2         = np.hstack((metacrayon.open[2:],          2 * (np.nan,)))
        close_shift_2        = np.hstack((metacrayon.close[2:],         2 * (np.nan,)))
        close_shift_1        = np.hstack((metacrayon.close[1:],             (np.nan,)))
        open_shift_1         = np.hstack((metacrayon.open[1:],              (np.nan,)))
        body_size_shift_2    = np.hstack((metacrayon.body_size[2:],     2 * (np.nan,)))
        body_size_shift_1    = np.hstack((metacrayon.body_size[1:],         (np.nan,)))
        average_size_shift_2 = np.hstack((metacrayon.body_size[2:],     2 * (np.nan,)))
        average_size_shift_1 = np.hstack((metacrayon.average_body_size[1:], (np.nan,)))

        # At the bottom?
        market_bottom = close_shift_1 == metacrayon.market_bottom

        # Large black crayon
        large_black = average_size_shift_2 < body_size_shift_2

        # Small with gap below
        small_gap = body_size_shift_1 < average_size_shift_1
        small_gap = (small_gap) & (close_shift_1 > open_shift_1)

        # White close within Large black
        color  = metacrayon.close > metacrayon.open
        within = metacrayon.close > close_shift_2 * 0.75
        within = (within) & open_shift_2 * 1.25 > metacrayon.close
        star   = (color) & (within)

        return np.where(
                (market_bottom) & (large_black) & (small_gap) & (star),
                1, 0)

