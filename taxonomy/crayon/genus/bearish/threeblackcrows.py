import numpy as np


class ThreeBlackCrows:


    def __call__(self, metacrayon):

        # Pattern metaproperties
        close_shift_2 = np.hstack((metacrayon.close[2:],     2 * (np.nan,)))
        close_shift_1 = np.hstack((metacrayon.close[1:],         (np.nan,)))
        low_shift_2   = np.hstack((metacrayon.low[2:],       2 * (np.nan,))) 
        low_shift_1   = np.hstack((metacrayon.low[1:],           (np.nan,)))
        size_shift_2  = np.hstack((metacrayon.body_size[2:], 2 * (np.nan,)))
        size_shift_1  = np.hstack((metacrayon.body_size[1:],     (np.nan,)))

        # Market top
        market_top = close_shift_2 == metacrayon.market_top

        # Consecutive Lower Closes
        shift_0 = metacrayon.close < close_shift_1
        shift_1 = close_shift_1 < close_shift_2
        lower_closes = (shift_0) & (shift_1)

        # Closes near lows?
        shift_0 = metacrayon.close - metacrayon.low < metacrayon.body_size * 0.1
        shift_1 = close_shift_1 - low_shift_1 < size_shift_1 * 0.1
        shift_2 = close_shift_2 - low_shift_2 < size_shift_2 * 0.1
        near_lows = (shift_0) & (shift_1) & (shift_2)

        return np.where((market_top) & (lower_closes) & (near_lows), 1, 0)
