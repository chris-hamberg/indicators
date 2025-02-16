import numpy as np


class ThreeWhiteSoldiers:


    def __call__(self, metacrayon):

        # Pattern metaproperties
        close_shift_2 = np.hstack((metacrayon.close[2:],     2 * (np.nan,)))
        close_shift_1 = np.hstack((metacrayon.close[1:],         (np.nan,)))
        high_shift_2  = np.hstack((metacrayon.high[2:],      2 * (np.nan,))) 
        high_shift_1  = np.hstack((metacrayon.high[1:],          (np.nan,)))
        size_shift_2  = np.hstack((metacrayon.body_size[2:], 2 * (np.nan,)))
        size_shift_1  = np.hstack((metacrayon.body_size[1:],     (np.nan,)))

        # Market bottom
        market_bottom = close_shift_2 == metacrayon.market_bottom

        # Consecutive Higher Closes
        shift_0 = metacrayon.close > close_shift_1
        shift_1 = close_shift_1 > close_shift_2
        higher_closes = (shift_0) & (shift_1)

        # Closes near highs?
        shift_0 = metacrayon.high - metacrayon.close < metacrayon.body_size * 0.1
        shift_1 = high_shift_1 - close_shift_1 < size_shift_1 * 0.1
        shift_2 = high_shift_2 - close_shift_2 < size_shift_2 * 0.1
        near_highs = (shift_0) & (shift_1) & (shift_2)

        return np.where((market_bottom) & (higher_closes) & (near_highs), 1, 0)
