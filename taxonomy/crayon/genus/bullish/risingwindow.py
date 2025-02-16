import numpy as np


class RisingWindow:


    def __call__(self, metacrayon):

        # Pattern metaproperties
        high_shift_1 = np.hstack((metacrayon.high[1:], (np.nan,)))

        # Window Gap
        window_gap = high_shift_1 < metacrayon.low

        return np.where(window_gap, 1, 0)
