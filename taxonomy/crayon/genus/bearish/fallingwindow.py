import numpy as np


class FallingWindow:


    def __call__(self, metacrayon):

        # Pattern metaproperties
        low_shift_1 = np.hstack((metacrayon.low[1:], (np.nan,)))

        # Window gap
        window_gap = metacrayon.high < low_shift_1

        return np.where(window_gap, 1, 0)
