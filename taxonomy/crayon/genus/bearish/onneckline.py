import numpy as np


class OnNeckline:


    def __call__(self, metacrayon):

        # Pattern metaproperties
        open_shift_2  = np.hstack((metacrayon.open[2:],                 2 * (np.nan,)))
        open_shift_1  = np.hstack((metacrayon.open[1:],                     (np.nan,)))
        close_shift_2 = np.hstack((metacrayon.close[2:],                2 * (np.nan,)))
        close_shift_1 = np.hstack((metacrayon.close[1:],                    (np.nan,)))
        low_shift_2   = np.hstack((metacrayon.low[2:],                  2 * (np.nan,)))
        low_shift_1   = np.hstack((metacrayon.low[1:],                      (np.nan,)))
        close_shift_1 = np.hstack((metacrayon.close[1:],                    (np.nan,)))
        open_shift_1  = np.hstack((metacrayon.open[1:],                     (np.nan,)))
        size_shift_1  = np.hstack((metacrayon.body_size[1:],                (np.nan,)))
        average_size_shift_1 = np.hstack((metacrayon.average_body_size[1:], (np.nan,)))

        # Trend
        trend = metacrayon.trend < -0.2

        # First is red
        color_shift_2 = close_shift_2 < open_shift_2

        # Second is green
        color_shift_1 = open_shift_1 < close_shift_1

        # Second is small
        size_shift_1  = size_shift_1 < average_size_shift_1

        # Second close near first low
        distance = np.abs(close_shift_1 - low_shift_2)
        distance = distance < metacrayon.body_size * 0.1

        # Penetrated Low
        penetrated = metacrayon.low < low_shift_1

        return np.where((trend) & (color_shift_2) & (color_shift_1) & (
            size_shift_1) & (distance) & (penetrated), 1, 0)
