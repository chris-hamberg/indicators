import numpy as np


class DarkCloudCover:


    def __call__(self, metacrayon):

        # Pattern metaproperties
        open_shift_1  = np.hstack((metacrayon.open[1:],      (np.nan,)))
        high_shift_1  = np.hstack((metacrayon.high[1:],      (np.nan,)))
        close_shift_1 = np.hstack((metacrayon.close[1:],     (np.nan,)))
        size_shift_1  = np.hstack((metacrayon.body_size[1:], (np.nan,)))

        # Trend
        trend = 0.2 < metacrayon.trend

        # First is green
        first_color = open_shift_1 < close_shift_1

        # Second is red
        second_color = metacrayon.close < metacrayon.open

        # First is large
        first_size = metacrayon.average_body_size < size_shift_1

        # Second opens above prev high
        open = high_shift_1 < metacrayon.open

        # Second closes within
        within = metacrayon.close < close_shift_1 * 0.75
        within = (within) & (open_shift_1 * 1.25 < metacrayon.close)

        return np.where((trend) & (first_color) & (second_color) & (
            first_size) & (open) & (within), 1, 0)
