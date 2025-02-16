import numpy as np


class JudasCandle:


    def __call__(self, metacrayon):

        # Pattern metaproperties
        open_shift_1  = np.hstack((metacrayon.open[1:],                     (np.nan,)))
        high_shift_1  = np.hstack((metacrayon.high[1:],                     (np.nan,)))
        low_shift_1   = np.hstack((metacrayon.low[1:],                      (np.nan,)))
        close_shift_1 = np.hstack((metacrayon.close[1:],                    (np.nan,)))
        size_shift_1  = np.hstack((metacrayon.body_size[1:],                (np.nan,)))
        average_size_shift_1 = np.hstack((metacrayon.average_body_size[1:], (np.nan,)))

        # First is red
        first_color = close_shift_1 < open_shift_1

        # First is large
        first_size = average_size_shift_1 < size_shift_1

        # Second is green
        second_color = metacrayon.open < metacrayon.close

        # Second is small
        second_size = metacrayon.body_size < metacrayon.average_body_size

        # Second lower shadow equals the first body size
        thres1 = metacrayon.lower_shadow <= size_shift_1 * 1.05
        thres2 = size_shift_1 * 0.95 <= metacrayon.lower_shadow
        equal  = (thres1) & (thres2)

        return np.where((first_color) & (second_color) & (first_size) & (
            second_size) & (equal), 1, 0)
