import numpy as np


class HangingMan:


    def __call__(self, metacrayon):

        # Uptrend
        trend = 0.2 < metacrayon.trend

        # Small body
        threshold = metacrayon.range_size * 0.1
        body_size = metacrayon.body_size <= threshold

        # Little or no upper shadow
        upper_shadow = metacrayon.upper_shadow <= metacrayon.body_size * 0.1

        # Long lower shadow
        lower_shadow = metacrayon.body_size * 2 <= metacrayon.lower_shadow

        return np.where(
            (trend) & (body_size) & (upper_shadow) & (lower_shadow),
            1, 0)
