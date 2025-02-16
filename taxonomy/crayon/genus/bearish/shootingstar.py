import numpy as np


class ShootingStar:


    def __call__(self, metacrayon):

        # Trend
        trend = 0.2 < metacrayon.trend

        # Small body
        threshold = metacrayon.range_size * 0.1
        body_size = metacrayon.body_size <= threshold

        # Little or small lower shadow
        lower_shadow = metacrayon.lower_shadow <= metacrayon.body_size * 0.1

        # Long upper shadow
        upper_shadow = metacrayon.body_size * 2 <= metacrayon.upper_shadow

        return np.where(
                (trend) & (body_size) & (lower_shadow) & (upper_shadow),
                1, 0)
