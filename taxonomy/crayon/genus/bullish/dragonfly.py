import numpy as np


class DragonflyDoji:


    def __call__(self, metacrayon):

        # Market Bottom
        bottom = metacrayon.close == metacrayon.market_bottom

        # Dragonfly
        shadow_ratio = 2
        shape = np.where(
            (metacrayon.open == metacrayon.high) | (metacrayon.close == metacrayon.high),
            True, False)
        shadow = metacrayon.body_size * shadow_ratio <= metacrayon.lower_shadow 

        # Doji
        doji = metacrayon.body_size <= metacrayon.range_size * 0.1

        return np.where((bottom) & (shape) & (shadow) & (doji), 1, 0)
