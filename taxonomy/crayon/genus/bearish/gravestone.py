import numpy as np


class GravestoneDoji:


    def __call__(self, metacrayon):

        # Market Top
        top = metacrayon.close == metacrayon.market_top

        # Gravestone
        shadow_ratio = 2
        shape = np.where(
            (metacrayon.open == metacrayon.high) | (metacrayon.close == metacrayon.high), 
            True, False)
        shadow = metacrayon.upper_shadow <= metacrayon.body_size * shadow_ratio

        # Doji
        threshold = metacrayon.range_size * 0.1
        doji = metacrayon.body_size <= threshold

        return np.where((top) & (shape) & (shadow) & (doji), 1, 0)
