import numpy as np


class LongLowerShadow:


    def __call__(self, metacrayon):

        # Support Levels
        S1, S2 = metacrayon.sr[:,0], metacrayon.sr[:,1]

        # Long lower Shadow
        lower_shadow = metacrayon.range_size * 2/3 <= metacrayon.lower_shadow

        # Near support
        multiplier = 1.5
        S1_upper_bound = S1 + (metacrayon.atr * multiplier)
        S1_lower_bound = S1 - (metacrayon.atr * multiplier)
        S2_upper_bound = S2 + (metacrayon.atr * multiplier)
        S2_lower_bound = S2 - (metacrayon.atr * multiplier)

        # Is close to support?
        C1 = (S1_lower_bound <= metacrayon.close) & (metacrayon.close <= S1_upper_bound)
        C2 = (S2_lower_bound <= metacrayon.close) & (metacrayon.close <= S2_upper_bound)
        support = np.where((C1) | (C2), True, False)

        return np.where((lower_shadow) & (support), 1, 0)
