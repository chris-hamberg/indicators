import numpy as np


class LongUpperShadow:


    def __call__(self, metacrayon):

        # Resistance Levels
        R1, R2 = metacrayon.sr[:,3], metacrayon.sr[:,4]

        # Long Upper Shadow
        upper_shadow = metacrayon.range_size * 2/3 <= metacrayon.upper_shadow

        # Near resistance
        multiplier = 1.5
        R1_upper_bound = R1 + (metacrayon.atr * multiplier)
        R1_lower_bound = R1 - (metacrayon.atr * multiplier)
        R2_upper_bound = R2 + (metacrayon.atr * multiplier)
        R2_lower_bound = R2 - (metacrayon.atr * multiplier)

        # Is close to resistance?
        C1 = (R1_lower_bound <= metacrayon.close) & (metacrayon.close <= R1_upper_bound)
        C2 = (R2_lower_bound <= metacrayon.close) & (metacrayon.close <= R2_upper_bound)
        resistance = np.where((C1) | (C2), True, False)

        return np.where((upper_shadow) & (resistance), 1, 0)
