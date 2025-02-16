import numpy as np


class BlackBody:


    def __call__(self, metacrayon):
        return np.where(metacrayon.close < metacrayon.open, 1, 0)
