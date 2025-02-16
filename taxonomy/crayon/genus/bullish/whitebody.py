import numpy as np


class WhiteBody:


    def __call__(self, metacrayon):
        return np.where(metacrayon.open < metacrayon.close, 1, 0)
