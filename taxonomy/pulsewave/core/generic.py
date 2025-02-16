from indicators.base.core import CoreIndicators
import numpy as np


class Generic(CoreIndicators):


    modes = ["flat", "rolling", "expanding"]


    def __init__(self):
        super().__init__()
        self.error = f"Mode must be one of: {Generic.modes}"


    def compute(self, dimension, n, function, mode):
        if mode not in Generic.modes: raise ValueError(self.error)
        elif mode == "rolling": 
            return self._windows.rolling_function(dimension, n, function)
        elif mode == "expanding":
            return self._windows.expanding_function(dimension, n, function)
        elif mode == "flat":
            return function(dimension)
