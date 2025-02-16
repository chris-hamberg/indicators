from indicators.base.core import CoreIndicators
import numpy as np


class QuantumVortex(CoreIndicators):
    """
    Quantum Vortex indicator based on the Alligator using EMA.

    This indicator is similar to the Alligator indicator but uses Exponential 
    Moving Averages (EMA) instead. It is designed to identify trends in the market.

    Parameters:
    - timeseries (pd.DataFrame or Stock): The input financial time series.
    - n (int or list or tuple or numpy.ndarray): If int, it represents the number 
    of Fibonacci numbers to use in the computation. If list, tuple, or ndarray, 
    it represents the specific Fibonacci numbers to use. If None, it uses a default 
    set of Fibonacci numbers (34, 55, 89).
    - dimension (str): The column name representing the dimension in the timeseries.

    Returns:
    numpy.ndarray: An array containing the Quantum Vortex values, which represent 
    the trends identified by the indicator.
    """
    def __init__(self):
        super().__init__()
        self.error = "parameter `n` argument must be greater than 5"


    def __call__(self, timeseries, n, dimension, _cls):
        series  = self._extractor.extract_dimension(timeseries, dimension)
        vectors = self.vectors(n, _cls)
        quantumVortex = np.full((series.shape[0], len(vectors)), np.nan)
        for dim, vector in enumerate(vectors):
            quantumVortex[:,dim] = self._ema(timeseries, vector, dimension)
        return quantumVortex


    def vectors(self, n, _cls):
        if isinstance(n,list) or isinstance(n,tuple) or isinstance(n,np.ndarray):
            vectors = n
        elif n is None: 
            vectors = np.array((34, 55, 89))
        elif isinstance(n, int):                               
            if n < 5: raise ValueError(self.error)
            elif len(_cls._fibonacci) != (n + 2):
                _cls._update_fibonacci(self, n)
            vectors = _cls._fibonacci[-3:]
        return vectors
