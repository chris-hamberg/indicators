from indicators.base.core import CoreIndicators
import numpy as np


class Stochastic(CoreIndicators):
    """
    Stochastic Oscillator with ATR adaptability, volatility weighting, and 
    %K line smoothing.

    The Stochastic Oscillator is a momentum indicator that compares a security's 
    closing price to its price range over a specific period. This implementation 
    includes support for ATR adaptability, which adjusts the Stochastic Oscillator 
    calculation based on market volatility, volatility weighting by variance, and 
    smoothing of the %K line.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): Historical price data.
    - n (int): Number of periods to consider for the calculations.
    - adaptive (bool): Flag indicating whether to use ATR adaptability.
    - weight (float): Weighting factor for volatility, ranging from 0.0 to 1.0.
    - ksmoothing (int): Smoothing parameter for the %K line.
    - dsmoothing (int): Smoothing parameter for the %D line.
    - kmode (str): Smoothing mode for the %K line, one of: "sma", "ema", "tema".
    - dmode (str): Smoothing mode for the %D line, one of: "sma", "ema", "tema".

    Returns:
    (np.ndarray, np.ndarray): Tuple containing the %K line values and the %D line values.
    """
    def __init__(self):
        super().__init__()
        self.kerror = f"`kmode` must be one of: {CoreIndicators._modes}"
        self.derror = f"`dmode` must be one of: {CoreIndicators._modes}"


    def __call__(self, timeseries, n, adaptive, weight, ksmoothing, dsmoothing,
            kmode, dmode):

        if kmode not in CoreIndicators._modes: raise ValueError(self.kerror)
        if dmode not in CoreIndicators._modes: raise ValueError(self.derror)

        matrix = self._extractor.extract_matrix(timeseries)
        high, low, close = matrix[:,2], matrix[:,3], matrix[:,4]
        
        # Adaptive (or classic) Stochastic Oscillator
        if adaptive: highest, lowest = self.adapt(matrix, high, low, n)
        else:        highest, lowest = self._maxima(high,n), self._minima(low,n)
        K = self._tools.divide(close - lowest, highest - lowest) * 100

        # Volatility Weighted
        K = self._std_weighted(timeseries, K, n, weight)

        # %K, %D Smoothing
        if ksmoothing: K = getattr(self, f"_{kmode}")(K, ksmoothing)
        D = getattr(self, f"_{dmode}")(K, dsmoothing)
        
        # Enforce correct scaling.
        if adaptive or weight:  
            K, D = self._tools.rescale(K), self._tools.rescale(D)

        return K, D


    def adapt(self, matrix, high, low, n):
        vectors = self._vectors(matrix, n)
        args    = [high, low, vectors, np.nanmax, np.nanmin]
        highest, lowest = self._bivariate_vector_space(*args) 
        return highest, lowest
