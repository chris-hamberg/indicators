from indicators.base.core import CoreIndicators
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class RelativeStrengthIndex(CoreIndicators):
    """
    Relative Strength Index (RSI) indicator with support for Adaptive RSI, 
    volatility weighting, and smoothing.

    The Relative Strength Index (RSI) is a momentum oscillator that measures the 
    speed and change of price movements. This implementation includes support for 
    Adaptive RSI, which adjusts the RSI calculation based on market volatility, 
    volatility weighting by variance, and smoothing using different methods.

    Parameters:
    - timeseries (pd.DataFrame, or Stock): Historical price data.
    - n (int): Number of periods to consider for the calculations.
    - dimension (str): Dimension of the timeseries to consider, e.g., "close" or "high".
    - adaptive (bool): Flag indicating whether to use Adaptive RSI.
    - weight (float): Weighting factor for volatility, 0.0 to 1.0 weight by variance.
    - smoothing (str): Smoothing method for the RSI, one of: "sma", "ema", "tema", or "wilders".

    Returns:
    np.ndarray: Array containing the RSI values.
    """
    def __init__(self):
        super().__init__()


    def __call__(self, timeseries, n, dimension, adaptive, weight, smoothing):
        
        # Traditional RSI
        dimension      = self._extractor.extract_dimension(timeseries, dimension)
        gain, loss     = self.mathA(dimension)
        muGain, muLoss = self._sma(gain, n), self._sma(loss, n)
        rsi            = self.mathB(muGain, muLoss)
        if not adaptive and not weight and not smoothing: return rsi
        
        # Used for enforcing threshold congruence.
        minima, maxima = np.nanmin(rsi), np.nanmax(rsi)
        scaler = MinMaxScaler(feature_range=(minima, maxima))
        scaler.fit(rsi.reshape(-1, 1))

        # Adaptive RSI
        if adaptive:
            muGain, muLoss = self.adapt(timeseries, n, gain, loss)
            rsi            = self.mathB(muGain, muLoss)

        # Weighted RSI
        rsi = self._std_weighted(timeseries, rsi, n, weight)

        # Apply smoothing.
        if smoothing: rsi = getattr(self, f"_{smoothing}")(rsi)

        # Enforce threshold congruence.
        #rsi = self.scaler(rsi, minima, maxima)
        rsi = scaler.transform(rsi.reshape(-1, 1)).flatten()

        return rsi


    def adapt(self, timeseries, n, gain, loss):
        matrix  = self._extractor.extract_matrix(timeseries)
        vectors = self._vectors(matrix, n)
        kws = {"dx": gain, "dy": loss, "vectors": vectors, "fx": np.mean}
        muGain, muLoss = self._bivariate_vector_space(**kws)
        return muGain, muLoss


    def mathA(self, dimension): 
        diff = np.diff(dimension)
        gain = np.maximum(0, diff)
        loss = np.abs(np.minimum(0, diff))
        gain = self._tools.resize(dimension, gain)
        loss = self._tools.resize(dimension, loss)
        return gain, loss


    def mathB(self, muGain, muLoss): 
        if not np.nansum(np.abs(muGain)) and not np.nansum(np.abs(muLoss)):
            return np.full_like(muGain, 50)
        rs  = self._tools.divide(muGain, muLoss)
        rsi = 100 - (100 / (1 + rs))
        return rsi


    def scaler(self, rsi, minima, maxima):
        rsi = self._tools.rescale(rsi, np.nanmin(rsi), np.nanmax(rsi))
        return rsi * (maxima - minima) + minima
